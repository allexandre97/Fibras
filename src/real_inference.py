import json
import os
import time
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Dict, List, Optional, Sequence

import numpy as np
import scipy.ndimage as ndi
import tifffile
import torch

try:
    from skimage.morphology import skeletonize
except Exception:  # pragma: no cover - exercised only when optional deps are missing.
    skeletonize = None

from src.inference_utils import normalize_image_percentile, predict_tiled_2d
from src.model import STEDResUNet2D
from src.sted import orientation_confidence_np, orientation_to_vector_map_np
from src.sted_calibration import (
    COMPARISON_METRICS,
    compute_image_metrics,
    load_calibration_profile,
    parse_sted_filename,
)
from src.tracking import StreamlineTracker
from src.visualization import AdvancedVisualizer


OUTPUT_SUFFIXES = {
    "skeleton": "_skeleton.tif",
    "pred_edt": "_pred_edt.tif",
    "pred_vis": "_pred_vis.tif",
    "pred_orient_conf": "_pred_orient_conf.tif",
    "preview": "_preview.png",
    "summary": "_summary.json",
}


@dataclass
class RealInferenceResult:
    image_path: str
    image: np.ndarray
    image_normalized: np.ndarray
    edt_map: np.ndarray
    orientation_map: np.ndarray
    vector_map: np.ndarray
    orientation_confidence: np.ndarray
    visibility_map: np.ndarray
    edt_for_tracking: np.ndarray
    seed_mask: np.ndarray
    skeleton: np.ndarray
    streamlines: List[np.ndarray]
    prediction_seconds: float
    tracking_seconds: float
    total_seconds: float
    tile_size: int
    tile_overlap: int
    min_edt: float
    visibility_floor: float
    used_amp: bool


def add_inference_arguments(
    parser,
    include_image_path: bool = True,
    include_output_dir: bool = True,
    include_visualize: bool = True,
    include_preview_options: bool = True,
):
    parser.add_argument("--model_path", type=str, required=True)
    if include_image_path:
        parser.add_argument("--image_path", type=str, required=True)
    parser.add_argument("--dim", type=int, choices=[2], default=2)
    parser.add_argument("--base_filters", type=int, default=32)
    parser.add_argument("--min_edt", type=float, default=0.15)
    parser.add_argument("--visibility_floor", type=float, default=0.25)
    parser.add_argument("--tile_size", type=int, default=512)
    parser.add_argument("--tile_overlap", type=int, default=128)
    parser.add_argument(
        "--downsample",
        type=float,
        default=1.0,
        help="Deprecated for 2D tiled STED inference and ignored when different from 1.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Torch device for inference. Use auto, cpu, cuda, or cuda:N.",
    )
    if include_output_dir:
        parser.add_argument("--output_dir", type=str, default="", help="Optional directory for prediction outputs.")
    parser.add_argument("--no_amp", action="store_true")
    if include_preview_options:
        parser.add_argument(
            "--save_preview",
            action="store_true",
            help="Save a compact QA preview PNG next to the TIFF outputs.",
        )
        parser.add_argument("--preview_path", type=str, default="", help="Optional override path for the preview PNG.")
    if include_visualize:
        parser.add_argument("--visualize", action="store_true")
    return parser


def resolve_device(device_spec: str = "auto") -> torch.device:
    if not device_spec or device_spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_spec)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA was requested via --device={device_spec}, but torch.cuda.is_available() is False.")
    return device


def load_sted_model(model_path: str, base_filters: int = 32, device_spec: str = "auto"):
    device = resolve_device(device_spec)
    model = STEDResUNet2D(in_channels=1, base_filters=base_filters)
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    state_dict = {k[7:] if k.startswith("module.") else k: v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, device


def load_image_for_inference(image_path: str, dim: int = 2) -> np.ndarray:
    if dim != 2:
        raise ValueError("The STED real-image inference path is 2D only. Use --dim 2.")

    if image_path.endswith(".pt"):
        data = torch.load(image_path, map_location="cpu", weights_only=True)
        image = data["volume"].squeeze(0).numpy()
        if image.shape[0] == 1:
            image = image.squeeze(0)
    elif image_path.endswith(".npy"):
        image = np.load(image_path)
    elif image_path.endswith((".tif", ".tiff")):
        image = tifffile.imread(image_path)
        image = np.squeeze(image)
        if image.ndim != 2:
            print(f"Warning: Image has shape {image.shape}. Reducing to 2D slice [0, ...]")
            image = image[0]
    else:
        raise ValueError(f"Unsupported input format for inference: {image_path}")

    image = np.asarray(image)
    if image.ndim != 2:
        raise ValueError(f"Expected a 2D image after loading, got shape {image.shape}.")
    return image


def run_real_image_inference(
    model: torch.nn.Module,
    image: np.ndarray,
    image_path: str,
    device: torch.device,
    tile_size: int = 512,
    tile_overlap: int = 128,
    min_edt: float = 0.15,
    visibility_floor: float = 0.25,
    use_amp: bool = True,
) -> RealInferenceResult:
    original = np.asarray(image)
    image_normalized = normalize_image_percentile(original)

    start_time = time.perf_counter()
    pred_start = time.perf_counter()
    pred = predict_tiled_2d(
        model,
        image_normalized,
        device=device,
        tile_size=tile_size,
        overlap=tile_overlap,
        output_channels=4,
        multiple=16,
        use_amp=use_amp,
    )
    prediction_seconds = time.perf_counter() - pred_start

    edt_map = pred[0]
    orientation_map = pred[1:3]
    vector_map = orientation_to_vector_map_np(orientation_map)
    orientation_confidence = orientation_confidence_np(orientation_map)
    visibility_logits = np.clip(pred[3], -20.0, 20.0)
    visibility_map = 1.0 / (1.0 + np.exp(-visibility_logits))
    edt_for_tracking = edt_map * np.clip(visibility_map, visibility_floor, 1.0)

    tracking_start = time.perf_counter()
    mask = (edt_for_tracking > min_edt).astype(np.uint8)
    if skeletonize is None:
        seed_mask = mask.astype(bool)
    else:
        seed_mask = skeletonize(mask).astype(bool)
    tracker = StreamlineTracker(step_size=0.5, min_edt=min_edt)
    streamlines = tracker.track(edt_for_tracking, vector_map, seed_mask=seed_mask)
    skeleton = tracker.to_binary_skeleton(streamlines, original.shape).astype(bool)
    tracking_seconds = time.perf_counter() - tracking_start

    return RealInferenceResult(
        image_path=image_path,
        image=original,
        image_normalized=image_normalized,
        edt_map=edt_map,
        orientation_map=orientation_map,
        vector_map=vector_map,
        orientation_confidence=orientation_confidence,
        visibility_map=visibility_map,
        edt_for_tracking=edt_for_tracking,
        seed_mask=np.asarray(seed_mask, dtype=bool),
        skeleton=skeleton,
        streamlines=streamlines,
        prediction_seconds=prediction_seconds,
        tracking_seconds=tracking_seconds,
        total_seconds=time.perf_counter() - start_time,
        tile_size=int(tile_size),
        tile_overlap=int(tile_overlap),
        min_edt=float(min_edt),
        visibility_floor=float(visibility_floor),
        used_amp=bool(use_amp and device.type == "cuda"),
    )


def resolve_output_base(image_path: str, output_dir: str = "", input_root: str = "") -> str:
    if output_dir:
        if input_root:
            relative = os.path.relpath(image_path, input_root)
            stem = os.path.splitext(relative)[0]
        else:
            stem = os.path.splitext(os.path.basename(image_path))[0]
        base_path = os.path.join(output_dir, stem)
    else:
        base_path = os.path.splitext(image_path)[0]

    directory = os.path.dirname(base_path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    return base_path


def build_output_paths(base_path: str) -> Dict[str, str]:
    return {key: f"{base_path}{suffix}" for key, suffix in OUTPUT_SUFFIXES.items()}


def required_output_paths(base_path: str) -> Dict[str, str]:
    paths = build_output_paths(base_path)
    return {key: paths[key] for key in ("skeleton", "pred_edt", "pred_vis", "pred_orient_conf", "summary")}


def outputs_exist(base_path: str) -> bool:
    return all(os.path.exists(path) for path in required_output_paths(base_path).values())


def save_inference_outputs(result: RealInferenceResult, base_path: str) -> Dict[str, str]:
    paths = build_output_paths(base_path)
    tifffile.imwrite(paths["skeleton"], (result.skeleton.astype(np.uint8) * 255))
    tifffile.imwrite(paths["pred_edt"], (np.clip(result.edt_map, 0.0, 1.0) * 255).astype(np.uint8))
    tifffile.imwrite(paths["pred_vis"], (np.clip(result.visibility_map, 0.0, 1.0) * 255).astype(np.uint8))
    tifffile.imwrite(
        paths["pred_orient_conf"],
        (np.clip(result.orientation_confidence, 0.0, 1.0) * 255).astype(np.uint8),
    )
    return paths


def load_saved_output_arrays(base_path: str) -> Dict[str, np.ndarray]:
    paths = build_output_paths(base_path)
    return {
        "skeleton": (tifffile.imread(paths["skeleton"]) > 0),
        "pred_edt": tifffile.imread(paths["pred_edt"]).astype(np.float32) / 255.0,
        "pred_vis": tifffile.imread(paths["pred_vis"]).astype(np.float32) / 255.0,
        "pred_orient_conf": tifffile.imread(paths["pred_orient_conf"]).astype(np.float32) / 255.0,
    }


def _streamline_length_summary(streamlines: Sequence[np.ndarray]) -> Dict[str, float]:
    lengths = []
    for path in streamlines:
        coords = np.asarray(path, dtype=np.float64)
        if coords.ndim != 2 or len(coords) < 2:
            continue
        step_lengths = np.linalg.norm(np.diff(coords, axis=0), axis=1)
        lengths.append(float(np.sum(step_lengths)))

    if not lengths:
        return {
            "streamline_length_mean": 0.0,
            "streamline_length_median": 0.0,
            "streamline_length_p95": 0.0,
        }

    arr = np.asarray(lengths, dtype=np.float64)
    return {
        "streamline_length_mean": float(np.mean(arr)),
        "streamline_length_median": float(np.median(arr)),
        "streamline_length_p95": float(np.percentile(arr, 95.0)),
    }


def _skeleton_component_count(skeleton: np.ndarray) -> int:
    labels, count = ndi.label(np.asarray(skeleton, dtype=bool))
    return int(count)


def _safe_ratio(numerator: float, denominator: float, default: float = 0.0) -> float:
    denominator = float(denominator)
    if abs(denominator) <= 1e-8:
        return float(default)
    return float(float(numerator) / denominator)


def _select_profile_metrics(profile: Dict[str, object], metadata: Dict[str, object]) -> Dict[str, object]:
    groups = profile.get("groups", {})
    condition = str(metadata.get("condition", "unknown"))
    if condition in groups.get("condition", {}):
        return groups["condition"][condition]

    div = metadata.get("div")
    if div is not None and str(div) in groups.get("div", {}):
        return groups["div"][str(div)]

    return profile["global"]


def _profile_oob_summary(profile: Optional[Dict[str, object]], metadata: Dict[str, object], metrics: Dict[str, object]) -> Dict[str, object]:
    if not profile:
        return {
            "input_profile_regime": "",
            "input_profile_oob_count": -1,
            "input_profile_oob_metrics": "",
        }

    metric_summary = _select_profile_metrics(profile, metadata)
    metric_table = metric_summary.get("metrics", {})
    regime = "global"
    condition = str(metadata.get("condition", "unknown"))
    div = metadata.get("div")
    if condition in profile.get("groups", {}).get("condition", {}):
        regime = f"condition:{condition}"
    elif div is not None and str(div) in profile.get("groups", {}).get("div", {}):
        regime = f"div:{div}"

    outside = []
    for metric in COMPARISON_METRICS:
        if metric not in metrics or metric not in metric_table:
            continue
        value = float(metrics[metric])
        q10 = float(metric_table[metric].get("q010", 0.0))
        q90 = float(metric_table[metric].get("q090", 0.0))
        if value < q10 or value > q90:
            outside.append(metric)

    return {
        "input_profile_regime": regime,
        "input_profile_oob_count": int(len(outside)),
        "input_profile_oob_metrics": ",".join(outside),
    }


def summarize_inference_result(
    result: RealInferenceResult,
    output_paths: Optional[Dict[str, str]] = None,
    calibration_profile: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    metadata = parse_sted_filename(result.image_path)
    input_metrics = compute_image_metrics(result.image, source=result.image_path, metadata=metadata)

    skeleton_mask = np.asarray(result.skeleton, dtype=bool)
    pred_mask = result.edt_for_tracking > result.min_edt
    stream_summary = _streamline_length_summary(result.streamlines)

    if np.any(skeleton_mask):
        raw_on = float(np.mean(result.image_normalized[skeleton_mask]))
        edt_on = float(np.mean(result.edt_map[skeleton_mask]))
        vis_on = float(np.mean(result.visibility_map[skeleton_mask]))
    else:
        raw_on = 0.0
        edt_on = 0.0
        vis_on = 0.0

    off_mask = ~skeleton_mask
    raw_off = float(np.mean(result.image_normalized[off_mask])) if np.any(off_mask) else 0.0
    visibility_floor_mask = result.visibility_map < result.visibility_floor
    low_support_fraction = float(np.mean(skeleton_mask & visibility_floor_mask))

    row: Dict[str, object] = {
        "source": result.image_path,
        "name": metadata["name"],
        "condition": metadata["condition"],
        "div": metadata["div"] if metadata["div"] is not None else -1,
        "replicate": metadata["replicate"],
        "series": metadata["series"] if metadata["series"] is not None else -1,
        "height": int(result.image.shape[0]),
        "width": int(result.image.shape[1]),
        "status": "ok",
        "tile_size": result.tile_size,
        "tile_overlap": result.tile_overlap,
        "min_edt": result.min_edt,
        "visibility_floor": result.visibility_floor,
        "used_amp": int(result.used_amp),
        "prediction_seconds": float(result.prediction_seconds),
        "tracking_seconds": float(result.tracking_seconds),
        "total_seconds": float(result.total_seconds),
        "seed_fraction": float(np.mean(result.seed_mask)),
        "pred_mask_fraction": float(np.mean(pred_mask)),
        "streamline_count": int(len(result.streamlines)),
        "skeleton_pixels": int(np.count_nonzero(skeleton_mask)),
        "skeleton_fraction": float(np.mean(skeleton_mask)),
        "skeleton_component_count": _skeleton_component_count(skeleton_mask),
        "pred_edt_mean": float(np.mean(result.edt_map)),
        "pred_edt_p95": float(np.percentile(result.edt_map, 95.0)),
        "pred_edt_p99": float(np.percentile(result.edt_map, 99.0)),
        "pred_vis_mean": float(np.mean(result.visibility_map)),
        "pred_vis_p95": float(np.percentile(result.visibility_map, 95.0)),
        "pred_vis_p99": float(np.percentile(result.visibility_map, 99.0)),
        "pred_orient_conf_mean": float(np.mean(result.orientation_confidence)),
        "pred_orient_conf_p95": float(np.percentile(result.orientation_confidence, 95.0)),
        "raw_on_skeleton_mean": raw_on,
        "raw_off_skeleton_mean": raw_off,
        "raw_skeleton_contrast": _safe_ratio(raw_on, raw_off, default=0.0),
        "edt_on_skeleton_mean": edt_on,
        "vis_on_skeleton_mean": vis_on,
        "low_support_skeleton_fraction": low_support_fraction,
        "pred_to_input_skeleton_ratio": _safe_ratio(float(np.mean(skeleton_mask)), float(input_metrics["skeleton_fraction"])),
        "pred_to_input_foreground_ratio": _safe_ratio(float(np.mean(pred_mask)), float(input_metrics["foreground_fraction"])),
    }
    row.update(stream_summary)

    for key, value in input_metrics.items():
        if key in {"source", "row_type", "patch_index", "patch_y", "patch_x"}:
            continue
        row[f"input_{key}"] = value

    row.update(_profile_oob_summary(calibration_profile, metadata, input_metrics))

    if output_paths:
        for key, value in output_paths.items():
            row[f"output_{key}"] = value

    return row


def render_preview_panel(
    image: np.ndarray,
    skeleton: np.ndarray,
    edt_map: np.ndarray,
    visibility_map: np.ndarray,
    orientation_confidence: np.ndarray,
    out_path: str,
    title: str = "",
    metrics: Optional[Dict[str, object]] = None,
) -> str:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    image_norm = normalize_image_percentile(image)
    skeleton_mask = np.asarray(skeleton, dtype=bool)

    overlay = np.stack([image_norm, image_norm, image_norm], axis=-1)
    overlay[skeleton_mask] = np.array([0.0, 1.0, 1.0], dtype=np.float32)

    fig, axes = plt.subplots(2, 3, figsize=(12, 8), dpi=140)
    axes = axes.ravel()

    axes[0].imshow(image_norm, cmap="magma", vmin=0.0, vmax=1.0)
    axes[0].set_title("Raw")
    axes[1].imshow(overlay, vmin=0.0, vmax=1.0)
    axes[1].set_title("Raw + Skeleton")
    axes[2].imshow(np.clip(edt_map, 0.0, 1.0), cmap="viridis", vmin=0.0, vmax=1.0)
    axes[2].set_title("Pred EDT")
    axes[3].imshow(np.clip(visibility_map, 0.0, 1.0), cmap="inferno", vmin=0.0, vmax=1.0)
    axes[3].set_title("Pred Visibility")
    axes[4].imshow(np.clip(orientation_confidence, 0.0, 1.0), cmap="cividis", vmin=0.0, vmax=1.0)
    axes[4].set_title("Orientation Confidence")
    axes[5].axis("off")

    if metrics:
        lines = [
            f"condition={metrics.get('condition', 'unknown')} div={metrics.get('div', -1)}",
            f"streamlines={metrics.get('streamline_count', 0)}",
            f"skeleton_fraction={float(metrics.get('skeleton_fraction', 0.0)):.6f}",
            f"raw_skeleton_contrast={float(metrics.get('raw_skeleton_contrast', 0.0)):.3f}",
            f"pred_vis_p99={float(metrics.get('pred_vis_p99', 0.0)):.3f}",
            f"pred_edt_p99={float(metrics.get('pred_edt_p99', 0.0)):.3f}",
        ]
        oob_count = int(metrics.get("input_profile_oob_count", -1))
        if oob_count >= 0:
            lines.append(f"profile_oob={oob_count} [{metrics.get('input_profile_oob_metrics', '')}]")
        axes[5].text(0.0, 1.0, "\n".join(lines), va="top", ha="left", family="monospace")

    for ax in axes[:5]:
        ax.axis("off")

    fig.suptitle(title or os.path.basename(out_path), fontsize=10)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def save_preview_panel(result: RealInferenceResult, out_path: str, metrics: Optional[Dict[str, object]] = None) -> str:
    title = metrics.get("source", result.image_path) if metrics else result.image_path
    return render_preview_panel(
        image=result.image,
        skeleton=result.skeleton,
        edt_map=result.edt_map,
        visibility_map=result.visibility_map,
        orientation_confidence=result.orientation_confidence,
        out_path=out_path,
        title=os.path.basename(title),
        metrics=metrics,
    )


def create_visualization_result(result: RealInferenceResult) -> SimpleNamespace:
    return SimpleNamespace(
        binary_mask=(result.edt_for_tracking > result.min_edt).astype(np.uint8),
        skeleton=result.skeleton.astype(np.uint8),
        hfa_map=result.edt_map,
        fa_macro_map=result.orientation_confidence,
    )


def show_interactive_result(result: RealInferenceResult) -> None:
    AdvancedVisualizer.show_interactive_napari(result.image, create_visualization_result(result))


def save_summary_json(row: Dict[str, object], path: str) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(row, handle, indent=2, sort_keys=True)
    return path


def load_summary_json(path: str) -> Dict[str, object]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_optional_profile(profile_path: str = "") -> Optional[Dict[str, object]]:
    if not profile_path:
        return None
    return load_calibration_profile(profile_path)
