import json
import os
import time
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Dict, List, Optional

import numpy as np
import scipy.ndimage as ndi
import tifffile
import torch

from src.decoder import CenterlineGraphDecoder
from src.inference_utils import normalize_image_percentile, predict_tiled_2d
from src.model import PREDICTION_HEAD_TYPE, STEDResUNet2D
from src.sted import orientation_confidence_np, orientation_to_vector_map_np
from src.sted_calibration import (
    COMPARISON_METRICS,
    compute_image_metrics,
    load_calibration_profile,
    parse_sted_filename,
)
from src.visualization import AdvancedVisualizer
from train import (
    checkpoint_aspp_dilations,
    checkpoint_head_hidden_channels,
    checkpoint_head_type,
    checkpoint_unet_depth,
    checkpoint_use_head_refinement,
    extract_model_state_dict,
    format_aspp_dilations,
)


OUTPUT_SUFFIXES = {
    "skeleton": "_skeleton.tif",
    "pred_centerline": "_pred_centerline.tif",
    "pred_traceability": "_pred_traceability.tif",
    "pred_radius": "_pred_radius.tif",
    "pred_bundle_count": "_pred_bundle_count.tif",
    "pred_orient_conf": "_pred_orient_conf.tif",
    "preview": "_preview.png",
    "summary": "_summary.json",
}

BUNDLE_COUNT_NORMALIZER = 6.0


@dataclass
class RealInferenceResult:
    image_path: str
    image: np.ndarray
    image_normalized: np.ndarray
    centerline_prob: np.ndarray
    orientation_map: np.ndarray
    vector_map: np.ndarray
    orientation_confidence: np.ndarray
    traceability_map: np.ndarray
    radius_map: np.ndarray
    bundle_count_map: np.ndarray
    decoder_score_map: np.ndarray
    decoder_candidate_mask: np.ndarray
    decoder_strong_seed_mask: np.ndarray
    skeleton: np.ndarray
    component_paths: List[np.ndarray]
    endpoint_mask: np.ndarray
    junction_mask: np.ndarray
    prediction_seconds: float
    decoding_seconds: float
    total_seconds: float
    tile_size: int
    tile_overlap: int
    centerline_threshold: float
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
    parser.add_argument(
        "--unet_depth",
        type=int,
        default=0,
        help="Optional U-Net depth override. Defaults to checkpoint config, or legacy depth 4 for old checkpoints.",
    )
    parser.add_argument(
        "--aspp_dilations",
        type=str,
        default="",
        help="Optional comma-separated ASPP dilation override. Defaults to checkpoint config, or legacy 2,4,8 for old checkpoints.",
    )
    parser.add_argument(
        "--head_type",
        type=str,
        default="",
        help="Optional prediction-head type override. Defaults to checkpoint config.",
    )
    parser.add_argument(
        "--head_hidden_channels",
        type=int,
        default=0,
        help="Optional bottleneck head width override. Defaults to checkpoint config.",
    )
    parser.add_argument(
        "--use_head_refinement",
        type=str,
        default="auto",
        help="Optional head-refinement override: auto, true, or false.",
    )
    parser.add_argument("--centerline_threshold", type=float, default=0.5)
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


def load_sted_model(
    model_path: str,
    base_filters: int = 32,
    device_spec: str = "auto",
    aspp_dilations=None,
    unet_depth=None,
    head_type=None,
    head_hidden_channels=None,
    use_head_refinement=None,
):
    device = resolve_device(device_spec)
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    resolved_aspp_dilations = checkpoint_aspp_dilations(checkpoint, override=aspp_dilations)
    resolved_unet_depth = checkpoint_unet_depth(checkpoint, override=unet_depth)
    resolved_head_type = checkpoint_head_type(checkpoint, override=head_type)
    resolved_head_hidden_channels = checkpoint_head_hidden_channels(checkpoint, override=head_hidden_channels)
    resolved_use_head_refinement = checkpoint_use_head_refinement(checkpoint, override=use_head_refinement)
    model = STEDResUNet2D(
        in_channels=1,
        base_filters=base_filters,
        aspp_dilations=resolved_aspp_dilations,
        unet_depth=resolved_unet_depth,
        head_type=resolved_head_type,
        head_hidden_channels=resolved_head_hidden_channels,
        use_head_refinement=resolved_use_head_refinement,
    )
    state_dict = extract_model_state_dict(checkpoint)
    try:
        model.load_state_dict(state_dict)
    except RuntimeError as error:
        message = str(error)
        if "edt_head" in message or "visibility_head" in message:
            raise RuntimeError(
                f"Checkpoint '{model_path}' uses the legacy EDT/visibility heads and is incompatible "
                "with the new structural centerline architecture. Train a new checkpoint with the "
                "upgraded model before running inference."
            ) from error
        if any(
            head_name in message
            for head_name in (
                "centerline_head",
                "orientation_head",
                "traceability_head",
                "radius_head",
                "bundle_count_head",
            )
        ):
            raise RuntimeError(
                f"Checkpoint '{model_path}' does not match the current {PREDICTION_HEAD_TYPE} prediction-head "
                "architecture. Train a new checkpoint with the current model before running inference."
            ) from error
        if "bundle_count_head" in message or "size mismatch" in message:
            raise RuntimeError(
                f"Checkpoint '{model_path}' does not match the current 6-channel bundle-count model. "
                "Train a new checkpoint with structural_v2 targets before running bundle-count inference."
            ) from error
        raise
    model.to(device)
    model.eval()
    print(f"Loaded model ASPP dilations: {format_aspp_dilations(resolved_aspp_dilations)}")
    print(f"Loaded model U-Net depth: {resolved_unet_depth}")
    print(f"Loaded model head type: {resolved_head_type}")
    print(f"Loaded model head hidden channels: {resolved_head_hidden_channels}")
    print(f"Loaded model head refinement: {resolved_use_head_refinement}")
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
    centerline_threshold: float = 0.5,
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
        output_channels=6,
        multiple=16,
        use_amp=use_amp,
    )
    prediction_seconds = time.perf_counter() - pred_start

    centerline_logits = np.clip(pred[0], -20.0, 20.0)
    centerline_prob = 1.0 / (1.0 + np.exp(-centerline_logits))
    orientation_map = pred[1:3]
    vector_map = orientation_to_vector_map_np(orientation_map)
    orientation_confidence = orientation_confidence_np(orientation_map)
    traceability_logits = np.clip(pred[3], -20.0, 20.0)
    traceability_map = 1.0 / (1.0 + np.exp(-traceability_logits))
    radius_logits = np.clip(pred[4], -20.0, 20.0)
    radius_map = 1.0 / (1.0 + np.exp(-radius_logits))
    bundle_count_logits = np.clip(pred[5], -20.0, 20.0)
    bundle_count_map = 1.0 / (1.0 + np.exp(-bundle_count_logits))

    decode_start = time.perf_counter()
    decoder = CenterlineGraphDecoder(centerline_threshold=centerline_threshold)
    decoded = decoder.decode(
        centerline_prob=centerline_prob,
        orientation_map=vector_map,
        traceability_map=traceability_map,
        orientation_confidence=orientation_confidence,
    )
    decoding_seconds = time.perf_counter() - decode_start

    return RealInferenceResult(
        image_path=image_path,
        image=original,
        image_normalized=image_normalized,
        centerline_prob=centerline_prob,
        orientation_map=orientation_map,
        vector_map=vector_map,
        orientation_confidence=orientation_confidence,
        traceability_map=traceability_map,
        radius_map=radius_map,
        bundle_count_map=bundle_count_map,
        decoder_score_map=decoded.score_map,
        decoder_candidate_mask=decoded.candidate_mask,
        decoder_strong_seed_mask=decoded.strong_seed_mask,
        skeleton=decoded.skeleton,
        component_paths=decoded.component_paths,
        endpoint_mask=decoded.endpoint_mask,
        junction_mask=decoded.junction_mask,
        prediction_seconds=prediction_seconds,
        decoding_seconds=decoding_seconds,
        total_seconds=time.perf_counter() - start_time,
        tile_size=int(tile_size),
        tile_overlap=int(tile_overlap),
        centerline_threshold=float(centerline_threshold),
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
    return {
        key: paths[key]
        for key in (
            "skeleton",
            "pred_centerline",
            "pred_traceability",
            "pred_radius",
            "pred_bundle_count",
            "pred_orient_conf",
            "summary",
        )
    }


def outputs_exist(base_path: str) -> bool:
    return all(os.path.exists(path) for path in required_output_paths(base_path).values())


def save_inference_outputs(result: RealInferenceResult, base_path: str) -> Dict[str, str]:
    paths = build_output_paths(base_path)
    tifffile.imwrite(paths["skeleton"], result.skeleton.astype(np.uint8) * 255)
    tifffile.imwrite(paths["pred_centerline"], (np.clip(result.centerline_prob, 0.0, 1.0) * 255).astype(np.uint8))
    tifffile.imwrite(paths["pred_traceability"], (np.clip(result.traceability_map, 0.0, 1.0) * 255).astype(np.uint8))
    tifffile.imwrite(paths["pred_radius"], (np.clip(result.radius_map, 0.0, 1.0) * 255).astype(np.uint8))
    tifffile.imwrite(paths["pred_bundle_count"], (np.clip(result.bundle_count_map, 0.0, 1.0) * 255).astype(np.uint8))
    tifffile.imwrite(paths["pred_orient_conf"], (np.clip(result.orientation_confidence, 0.0, 1.0) * 255).astype(np.uint8))
    return paths


def load_saved_output_arrays(base_path: str) -> Dict[str, np.ndarray]:
    paths = build_output_paths(base_path)
    return {
        "skeleton": tifffile.imread(paths["skeleton"]) > 0,
        "pred_centerline": tifffile.imread(paths["pred_centerline"]).astype(np.float32) / 255.0,
        "pred_traceability": tifffile.imread(paths["pred_traceability"]).astype(np.float32) / 255.0,
        "pred_radius": tifffile.imread(paths["pred_radius"]).astype(np.float32) / 255.0,
        "pred_bundle_count": tifffile.imread(paths["pred_bundle_count"]).astype(np.float32) / 255.0,
        "pred_orient_conf": tifffile.imread(paths["pred_orient_conf"]).astype(np.float32) / 255.0,
    }


def _component_length_summary(component_paths: List[np.ndarray]) -> Dict[str, float]:
    lengths = [float(len(path)) for path in component_paths if len(path) > 0]
    if not lengths:
        return {
            "component_length_mean": 0.0,
            "component_length_median": 0.0,
            "component_length_p95": 0.0,
        }
    arr = np.asarray(lengths, dtype=np.float64)
    return {
        "component_length_mean": float(np.mean(arr)),
        "component_length_median": float(np.median(arr)),
        "component_length_p95": float(np.percentile(arr, 95.0)),
    }


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


def _binary_iou(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=bool)
    b = np.asarray(b, dtype=bool)
    union = np.count_nonzero(a | b)
    if union == 0:
        return 1.0
    return float(np.count_nonzero(a & b) / union)


def _decoder_self_consistency(result: RealInferenceResult) -> float:
    thresholds = [
        max(0.05, result.centerline_threshold - 0.10),
        result.centerline_threshold,
        min(0.95, result.centerline_threshold + 0.10),
    ]
    skeletons = []
    for threshold in thresholds:
        decoded = CenterlineGraphDecoder(centerline_threshold=threshold).decode(
            centerline_prob=result.centerline_prob,
            orientation_map=result.vector_map,
            traceability_map=result.traceability_map,
            orientation_confidence=result.orientation_confidence,
        )
        skeletons.append(decoded.skeleton)
    return min(_binary_iou(skeletons[0], skeletons[1]), _binary_iou(skeletons[1], skeletons[2]))


def summarize_inference_result(
    result: RealInferenceResult,
    output_paths: Optional[Dict[str, str]] = None,
    calibration_profile: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    metadata = parse_sted_filename(result.image_path)
    input_metrics = compute_image_metrics(result.image, source=result.image_path, metadata=metadata)

    skeleton_mask = np.asarray(result.skeleton, dtype=bool)
    component_summary = _component_length_summary(result.component_paths)

    if np.any(skeleton_mask):
        raw_on = float(np.mean(result.image_normalized[skeleton_mask]))
        centerline_on = float(np.mean(result.centerline_prob[skeleton_mask]))
        traceability_on = float(np.mean(result.traceability_map[skeleton_mask]))
        radius_on = float(np.mean(result.radius_map[skeleton_mask]))
        bundle_count_on = float(np.mean(result.bundle_count_map[skeleton_mask]) * BUNDLE_COUNT_NORMALIZER)
    else:
        raw_on = 0.0
        centerline_on = 0.0
        traceability_on = 0.0
        radius_on = 0.0
        bundle_count_on = 0.0

    off_mask = ~skeleton_mask
    raw_off = float(np.mean(result.image_normalized[off_mask])) if np.any(off_mask) else 0.0
    decoder_score_mask = np.asarray(result.decoder_candidate_mask, dtype=bool)

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
        "centerline_threshold": result.centerline_threshold,
        "used_amp": int(result.used_amp),
        "prediction_seconds": float(result.prediction_seconds),
        "decoding_seconds": float(result.decoding_seconds),
        "total_seconds": float(result.total_seconds),
        "decoder_mask_fraction": float(np.mean(decoder_score_mask)),
        "component_count": int(len(result.component_paths)),
        "skeleton_pixels": int(np.count_nonzero(skeleton_mask)),
        "skeleton_fraction": float(np.mean(skeleton_mask)),
        "endpoint_count": int(np.count_nonzero(result.endpoint_mask)),
        "junction_count": int(np.count_nonzero(result.junction_mask)),
        "pred_centerline_mean": float(np.mean(result.centerline_prob)),
        "pred_centerline_p95": float(np.percentile(result.centerline_prob, 95.0)),
        "pred_centerline_p99": float(np.percentile(result.centerline_prob, 99.0)),
        "pred_traceability_mean": float(np.mean(result.traceability_map)),
        "pred_traceability_p95": float(np.percentile(result.traceability_map, 95.0)),
        "pred_traceability_p99": float(np.percentile(result.traceability_map, 99.0)),
        "pred_radius_mean": float(np.mean(result.radius_map)),
        "pred_radius_p95": float(np.percentile(result.radius_map, 95.0)),
        "pred_radius_p99": float(np.percentile(result.radius_map, 99.0)),
        "pred_bundle_count_mean": float(np.mean(result.bundle_count_map) * BUNDLE_COUNT_NORMALIZER),
        "pred_bundle_count_p95": float(np.percentile(result.bundle_count_map, 95.0) * BUNDLE_COUNT_NORMALIZER),
        "pred_bundle_count_p99": float(np.percentile(result.bundle_count_map, 99.0) * BUNDLE_COUNT_NORMALIZER),
        "pred_orient_conf_mean": float(np.mean(result.orientation_confidence)),
        "pred_orient_conf_p95": float(np.percentile(result.orientation_confidence, 95.0)),
        "raw_on_skeleton_mean": raw_on,
        "raw_off_skeleton_mean": raw_off,
        "raw_skeleton_contrast": _safe_ratio(raw_on, raw_off, default=0.0),
        "centerline_on_skeleton_mean": centerline_on,
        "traceability_on_skeleton_mean": traceability_on,
        "radius_on_skeleton_mean": radius_on,
        "bundle_count_on_skeleton_mean": bundle_count_on,
        "decoder_self_consistency": _decoder_self_consistency(result),
        "pred_to_input_skeleton_ratio": _safe_ratio(float(np.mean(skeleton_mask)), float(input_metrics["skeleton_fraction"])),
        "pred_to_input_foreground_ratio": _safe_ratio(float(np.mean(decoder_score_mask)), float(input_metrics["foreground_fraction"])),
    }
    row.update(component_summary)

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
    centerline_prob: np.ndarray,
    traceability_map: np.ndarray,
    radius_map: np.ndarray,
    bundle_count_map: np.ndarray,
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
    axes[2].imshow(np.clip(centerline_prob, 0.0, 1.0), cmap="viridis", vmin=0.0, vmax=1.0)
    axes[2].set_title("Centerline Probability")
    axes[3].imshow(np.clip(traceability_map, 0.0, 1.0), cmap="inferno", vmin=0.0, vmax=1.0)
    axes[3].set_title("Traceability")
    axes[4].imshow(np.clip(radius_map, 0.0, 1.0), cmap="cividis", vmin=0.0, vmax=1.0)
    axes[4].set_title("Radius")
    axes[5].imshow(
        np.clip(bundle_count_map, 0.0, 1.0) * BUNDLE_COUNT_NORMALIZER,
        cmap="plasma",
        vmin=0.0,
        vmax=BUNDLE_COUNT_NORMALIZER,
    )
    axes[5].set_title("Bundle Count")

    if metrics:
        lines = [
            f"condition={metrics.get('condition', 'unknown')} div={metrics.get('div', -1)}",
            f"components={metrics.get('component_count', 0)}",
            f"skeleton_fraction={float(metrics.get('skeleton_fraction', 0.0)):.6f}",
            f"raw_skeleton_contrast={float(metrics.get('raw_skeleton_contrast', 0.0)):.3f}",
            f"trace_p99={float(metrics.get('pred_traceability_p99', 0.0)):.3f}",
            f"centerline_p99={float(metrics.get('pred_centerline_p99', 0.0)):.3f}",
            f"self_consistency={float(metrics.get('decoder_self_consistency', 0.0)):.3f}",
        ]
        oob_count = int(metrics.get("input_profile_oob_count", -1))
        if oob_count >= 0:
            lines.append(f"profile_oob={oob_count} [{metrics.get('input_profile_oob_metrics', '')}]")
        axes[1].text(0.01, 0.99, "\n".join(lines), va="top", ha="left", family="monospace", color="white")

    for ax in axes:
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
        centerline_prob=result.centerline_prob,
        traceability_map=result.traceability_map,
        radius_map=result.radius_map,
        bundle_count_map=result.bundle_count_map,
        orientation_confidence=result.orientation_confidence,
        out_path=out_path,
        title=os.path.basename(title),
        metrics=metrics,
    )


def create_visualization_result(result: RealInferenceResult) -> SimpleNamespace:
    return SimpleNamespace(
        binary_mask=result.decoder_candidate_mask.astype(np.uint8),
        strong_seed_mask=result.decoder_strong_seed_mask.astype(np.uint8),
        skeleton=result.skeleton.astype(np.uint8),
        hfa_map=result.centerline_prob,
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
