import argparse
import concurrent.futures
import csv
import json
import math
import multiprocessing
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

try:
    from skimage.morphology import skeletonize
except Exception:  # pragma: no cover - exercised only when optional deps are missing.
    skeletonize = None

from src.inference_utils import normalize_image_percentile, predict_tiled_2d
from src.real_inference import load_image_for_inference, load_optional_profile, load_sted_model
from src.sted import orientation_confidence_np, orientation_to_vector_map_np
from src.sted_calibration import compute_image_metrics, parse_sted_filename
from src.tracking import StreamlineTracker


_WORKER_SYNTHETIC_SAMPLES: Sequence["CachedSyntheticSample"] = ()
_WORKER_FIBER_SAMPLES: Sequence["CachedRealSample"] = ()
_WORKER_BLANK_SAMPLES: Sequence["CachedRealSample"] = ()
_WORKER_MAX_BLANK_NONEMPTY_RATE: float = 0.10
_WORKER_MAX_BLANK_SKELETON_FRACTION: float = 0.00005


@dataclass
class CachedPrediction:
    image_path: str
    image: np.ndarray
    image_normalized: np.ndarray
    edt_map: np.ndarray
    orientation_map: np.ndarray
    vector_map: np.ndarray
    orientation_confidence: np.ndarray
    visibility_map: np.ndarray
    prediction_seconds: float
    tile_size: int
    tile_overlap: int
    used_amp: bool


@dataclass
class CachedSyntheticSample:
    image_path: str
    prediction: CachedPrediction
    target_centerline: np.ndarray
    target_skeleton_fraction: float


@dataclass
class CachedRealSample:
    image_path: str
    prediction: CachedPrediction
    input_foreground_fraction: float
    input_skeleton_fraction: float
    pred_edt_p99: float
    pred_vis_p99: float


@dataclass
class TrackedDecodeResult:
    skeleton: np.ndarray
    streamlines: List[np.ndarray]
    edt_for_tracking: np.ndarray
    seed_mask: np.ndarray
    tracking_seconds: float


def _discover_tiff_files(input_dir: str, recursive: bool) -> List[str]:
    files = []
    if recursive:
        walker = os.walk(input_dir)
    else:
        walker = [(input_dir, [], os.listdir(input_dir))]

    for root, _, names in walker:
        for name in names:
            if name.lower().endswith((".tif", ".tiff")):
                files.append(os.path.join(root, name))
    files.sort()
    return files


def _discover_synthetic_files(data_dir: str, split: str) -> List[str]:
    split_dir = os.path.join(data_dir, split)
    root = split_dir if os.path.isdir(split_dir) else data_dir
    files = [os.path.join(root, name) for name in os.listdir(root) if name.endswith(".pt")]
    files.sort()
    return files


def _limit_paths(paths: Sequence[str], max_count: int) -> List[str]:
    if max_count <= 0:
        return list(paths)
    return list(paths[:max_count])


def _predict_cached_fields(
    model: torch.nn.Module,
    image: np.ndarray,
    image_path: str,
    device: torch.device,
    tile_size: int,
    tile_overlap: int,
    use_amp: bool,
) -> CachedPrediction:
    original = np.asarray(image)
    image_normalized = normalize_image_percentile(original)

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

    edt_map = np.asarray(pred[0], dtype=np.float32)
    orientation_map = np.asarray(pred[1:3], dtype=np.float32)
    vector_map = orientation_to_vector_map_np(orientation_map)
    orientation_confidence = orientation_confidence_np(orientation_map)
    visibility_logits = np.clip(pred[3], -20.0, 20.0)
    visibility_map = 1.0 / (1.0 + np.exp(-visibility_logits))

    return CachedPrediction(
        image_path=image_path,
        image=original,
        image_normalized=image_normalized,
        edt_map=edt_map,
        orientation_map=orientation_map,
        vector_map=vector_map,
        orientation_confidence=np.asarray(orientation_confidence, dtype=np.float32),
        visibility_map=np.asarray(visibility_map, dtype=np.float32),
        prediction_seconds=float(prediction_seconds),
        tile_size=int(tile_size),
        tile_overlap=int(tile_overlap),
        used_amp=bool(use_amp and device.type == "cuda"),
    )


def _track_cached_prediction(
    cached: CachedPrediction,
    min_edt: float,
    visibility_floor: float,
) -> TrackedDecodeResult:
    edt_for_tracking = cached.edt_map * np.clip(cached.visibility_map, visibility_floor, 1.0)

    tracking_start = time.perf_counter()
    mask = (edt_for_tracking > min_edt).astype(np.uint8)
    if skeletonize is None:
        seed_mask = mask.astype(bool)
    else:
        seed_mask = skeletonize(mask).astype(bool)
    tracker = StreamlineTracker(step_size=0.5, min_edt=min_edt)
    streamlines = tracker.track(edt_for_tracking, cached.vector_map, seed_mask=seed_mask)
    skeleton = tracker.to_binary_skeleton(streamlines, cached.image.shape).astype(bool)
    tracking_seconds = time.perf_counter() - tracking_start

    return TrackedDecodeResult(
        skeleton=skeleton,
        streamlines=streamlines,
        edt_for_tracking=np.asarray(edt_for_tracking, dtype=np.float32),
        seed_mask=np.asarray(seed_mask, dtype=bool),
        tracking_seconds=float(tracking_seconds),
    )


def _binary_precision_recall_f1(pred: np.ndarray, target: np.ndarray) -> Dict[str, float]:
    pred = np.asarray(pred, dtype=bool)
    target = np.asarray(target, dtype=bool)
    tp = float(np.count_nonzero(pred & target))
    fp = float(np.count_nonzero(pred & ~target))
    fn = float(np.count_nonzero(~pred & target))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2.0 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 1.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou": iou,
    }


def _target_centerline_from_targets(targets: torch.Tensor, edt_threshold: float) -> np.ndarray:
    target_edt = targets[0]
    if target_edt.ndim == 3:
        target_edt = target_edt[0]
    target_edt = np.asarray(target_edt, dtype=np.float32)
    mask = target_edt > float(edt_threshold)
    if skeletonize is None:
        return mask.astype(bool)
    return np.asarray(skeletonize(mask), dtype=bool)


def _load_cached_synthetic_sample(
    model: torch.nn.Module,
    file_path: str,
    device: torch.device,
    tile_size: int,
    tile_overlap: int,
    use_amp: bool,
    target_edt_threshold: float,
) -> CachedSyntheticSample:
    data = torch.load(file_path, map_location="cpu", weights_only=True)
    image = data["volume"].squeeze().numpy()
    if image.ndim != 2:
        raise ValueError(f"Expected a 2D synthetic image in {file_path}, got shape {image.shape}")
    prediction = _predict_cached_fields(
        model=model,
        image=image,
        image_path=file_path,
        device=device,
        tile_size=tile_size,
        tile_overlap=tile_overlap,
        use_amp=use_amp,
    )
    target_centerline = _target_centerline_from_targets(data["targets"], edt_threshold=target_edt_threshold)
    return CachedSyntheticSample(
        image_path=file_path,
        prediction=prediction,
        target_centerline=target_centerline,
        target_skeleton_fraction=float(np.mean(target_centerline)),
    )


def _load_cached_real_sample(
    model: torch.nn.Module,
    image_path: str,
    device: torch.device,
    tile_size: int,
    tile_overlap: int,
    use_amp: bool,
) -> CachedRealSample:
    image = load_image_for_inference(image_path, dim=2)
    prediction = _predict_cached_fields(
        model=model,
        image=image,
        image_path=image_path,
        device=device,
        tile_size=tile_size,
        tile_overlap=tile_overlap,
        use_amp=use_amp,
    )
    metadata = parse_sted_filename(image_path)
    input_metrics = compute_image_metrics(image, source=image_path, metadata=metadata)
    return CachedRealSample(
        image_path=image_path,
        prediction=prediction,
        input_foreground_fraction=float(input_metrics.get("foreground_fraction", 0.0)),
        input_skeleton_fraction=float(input_metrics.get("skeleton_fraction", 0.0)),
        pred_edt_p99=float(np.percentile(prediction.edt_map, 99.0)),
        pred_vis_p99=float(np.percentile(prediction.visibility_map, 99.0)),
    )


def _median(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return float(np.median(np.asarray(values, dtype=np.float64)))


def _mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def _fraction(condition_count: int, total_count: int) -> float:
    if total_count <= 0:
        return 0.0
    return float(condition_count / total_count)


def _log1p_safe(value: float) -> float:
    return float(math.log1p(max(float(value), 0.0)))


def _combined_score(row: Dict[str, float]) -> float:
    return (
        2.0 * float(row.get("synthetic_f1_median", 0.0))
        + 0.35 * float(row.get("fiber_nonempty_rate", 0.0))
        + 0.20 * _log1p_safe(row.get("fiber_raw_skeleton_contrast_median", 0.0))
        + 0.10 * _log1p_safe(row.get("fiber_streamline_length_median_median", 0.0))
        - 2.50 * float(row.get("blank_nonempty_rate", 0.0))
        - 150.0 * float(row.get("blank_skeleton_fraction_median", 0.0))
        - 0.50 * float(row.get("fiber_low_support_fraction_median", 0.0))
    )


def _pareto_front(rows: Sequence[Dict[str, float]]) -> List[int]:
    front = []
    for i, candidate in enumerate(rows):
        dominated = False
        for j, other in enumerate(rows):
            if i == j:
                continue
            no_worse = (
                float(other.get("synthetic_f1_median", 0.0)) >= float(candidate.get("synthetic_f1_median", 0.0))
                and float(other.get("fiber_raw_skeleton_contrast_median", 0.0)) >= float(candidate.get("fiber_raw_skeleton_contrast_median", 0.0))
                and float(other.get("fiber_nonempty_rate", 0.0)) >= float(candidate.get("fiber_nonempty_rate", 0.0))
                and float(other.get("blank_nonempty_rate", 1.0)) <= float(candidate.get("blank_nonempty_rate", 1.0))
                and float(other.get("blank_skeleton_fraction_median", 1.0)) <= float(candidate.get("blank_skeleton_fraction_median", 1.0))
            )
            strictly_better = (
                float(other.get("synthetic_f1_median", 0.0)) > float(candidate.get("synthetic_f1_median", 0.0))
                or float(other.get("fiber_raw_skeleton_contrast_median", 0.0)) > float(candidate.get("fiber_raw_skeleton_contrast_median", 0.0))
                or float(other.get("fiber_nonempty_rate", 0.0)) > float(candidate.get("fiber_nonempty_rate", 0.0))
                or float(other.get("blank_nonempty_rate", 1.0)) < float(candidate.get("blank_nonempty_rate", 1.0))
                or float(other.get("blank_skeleton_fraction_median", 1.0)) < float(candidate.get("blank_skeleton_fraction_median", 1.0))
            )
            if no_worse and strictly_better:
                dominated = True
                break
        if not dominated:
            front.append(i)
    return front


def _write_csv(rows: Sequence[Dict[str, object]], path: str) -> str:
    fieldnames = sorted({key for row in rows for key in row.keys()})
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


def _write_json(data: Dict[str, object], path: str) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
    return path


def _evaluate_synthetic_setting(
    samples: Sequence[CachedSyntheticSample],
    min_edt: float,
    visibility_floor: float,
) -> Dict[str, float]:
    if not samples:
        return {
            "synthetic_count": 0,
            "synthetic_f1_median": 0.0,
            "synthetic_precision_median": 0.0,
            "synthetic_recall_median": 0.0,
            "synthetic_iou_median": 0.0,
            "synthetic_pred_to_target_skeleton_ratio_median": 0.0,
        }

    f1_values = []
    precision_values = []
    recall_values = []
    iou_values = []
    ratio_values = []

    for sample in samples:
        result = _track_cached_prediction(sample.prediction, min_edt=min_edt, visibility_floor=visibility_floor)
        metrics = _binary_precision_recall_f1(result.skeleton, sample.target_centerline)
        f1_values.append(metrics["f1"])
        precision_values.append(metrics["precision"])
        recall_values.append(metrics["recall"])
        iou_values.append(metrics["iou"])
        ratio_values.append(float(np.mean(result.skeleton)) / max(sample.target_skeleton_fraction, 1e-8))

    return {
        "synthetic_count": len(samples),
        "synthetic_f1_median": _median(f1_values),
        "synthetic_precision_median": _median(precision_values),
        "synthetic_recall_median": _median(recall_values),
        "synthetic_iou_median": _median(iou_values),
        "synthetic_pred_to_target_skeleton_ratio_median": _median(ratio_values),
    }


def _evaluate_real_setting(
    cached_samples: Sequence[CachedRealSample],
    min_edt: float,
    visibility_floor: float,
    prefix: str,
) -> Dict[str, float]:
    if not cached_samples:
        return {
            f"{prefix}_count": 0,
            f"{prefix}_nonempty_rate": 0.0,
            f"{prefix}_streamline_count_median": 0.0,
            f"{prefix}_skeleton_fraction_median": 0.0,
            f"{prefix}_raw_skeleton_contrast_median": 0.0,
            f"{prefix}_low_support_fraction_median": 0.0,
            f"{prefix}_streamline_length_median_median": 0.0,
            f"{prefix}_pred_edt_p99_median": 0.0,
            f"{prefix}_pred_vis_p99_median": 0.0,
        }

    streamline_counts = []
    skeleton_fractions = []
    raw_skeleton_contrasts = []
    low_support_fractions = []
    streamline_length_medians = []
    pred_to_input_skeleton_ratios = []
    pred_to_input_foreground_ratios = []
    nonempty_count = 0

    for sample in cached_samples:
        result = _track_cached_prediction(sample.prediction, min_edt=min_edt, visibility_floor=visibility_floor)
        skeleton_mask = np.asarray(result.skeleton, dtype=bool)
        pred_mask = result.edt_for_tracking > min_edt

        streamline_counts.append(float(len(result.streamlines)))
        skeleton_fraction = float(np.mean(skeleton_mask))
        skeleton_fractions.append(skeleton_fraction)
        nonempty_count += int(len(result.streamlines) > 0)

        if np.any(skeleton_mask):
            raw_on = float(np.mean(sample.prediction.image_normalized[skeleton_mask]))
            raw_off = float(np.mean(sample.prediction.image_normalized[~skeleton_mask])) if np.any(~skeleton_mask) else 0.0
        else:
            raw_on = 0.0
            raw_off = float(np.mean(sample.prediction.image_normalized))
        raw_skeleton_contrasts.append(_safe_ratio(raw_on, raw_off, default=0.0))

        visibility_floor_mask = sample.prediction.visibility_map < visibility_floor
        low_support_fractions.append(float(np.mean(skeleton_mask & visibility_floor_mask)))

        lengths = []
        for path in result.streamlines:
            coords = np.asarray(path, dtype=np.float64)
            if coords.ndim == 2 and len(coords) >= 2:
                lengths.append(float(np.sum(np.linalg.norm(np.diff(coords, axis=0), axis=1))))
        streamline_length_medians.append(_median(lengths))

        pred_to_input_skeleton_ratios.append(_safe_ratio(skeleton_fraction, sample.input_skeleton_fraction))
        pred_to_input_foreground_ratios.append(_safe_ratio(float(np.mean(pred_mask)), sample.input_foreground_fraction))

    return {
        f"{prefix}_count": len(cached_samples),
        f"{prefix}_nonempty_rate": _fraction(nonempty_count, len(cached_samples)),
        f"{prefix}_streamline_count_median": _median(streamline_counts),
        f"{prefix}_skeleton_fraction_median": _median(skeleton_fractions),
        f"{prefix}_raw_skeleton_contrast_median": _median(raw_skeleton_contrasts),
        f"{prefix}_low_support_fraction_median": _median(low_support_fractions),
        f"{prefix}_streamline_length_median_median": _median(streamline_length_medians),
        f"{prefix}_pred_edt_p99_median": _median([sample.pred_edt_p99 for sample in cached_samples]),
        f"{prefix}_pred_vis_p99_median": _median([sample.pred_vis_p99 for sample in cached_samples]),
        f"{prefix}_pred_to_input_skeleton_ratio_median": _median(pred_to_input_skeleton_ratios),
        f"{prefix}_pred_to_input_foreground_ratio_median": _median(pred_to_input_foreground_ratios),
    }


def _best_row(rows: Sequence[Dict[str, float]], key: str, largest: bool = True, feasible_only: bool = False) -> Optional[Dict[str, float]]:
    candidates = [row for row in rows if (not feasible_only or bool(row.get("blank_feasible", False)))]
    if not candidates:
        return None
    return sorted(candidates, key=lambda row: float(row.get(key, 0.0)), reverse=largest)[0]


def _build_summary(rows: Sequence[Dict[str, float]]) -> Dict[str, object]:
    front_indices = _pareto_front(rows)
    pareto_rows = sorted(
        [rows[index] for index in front_indices],
        key=lambda row: float(row.get("combined_score", 0.0)),
        reverse=True,
    )
    return {
        "setting_count": len(rows),
        "pareto_front": pareto_rows,
        "best_combined": _best_row(rows, "combined_score", largest=True),
        "best_feasible_combined": _best_row(rows, "combined_score", largest=True, feasible_only=True),
        "best_synthetic_f1": _best_row(rows, "synthetic_f1_median", largest=True),
        "best_blank_specificity": _best_row(rows, "blank_nonempty_rate", largest=False),
        "best_fiber_support": _best_row(rows, "fiber_raw_skeleton_contrast_median", largest=True),
    }


def _set_worker_state(
    synthetic_samples: Sequence[CachedSyntheticSample],
    fiber_samples: Sequence[CachedRealSample],
    blank_samples: Sequence[CachedRealSample],
    max_blank_nonempty_rate: float,
    max_blank_skeleton_fraction: float,
) -> None:
    global _WORKER_SYNTHETIC_SAMPLES
    global _WORKER_FIBER_SAMPLES
    global _WORKER_BLANK_SAMPLES
    global _WORKER_MAX_BLANK_NONEMPTY_RATE
    global _WORKER_MAX_BLANK_SKELETON_FRACTION

    _WORKER_SYNTHETIC_SAMPLES = synthetic_samples
    _WORKER_FIBER_SAMPLES = fiber_samples
    _WORKER_BLANK_SAMPLES = blank_samples
    _WORKER_MAX_BLANK_NONEMPTY_RATE = float(max_blank_nonempty_rate)
    _WORKER_MAX_BLANK_SKELETON_FRACTION = float(max_blank_skeleton_fraction)


def _evaluate_single_setting(task: tuple[int, float, float]) -> tuple[int, Dict[str, float]]:
    setting_index, min_edt, visibility_floor = task
    row: Dict[str, float] = {
        "min_edt": float(min_edt),
        "visibility_floor": float(visibility_floor),
    }
    row.update(_evaluate_synthetic_setting(_WORKER_SYNTHETIC_SAMPLES, min_edt=min_edt, visibility_floor=visibility_floor))
    row.update(_evaluate_real_setting(_WORKER_FIBER_SAMPLES, min_edt=min_edt, visibility_floor=visibility_floor, prefix="fiber"))
    row.update(_evaluate_real_setting(_WORKER_BLANK_SAMPLES, min_edt=min_edt, visibility_floor=visibility_floor, prefix="blank"))
    row["blank_feasible"] = (
        float(row.get("blank_nonempty_rate", 0.0)) <= _WORKER_MAX_BLANK_NONEMPTY_RATE
        and float(row.get("blank_skeleton_fraction_median", 0.0)) <= _WORKER_MAX_BLANK_SKELETON_FRACTION
    )
    row["combined_score"] = _combined_score(row)
    return setting_index, row


def _cpu_job_count(requested_jobs: int) -> int:
    if requested_jobs > 0:
        return int(requested_jobs)
    cpu_count = os.cpu_count() or 1
    return max(1, cpu_count - 1)


def main(args):
    model, device = load_sted_model(
        model_path=args.model_path,
        base_filters=args.base_filters,
        device_spec=args.device,
        aspp_dilations=args.aspp_dilations,
        unet_depth=args.unet_depth,
    )
    use_amp = not args.no_amp
    _ = load_optional_profile(args.profile)

    synthetic_samples: List[CachedSyntheticSample] = []
    if args.synthetic_data_dir:
        synthetic_files = _limit_paths(
            _discover_synthetic_files(args.synthetic_data_dir, split=args.synthetic_split),
            args.max_synthetic,
        )
        print(f"Caching predictions for {len(synthetic_files)} synthetic samples...")
        for index, file_path in enumerate(synthetic_files, start=1):
            synthetic_samples.append(
                _load_cached_synthetic_sample(
                    model=model,
                    file_path=file_path,
                    device=device,
                    tile_size=args.tile_size,
                    tile_overlap=args.tile_overlap,
                    use_amp=use_amp,
                    target_edt_threshold=args.synthetic_target_edt_threshold,
                )
            )
            if index % 25 == 0 or index == len(synthetic_files):
                print(f"  synthetic cached {index}/{len(synthetic_files)}")

    fiber_files = _limit_paths(
        _discover_tiff_files(args.real_fibers_dir, recursive=args.recursive_real),
        args.max_fibers,
    )
    blank_files = _limit_paths(
        _discover_tiff_files(args.real_blanks_dir, recursive=args.recursive_real),
        args.max_blanks,
    )
    print(f"Caching predictions for {len(fiber_files)} fiber TIFFs...")
    fiber_predictions = [
        _load_cached_real_sample(
            model=model,
            image_path=path,
            device=device,
            tile_size=args.tile_size,
            tile_overlap=args.tile_overlap,
            use_amp=use_amp,
        )
        for path in fiber_files
    ]
    print(f"Caching predictions for {len(blank_files)} blank TIFFs...")
    blank_predictions = [
        _load_cached_real_sample(
            model=model,
            image_path=path,
            device=device,
            tile_size=args.tile_size,
            tile_overlap=args.tile_overlap,
            use_amp=use_amp,
        )
        for path in blank_files
    ]

    tasks = []
    setting_index = 0
    for min_edt in args.min_edt_values:
        for visibility_floor in args.visibility_floor_values:
            setting_index += 1
            tasks.append((setting_index, float(min_edt), float(visibility_floor)))
    grid_size = len(tasks)

    _set_worker_state(
        synthetic_samples=synthetic_samples,
        fiber_samples=fiber_predictions,
        blank_samples=blank_predictions,
        max_blank_nonempty_rate=args.max_blank_nonempty_rate,
        max_blank_skeleton_fraction=args.max_blank_skeleton_fraction,
    )

    rows_by_index: Dict[int, Dict[str, float]] = {}
    jobs = _cpu_job_count(args.jobs)
    if jobs <= 1 or grid_size <= 1:
        for task in tasks:
            idx, row = _evaluate_single_setting(task)
            rows_by_index[idx] = row
            print(
                f"[{idx}/{grid_size}] min_edt={row['min_edt']:.4f} visibility_floor={row['visibility_floor']:.4f} "
                f"| synth_f1={row.get('synthetic_f1_median', 0.0):.4f} "
                f"| fiber_contrast={row.get('fiber_raw_skeleton_contrast_median', 0.0):.3f} "
                f"| blank_nonempty={row.get('blank_nonempty_rate', 0.0):.3f}"
            )
    else:
        try:
            mp_context = multiprocessing.get_context("fork")
        except ValueError:
            mp_context = None

        if mp_context is None:
            print("Parallel evaluation requested, but no 'fork' multiprocessing context is available. Falling back to sequential evaluation.")
            for task in tasks:
                idx, row = _evaluate_single_setting(task)
                rows_by_index[idx] = row
                print(
                    f"[{idx}/{grid_size}] min_edt={row['min_edt']:.4f} visibility_floor={row['visibility_floor']:.4f} "
                    f"| synth_f1={row.get('synthetic_f1_median', 0.0):.4f} "
                    f"| fiber_contrast={row.get('fiber_raw_skeleton_contrast_median', 0.0):.3f} "
                    f"| blank_nonempty={row.get('blank_nonempty_rate', 0.0):.3f}"
                )
        else:
            print(f"Evaluating {grid_size} settings with {jobs} worker processes...")
            with concurrent.futures.ProcessPoolExecutor(max_workers=jobs, mp_context=mp_context) as executor:
                future_to_task = {executor.submit(_evaluate_single_setting, task): task for task in tasks}
                for future in concurrent.futures.as_completed(future_to_task):
                    idx, row = future.result()
                    rows_by_index[idx] = row
                    print(
                        f"[{idx}/{grid_size}] min_edt={row['min_edt']:.4f} visibility_floor={row['visibility_floor']:.4f} "
                        f"| synth_f1={row.get('synthetic_f1_median', 0.0):.4f} "
                        f"| fiber_contrast={row.get('fiber_raw_skeleton_contrast_median', 0.0):.3f} "
                        f"| blank_nonempty={row.get('blank_nonempty_rate', 0.0):.3f}"
                    )

    rows = [rows_by_index[index] for index in sorted(rows_by_index)]

    summary = _build_summary(rows)
    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = _write_csv(rows, os.path.join(args.output_dir, "inference_calibration_grid.csv"))
    json_path = _write_json(summary, os.path.join(args.output_dir, "inference_calibration_summary.json"))
    print(f"Wrote grid: {csv_path}")
    print(f"Wrote summary: {json_path}")

    best = summary.get("best_feasible_combined") or summary.get("best_combined")
    if best:
        print(
            "Best setting: "
            f"min_edt={float(best['min_edt']):.4f} "
            f"visibility_floor={float(best['visibility_floor']):.4f} "
            f"combined_score={float(best['combined_score']):.4f} "
            f"blank_nonempty={float(best.get('blank_nonempty_rate', 0.0)):.3f} "
            f"synthetic_f1={float(best.get('synthetic_f1_median', 0.0)):.4f}"
        )


def add_calibration_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model_path", type=str, required=True)
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
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--tile_size", type=int, default=512)
    parser.add_argument("--tile_overlap", type=int, default=128)
    parser.add_argument("--synthetic_data_dir", type=str, default="", help="Optional synthetic dataset root. Uses --synthetic_split if present.")
    parser.add_argument("--synthetic_split", type=str, default="test")
    parser.add_argument("--synthetic_target_edt_threshold", type=float, default=0.85)
    parser.add_argument("--real_fibers_dir", type=str, default="/ssd/STED_dataset/data")
    parser.add_argument("--real_blanks_dir", type=str, default="/ssd/STED_dataset/data/blanks")
    parser.add_argument("--recursive_real", action="store_true")
    parser.add_argument("--profile", type=str, default="")
    parser.add_argument("--output_dir", type=str, default="reports/inference_calibration")
    parser.add_argument("--max_synthetic", type=int, default=0)
    parser.add_argument("--max_fibers", type=int, default=0)
    parser.add_argument("--max_blanks", type=int, default=0)
    parser.add_argument("--jobs", type=int, default=1, help="Number of CPU worker processes for parameter sweep evaluation. Use 0 for cpu_count-1.")
    parser.add_argument("--min_edt_values", type=float, nargs="+", default=[0.05, 0.075, 0.10, 0.125, 0.15, 0.175, 0.20, 0.25, 0.30])
    parser.add_argument("--visibility_floor_values", type=float, nargs="+", default=[0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40])
    parser.add_argument("--max_blank_nonempty_rate", type=float, default=0.10)
    parser.add_argument("--max_blank_skeleton_fraction", type=float, default=0.00005)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calibrate EDT+visibility decoder parameters against synthetic and real data.")
    add_calibration_arguments(parser)
    args = parser.parse_args()
    main(args)
