import json
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import scipy.ndimage as ndi

try:
    from skimage.filters import threshold_otsu
    from skimage.morphology import remove_small_objects, skeletonize
except Exception:  # pragma: no cover - exercised only when optional deps are missing.
    threshold_otsu = None
    remove_small_objects = None
    skeletonize = None


PROFILE_VERSION = 1
DEFAULT_PATCH_SIZE = 512
PROFILE_QUANTILES = (0.0, 0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 1.0)
COMPARISON_METRICS = (
    "foreground_fraction",
    "skeleton_fraction",
    "p50",
    "p90",
    "p99",
    "bg_highfreq_std",
    "lowfreq_cv",
    "stripe_strength",
    "width_median",
)


def parse_sted_filename(path: str) -> Dict[str, object]:
    name = os.path.basename(path)
    condition_match = re.search(r"_(AD|PID|PSP)_", name, flags=re.IGNORECASE)
    div_match = re.search(r"DIV\s*0?(\d+)", name, flags=re.IGNORECASE)
    replicate_match = re.search(r"(?:^|_)(\d+R|0N\d+R)(?:_|-)", name, flags=re.IGNORECASE)
    series_match = re.search(r"Series\s+(\d+)", name, flags=re.IGNORECASE)

    return {
        "name": name,
        "condition": condition_match.group(1).upper() if condition_match else "unknown",
        "div": int(div_match.group(1)) if div_match else None,
        "replicate": replicate_match.group(1).upper() if replicate_match else "unknown",
        "series": int(series_match.group(1)) if series_match else None,
    }


def normalize_image(image: np.ndarray) -> Tuple[np.ndarray, Dict[str, object]]:
    arr = np.asarray(image)
    arr = np.squeeze(arr)
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2D image after squeeze, got shape {arr.shape}.")

    original_dtype = str(arr.dtype)
    arr = arr.astype(np.float64, copy=False)
    if np.issubdtype(np.asarray(image).dtype, np.integer):
        max_value = float(np.iinfo(np.asarray(image).dtype).max)
    else:
        finite = arr[np.isfinite(arr)]
        observed_max = float(np.max(finite)) if finite.size else 1.0
        observed_min = float(np.min(finite)) if finite.size else 0.0
        if observed_min >= 0.0 and observed_max <= 1.0:
            max_value = 1.0
        else:
            max_value = float(np.nanpercentile(arr, 99.9))
            max_value = max(max_value, 1.0)

    normalized = np.clip(arr / max(max_value, 1e-8), 0.0, 1.0)
    return normalized, {"dtype": original_dtype, "scale_max": max_value}


def iter_patches(image: np.ndarray, patch_size: int, stride: Optional[int] = None):
    if patch_size <= 0:
        return

    h, w = image.shape
    if h < patch_size or w < patch_size:
        return

    stride = int(stride or patch_size)
    y_positions = list(range(0, h - patch_size + 1, stride))
    x_positions = list(range(0, w - patch_size + 1, stride))
    if y_positions[-1] != h - patch_size:
        y_positions.append(h - patch_size)
    if x_positions[-1] != w - patch_size:
        x_positions.append(w - patch_size)

    for y0 in y_positions:
        for x0 in x_positions:
            yield y0, x0, image[y0 : y0 + patch_size, x0 : x0 + patch_size]


def _safe_otsu_threshold(image: np.ndarray, fallback: float) -> float:
    if threshold_otsu is None or float(np.max(image) - np.min(image)) < 1e-8:
        return fallback
    try:
        return float(threshold_otsu(image))
    except ValueError:
        return fallback


def robust_foreground_mask(image: np.ndarray, min_component_area: int = 8) -> Tuple[np.ndarray, Dict[str, float]]:
    img = np.asarray(image, dtype=np.float64)
    median = float(np.median(img))
    mad = float(np.median(np.abs(img - median)))
    bg_sigma = float(max(1.4826 * mad, 1e-8))
    robust_threshold = median + max(4.0 * bg_sigma, 1.0 / 255.0)
    otsu_threshold = _safe_otsu_threshold(img, robust_threshold)
    p995 = float(np.percentile(img, 99.5))
    threshold = min(max(robust_threshold, otsu_threshold), p995)

    mask = img > threshold
    if remove_small_objects is not None and np.any(mask):
        mask = remove_small_objects(mask, min_size=max(1, int(min_component_area)))

    return np.asarray(mask, dtype=bool), {
        "background_median": median,
        "background_sigma": bg_sigma,
        "foreground_threshold": float(threshold),
        "otsu_threshold": float(otsu_threshold),
        "robust_threshold": float(robust_threshold),
    }


def _component_stats(mask: np.ndarray) -> Dict[str, float]:
    labels, component_count = ndi.label(mask)
    if component_count == 0:
        return {
            "component_count": 0.0,
            "component_area_median": 0.0,
            "component_area_p95": 0.0,
        }

    areas = ndi.sum(mask, labels, index=np.arange(1, component_count + 1))
    return {
        "component_count": float(component_count),
        "component_area_median": float(np.median(areas)),
        "component_area_p95": float(np.percentile(areas, 95.0)),
    }


def _skeleton_stats(mask: np.ndarray) -> Dict[str, float]:
    if not np.any(mask):
        return {
            "skeleton_fraction": 0.0,
            "width_median": 0.0,
            "width_p90": 0.0,
            "endpoint_density": 0.0,
            "junction_density": 0.0,
            "curvature_proxy": 0.0,
        }

    if skeletonize is None:
        skel = ndi.binary_erosion(mask) ^ mask
    else:
        skel = skeletonize(mask)

    skeleton_fraction = float(np.mean(skel))
    if not np.any(skel):
        return {
            "skeleton_fraction": skeleton_fraction,
            "width_median": 0.0,
            "width_p90": 0.0,
            "endpoint_density": 0.0,
            "junction_density": 0.0,
            "curvature_proxy": 0.0,
        }

    distance = ndi.distance_transform_edt(mask)
    widths = 2.0 * distance[skel]
    neighbors = ndi.convolve(skel.astype(np.uint8), np.ones((3, 3), dtype=np.uint8), mode="constant") - skel
    endpoints = skel & (neighbors == 1)
    junctions = skel & (neighbors >= 3)
    endpoint_density = float(np.mean(endpoints))
    junction_density = float(np.mean(junctions))

    return {
        "skeleton_fraction": skeleton_fraction,
        "width_median": float(np.median(widths)),
        "width_p90": float(np.percentile(widths, 90.0)),
        "endpoint_density": endpoint_density,
        "junction_density": junction_density,
        "curvature_proxy": float(junction_density + (0.5 * endpoint_density)),
    }


def _artifact_stats(image: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
    img = np.asarray(image, dtype=np.float64)
    low_sigma = max(4.0, min(img.shape) / 16.0)
    lowfreq = ndi.gaussian_filter(img, sigma=low_sigma)
    lowfreq_cv = float(np.std(lowfreq) / max(float(np.mean(lowfreq)), 1e-8))

    border = max(1, min(img.shape) // 16)
    edge_mask = np.zeros_like(mask, dtype=bool)
    edge_mask[:border, :] = True
    edge_mask[-border:, :] = True
    edge_mask[:, :border] = True
    edge_mask[:, -border:] = True
    center_mask = np.zeros_like(mask, dtype=bool)
    center_margin_y = max(1, img.shape[0] // 4)
    center_margin_x = max(1, img.shape[1] // 4)
    center_mask[center_margin_y:-center_margin_y, center_margin_x:-center_margin_x] = True
    edge_mean = float(np.mean(lowfreq[edge_mask])) if np.any(edge_mask) else 0.0
    center_mean = float(np.mean(lowfreq[center_mask])) if np.any(center_mask) else float(np.mean(lowfreq))
    vignette_strength = float((center_mean - edge_mean) / max(center_mean, 1e-8))

    row_profile = np.mean(img, axis=1)
    col_profile = np.mean(img, axis=0)
    row_hf = row_profile - ndi.gaussian_filter1d(row_profile, sigma=max(1.0, len(row_profile) / 32.0))
    col_hf = col_profile - ndi.gaussian_filter1d(col_profile, sigma=max(1.0, len(col_profile) / 32.0))
    stripe_strength = max(float(np.std(row_hf)), float(np.std(col_hf))) / max(float(np.mean(img)), 1e-8)

    background_mask = ~mask
    if float(np.mean(background_mask)) < 0.10:
        background_mask = img <= np.percentile(img, 50.0)
    highfreq = img - ndi.gaussian_filter(img, sigma=1.0)
    bg_highfreq_std = float(np.std(highfreq[background_mask])) if np.any(background_mask) else float(np.std(highfreq))

    return {
        "lowfreq_cv": lowfreq_cv,
        "vignette_strength": vignette_strength,
        "stripe_strength": float(stripe_strength),
        "bg_highfreq_std": bg_highfreq_std,
    }


def _orientation_stats(image: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
    if not np.any(mask):
        return {"orientation_anisotropy": 0.0}

    smoothed = ndi.gaussian_filter(image, sigma=1.0)
    grad_y, grad_x = np.gradient(smoothed)
    orientation = np.mod(np.arctan2(grad_y, grad_x) + (np.pi / 2.0), np.pi)
    weights = np.hypot(grad_x, grad_y) * mask
    weight_sum = float(np.sum(weights))
    if weight_sum <= 1e-8:
        return {"orientation_anisotropy": 0.0}

    resultant = np.sum(weights * np.exp(2j * orientation)) / weight_sum
    return {"orientation_anisotropy": float(np.abs(resultant))}


def compute_image_metrics(
    image: np.ndarray,
    source: str = "",
    row_type: str = "image",
    patch_index: Optional[int] = None,
    patch_origin: Optional[Tuple[int, int]] = None,
    min_component_area: int = 8,
    metadata: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    img, norm_info = normalize_image(image)
    mask, threshold_stats = robust_foreground_mask(img, min_component_area=min_component_area)

    percentiles = {
        f"p{str(p).replace('.', '')}": float(np.percentile(img, p))
        for p in (0.0, 0.5, 1.0, 5.0, 10.0, 50.0, 90.0, 95.0, 99.0, 99.5, 99.9, 100.0)
    }
    # Stable aliases used by comparison and generation code.
    percentiles["min"] = percentiles.pop("p00")
    percentiles["p50"] = percentiles.pop("p500")
    percentiles["p90"] = percentiles.pop("p900")
    percentiles["p95"] = percentiles.pop("p950")
    percentiles["p99"] = percentiles.pop("p990")
    percentiles["p995"] = percentiles.pop("p995")
    percentiles["p999"] = percentiles.pop("p999")
    percentiles["max"] = percentiles.pop("p1000")

    stats: Dict[str, object] = {
        "source": source,
        "row_type": row_type,
        "patch_index": -1 if patch_index is None else int(patch_index),
        "patch_y": -1 if patch_origin is None else int(patch_origin[0]),
        "patch_x": -1 if patch_origin is None else int(patch_origin[1]),
        "height": int(img.shape[0]),
        "width": int(img.shape[1]),
        "dtype": norm_info["dtype"],
        "scale_max": float(norm_info["scale_max"]),
        "mean": float(np.mean(img)),
        "std": float(np.std(img)),
        "nonzero_fraction": float(np.mean(img > 0.0)),
        "foreground_fraction": float(np.mean(mask)),
    }
    if metadata:
        stats.update(metadata)

    stats.update(percentiles)
    stats.update(threshold_stats)
    stats.update(_component_stats(mask))
    stats.update(_skeleton_stats(mask))
    stats.update(_artifact_stats(img, mask))
    stats.update(_orientation_stats(img, mask))

    bg_sigma = float(stats["background_sigma"])
    stats["snr_p99"] = float((float(stats["p99"]) - float(stats["background_median"])) / max(bg_sigma, 1e-8))
    return stats


def quantile_key(q: float) -> str:
    return f"q{int(round(float(q) * 100.0)):03d}"


def quantile_summary(values: Sequence[float], quantiles: Sequence[float] = PROFILE_QUANTILES) -> Dict[str, float]:
    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=np.float64)
    if arr.size == 0:
        return {quantile_key(q): 0.0 for q in quantiles}
    return {quantile_key(q): float(np.quantile(arr, q)) for q in quantiles}


def _numeric_metric_keys(rows: Sequence[Dict[str, object]]) -> List[str]:
    keys = set()
    for row in rows:
        for key, value in row.items():
            if isinstance(value, (int, float, np.integer, np.floating)) and key not in {
                "patch_index",
                "patch_y",
                "patch_x",
                "height",
                "width",
                "div",
                "series",
            }:
                keys.add(key)
    return sorted(keys)


def _summarize_rows(rows: Sequence[Dict[str, object]]) -> Dict[str, object]:
    metrics = {}
    for key in _numeric_metric_keys(rows):
        metrics[key] = quantile_summary([float(row[key]) for row in rows if key in row])
    return {
        "count": len(rows),
        "metrics": metrics,
    }


def build_calibration_profile(
    rows: Sequence[Dict[str, object]],
    source_dir: str = "",
    patch_size: int = DEFAULT_PATCH_SIZE,
) -> Dict[str, object]:
    rows = list(rows)
    image_rows = [row for row in rows if row.get("row_type") == "image"]
    patch_rows = [row for row in rows if row.get("row_type") == "patch"]
    calibration_rows = patch_rows if patch_rows else image_rows

    profile = {
        "version": PROFILE_VERSION,
        "source_dir": source_dir,
        "patch_size": int(patch_size),
        "image_count": len(image_rows),
        "patch_count": len(patch_rows),
        "calibration_row_type": "patch" if patch_rows else "image",
        "global": _summarize_rows(calibration_rows),
        "groups": {"condition": {}, "div": {}},
    }

    for group_key in ("condition", "div"):
        values = sorted({row.get(group_key) for row in calibration_rows if row.get(group_key) not in (None, "unknown")})
        for value in values:
            grouped = [row for row in calibration_rows if row.get(group_key) == value]
            if grouped:
                profile["groups"][group_key][str(value)] = _summarize_rows(grouped)

    return profile


def load_calibration_profile(path: str) -> Dict[str, object]:
    with open(path, "r", encoding="utf-8") as handle:
        profile = json.load(handle)
    validate_calibration_profile(profile)
    return profile


def validate_calibration_profile(profile: Dict[str, object]) -> None:
    if not isinstance(profile, dict):
        raise ValueError("Calibration profile must be a JSON object.")
    if profile.get("version") != PROFILE_VERSION:
        raise ValueError(f"Unsupported calibration profile version: {profile.get('version')!r}.")
    if "global" not in profile or "metrics" not in profile["global"]:
        raise ValueError("Calibration profile is missing global metric summaries.")
    missing = [metric for metric in ("foreground_fraction", "skeleton_fraction", "p99") if metric not in profile["global"]["metrics"]]
    if missing:
        raise ValueError(f"Calibration profile is missing required metrics: {', '.join(missing)}.")


def _summary_points(summary: Dict[str, float]) -> Tuple[np.ndarray, np.ndarray]:
    qs = []
    values = []
    for key, value in summary.items():
        if not key.startswith("q"):
            continue
        try:
            q = int(key[1:]) / 100.0
        except ValueError:
            continue
        qs.append(q)
        values.append(float(value))

    if not qs:
        return np.asarray([0.0, 1.0]), np.asarray([0.0, 0.0])

    order = np.argsort(qs)
    return np.asarray(qs, dtype=np.float64)[order], np.asarray(values, dtype=np.float64)[order]


def sample_from_summary(summary: Dict[str, float], low_q: float = 0.05, high_q: float = 0.95) -> float:
    qs, values = _summary_points(summary)
    q = float(np.random.uniform(low_q, high_q))
    return float(np.interp(q, qs, values))


def _clip(value: float, low: float, high: float) -> float:
    return float(np.clip(float(value), low, high))


@dataclass
class CalibrationSampler:
    profile: Dict[str, object]
    real_regime: str = "global"

    @classmethod
    def from_file(cls, path: str, real_regime: str = "global"):
        return cls(load_calibration_profile(path), real_regime=real_regime)

    def _choose_profile(self) -> Tuple[str, Dict[str, object]]:
        if self.real_regime == "global":
            return "global", self.profile["global"]

        group_profiles = self.profile.get("groups", {}).get(self.real_regime, {})
        if not group_profiles:
            return "global", self.profile["global"]

        names = sorted(group_profiles.keys())
        weights = np.asarray([max(1, int(group_profiles[name].get("count", 1))) for name in names], dtype=np.float64)
        weights /= weights.sum()
        index = int(np.random.choice(np.arange(len(names)), p=weights))
        name = names[index]
        return f"{self.real_regime}:{name}", group_profiles[name]

    @staticmethod
    def _metric(metric_profile: Dict[str, object], name: str, fallback: float, low_q: float = 0.05, high_q: float = 0.95) -> float:
        summary = metric_profile.get("metrics", {}).get(name)
        if summary is None:
            return float(fallback)
        return sample_from_summary(summary, low_q=low_q, high_q=high_q)

    def sample_scene_config(self, bounds: Tuple[int, int, int]) -> Dict[str, object]:
        profile_name, metric_profile = self._choose_profile()
        x_size, y_size, _ = bounds
        area = float(x_size * y_size)

        target_foreground = _clip(self._metric(metric_profile, "foreground_fraction", 0.016, 0.01, 0.95), 0.0, 0.30)
        target_skeleton = _clip(self._metric(metric_profile, "skeleton_fraction", 0.006, 0.01, 0.95), 0.0, 0.12)
        width_median = _clip(self._metric(metric_profile, "width_median", 2.0, 0.10, 0.90), 0.6, 12.0)
        width_p90 = _clip(self._metric(metric_profile, "width_p90", max(2.0, width_median), 0.25, 0.90), width_median, 30.0)
        bg_highfreq = _clip(self._metric(metric_profile, "bg_highfreq_std", 0.012, 0.10, 0.90), 0.0, 0.08)
        lowfreq_cv = _clip(self._metric(metric_profile, "lowfreq_cv", 0.55, 0.10, 0.90), 0.0, 3.0)
        stripe_metric = _clip(self._metric(metric_profile, "stripe_strength", 0.08, 0.10, 0.90), 0.0, 0.25)
        vignette_strength = _clip(self._metric(metric_profile, "vignette_strength", 0.12, 0.10, 0.90), -0.50, 0.80)
        component_count = max(0.0, self._metric(metric_profile, "component_count", 100.0, 0.25, 0.90))

        p50 = _clip(self._metric(metric_profile, "p50", 4.0 / 255.0, 0.10, 0.90), 0.0, 1.0)
        p90 = _clip(self._metric(metric_profile, "p90", 13.0 / 255.0, 0.10, 0.90), 0.0, 1.0)
        p99 = _clip(self._metric(metric_profile, "p99", 26.0 / 255.0, 0.10, 0.90), 0.0, 1.0)
        p999 = _clip(self._metric(metric_profile, "p999", 41.0 / 255.0, 0.10, 0.95), 0.0, 1.0)

        dense = target_skeleton >= self._metric(metric_profile, "skeleton_fraction", 0.006, 0.75, 0.75)
        very_sparse = target_skeleton <= max(1e-5, self._metric(metric_profile, "skeleton_fraction", 0.001, 0.10, 0.10))
        step_scale_range = (0.006, 0.030) if very_sparse else ((0.008, 0.040) if dense else (0.008, 0.035))
        num_steps_range = (6, 30) if very_sparse else ((20, 70) if dense else (10, 55))
        turn_degrees_range = (0.5, 28.0) if very_sparse else (0.5, 40.0)

        if width_p90 >= 6.0:
            bundle_probs = (0.45, 0.40, 0.15)
        elif width_p90 >= 3.5:
            bundle_probs = (0.60, 0.32, 0.08)
        else:
            bundle_probs = (0.78, 0.20, 0.02)

        base_sigma = _clip(width_median / 2.35, 0.55, 2.20)
        noise_level = _clip(0.005 + (bg_highfreq / 0.030) * 0.040, 0.005, 0.045)
        debris_density = component_count / max(float(self.profile.get("patch_size", DEFAULT_PATCH_SIZE)) ** 2, 1.0)
        debris_count = int(np.clip(area * debris_density * 0.15, 0, max(1, area / 1024.0)))
        gap_prob = _clip(np.random.uniform(0.0, 0.06) + (0.04 if very_sparse else 0.0), 0.0, 0.12)

        q75_lowfreq = self._metric(metric_profile, "lowfreq_cv", 0.75, 0.75, 0.75)
        q90_lowfreq = self._metric(metric_profile, "lowfreq_cv", 1.0, 0.90, 0.90)
        if lowfreq_cv >= q90_lowfreq:
            haze_regime = "strong"
        elif lowfreq_cv >= q75_lowfreq:
            haze_regime = "moderate"
        elif lowfreq_cv >= 0.35:
            haze_regime = "subtle"
        else:
            haze_regime = "none"

        edge_center = _clip(1.0 - max(0.0, vignette_strength), 0.55, 0.98)
        vignette_edge_range = (_clip(edge_center - 0.05, 0.50, 0.98), _clip(edge_center + 0.05, 0.55, 1.0))
        dynamic_min = _clip(max(p50 * 0.8, p90 * 0.35), 0.005, 0.45)
        dynamic_max = _clip(max(dynamic_min + 0.04, p999 * 1.25), dynamic_min + 0.02, 1.0)

        return {
            "profile_name": profile_name,
            "target_foreground_fraction": float(target_foreground),
            "target_skeleton_fraction": float(target_skeleton),
            "base_sigma": float(base_sigma),
            "z_anisotropy": float(np.random.uniform(1.4, 2.8)),
            "noise_level": float(noise_level),
            "debris_count": int(debris_count),
            "gap_prob": float(gap_prob),
            "haze_regime": haze_regime,
            "dynamic_range": (float(dynamic_min), float(dynamic_max)),
            "target_p50": float(p50),
            "target_p90": float(p90),
            "target_p99": float(p99),
            "target_p999": float(p999),
            "target_bg_highfreq_std": float(bg_highfreq),
            "step_scale_range": step_scale_range,
            "num_steps_range": num_steps_range,
            "turn_degrees_range": turn_degrees_range,
            "bundle_probs": bundle_probs,
            "jitter_range": (float(max(0.10, base_sigma * 0.20)), float(max(0.50, base_sigma * 1.20))),
            "stripe_strength": float(_clip(stripe_metric * 0.20, 0.0, 0.04)),
            "target_stripe_strength": float(stripe_metric),
            "target_lowfreq_cv": float(lowfreq_cv),
            "vignette_edge_range": vignette_edge_range,
        }


def estimate_calibrated_fiber_count(
    bounds: Tuple[int, int, int],
    scene_config: Dict[str, object],
    depth_of_field: float,
) -> int:
    x_size, y_size, z_size = bounds
    area = float(x_size * y_size)
    target_skeleton = float(scene_config.get("target_skeleton_fraction", 0.006))
    target_length = area * target_skeleton
    if target_length < 1.0:
        return 0

    step_min, step_max = scene_config.get("step_scale_range", (0.008, 0.035))
    step_scale = 0.5 * (float(step_min) + float(step_max))
    step_length = max(0.8, max(x_size, y_size) * step_scale)
    steps_min, steps_max = scene_config.get("num_steps_range", (10, 55))
    num_steps = 0.5 * (int(steps_min) + int(steps_max))
    visible_fraction = _clip((2.0 * float(depth_of_field)) / max(float(z_size), 1.0), 0.15, 1.0)
    expected_visible_length = max(1.0, step_length * num_steps * visible_fraction * 0.35)
    estimated = int(round(target_length / expected_visible_length))

    max_fibers = max(1, int(area / 16000.0))
    return int(np.clip(estimated, 0, max_fibers))


def match_intensity_to_scene_profile(image: np.ndarray, scene_config: Dict[str, object]) -> np.ndarray:
    """Affine-match synthetic contrast to sampled real percentiles, then add missing background noise."""
    out = np.asarray(image, dtype=np.float64)
    target_p50 = _clip(float(scene_config.get("target_p50", np.percentile(out, 50.0))), 0.0, 1.0)
    target_p99 = _clip(float(scene_config.get("target_p99", np.percentile(out, 99.0))), target_p50 + 1e-4, 1.0)

    source_p50 = float(np.percentile(out, 50.0))
    source_p99 = float(np.percentile(out, 99.0))
    if source_p99 - source_p50 > 1e-5:
        scale = (target_p99 - target_p50) / (source_p99 - source_p50)
        out = (out - source_p50) * scale + target_p50
    else:
        out = out + (target_p50 - source_p50)
    out = np.clip(out, 0.0, 1.0)

    target_lowfreq_cv = max(0.0, float(scene_config.get("target_lowfreq_cv", 0.0)))
    if target_lowfreq_cv > 0.0:
        low_sigma = max(4.0, min(out.shape) / 16.0)
        lowfreq = ndi.gaussian_filter(out, sigma=low_sigma)
        current_lowfreq_cv = float(np.std(lowfreq) / max(float(np.mean(lowfreq)), 1e-8))
        missing_cv = max(0.0, target_lowfreq_cv - current_lowfreq_cv)
        if missing_cv > 0.02:
            field = np.random.normal(0.0, 1.0, size=out.shape)
            field = ndi.gaussian_filter(field, sigma=max(4.0, min(out.shape) / 8.0))
            field = field - float(np.mean(field))
            field_std = float(np.std(field))
            if field_std > 1e-8:
                field /= field_std
                out = out + (field * float(np.mean(out)) * min(0.75, missing_cv))
                out = np.clip(out, 0.0, 1.0)

    target_bg_hf = max(0.0, float(scene_config.get("target_bg_highfreq_std", 0.0)))
    if target_bg_hf > 0.0:
        highfreq = out - ndi.gaussian_filter(out, sigma=1.0)
        background = out <= np.percentile(out, 60.0)
        current_bg_hf = float(np.std(highfreq[background])) if np.any(background) else float(np.std(highfreq))
        missing_sigma = max(0.0, target_bg_hf - current_bg_hf)
        if missing_sigma > 1e-5:
            out = out + np.random.normal(0.0, missing_sigma, size=out.shape)

    return np.clip(out, 0.0, 1.0)
