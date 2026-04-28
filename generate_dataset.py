import argparse
import concurrent.futures
import json
import math
import os
import time
from functools import partial
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.core import FiberSegment, ReflectiveBoundary
from src.rasterization import EmpiricalRasterizer
from src.sted import vector_to_orientation_channels_np
from src.sted_calibration import (
    CalibrationSampler,
    load_calibration_profile,
    match_intensity_to_scene_profile,
    sample_from_summary,
)
from src.synthesis import RandomWalkGenerator
from src.targets import (
    ScalarTargetGenerator2D,
    StructuralTargetGenerator2D,
    TargetFieldGenerator,
    WeightedVisibilityTargetGenerator,
)


# -----------------------------------------------------------------------------
# Defaults chosen for the current 2D STED training pipeline.
#
# Important design choices:
#   * 2D samples by default, because train.py expects 2D structural targets.
#   * A 64 px synthetic z-depth by default, because thick apparent fibers imply a
#     broader optical section than the old thin-fiber defaults.
#   * Calibrated synthesis when a real-data profile is provided, with an
#     uncalibrated smoke/debug path for tests and visualization.
#   * Fixed-pixel centerline-label widths, decoupled from rendered fiber width.
#   * Fixed-pixel radius normalization, so the radius head learns a stable pixel
#     scale rather than a scene-relative scale.
#   * Explicit structural_v2 targets with six channels, including bundle count.
# -----------------------------------------------------------------------------

SHORT_FIBER_STEPS = (10, 60)
SHORT_TURN_DEGREES = (0.5, 35.0)
TARGET_MAX_DISTANCE = 5.0

DEFAULT_LABEL_SLAB_SCALE = 1.3
DEFAULT_SOFT_SKELETON_ALPHA = 0.25
DEFAULT_ANNOTATION_WEIGHT_FLOOR = 0.10
DEFAULT_VISIBILITY_WEIGHT_FLOOR = 0.05
DEFAULT_STRUCTURAL_ANNOTATION_ALPHA = 0.25
DEFAULT_STRUCTURAL_CENTERLINE_SIGMA_PX = 1.05
DEFAULT_STRUCTURAL_ANNOTATION_SIGMA_PX = 1.60
DEFAULT_RADIUS_SIGMA_NORMALIZER_PX = 6.0
DEFAULT_BUNDLE_COUNT_NORMALIZER = 6.0
DEFAULT_UNCALIBRATED_BASE_SIGMA_RANGE = (1.2, 2.2)

# Lower than the old default. The old range became extremely cluttered at
# 512/1024 px because it scaled directly with image area.
LEGACY_2D_FIBER_DENSITY_RANGE = (4.0e-5, 1.4e-4)
STED_2D_FIBER_DENSITY_RANGE = (4.0e-5, 7.0e-4)
STED_2D_STEP_SCALE_RANGE = (0.008, 0.035)
STED_2D_INITIAL_DIR_SCALE = np.array([1.0, 1.0, 0.25], dtype=float)
STED_2D_ORTHOGONAL_SCALE = np.array([1.0, 1.0, 0.35], dtype=float)
STED_2D_HAZE_REGIMES = np.array(["none", "subtle", "moderate", "strong"], dtype=object)
STED_2D_HAZE_PROBS = np.array([0.25, 0.35, 0.28, 0.12], dtype=np.float64)

DEFAULT_COHERENT_BUNDLE_PROBABILITY = 0.35
DEFAULT_COHERENT_BUNDLE_SIZE_RANGE = (2, 6)
DEFAULT_COHERENT_BUNDLE_SEPARATION_RANGE = (1.0, 4.0)
DEFAULT_OPTICAL_JITTER_RANGE = (0.25, 1.50)
DEFAULT_MAX_FIBER_AREA_PX = 8000.0
DEFAULT_MIN_FIBERS = 1
DEFAULT_MAX_FIBERS = 0  # 0 means area-derived cap only.

DEFAULT_APPARENT_WIDTH_P90_WEIGHT = 0.75
DEFAULT_BASE_SIGMA_MIN = 0.90
DEFAULT_BASE_SIGMA_MAX = 2.80


# -----------------------------------------------------------------------------
# Generic helpers
# -----------------------------------------------------------------------------


def _validate_probability(name: str, value: float) -> None:
    if value < 0.0 or value > 1.0:
        raise ValueError(f"{name} must be in the interval [0, 1].")


def _validate_positive(name: str, value: float) -> None:
    if value <= 0.0:
        raise ValueError(f"{name} must be greater than 0.")


def _validate_range(name: str, values: Sequence[float], *, positive: bool = False) -> Tuple[float, float]:
    if len(values) != 2:
        raise ValueError(f"{name} must contain exactly two values.")
    lo, hi = float(values[0]), float(values[1])
    if lo > hi:
        raise ValueError(f"{name} lower bound must be <= upper bound.")
    if positive and lo <= 0.0:
        raise ValueError(f"{name} values must be positive.")
    return lo, hi


def _clip(value: float, low: float, high: float) -> float:
    return float(np.clip(float(value), float(low), float(high)))


def _sample_sted_haze_regime() -> str:
    return str(np.random.choice(STED_2D_HAZE_REGIMES, p=STED_2D_HAZE_PROBS))


def _resolve_sted_monomer_config(haze_regime: str):
    if haze_regime == "none":
        return False, (0.70, 0.20, 0.10)
    if haze_regime == "subtle":
        return True, (1.0, 0.0, 0.0)
    if haze_regime == "moderate":
        return True, (0.0, 1.0, 0.0)
    if haze_regime == "strong":
        return True, (0.0, 0.0, 1.0)
    raise ValueError(f"Unsupported haze regime '{haze_regime}'.")


def _safe_unit_vector(v: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(v))
    if norm < 1e-8:
        return np.asarray(fallback, dtype=np.float64)
    return np.asarray(v, dtype=np.float64) / norm


def _clip_segment_to_bounds(seg: FiberSegment, bounds: Tuple[int, int, int]) -> FiberSegment:
    upper = np.asarray(bounds, dtype=np.float64) - 1.0
    start = np.clip(seg.start, 0.0, upper[: seg.start.shape[0]])
    end = np.clip(seg.end, 0.0, upper[: seg.end.shape[0]])
    return FiberSegment(start=start, end=end, thickness_mult=seg.thickness_mult)


# -----------------------------------------------------------------------------
# Real-data calibration repair
# -----------------------------------------------------------------------------


def _profile_from_profile_name(profile: Dict[str, object], profile_name: str) -> Dict[str, object]:
    if profile_name == "global":
        return profile["global"]
    if ":" not in profile_name:
        return profile["global"]
    group_key, group_value = profile_name.split(":", 1)
    return profile.get("groups", {}).get(group_key, {}).get(group_value, profile["global"])


def _sample_metric(
    metric_profile: Dict[str, object],
    metric_name: str,
    fallback: float,
    low_q: float,
    high_q: float,
) -> float:
    summary = metric_profile.get("metrics", {}).get(metric_name)
    if summary is None:
        return float(fallback)
    return float(sample_from_summary(summary, low_q=low_q, high_q=high_q))


def _repair_scene_config_width_calibration(
    scene_config: Dict[str, object],
    calibration_profile: Dict[str, object],
    apparent_width_p90_weight: float,
    base_sigma_min: float,
    base_sigma_max: float,
    base_sigma_override: Optional[float],
) -> Dict[str, object]:
    """Use an apparent-width statistic instead of only width_median.

    The original calibration code maps width_median / 2.35 to base_sigma. For
    real STED images where the skeleton-derived median width collapses toward
    two pixels, this underestimates visibly thick fibers. Here we use
    max(width_median, apparent_width_p90_weight * width_p90), then convert this
    apparent FWHM-like width to a Gaussian sigma.
    """
    cfg = dict(scene_config)
    if base_sigma_override is not None:
        cfg["base_sigma_original"] = float(scene_config.get("base_sigma", base_sigma_override))
        cfg["base_sigma"] = float(base_sigma_override)
        cfg["base_sigma_source"] = "override"
        return cfg

    profile_name = str(cfg.get("profile_name", "global"))
    metric_profile = _profile_from_profile_name(calibration_profile, profile_name)
    width_median = _clip(_sample_metric(metric_profile, "width_median", 2.0, 0.10, 0.90), 0.6, 16.0)
    width_p90 = _clip(_sample_metric(metric_profile, "width_p90", max(2.0, width_median), 0.25, 0.95), width_median, 32.0)
    apparent_width = max(width_median, float(apparent_width_p90_weight) * width_p90)
    base_sigma = _clip(apparent_width / 2.35, base_sigma_min, base_sigma_max)

    cfg["base_sigma_original"] = float(scene_config.get("base_sigma", base_sigma))
    cfg["base_sigma"] = float(base_sigma)
    cfg["base_sigma_source"] = "apparent_width"
    cfg["calibrated_width_median_px"] = float(width_median)
    cfg["calibrated_width_p90_px"] = float(width_p90)
    cfg["calibrated_apparent_width_px"] = float(apparent_width)

    # Keep related priors consistent with the repaired fiber width.
    cfg["jitter_range"] = (
        float(max(0.15, base_sigma * 0.25)),
        float(max(0.75, base_sigma * 1.35)),
    )
    return cfg


# -----------------------------------------------------------------------------
# Fiber generation and optical bundle rendering
# -----------------------------------------------------------------------------


def _generate_constrained_random_walk(bounds: Tuple[int, int, int], scene_config: Optional[dict] = None):
    x_size, y_size, z_size = bounds
    scene_config = scene_config or {}
    step_scale_range = scene_config.get("step_scale_range", STED_2D_STEP_SCALE_RANGE)
    num_steps_range = scene_config.get("num_steps_range", SHORT_FIBER_STEPS)
    turn_degrees_range = scene_config.get("turn_degrees_range", SHORT_TURN_DEGREES)

    step_length = max(0.8, max(x_size, y_size) * np.random.uniform(*step_scale_range))
    num_steps = np.random.randint(int(num_steps_range[0]), int(num_steps_range[1]) + 1)
    max_turn_angle = np.deg2rad(np.random.uniform(float(turn_degrees_range[0]), float(turn_degrees_range[1])))

    start_pos = (
        np.random.uniform(x_size * 0.1, x_size * 0.9),
        np.random.uniform(y_size * 0.1, y_size * 0.9),
        np.random.uniform(z_size * 0.2, z_size * 0.8),
    )
    initial_direction = np.random.normal(size=3) * scene_config.get("initial_dir_scale", STED_2D_INITIAL_DIR_SCALE)
    orthogonal_scale = scene_config.get("orthogonal_scale", STED_2D_ORTHOGONAL_SCALE)

    generator = RandomWalkGenerator(
        start_pos=start_pos,
        num_steps=num_steps,
        step_length=step_length,
        max_turn_angle=max_turn_angle,
        boundary=ReflectiveBoundary(bounds),
        initial_direction=initial_direction,
        orthogonal_scale=orthogonal_scale,
    )
    return generator.generate(), {
        "num_steps": int(num_steps),
        "step_length": float(step_length),
        "max_turn_degrees": float(np.rad2deg(max_turn_angle)),
    }


def apply_optical_jitter(core_segments, bundle_size: int = 3, jitter_amount: float = 1.5, lock_z: bool = False):
    """Spawns incoherently jittered strands for rasterization.

    The target geometry remains the single backbone. Bundle multiplicity is
    stored separately in the bundle-count target.
    """
    optical_segments = []
    for seg in core_segments:
        optical_segments.append(seg)
        for _ in range(int(bundle_size) - 1):
            jitter_start = np.random.normal(0, jitter_amount, 3)
            jitter_end = np.random.normal(0, jitter_amount, 3)
            if lock_z:
                jitter_start[2] = 0.0
                jitter_end[2] = 0.0
            optical_segments.append(
                FiberSegment(
                    start=seg.start + jitter_start,
                    end=seg.end + jitter_end,
                    thickness_mult=seg.thickness_mult * np.random.uniform(0.45, 0.90),
                )
            )
    return optical_segments


def apply_coherent_optical_bundle(
    core_segments,
    bounds: Tuple[int, int, int],
    bundle_size: int,
    separation_range: Tuple[float, float],
    local_jitter_scale: float = 0.20,
):
    """Spawns laterally separated, coherent strands around one target backbone.

    This produces visually resolvable or semi-resolvable fiber bundles while the
    supervision still describes the local bundle multiplicity of the backbone.
    """
    optical_segments = list(core_segments)
    if bundle_size <= 1:
        return optical_segments

    lo, hi = separation_range
    for strand_index in range(int(bundle_size) - 1):
        angle = np.random.uniform(0.0, 2.0 * np.pi)
        offset_mag = np.random.uniform(lo, hi)
        offset = np.array([
            math.cos(angle) * offset_mag,
            math.sin(angle) * offset_mag,
            np.random.normal(0.0, 0.35 * offset_mag),
        ])
        strand_thickness = np.random.uniform(0.50, 0.95)
        for seg in core_segments:
            local_start = np.random.normal(0.0, local_jitter_scale, 3)
            local_end = np.random.normal(0.0, local_jitter_scale, 3)
            bundled = FiberSegment(
                start=seg.start + offset + local_start,
                end=seg.end + offset + local_end,
                thickness_mult=seg.thickness_mult * strand_thickness,
            )
            optical_segments.append(_clip_segment_to_bounds(bundled, bounds))
    return optical_segments


def apply_coherent_bundle(
    core_segments,
    bundle_size: int,
    separation: float,
    direction_jitter_degrees: float = 0.0,
    lateral_jitter_fraction: float = 0.0,
    axial_jitter_fraction: float = 0.0,
    lock_z: bool = False,
    bounds: Tuple[int, int, int] = (64, 64, 16),
):
    optical_segments = []
    center = 0.5 * (int(bundle_size) - 1)
    for strand_index in range(int(bundle_size)):
        offset = np.array([0.0, (strand_index - center) * float(separation), 0.0])
        for seg in core_segments:
            if direction_jitter_degrees or lateral_jitter_fraction or axial_jitter_fraction:
                jitter_scale = float(separation) * float(lateral_jitter_fraction)
                jitter = np.random.normal(0.0, jitter_scale, 3)
                if lock_z:
                    jitter[2] = 0.0
            else:
                jitter = np.zeros(3)
            bundled = FiberSegment(
                start=seg.start + offset + jitter,
                end=seg.end + offset + jitter,
                thickness_mult=seg.thickness_mult,
            )
            optical_segments.append(_clip_segment_to_bounds(bundled, bounds))
    return optical_segments


def _sample_bundle_size(
    scene_config: Optional[dict],
    coherent_bundle_probability: float,
    coherent_bundle_size_range: Tuple[int, int],
) -> Tuple[int, bool]:
    if np.random.random() < coherent_bundle_probability:
        lo, hi = coherent_bundle_size_range
        return int(np.random.randint(int(lo), int(hi) + 1)), True

    if scene_config is not None and scene_config.get("bundle_probs") is not None:
        probs = np.asarray(scene_config.get("bundle_probs"), dtype=np.float64)
        probs = probs / max(float(probs.sum()), 1e-12)
        sizes = np.arange(1, len(probs) + 1)
        return int(np.random.choice(sizes, p=probs)), False

    # Conservative non-coherent optical bundle prior for uncalibrated smoke/debug scenes.
    sizes = np.asarray([1, 2, 3], dtype=int)
    probs = np.asarray([0.70, 0.25, 0.05], dtype=np.float64)
    return int(np.random.choice(sizes, p=probs)), False


# -----------------------------------------------------------------------------
# Axial projection and target generation
# -----------------------------------------------------------------------------


def _clip_segment_to_z_slab(segment: FiberSegment, lower_z: float, upper_z: float):
    start = segment.start
    end = segment.end
    delta = end - start
    dz = delta[2]

    if abs(dz) < 1e-8:
        if lower_z <= start[2] <= upper_z:
            return start.copy(), end.copy()
        return None

    t0 = (lower_z - start[2]) / dz
    t1 = (upper_z - start[2]) / dz
    t_enter = max(0.0, min(t0, t1))
    t_exit = min(1.0, max(t0, t1))
    if t_enter > t_exit:
        return None
    return start + (delta * t_enter), start + (delta * t_exit)


def _project_segment_to_xy(segment: FiberSegment):
    start_xy = np.array([segment.start[0], segment.start[1]], dtype=np.float64)
    end_xy = np.array([segment.end[0], segment.end[1]], dtype=np.float64)
    if np.linalg.norm(end_xy - start_xy) < 1e-6:
        return None
    return FiberSegment(start=start_xy, end=end_xy, thickness_mult=segment.thickness_mult)


def _project_segments_to_z_band(core_segments, lower_z: float, upper_z: float):
    projected_segments = []
    clipped_segments = []
    for segment in core_segments:
        clipped = _clip_segment_to_z_slab(segment, lower_z, upper_z)
        if clipped is None:
            continue
        clipped_start, clipped_end = clipped
        clipped_segment = FiberSegment(
            start=np.asarray(clipped_start, dtype=np.float64),
            end=np.asarray(clipped_end, dtype=np.float64),
            thickness_mult=segment.thickness_mult,
        )
        projected = _project_segment_to_xy(clipped_segment)
        if projected is None:
            continue
        projected_segments.append(projected)
        clipped_segments.append(clipped_segment)
    return projected_segments, clipped_segments


def _project_segments_to_label_slab(core_segments, slice_center: float, slab_thickness: float):
    lower_z = slice_center - (slab_thickness / 2.0)
    upper_z = slice_center + (slab_thickness / 2.0)
    projected_segments, _ = _project_segments_to_z_band(core_segments, lower_z, upper_z)
    return projected_segments


def _project_segments_and_values_to_z_band(core_segments, values, lower_z: float, upper_z: float):
    if len(core_segments) != len(values):
        raise ValueError("core_segments and values must have the same length.")
    projected_segments = []
    projected_values = []
    clipped_segments = []
    for segment, value in zip(core_segments, values):
        clipped = _clip_segment_to_z_slab(segment, lower_z, upper_z)
        if clipped is None:
            continue
        clipped_start, clipped_end = clipped
        clipped_segment = FiberSegment(
            start=np.asarray(clipped_start, dtype=np.float64),
            end=np.asarray(clipped_end, dtype=np.float64),
            thickness_mult=segment.thickness_mult,
        )
        projected = _project_segment_to_xy(clipped_segment)
        if projected is None:
            continue
        projected_segments.append(projected)
        projected_values.append(float(value))
        clipped_segments.append(clipped_segment)
    return projected_segments, np.asarray(projected_values, dtype=np.float64), clipped_segments


def _resolve_axial_weight_band_half_width(depth_of_field: float, weight_floor: float) -> float:
    if weight_floor <= 0.0 or weight_floor > 1.0:
        raise ValueError("weight_floor must be in the interval (0, 1].")
    return float(depth_of_field * np.sqrt((1.0 / float(weight_floor)) - 1.0))


def _mean_axial_weight_over_segment(clipped_segment: FiberSegment, slice_center: float, depth_of_field: float) -> float:
    z0 = float(clipped_segment.start[2])
    z1 = float(clipped_segment.end[2])
    dz = z1 - z0
    if abs(dz) < 1e-8:
        relative_z = (z0 - slice_center) / max(depth_of_field, 1e-8)
        return float(np.clip(1.0 / (1.0 + (relative_z**2)), 0.0, 1.0))

    rel0 = (z0 - slice_center) / max(depth_of_field, 1e-8)
    rel1 = (z1 - slice_center) / max(depth_of_field, 1e-8)
    integral = depth_of_field * (np.arctan(rel1) - np.arctan(rel0))
    mean_weight = abs(integral / dz)
    return float(np.clip(mean_weight, 0.0, 1.0))


def _project_segments_to_visibility(core_segments, slice_center, rasterizer, min_weight: float):
    depth_of_field, _, _ = rasterizer._sted_optical_section_params()
    half_width = _resolve_axial_weight_band_half_width(depth_of_field, min_weight)
    lower_z = slice_center - half_width
    upper_z = slice_center + half_width
    projected_segments, clipped_segments = _project_segments_to_z_band(core_segments, lower_z, upper_z)
    visibility_weights = [
        _mean_axial_weight_over_segment(clipped_segment, slice_center, depth_of_field)
        for clipped_segment in clipped_segments
    ]
    return projected_segments, np.asarray(visibility_weights, dtype=np.float64)


def _resolve_localization_slab_thickness(rasterizer, label_slab_thickness: Optional[float], label_slab_scale: float) -> float:
    _validate_positive("label_slab_scale", label_slab_scale)
    if label_slab_thickness is not None:
        _validate_positive("label_slab_thickness", label_slab_thickness)
        return float(label_slab_thickness)
    depth_of_field, _, _ = rasterizer._sted_optical_section_params()
    return float(depth_of_field * label_slab_scale)


def _build_2d_targets(
    core_segments,
    core_segment_bundle_counts: Sequence[float],
    slice_center: float,
    localization_slab_thickness: float,
    rasterizer,
    target_gen,
    visibility_target_gen,
    annotation_weight_floor: float,
    soft_skeleton_alpha: float,
    visibility_weight_floor: float,
    structural_annotation_alpha: float,
    structural_centerline_sigma_px: float,
    structural_annotation_sigma_px: float,
    radius_sigma_normalizer_px: float,
    bundle_count_normalizer: float,
):
    _validate_probability("soft_skeleton_alpha", soft_skeleton_alpha)
    _validate_probability("structural_annotation_alpha", structural_annotation_alpha)
    _validate_positive("structural_centerline_sigma_px", structural_centerline_sigma_px)
    _validate_positive("structural_annotation_sigma_px", structural_annotation_sigma_px)
    _validate_positive("radius_sigma_normalizer_px", radius_sigma_normalizer_px)
    _validate_positive("bundle_count_normalizer", bundle_count_normalizer)
    if annotation_weight_floor <= 0.0 or annotation_weight_floor > 1.0:
        raise ValueError("annotation_weight_floor must be in the interval (0, 1].")
    if visibility_weight_floor <= 0.0 or visibility_weight_floor > 1.0:
        raise ValueError("visibility_weight_floor must be in the interval (0, 1].")

    if len(core_segments) != len(core_segment_bundle_counts):
        raise ValueError("core_segments and core_segment_bundle_counts must have the same length.")

    focus_lower_z = slice_center - (localization_slab_thickness / 2.0)
    focus_upper_z = slice_center + (localization_slab_thickness / 2.0)
    focus_segments, focus_bundle_values, _ = _project_segments_and_values_to_z_band(
        core_segments,
        core_segment_bundle_counts,
        focus_lower_z,
        focus_upper_z,
    )

    depth_of_field, _, _ = rasterizer._sted_optical_section_params()
    annotation_half_width = _resolve_axial_weight_band_half_width(depth_of_field, annotation_weight_floor)
    annotation_segments, annotation_bundle_values, _ = _project_segments_and_values_to_z_band(
        core_segments,
        core_segment_bundle_counts,
        slice_center - annotation_half_width,
        slice_center + annotation_half_width,
    )

    edt_focus, vector_focus = target_gen.generate(focus_segments)
    edt_annotation, vector_annotation = target_gen.generate(annotation_segments)
    edt_soft = np.clip(edt_annotation * float(soft_skeleton_alpha), 0.0, 1.0)
    edt_target = np.maximum(edt_focus, edt_soft)
    vector_target = np.array(vector_focus, copy=True)
    soft_overwrite_mask = edt_soft > edt_focus
    if np.any(soft_overwrite_mask):
        vector_target[:, soft_overwrite_mask] = vector_annotation[:, soft_overwrite_mask]

    visibility_segments, visibility_weights = _project_segments_to_visibility(
        core_segments,
        slice_center,
        rasterizer,
        min_weight=visibility_weight_floor,
    )
    visibility_target = visibility_target_gen.generate(visibility_segments, visibility_weights)

    structural_focus_target_gen = StructuralTargetGenerator2D(
        target_gen.grid_shape,
        base_sigma=rasterizer.base_sigma,
        centerline_sigma=structural_centerline_sigma_px,
        radius_normalizer=radius_sigma_normalizer_px,
    )
    structural_annotation_target_gen = StructuralTargetGenerator2D(
        target_gen.grid_shape,
        base_sigma=rasterizer.base_sigma,
        centerline_sigma=max(structural_centerline_sigma_px, structural_annotation_sigma_px),
        radius_normalizer=radius_sigma_normalizer_px,
    )
    centerline_focus, vector_focus_structural, radius_focus = structural_focus_target_gen.generate(focus_segments)
    centerline_annotation, vector_annotation_structural, radius_annotation = structural_annotation_target_gen.generate(annotation_segments)
    centerline_soft = np.clip(centerline_annotation * float(structural_annotation_alpha), 0.0, 1.0)
    centerline_target = np.maximum(centerline_focus, centerline_soft)
    structural_vector_target = np.array(vector_focus_structural, copy=True)
    radius_target = np.array(radius_focus, copy=True)

    bundle_focus_gen = ScalarTargetGenerator2D(
        target_gen.grid_shape,
        centerline_sigma=structural_centerline_sigma_px,
        value_normalizer=bundle_count_normalizer,
    )
    bundle_annotation_gen = ScalarTargetGenerator2D(
        target_gen.grid_shape,
        centerline_sigma=max(structural_centerline_sigma_px, structural_annotation_sigma_px),
        value_normalizer=bundle_count_normalizer,
    )
    bundle_focus = bundle_focus_gen.generate(focus_segments, focus_bundle_values)
    bundle_annotation = bundle_annotation_gen.generate(annotation_segments, annotation_bundle_values)
    bundle_count_target = np.array(bundle_focus, copy=True)

    centerline_soft_overwrite_mask = centerline_soft > centerline_focus
    if np.any(centerline_soft_overwrite_mask):
        structural_vector_target[:, centerline_soft_overwrite_mask] = vector_annotation_structural[:, centerline_soft_overwrite_mask]
        radius_target[centerline_soft_overwrite_mask] = radius_annotation[centerline_soft_overwrite_mask]
        bundle_count_target[centerline_soft_overwrite_mask] = bundle_annotation[centerline_soft_overwrite_mask]

    return {
        "focus_segments": focus_segments,
        "annotation_segments": annotation_segments,
        "edt_focus": edt_focus,
        "vector_focus": vector_focus,
        "edt_annotation": edt_annotation,
        "vector_annotation": vector_annotation,
        "edt_soft": edt_soft,
        "edt_target": edt_target,
        "vector_target": vector_target,
        "visibility_target": np.clip(visibility_target, 0.0, 1.0),
        "centerline_focus": centerline_focus,
        "centerline_soft": centerline_soft,
        "centerline_target": centerline_target,
        "structural_vector_target": structural_vector_target,
        "traceability_target": np.clip(visibility_target, 0.0, 1.0),
        "radius_target": np.clip(radius_target, 0.0, 1.0),
        "bundle_count_target": np.clip(bundle_count_target, 0.0, 1.0),
        "visibility_segments": visibility_segments,
        "visibility_weights": visibility_weights,
        "bundle_focus": bundle_focus,
        "bundle_annotation": bundle_annotation,
    }


def _build_2d_focus_and_visibility_targets(
    core_segments,
    slice_center: float,
    localization_slab_thickness: float,
    rasterizer,
    target_gen,
    visibility_target_gen,
    annotation_weight_floor: float = DEFAULT_ANNOTATION_WEIGHT_FLOOR,
    soft_skeleton_alpha: float = DEFAULT_SOFT_SKELETON_ALPHA,
    visibility_weight_floor: float = DEFAULT_VISIBILITY_WEIGHT_FLOOR,
    structural_annotation_alpha: float = DEFAULT_STRUCTURAL_ANNOTATION_ALPHA,
    structural_centerline_sigma_px: float = DEFAULT_STRUCTURAL_CENTERLINE_SIGMA_PX,
    structural_annotation_sigma_px: float = DEFAULT_STRUCTURAL_ANNOTATION_SIGMA_PX,
    radius_sigma_normalizer_px: float = DEFAULT_RADIUS_SIGMA_NORMALIZER_PX,
    bundle_count_normalizer: float = DEFAULT_BUNDLE_COUNT_NORMALIZER,
):
    bundle_counts = np.ones(len(core_segments), dtype=np.float64)
    return _build_2d_targets(
        core_segments,
        bundle_counts,
        slice_center,
        localization_slab_thickness,
        rasterizer,
        target_gen,
        visibility_target_gen,
        annotation_weight_floor,
        soft_skeleton_alpha,
        visibility_weight_floor,
        structural_annotation_alpha,
        structural_centerline_sigma_px,
        structural_annotation_sigma_px,
        radius_sigma_normalizer_px,
        bundle_count_normalizer,
    )


# -----------------------------------------------------------------------------
# Scene synthesis
# -----------------------------------------------------------------------------


def _estimate_fiber_count(
    bounds: Tuple[int, int, int],
    scene_config: Optional[Dict[str, object]],
    depth_of_field: float,
    max_fiber_area_px: float,
    min_fibers: int,
    max_fibers: int,
) -> int:
    x_size, y_size, z_size = bounds
    area = float(x_size * y_size)
    if scene_config is None:
        lo, hi = LEGACY_2D_FIBER_DENSITY_RANGE
        estimated = int(round(area * np.random.uniform(lo, hi)))
    else:
        target_skeleton = float(scene_config.get("target_skeleton_fraction", 0.006))
        target_length = area * target_skeleton
        if target_length < 1.0:
            return 0
        step_min, step_max = scene_config.get("step_scale_range", STED_2D_STEP_SCALE_RANGE)
        step_scale = 0.5 * (float(step_min) + float(step_max))
        step_length = max(0.8, max(x_size, y_size) * step_scale)
        steps_min, steps_max = scene_config.get("num_steps_range", SHORT_FIBER_STEPS)
        num_steps = 0.5 * (int(steps_min) + int(steps_max))
        visible_fraction = _clip((2.0 * float(depth_of_field)) / max(float(z_size), 1.0), 0.12, 1.0)
        expected_visible_length = max(1.0, step_length * num_steps * visible_fraction * 0.35)
        estimated = int(round(target_length / expected_visible_length))
        estimated = int(round(estimated * np.random.uniform(0.75, 1.35)))

    area_cap = max(1, int(area / max(float(max_fiber_area_px), 1.0)))
    upper = int(max_fibers) if int(max_fibers) > 0 else area_cap
    upper = max(int(min_fibers), upper)
    return int(np.clip(estimated, int(min_fibers), upper))


def _prepare_2d_sted_scene(
    bounds: Tuple[int, int, int],
    label_slab_thickness: Optional[float],
    label_slab_scale: float = DEFAULT_LABEL_SLAB_SCALE,
    annotation_weight_floor: float = DEFAULT_ANNOTATION_WEIGHT_FLOOR,
    soft_skeleton_alpha: float = DEFAULT_SOFT_SKELETON_ALPHA,
    visibility_weight_floor: float = DEFAULT_VISIBILITY_WEIGHT_FLOOR,
    structural_annotation_alpha: float = DEFAULT_STRUCTURAL_ANNOTATION_ALPHA,
    structural_centerline_sigma_px: float = DEFAULT_STRUCTURAL_CENTERLINE_SIGMA_PX,
    structural_annotation_sigma_px: float = DEFAULT_STRUCTURAL_ANNOTATION_SIGMA_PX,
    radius_sigma_normalizer_px: float = DEFAULT_RADIUS_SIGMA_NORMALIZER_PX,
    bundle_count_normalizer: float = DEFAULT_BUNDLE_COUNT_NORMALIZER,
    calibration_sampler: Optional[CalibrationSampler] = None,
    calibration_profile: Optional[Dict[str, object]] = None,
    apparent_width_p90_weight: float = DEFAULT_APPARENT_WIDTH_P90_WEIGHT,
    base_sigma_min: float = DEFAULT_BASE_SIGMA_MIN,
    base_sigma_max: float = DEFAULT_BASE_SIGMA_MAX,
    base_sigma_override: Optional[float] = None,
    coherent_bundle_probability: float = DEFAULT_COHERENT_BUNDLE_PROBABILITY,
    coherent_bundle_size_range: Tuple[int, int] = DEFAULT_COHERENT_BUNDLE_SIZE_RANGE,
    coherent_bundle_separation_range: Tuple[float, float] = DEFAULT_COHERENT_BUNDLE_SEPARATION_RANGE,
    optical_jitter_range: Tuple[float, float] = DEFAULT_OPTICAL_JITTER_RANGE,
    max_fiber_area_px: float = DEFAULT_MAX_FIBER_AREA_PX,
    min_fibers: int = DEFAULT_MIN_FIBERS,
    max_fibers: int = DEFAULT_MAX_FIBERS,
    max_generation_attempts: int = 6,
):
    x_size, y_size, z_size = bounds
    xy_area = x_size * y_size
    volume_size = x_size * y_size * z_size

    scene_config = None
    if calibration_sampler is not None:
        if calibration_profile is None:
            calibration_profile = calibration_sampler.profile
        scene_config = calibration_sampler.sample_scene_config(bounds)
        scene_config = _repair_scene_config_width_calibration(
            scene_config,
            calibration_profile,
            apparent_width_p90_weight=apparent_width_p90_weight,
            base_sigma_min=base_sigma_min,
            base_sigma_max=base_sigma_max,
            base_sigma_override=base_sigma_override,
        )

    haze_regime = scene_config["haze_regime"] if scene_config is not None else _sample_sted_haze_regime()
    enable_monomer_cloud, monomer_mix = _resolve_sted_monomer_config(haze_regime)

    if scene_config is None:
        base_sigma = (
            float(base_sigma_override)
            if base_sigma_override is not None
            else float(np.random.uniform(*DEFAULT_UNCALIBRATED_BASE_SIGMA_RANGE))
        )
        z_anisotropy = float(np.random.uniform(1.6, 2.8))
        noise_level = float(np.random.uniform(0.005, 0.045))
        debris_count = int(volume_size * np.random.uniform(0.00001, 0.00007))
        gap_prob = float(np.random.uniform(0.0, 0.08))
        stripe_strength = None
        vignette_edge_range = None
        dynamic_range = None
    else:
        base_sigma = float(scene_config.get("base_sigma", 1.4))
        z_anisotropy = float(scene_config.get("z_anisotropy", np.random.uniform(1.6, 2.8)))
        noise_level = float(scene_config.get("noise_level", np.random.uniform(0.005, 0.045)))
        debris_count = int(scene_config.get("debris_count", 0))
        gap_prob = float(scene_config.get("gap_prob", np.random.uniform(0.0, 0.08)))
        stripe_strength = scene_config.get("stripe_strength")
        vignette_edge_range = scene_config.get("vignette_edge_range")
        dynamic_range = scene_config.get("dynamic_range")

    rasterizer = EmpiricalRasterizer(
        bounds=bounds,
        base_sigma=base_sigma,
        z_anisotropy=z_anisotropy,
        noise_level=noise_level,
        debris_count=max(0, int(debris_count)),
        gap_prob=gap_prob,
        enable_sted_monomer_cloud=enable_monomer_cloud,
        sted_monomer_mix=monomer_mix,
        stripe_strength=stripe_strength,
        vignette_edge_range=vignette_edge_range,
    )
    depth_of_field, _, _ = rasterizer._sted_optical_section_params()
    axial_fwhm = rasterizer._sted_axial_fwhm(depth_of_field)
    localization_slab_thickness = _resolve_localization_slab_thickness(
        rasterizer,
        label_slab_thickness,
        label_slab_scale=label_slab_scale,
    )

    target_gen = TargetFieldGenerator((x_size, y_size), max_distance=TARGET_MAX_DISTANCE)
    visibility_target_gen = WeightedVisibilityTargetGenerator((x_size, y_size), base_sigma=rasterizer.base_sigma)

    if dynamic_range is None:
        img_min = np.random.uniform(0.12, 0.35)
        img_max = np.random.uniform(img_min + 0.15, 1.0)
        dynamic_range = (float(img_min), float(img_max))

    best_scene = None
    for attempt in range(int(max_generation_attempts)):
        optical_bundle_lists = []
        core_segments_flat = []
        core_segment_bundle_counts = []
        walk_parameter_samples = []
        bundle_sizes = []
        coherent_bundle_flags = []

        num_fibers = _estimate_fiber_count(
            bounds,
            scene_config,
            depth_of_field,
            max_fiber_area_px=max_fiber_area_px,
            min_fibers=min_fibers,
            max_fibers=max_fibers,
        )

        for _ in range(num_fibers):
            core_segments, walk_params = _generate_constrained_random_walk(bounds, scene_config=scene_config)
            if not core_segments:
                continue

            bundle_size, coherent_bundle = _sample_bundle_size(
                scene_config,
                coherent_bundle_probability=coherent_bundle_probability,
                coherent_bundle_size_range=coherent_bundle_size_range,
            )
            core_segments_flat.extend(core_segments)
            core_segment_bundle_counts.extend([float(bundle_size)] * len(core_segments))
            walk_parameter_samples.append(walk_params)
            bundle_sizes.append(int(bundle_size))
            coherent_bundle_flags.append(bool(coherent_bundle))

            if bundle_size <= 1:
                optical_segments = core_segments
            elif coherent_bundle:
                optical_segments = apply_coherent_optical_bundle(
                    core_segments,
                    bounds=bounds,
                    bundle_size=bundle_size,
                    separation_range=coherent_bundle_separation_range,
                )
            else:
                jitter_range = scene_config.get("jitter_range", optical_jitter_range) if scene_config is not None else optical_jitter_range
                jitter_amount = float(np.random.uniform(*jitter_range))
                optical_segments = apply_optical_jitter(
                    core_segments,
                    bundle_size=bundle_size,
                    jitter_amount=jitter_amount,
                    lock_z=False,
                )
            optical_bundle_lists.append(optical_segments)

        slice_center = float(np.random.uniform(z_size * 0.2, z_size * 0.8))
        target_data = _build_2d_targets(
            core_segments_flat,
            core_segment_bundle_counts,
            slice_center,
            localization_slab_thickness,
            rasterizer,
            target_gen,
            visibility_target_gen,
            annotation_weight_floor=annotation_weight_floor,
            soft_skeleton_alpha=soft_skeleton_alpha,
            visibility_weight_floor=visibility_weight_floor,
            structural_annotation_alpha=structural_annotation_alpha,
            structural_centerline_sigma_px=structural_centerline_sigma_px,
            structural_annotation_sigma_px=structural_annotation_sigma_px,
            radius_sigma_normalizer_px=radius_sigma_normalizer_px,
            bundle_count_normalizer=bundle_count_normalizer,
        )

        scene = {
            "bounds": bounds,
            "optical_bundle_lists": optical_bundle_lists,
            "core_segments": core_segments_flat,
            "core_segment_bundle_counts": np.asarray(core_segment_bundle_counts, dtype=np.float64),
            "projected_segments": target_data["focus_segments"],
            "annotation_segments": target_data["annotation_segments"],
            "visibility_segments": target_data["visibility_segments"],
            "visibility_weights": target_data["visibility_weights"],
            "requested_fiber_count": int(num_fibers),
            "actual_fiber_count": int(len(optical_bundle_lists)),
            "walk_parameter_samples": walk_parameter_samples,
            "bundle_sizes": bundle_sizes,
            "coherent_bundle_flags": coherent_bundle_flags,
            "slice_center": slice_center,
            "haze_regime": haze_regime,
            "label_slab_scale": float(label_slab_scale),
            "label_slab_thickness": float(localization_slab_thickness),
            "annotation_weight_floor": float(annotation_weight_floor),
            "soft_skeleton_alpha": float(soft_skeleton_alpha),
            "visibility_weight_floor": float(visibility_weight_floor),
            "depth_of_field": float(depth_of_field),
            "depth_to_dof_ratio": float(z_size / max(depth_of_field, 1e-8)),
            "axial_fwhm": float(axial_fwhm),
            "dynamic_range": tuple(float(v) for v in dynamic_range),
            "calibration_scene_config": scene_config,
            "rasterizer": rasterizer,
            "structural_centerline_sigma_px": float(structural_centerline_sigma_px),
            "structural_annotation_sigma_px": float(structural_annotation_sigma_px),
            "radius_sigma_normalizer_px": float(radius_sigma_normalizer_px),
            "bundle_count_normalizer": float(bundle_count_normalizer),
            **target_data,
        }
        best_scene = scene

        # Avoid accepting empty/non-visible scenes. For calibrated sparse images,
        # do not force the synthetic skeleton fraction too high; only reject clear
        # failures and grossly over-cluttered scenes.
        projected_count = len(target_data["focus_segments"])
        centerline_proxy = float(np.mean(target_data["centerline_target"] > 0.5))
        if scene_config is not None:
            target_skeleton_fraction = float(scene_config.get("target_skeleton_fraction", 0.0))
            too_cluttered = target_skeleton_fraction > 0.0 and centerline_proxy > max(0.12, 12.0 * target_skeleton_fraction)
            if projected_count > 0 and not too_cluttered:
                return scene
        else:
            if projected_count > 0:
                return scene

    return best_scene


def _build_2d_sample(
    bounds: Tuple[int, int, int],
    label_slab_thickness: Optional[float],
    label_slab_scale: float = DEFAULT_LABEL_SLAB_SCALE,
    annotation_weight_floor: float = DEFAULT_ANNOTATION_WEIGHT_FLOOR,
    soft_skeleton_alpha: float = DEFAULT_SOFT_SKELETON_ALPHA,
    visibility_weight_floor: float = DEFAULT_VISIBILITY_WEIGHT_FLOOR,
    structural_annotation_alpha: float = DEFAULT_STRUCTURAL_ANNOTATION_ALPHA,
    structural_centerline_sigma_px: float = DEFAULT_STRUCTURAL_CENTERLINE_SIGMA_PX,
    structural_annotation_sigma_px: float = DEFAULT_STRUCTURAL_ANNOTATION_SIGMA_PX,
    radius_sigma_normalizer_px: float = DEFAULT_RADIUS_SIGMA_NORMALIZER_PX,
    bundle_count_normalizer: float = DEFAULT_BUNDLE_COUNT_NORMALIZER,
    calibration_sampler: Optional[CalibrationSampler] = None,
    calibration_profile: Optional[Dict[str, object]] = None,
    apparent_width_p90_weight: float = DEFAULT_APPARENT_WIDTH_P90_WEIGHT,
    base_sigma_min: float = DEFAULT_BASE_SIGMA_MIN,
    base_sigma_max: float = DEFAULT_BASE_SIGMA_MAX,
    base_sigma_override: Optional[float] = None,
    coherent_bundle_probability: float = DEFAULT_COHERENT_BUNDLE_PROBABILITY,
    coherent_bundle_size_range: Tuple[int, int] = DEFAULT_COHERENT_BUNDLE_SIZE_RANGE,
    coherent_bundle_separation_range: Tuple[float, float] = DEFAULT_COHERENT_BUNDLE_SEPARATION_RANGE,
    optical_jitter_range: Tuple[float, float] = DEFAULT_OPTICAL_JITTER_RANGE,
    max_fiber_area_px: float = DEFAULT_MAX_FIBER_AREA_PX,
    min_fibers: int = DEFAULT_MIN_FIBERS,
    max_fibers: int = DEFAULT_MAX_FIBERS,
    max_generation_attempts: int = 6,
    return_metadata: bool = False,
):
    scene = _prepare_2d_sted_scene(
        bounds,
        label_slab_thickness,
        label_slab_scale=label_slab_scale,
        annotation_weight_floor=annotation_weight_floor,
        soft_skeleton_alpha=soft_skeleton_alpha,
        visibility_weight_floor=visibility_weight_floor,
        structural_annotation_alpha=structural_annotation_alpha,
        structural_centerline_sigma_px=structural_centerline_sigma_px,
        structural_annotation_sigma_px=structural_annotation_sigma_px,
        radius_sigma_normalizer_px=radius_sigma_normalizer_px,
        bundle_count_normalizer=bundle_count_normalizer,
        calibration_sampler=calibration_sampler,
        calibration_profile=calibration_profile,
        apparent_width_p90_weight=apparent_width_p90_weight,
        base_sigma_min=base_sigma_min,
        base_sigma_max=base_sigma_max,
        base_sigma_override=base_sigma_override,
        coherent_bundle_probability=coherent_bundle_probability,
        coherent_bundle_size_range=coherent_bundle_size_range,
        coherent_bundle_separation_range=coherent_bundle_separation_range,
        optical_jitter_range=optical_jitter_range,
        max_fiber_area_px=max_fiber_area_px,
        min_fibers=min_fibers,
        max_fibers=max_fibers,
        max_generation_attempts=max_generation_attempts,
    )
    image = scene["rasterizer"].render_sted_slice(
        scene["optical_bundle_lists"],
        slice_center=scene["slice_center"],
        dynamic_range=scene["dynamic_range"],
    )
    if scene["calibration_scene_config"] is not None:
        image = match_intensity_to_scene_profile(image, scene["calibration_scene_config"])

    metadata = None
    if return_metadata:
        metadata = {
            "bounds": tuple(int(v) for v in bounds),
            "requested_fiber_count": scene["requested_fiber_count"],
            "actual_fiber_count": scene["actual_fiber_count"],
            "projected_segment_count": len(scene["projected_segments"]),
            "annotation_segment_count": len(scene["annotation_segments"]),
            "visibility_segment_count": len(scene["visibility_segments"]),
            "bundle_sizes": scene["bundle_sizes"],
            "coherent_bundle_flags": scene["coherent_bundle_flags"],
            "haze_regime": scene["haze_regime"],
            "slice_center": scene["slice_center"],
            "depth_of_field": scene["depth_of_field"],
            "depth_to_dof_ratio": scene["depth_to_dof_ratio"],
            "axial_fwhm": scene["axial_fwhm"],
            "dynamic_range": scene["dynamic_range"],
            "label_slab_scale": scene["label_slab_scale"],
            "label_slab_thickness": scene["label_slab_thickness"],
            "annotation_weight_floor": scene["annotation_weight_floor"],
            "visibility_weight_floor": scene["visibility_weight_floor"],
            "structural_centerline_sigma_px": scene["structural_centerline_sigma_px"],
            "structural_annotation_sigma_px": scene["structural_annotation_sigma_px"],
            "radius_sigma_normalizer_px": scene["radius_sigma_normalizer_px"],
            "bundle_count_normalizer": scene["bundle_count_normalizer"],
            "base_sigma": float(scene["rasterizer"].base_sigma),
            "z_anisotropy": float(scene["rasterizer"].z_anisotropy),
            "calibration_scene_config": scene["calibration_scene_config"],
        }

    result = (
        image,
        scene["centerline_target"],
        scene["structural_vector_target"],
        scene["traceability_target"],
        scene["radius_target"],
        scene["bundle_count_target"],
    )
    if return_metadata:
        return (*result, metadata)
    return result


def build_sted_debug_sample(
    bounds: tuple,
    synth_depth: int = 64,
    label_slab_thickness: Optional[float] = None,
    label_slab_scale: float = DEFAULT_LABEL_SLAB_SCALE,
    annotation_weight_floor: float = DEFAULT_ANNOTATION_WEIGHT_FLOOR,
    soft_skeleton_alpha: float = DEFAULT_SOFT_SKELETON_ALPHA,
    visibility_weight_floor: float = DEFAULT_VISIBILITY_WEIGHT_FLOOR,
    structural_annotation_alpha: float = DEFAULT_STRUCTURAL_ANNOTATION_ALPHA,
    structural_centerline_sigma_px: float = DEFAULT_STRUCTURAL_CENTERLINE_SIGMA_PX,
    structural_annotation_sigma_px: float = DEFAULT_STRUCTURAL_ANNOTATION_SIGMA_PX,
    radius_sigma_normalizer_px: float = DEFAULT_RADIUS_SIGMA_NORMALIZER_PX,
    bundle_count_normalizer: float = DEFAULT_BUNDLE_COUNT_NORMALIZER,
    seed: int = None,
    calibration_sampler: Optional[CalibrationSampler] = None,
    calibration_profile: Optional[Dict[str, object]] = None,
    coherent_bundle_config: Optional[Dict[str, object]] = None,
):
    if len(bounds) == 2:
        synth_bounds = (bounds[0], bounds[1], synth_depth)
    elif len(bounds) == 3:
        synth_bounds = tuple(bounds)
    else:
        raise ValueError("Bounds must contain either 2 or 3 integers.")
    if seed is not None:
        np.random.seed(seed)

    coherent_bundle_probability = DEFAULT_COHERENT_BUNDLE_PROBABILITY
    coherent_bundle_size_range = DEFAULT_COHERENT_BUNDLE_SIZE_RANGE
    coherent_bundle_separation_range = DEFAULT_COHERENT_BUNDLE_SEPARATION_RANGE
    if coherent_bundle_config is not None:
        coherent_bundle_probability = float(coherent_bundle_config.get("probability", coherent_bundle_probability))
        coherent_bundle_size_range = tuple(coherent_bundle_config.get("size_range", coherent_bundle_size_range))
        coherent_bundle_separation_range = tuple(
            coherent_bundle_config.get("separation_range", coherent_bundle_separation_range)
        )

    scene = _prepare_2d_sted_scene(
        synth_bounds,
        label_slab_thickness,
        label_slab_scale=label_slab_scale,
        annotation_weight_floor=annotation_weight_floor,
        soft_skeleton_alpha=soft_skeleton_alpha,
        visibility_weight_floor=visibility_weight_floor,
        structural_annotation_alpha=structural_annotation_alpha,
        structural_centerline_sigma_px=structural_centerline_sigma_px,
        structural_annotation_sigma_px=structural_annotation_sigma_px,
        radius_sigma_normalizer_px=radius_sigma_normalizer_px,
        bundle_count_normalizer=bundle_count_normalizer,
        calibration_sampler=calibration_sampler,
        calibration_profile=calibration_profile,
        apparent_width_p90_weight=DEFAULT_APPARENT_WIDTH_P90_WEIGHT,
        base_sigma_min=DEFAULT_BASE_SIGMA_MIN,
        base_sigma_max=DEFAULT_BASE_SIGMA_MAX,
        base_sigma_override=None,
        coherent_bundle_probability=coherent_bundle_probability,
        coherent_bundle_size_range=coherent_bundle_size_range,
        coherent_bundle_separation_range=coherent_bundle_separation_range,
        optical_jitter_range=DEFAULT_OPTICAL_JITTER_RANGE,
        max_fiber_area_px=DEFAULT_MAX_FIBER_AREA_PX,
        min_fibers=DEFAULT_MIN_FIBERS,
        max_fibers=DEFAULT_MAX_FIBERS,
        max_generation_attempts=6,
    )
    debug_render = scene["rasterizer"].render_sted_slice_debug(
        scene["optical_bundle_lists"],
        slice_center=scene["slice_center"],
        dynamic_range=scene["dynamic_range"],
    )
    debug_render.update({k: v for k, v in scene.items() if k != "rasterizer" and k != "optical_bundle_lists"})
    debug_render["projected_segment_count"] = len(scene.get("projected_segments", []))
    debug_render["annotation_segment_count"] = len(scene.get("annotation_segments", []))
    if coherent_bundle_config is not None:
        debug_render["coherent_bundle_config"] = dict(coherent_bundle_config)
        separation = float(coherent_bundle_separation_range[0])
        debug_render["coherent_bundle_details"] = [
            {
                "mode": "coherent" if flag else "single",
                "bundle_size": int(size),
                "separation": separation,
            }
            for size, flag in zip(scene.get("bundle_sizes", []), scene.get("coherent_bundle_flags", []))
        ]
    return debug_render


# -----------------------------------------------------------------------------
# Serialization and optional crop extraction
# -----------------------------------------------------------------------------


def _crop_array(arr: np.ndarray, x0: int, y0: int, crop_x: int, crop_y: int) -> np.ndarray:
    if arr.ndim == 2:
        return arr[x0 : x0 + crop_x, y0 : y0 + crop_y]
    if arr.ndim == 3:
        return arr[:, x0 : x0 + crop_x, y0 : y0 + crop_y]
    raise ValueError(f"Unsupported array dimensionality for crop: {arr.ndim}")


def _random_crop_origin(scene_shape: Tuple[int, int], crop_shape: Tuple[int, int]) -> Tuple[int, int]:
    scene_x, scene_y = scene_shape
    crop_x, crop_y = crop_shape
    if crop_x > scene_x or crop_y > scene_y:
        raise ValueError(f"Crop shape {crop_shape} cannot exceed scene shape {scene_shape}.")
    max_x = scene_x - crop_x
    max_y = scene_y - crop_y
    x0 = 0 if max_x == 0 else int(np.random.randint(0, max_x + 1))
    y0 = 0 if max_y == 0 else int(np.random.randint(0, max_y + 1))
    return x0, y0


def _to_2d_tensors(
    image,
    centerline_target,
    vector_target,
    traceability_target,
    radius_target,
    bundle_count_target,
):
    import torch

    image = np.asarray(image).transpose(1, 0)
    centerline_target = np.asarray(centerline_target).transpose(1, 0)
    orientation_target = vector_to_orientation_channels_np(np.asarray(vector_target)).transpose(0, 2, 1)
    traceability_target = np.asarray(traceability_target).transpose(1, 0)
    radius_target = np.asarray(radius_target).transpose(1, 0)
    bundle_count_target = np.asarray(bundle_count_target).transpose(1, 0)

    volume_tensor = torch.tensor(image[np.newaxis, np.newaxis, :, :], dtype=torch.float32)
    targets = np.zeros((6, 1, image.shape[0], image.shape[1]), dtype=np.float32)
    targets[0, 0, :, :] = centerline_target.astype(np.float32)
    targets[1, 0, :, :] = orientation_target[0].astype(np.float32)
    targets[2, 0, :, :] = orientation_target[1].astype(np.float32)
    targets[3, 0, :, :] = traceability_target.astype(np.float32)
    targets[4, 0, :, :] = radius_target.astype(np.float32)
    targets[5, 0, :, :] = bundle_count_target.astype(np.float32)
    return volume_tensor, torch.tensor(targets, dtype=torch.float32)


def process_scene_and_save_crops(
    parent_idx: int,
    file_offset: int,
    split_size: int,
    output_dir: str,
    scene_bounds: Tuple[int, int, int],
    crop_bounds: Tuple[int, int],
    crops_per_scene: int,
    label_slab_thickness: Optional[float],
    label_slab_scale: float,
    annotation_weight_floor: float,
    soft_skeleton_alpha: float,
    visibility_weight_floor: float,
    structural_annotation_alpha: float,
    structural_centerline_sigma_px: float,
    structural_annotation_sigma_px: float,
    radius_sigma_normalizer_px: float,
    bundle_count_normalizer: float,
    calibration_profile: Optional[dict],
    real_regime: str,
    apparent_width_p90_weight: float,
    base_sigma_min: float,
    base_sigma_max: float,
    base_sigma_override: Optional[float],
    coherent_bundle_probability: float,
    coherent_bundle_size_range: Tuple[int, int],
    coherent_bundle_separation_range: Tuple[float, float],
    optical_jitter_range: Tuple[float, float],
    max_fiber_area_px: float,
    min_fibers: int,
    max_fibers: int,
    max_generation_attempts: int,
    emit_synthesis_metadata: bool,
):
    import torch

    seed = (os.getpid() + time.time_ns() + parent_idx * 1000003) % (2**32 - 1)
    np.random.seed(seed)

    calibration_sampler = None
    if calibration_profile is not None:
        calibration_sampler = CalibrationSampler(calibration_profile, real_regime=real_regime)

    (
        image,
        centerline_target,
        vector_target,
        traceability_target,
        radius_target,
        bundle_count_target,
        metadata,
    ) = _build_2d_sample(
        scene_bounds,
        label_slab_thickness,
        label_slab_scale=label_slab_scale,
        annotation_weight_floor=annotation_weight_floor,
        soft_skeleton_alpha=soft_skeleton_alpha,
        visibility_weight_floor=visibility_weight_floor,
        structural_annotation_alpha=structural_annotation_alpha,
        structural_centerline_sigma_px=structural_centerline_sigma_px,
        structural_annotation_sigma_px=structural_annotation_sigma_px,
        radius_sigma_normalizer_px=radius_sigma_normalizer_px,
        bundle_count_normalizer=bundle_count_normalizer,
        calibration_sampler=calibration_sampler,
        calibration_profile=calibration_profile,
        apparent_width_p90_weight=apparent_width_p90_weight,
        base_sigma_min=base_sigma_min,
        base_sigma_max=base_sigma_max,
        base_sigma_override=base_sigma_override,
        coherent_bundle_probability=coherent_bundle_probability,
        coherent_bundle_size_range=coherent_bundle_size_range,
        coherent_bundle_separation_range=coherent_bundle_separation_range,
        optical_jitter_range=optical_jitter_range,
        max_fiber_area_px=max_fiber_area_px,
        min_fibers=min_fibers,
        max_fibers=max_fibers,
        max_generation_attempts=max_generation_attempts,
        return_metadata=True,
    )

    saved_ids = []
    base_index = parent_idx * int(crops_per_scene)
    n_to_write = min(int(crops_per_scene), int(split_size) - base_index)
    if n_to_write <= 0:
        return saved_ids

    scene_x, scene_y = int(image.shape[0]), int(image.shape[1])
    crop_x, crop_y = crop_bounds
    for crop_idx in range(n_to_write):
        x0, y0 = _random_crop_origin((scene_x, scene_y), crop_bounds)
        image_crop = _crop_array(image, x0, y0, crop_x, crop_y)
        centerline_crop = _crop_array(centerline_target, x0, y0, crop_x, crop_y)
        vector_crop = _crop_array(vector_target, x0, y0, crop_x, crop_y)
        traceability_crop = _crop_array(traceability_target, x0, y0, crop_x, crop_y)
        radius_crop = _crop_array(radius_target, x0, y0, crop_x, crop_y)
        bundle_count_crop = _crop_array(bundle_count_target, x0, y0, crop_x, crop_y)

        volume_tensor, targets_tensor = _to_2d_tensors(
            image_crop,
            centerline_crop,
            vector_crop,
            traceability_crop,
            radius_crop,
            bundle_count_crop,
        )

        file_id = file_offset + base_index + crop_idx
        record = {
            "volume": volume_tensor,
            "targets": targets_tensor,
            "target_schema": "structural_v2",
        }
        if emit_synthesis_metadata:
            crop_metadata = dict(metadata or {})
            crop_metadata.update(
                {
                    "parent_scene_index": int(parent_idx),
                    "crop_index_within_scene": int(crop_idx),
                    "crop_origin_xy": (int(x0), int(y0)),
                    "crop_bounds_xy": tuple(int(v) for v in crop_bounds),
                    "scene_bounds_xyz": tuple(int(v) for v in scene_bounds),
                    "target_schema": "structural_v2",
                }
            )
            record["metadata"] = crop_metadata
        torch.save(record, os.path.join(output_dir, f"sample_{file_id}.pt"))
        saved_ids.append(file_id)
    return saved_ids


def build_dataset_split(
    split_name: str,
    size: int,
    file_offset: int,
    scene_bounds: Tuple[int, int, int],
    crop_bounds: Tuple[int, int],
    base_dir: str,
    workers: int,
    crops_per_scene: int,
    **kwargs,
):
    split_dir = os.path.join(base_dir, split_name)
    os.makedirs(split_dir, exist_ok=True)
    parent_count = int(math.ceil(float(size) / float(crops_per_scene))) if size > 0 else 0
    print(
        f"Building '{split_name}' split "
        f"({size} saved samples | {parent_count} parent scenes | "
        f"scene={scene_bounds}, crop={crop_bounds}) at {split_dir}..."
    )
    if size <= 0:
        return

    worker_func = partial(
        process_scene_and_save_crops,
        file_offset=file_offset,
        split_size=size,
        output_dir=split_dir,
        scene_bounds=scene_bounds,
        crop_bounds=crop_bounds,
        crops_per_scene=crops_per_scene,
        **kwargs,
    )

    completed_scenes = 0
    completed_samples = 0
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(worker_func, parent_idx): parent_idx for parent_idx in range(parent_count)}
        for future in concurrent.futures.as_completed(futures):
            parent_idx = futures[future]
            try:
                saved_ids = future.result()
            except Exception as exc:
                raise RuntimeError(f"Error generating parent scene {parent_idx} for split '{split_name}': {exc}") from exc
            completed_scenes += 1
            completed_samples += len(saved_ids)
            if completed_scenes % max(1, parent_count // 10) == 0 or completed_scenes == parent_count:
                print(f"  [{completed_samples}/{size}] samples saved ({completed_scenes}/{parent_count} scenes).")

    actual_files = len([f for f in os.listdir(split_dir) if f.endswith(".pt")])
    if actual_files != size:
        raise RuntimeError(f"Split '{split_name}' expected {size} .pt files, found {actual_files}.")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Generate calibrated synthetic 2D STED fiber datasets with structural_v2 targets."
    )
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument(
        "--bounds",
        type=int,
        nargs=2,
        default=[512, 512],
        help="Saved sample/crop size as X Y. Default: 512 512.",
    )
    parser.add_argument(
        "--scene_bounds",
        type=int,
        nargs=2,
        default=None,
        help="Optional internal full-scene size as X Y. Use e.g. 1024 1024 with --bounds 512 512 to save crops.",
    )
    parser.add_argument("--synth_depth", type=int, default=64)
    parser.add_argument("--crops_per_scene", type=int, default=1)

    parser.add_argument("--label_slab_thickness", type=float, default=None)
    parser.add_argument("--label_slab_scale", type=float, default=DEFAULT_LABEL_SLAB_SCALE)
    parser.add_argument("--soft_skeleton_alpha", type=float, default=DEFAULT_SOFT_SKELETON_ALPHA)
    parser.add_argument("--annotation_weight_floor", type=float, default=DEFAULT_ANNOTATION_WEIGHT_FLOOR)
    parser.add_argument("--visibility_weight_floor", type=float, default=DEFAULT_VISIBILITY_WEIGHT_FLOOR)
    parser.add_argument("--structural_annotation_alpha", type=float, default=DEFAULT_STRUCTURAL_ANNOTATION_ALPHA)
    parser.add_argument("--structural_centerline_sigma_px", type=float, default=DEFAULT_STRUCTURAL_CENTERLINE_SIGMA_PX)
    parser.add_argument("--structural_annotation_sigma_px", type=float, default=DEFAULT_STRUCTURAL_ANNOTATION_SIGMA_PX)
    parser.add_argument("--radius_sigma_normalizer_px", type=float, default=DEFAULT_RADIUS_SIGMA_NORMALIZER_PX)
    parser.add_argument("--bundle_count_normalizer", type=float, default=DEFAULT_BUNDLE_COUNT_NORMALIZER)

    parser.add_argument(
        "--calibration_profile",
        type=str,
        default=None,
        help="Optional path to sted_real_profile.json produced from real 1024 px STED images/patches.",
    )
    parser.add_argument("--real_regime", type=str, choices=["global", "condition", "div"], default="global")
    parser.add_argument("--apparent_width_p90_weight", type=float, default=DEFAULT_APPARENT_WIDTH_P90_WEIGHT)
    parser.add_argument("--base_sigma_min", type=float, default=DEFAULT_BASE_SIGMA_MIN)
    parser.add_argument("--base_sigma_max", type=float, default=DEFAULT_BASE_SIGMA_MAX)
    parser.add_argument("--base_sigma_override", type=float, default=None)

    parser.add_argument("--coherent_bundle_probability", type=float, default=DEFAULT_COHERENT_BUNDLE_PROBABILITY)
    parser.add_argument("--coherent_bundle_size_range", type=int, nargs=2, default=list(DEFAULT_COHERENT_BUNDLE_SIZE_RANGE))
    parser.add_argument("--coherent_bundle_separation_range", type=float, nargs=2, default=list(DEFAULT_COHERENT_BUNDLE_SEPARATION_RANGE))
    parser.add_argument("--optical_jitter_range", type=float, nargs=2, default=list(DEFAULT_OPTICAL_JITTER_RANGE))
    parser.add_argument("--max_fiber_area_px", type=float, default=DEFAULT_MAX_FIBER_AREA_PX)
    parser.add_argument("--min_fibers", type=int, default=DEFAULT_MIN_FIBERS)
    parser.add_argument("--max_fibers", type=int, default=DEFAULT_MAX_FIBERS)
    parser.add_argument("--max_generation_attempts", type=int, default=6)

    parser.add_argument("--train_size", type=int, default=10000)
    parser.add_argument("--val_size", type=int, default=1000)
    parser.add_argument("--test_size", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--emit_synthesis_metadata", action="store_true")
    return parser.parse_args()


def _validate_args(args) -> None:
    if args.synth_depth < 2:
        raise ValueError("--synth_depth must be at least 2.")
    if args.crops_per_scene < 1:
        raise ValueError("--crops_per_scene must be at least 1.")
    if args.workers < 1:
        raise ValueError("--workers must be at least 1.")
    if args.train_size < 0 or args.val_size < 0 or args.test_size < 0:
        raise ValueError("Split sizes must be non-negative.")

    _validate_positive("label_slab_scale", args.label_slab_scale)
    _validate_probability("soft_skeleton_alpha", args.soft_skeleton_alpha)
    _validate_probability("structural_annotation_alpha", args.structural_annotation_alpha)
    _validate_positive("structural_centerline_sigma_px", args.structural_centerline_sigma_px)
    _validate_positive("structural_annotation_sigma_px", args.structural_annotation_sigma_px)
    _validate_positive("radius_sigma_normalizer_px", args.radius_sigma_normalizer_px)
    _validate_positive("bundle_count_normalizer", args.bundle_count_normalizer)
    _validate_probability("coherent_bundle_probability", args.coherent_bundle_probability)
    _validate_positive("max_fiber_area_px", args.max_fiber_area_px)

    if args.annotation_weight_floor <= 0.0 or args.annotation_weight_floor > 1.0:
        raise ValueError("--annotation_weight_floor must be in the interval (0, 1].")
    if args.visibility_weight_floor <= 0.0 or args.visibility_weight_floor > 1.0:
        raise ValueError("--visibility_weight_floor must be in the interval (0, 1].")
    if args.base_sigma_min > args.base_sigma_max:
        raise ValueError("--base_sigma_min must be <= --base_sigma_max.")
    if args.base_sigma_override is not None and args.base_sigma_override <= 0.0:
        raise ValueError("--base_sigma_override must be positive.")

    scene_xy = tuple(args.scene_bounds) if args.scene_bounds is not None else tuple(args.bounds)
    if args.bounds[0] > scene_xy[0] or args.bounds[1] > scene_xy[1]:
        raise ValueError("--bounds crop size cannot exceed --scene_bounds.")


def main() -> None:
    args = _parse_args()
    _validate_args(args)

    crop_bounds = (int(args.bounds[0]), int(args.bounds[1]))
    scene_xy = tuple(args.scene_bounds) if args.scene_bounds is not None else crop_bounds
    scene_bounds = (int(scene_xy[0]), int(scene_xy[1]), int(args.synth_depth))

    calibration_profile = None
    if args.calibration_profile is not None:
        calibration_profile = load_calibration_profile(args.calibration_profile)

    os.makedirs(args.output_dir, exist_ok=True)

    config_summary = {
        "target_schema": "structural_v2",
        "scene_bounds": scene_bounds,
        "crop_bounds": crop_bounds,
        "crops_per_scene": int(args.crops_per_scene),
        "calibration_profile": args.calibration_profile,
        "real_regime": args.real_regime,
        "synth_depth": int(args.synth_depth),
        "structural_centerline_sigma_px": float(args.structural_centerline_sigma_px),
        "structural_annotation_sigma_px": float(args.structural_annotation_sigma_px),
        "radius_sigma_normalizer_px": float(args.radius_sigma_normalizer_px),
        "bundle_count_normalizer": float(args.bundle_count_normalizer),
        "coherent_bundle_probability": float(args.coherent_bundle_probability),
        "coherent_bundle_size_range": tuple(int(v) for v in args.coherent_bundle_size_range),
        "coherent_bundle_separation_range": tuple(float(v) for v in args.coherent_bundle_separation_range),
    }
    with open(os.path.join(args.output_dir, "generation_config.json"), "w", encoding="utf-8") as handle:
        json.dump(config_summary, handle, indent=2)

    common_kwargs = dict(
        label_slab_thickness=args.label_slab_thickness,
        label_slab_scale=args.label_slab_scale,
        annotation_weight_floor=args.annotation_weight_floor,
        soft_skeleton_alpha=args.soft_skeleton_alpha,
        visibility_weight_floor=args.visibility_weight_floor,
        structural_annotation_alpha=args.structural_annotation_alpha,
        structural_centerline_sigma_px=args.structural_centerline_sigma_px,
        structural_annotation_sigma_px=args.structural_annotation_sigma_px,
        radius_sigma_normalizer_px=args.radius_sigma_normalizer_px,
        bundle_count_normalizer=args.bundle_count_normalizer,
        calibration_profile=calibration_profile,
        real_regime=args.real_regime,
        apparent_width_p90_weight=args.apparent_width_p90_weight,
        base_sigma_min=args.base_sigma_min,
        base_sigma_max=args.base_sigma_max,
        base_sigma_override=args.base_sigma_override,
        coherent_bundle_probability=args.coherent_bundle_probability,
        coherent_bundle_size_range=tuple(int(v) for v in args.coherent_bundle_size_range),
        coherent_bundle_separation_range=_validate_range(
            "coherent_bundle_separation_range",
            args.coherent_bundle_separation_range,
            positive=True,
        ),
        optical_jitter_range=_validate_range("optical_jitter_range", args.optical_jitter_range, positive=True),
        max_fiber_area_px=args.max_fiber_area_px,
        min_fibers=args.min_fibers,
        max_fibers=args.max_fibers,
        max_generation_attempts=args.max_generation_attempts,
        emit_synthesis_metadata=args.emit_synthesis_metadata,
    )

    build_dataset_split(
        "train",
        args.train_size,
        file_offset=0,
        scene_bounds=scene_bounds,
        crop_bounds=crop_bounds,
        base_dir=args.output_dir,
        workers=args.workers,
        crops_per_scene=args.crops_per_scene,
        **common_kwargs,
    )
    build_dataset_split(
        "val",
        args.val_size,
        file_offset=args.train_size,
        scene_bounds=scene_bounds,
        crop_bounds=crop_bounds,
        base_dir=args.output_dir,
        workers=args.workers,
        crops_per_scene=args.crops_per_scene,
        **common_kwargs,
    )
    build_dataset_split(
        "test",
        args.test_size,
        file_offset=args.train_size + args.val_size,
        scene_bounds=scene_bounds,
        crop_bounds=crop_bounds,
        base_dir=args.output_dir,
        workers=args.workers,
        crops_per_scene=args.crops_per_scene,
        **common_kwargs,
    )


if __name__ == "__main__":
    main()
