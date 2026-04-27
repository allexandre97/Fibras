from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import scipy.ndimage as ndi

try:
    from skimage.morphology import skeletonize
except Exception:  # pragma: no cover - exercised only when optional deps are missing.
    skeletonize = None


@dataclass
class DecodedCenterlineGraph:
    skeleton: np.ndarray
    score_map: np.ndarray
    candidate_mask: np.ndarray
    strong_seed_mask: np.ndarray
    component_paths: List[np.ndarray]
    endpoint_mask: np.ndarray
    junction_mask: np.ndarray


def _hard_skeletonize(mask: np.ndarray) -> np.ndarray:
    binary = np.asarray(mask, dtype=bool)
    if not np.any(binary):
        return binary
    if skeletonize is None:
        return binary
    return np.asarray(skeletonize(binary), dtype=bool)


def _neighbor_count(mask: np.ndarray) -> np.ndarray:
    binary = np.asarray(mask, dtype=np.uint8)
    return ndi.convolve(binary, np.ones((3, 3), dtype=np.uint8), mode="constant") - binary


def _label_components(mask: np.ndarray) -> tuple[np.ndarray, int]:
    return ndi.label(np.asarray(mask, dtype=bool), structure=np.ones((3, 3), dtype=np.uint8))


def _sample_line(start_rc: np.ndarray, end_rc: np.ndarray) -> np.ndarray:
    delta = end_rc - start_rc
    steps = int(max(abs(delta[0]), abs(delta[1]), 1))
    coords = np.linspace(start_rc, end_rc, steps + 1)
    coords = np.round(coords).astype(int)
    dedup = np.concatenate([[True], np.any(coords[1:] != coords[:-1], axis=1)])
    return coords[dedup]


def _clip_coords(coords: np.ndarray, shape) -> np.ndarray:
    clipped = np.array(coords, copy=True)
    clipped[:, 0] = np.clip(clipped[:, 0], 0, shape[0] - 1)
    clipped[:, 1] = np.clip(clipped[:, 1], 0, shape[1] - 1)
    return clipped


def _orientation_rc(orientation_map: np.ndarray, row: int, col: int) -> np.ndarray:
    vec_x = float(orientation_map[0, row, col])
    vec_y = float(orientation_map[1, row, col])
    vec_rc = np.asarray([vec_y, vec_x], dtype=np.float64)
    norm = np.linalg.norm(vec_rc)
    if norm <= 1e-8:
        return np.zeros(2, dtype=np.float64)
    return vec_rc / norm


def _ridge_non_maximum_suppression(
    centerline_prob: np.ndarray,
    orientation_map: np.ndarray,
    sample_radius: float = 1.0,
) -> np.ndarray:
    centerline_prob = np.asarray(centerline_prob, dtype=np.float32)
    if not np.any(centerline_prob > 0.0):
        return np.zeros_like(centerline_prob, dtype=np.float32)

    tangent_r = np.asarray(orientation_map[1], dtype=np.float32)
    tangent_c = np.asarray(orientation_map[0], dtype=np.float32)
    tangent_norm = np.sqrt((tangent_r * tangent_r) + (tangent_c * tangent_c))
    valid_orientation = tangent_norm > 1e-6

    tangent_r = np.where(valid_orientation, tangent_r / np.maximum(tangent_norm, 1e-6), 0.0)
    tangent_c = np.where(valid_orientation, tangent_c / np.maximum(tangent_norm, 1e-6), 0.0)
    normal_r = -tangent_c
    normal_c = tangent_r

    rr, cc = np.indices(centerline_prob.shape, dtype=np.float32)
    pos = ndi.map_coordinates(
        centerline_prob,
        [rr + sample_radius * normal_r, cc + sample_radius * normal_c],
        order=1,
        mode="nearest",
    )
    neg = ndi.map_coordinates(
        centerline_prob,
        [rr - sample_radius * normal_r, cc - sample_radius * normal_c],
        order=1,
        mode="nearest",
    )
    local_max = centerline_prob >= (ndi.maximum_filter(centerline_prob, size=3, mode="nearest") - 1e-6)
    ridge_mask = (centerline_prob >= pos) & (centerline_prob >= neg) & valid_orientation
    ridge_mask |= (~valid_orientation) & local_max

    ridge = np.zeros_like(centerline_prob, dtype=np.float32)
    ridge[ridge_mask] = centerline_prob[ridge_mask]
    return ridge


def _support_map(
    traceability_map: np.ndarray,
    orientation_confidence: Optional[np.ndarray] = None,
) -> np.ndarray:
    support = np.sqrt(np.clip(np.asarray(traceability_map, dtype=np.float32), 0.0, 1.0))
    if orientation_confidence is not None:
        support *= np.sqrt(np.clip(np.asarray(orientation_confidence, dtype=np.float32), 0.0, 1.0))
    return np.clip(support, 0.0, 1.0)


def _select_hysteresis_mask(
    ridge_map: np.ndarray,
    support_map: np.ndarray,
    high_threshold: float,
    low_threshold_ratio: float = 0.35,
    strong_support_floor: float = 0.20,
    weak_support_floor: float = 0.05,
    adaptive_seed_quantile: float = 0.995,
) -> tuple[np.ndarray, np.ndarray]:
    low_threshold = max(1e-4, float(high_threshold) * float(low_threshold_ratio))

    weak_mask = (ridge_map >= low_threshold) & (support_map >= weak_support_floor)
    strong_seed_mask = (ridge_map >= high_threshold) & (support_map >= strong_support_floor)

    if not np.any(strong_seed_mask) and np.any(weak_mask):
        weak_values = ridge_map[weak_mask]
        adaptive_threshold = max(low_threshold, float(np.quantile(weak_values, adaptive_seed_quantile)))
        strong_seed_mask = weak_mask & (ridge_map >= adaptive_threshold)

    if not np.any(strong_seed_mask):
        return np.zeros_like(weak_mask, dtype=bool), strong_seed_mask

    candidate_mask = ndi.binary_propagation(strong_seed_mask, mask=weak_mask)
    return np.asarray(candidate_mask, dtype=bool), np.asarray(strong_seed_mask, dtype=bool)


def _bridge_endpoints(
    skeleton: np.ndarray,
    orientation_map: np.ndarray,
    support_map: np.ndarray,
    score_map: np.ndarray,
    max_gap: int = 10,
    orientation_cos_min: float = 0.45,
    support_min: float = 0.20,
    score_min: float = 0.10,
) -> np.ndarray:
    skeleton = np.asarray(skeleton, dtype=bool)
    if not np.any(skeleton):
        return skeleton

    labels, component_count = _label_components(skeleton)
    if component_count <= 1:
        return skeleton

    neighbors = _neighbor_count(skeleton)
    endpoints = np.argwhere(skeleton & (neighbors == 1))
    if len(endpoints) < 2:
        return skeleton

    bridged = skeleton.copy()
    for idx in range(len(endpoints)):
        a = endpoints[idx]
        label_a = labels[a[0], a[1]]
        if label_a == 0:
            continue

        best_candidate = None
        best_distance = float("inf")
        for jdx in range(idx + 1, len(endpoints)):
            b = endpoints[jdx]
            label_b = labels[b[0], b[1]]
            if label_b == 0 or label_b == label_a:
                continue

            distance = float(np.linalg.norm(a.astype(np.float64) - b.astype(np.float64)))
            if distance > max_gap or distance >= best_distance:
                continue

            direction = b.astype(np.float64) - a.astype(np.float64)
            norm = np.linalg.norm(direction)
            if norm <= 1e-8:
                continue
            direction /= norm

            ori_a = _orientation_rc(orientation_map, int(a[0]), int(a[1]))
            ori_b = _orientation_rc(orientation_map, int(b[0]), int(b[1]))
            if np.linalg.norm(ori_a) <= 1e-8 or np.linalg.norm(ori_b) <= 1e-8:
                continue

            ori_score = 0.5 * (
                abs(float(np.dot(ori_a, direction))) +
                abs(float(np.dot(ori_b, direction)))
            )
            if ori_score < orientation_cos_min:
                continue

            line_coords = _clip_coords(_sample_line(a.astype(np.float64), b.astype(np.float64)), skeleton.shape)
            corridor_support = float(np.mean(support_map[line_coords[:, 0], line_coords[:, 1]]))
            corridor_score = float(np.mean(score_map[line_coords[:, 0], line_coords[:, 1]]))
            if corridor_support < support_min or corridor_score < score_min:
                continue

            best_distance = distance
            best_candidate = line_coords

        if best_candidate is not None:
            bridged[best_candidate[:, 0], best_candidate[:, 1]] = True

    return _hard_skeletonize(bridged)


def _prune_small_components(
    skeleton: np.ndarray,
    score_map: np.ndarray,
    support_map: np.ndarray,
    min_pixels: int = 2,
    min_mean_score: float = 0.08,
    min_mean_support: float = 0.20,
    min_peak_score: float = 0.12,
) -> np.ndarray:
    labels, component_count = _label_components(skeleton)
    if component_count == 0:
        return np.asarray(skeleton, dtype=bool)

    pruned = np.zeros_like(skeleton, dtype=bool)
    for component_id in range(1, component_count + 1):
        coords = np.argwhere(labels == component_id)
        if coords.size == 0:
            continue
        length = int(coords.shape[0])
        mean_score = float(np.mean(score_map[coords[:, 0], coords[:, 1]]))
        mean_support = float(np.mean(support_map[coords[:, 0], coords[:, 1]]))
        peak_score = float(np.max(score_map[coords[:, 0], coords[:, 1]]))
        if (
            length >= min_pixels
            and mean_score >= min_mean_score
            and mean_support >= min_mean_support
            and peak_score >= min_peak_score
        ):
            pruned[coords[:, 0], coords[:, 1]] = True
    return pruned


def _component_paths_from_skeleton(skeleton: np.ndarray) -> List[np.ndarray]:
    labels, component_count = _label_components(skeleton)
    paths = []
    for component_id in range(1, component_count + 1):
        coords = np.argwhere(labels == component_id)
        if coords.size == 0:
            continue
        paths.append(coords.astype(np.float64))
    return paths


class CenterlineGraphDecoder:
    def __init__(
        self,
        centerline_threshold: float = 0.5,
        weak_threshold_ratio: float = 0.35,
        bridge_gap: int = 10,
        min_component_pixels: int = 2,
        orientation_cos_min: float = 0.45,
        strong_support_floor: float = 0.20,
        weak_support_floor: float = 0.05,
        bridge_support_min: float = 0.20,
        bridge_score_min: float = 0.10,
        min_component_mean_score: float = 0.08,
        min_component_mean_support: float = 0.20,
        min_component_peak_score: float = 0.12,
        ridge_nms_radius: float = 1.0,
        adaptive_seed_quantile: float = 0.995,
    ):
        self.centerline_threshold = float(centerline_threshold)
        self.weak_threshold_ratio = float(weak_threshold_ratio)
        self.bridge_gap = int(bridge_gap)
        self.min_component_pixels = int(min_component_pixels)
        self.orientation_cos_min = float(orientation_cos_min)
        self.strong_support_floor = float(strong_support_floor)
        self.weak_support_floor = float(weak_support_floor)
        self.bridge_support_min = float(bridge_support_min)
        self.bridge_score_min = float(bridge_score_min)
        self.min_component_mean_score = float(min_component_mean_score)
        self.min_component_mean_support = float(min_component_mean_support)
        self.min_component_peak_score = float(min_component_peak_score)
        self.ridge_nms_radius = float(ridge_nms_radius)
        self.adaptive_seed_quantile = float(adaptive_seed_quantile)

    def decode(
        self,
        centerline_prob: np.ndarray,
        orientation_map: np.ndarray,
        traceability_map: np.ndarray,
        orientation_confidence: Optional[np.ndarray] = None,
    ) -> DecodedCenterlineGraph:
        centerline_prob = np.asarray(centerline_prob, dtype=np.float32)
        traceability_map = np.asarray(traceability_map, dtype=np.float32)
        support = _support_map(traceability_map, orientation_confidence=orientation_confidence)
        bridge_score_map = centerline_prob * support

        ridge_map = _ridge_non_maximum_suppression(
            centerline_prob,
            orientation_map=orientation_map,
            sample_radius=self.ridge_nms_radius,
        )
        score_map = ridge_map * support
        candidate_mask, strong_seed_mask = _select_hysteresis_mask(
            score_map,
            support_map=support,
            high_threshold=self.centerline_threshold,
            low_threshold_ratio=self.weak_threshold_ratio,
            strong_support_floor=self.strong_support_floor,
            weak_support_floor=self.weak_support_floor,
            adaptive_seed_quantile=self.adaptive_seed_quantile,
        )

        skeleton = _hard_skeletonize(candidate_mask)
        skeleton = _bridge_endpoints(
            skeleton,
            orientation_map=orientation_map,
            support_map=support,
            score_map=bridge_score_map,
            max_gap=self.bridge_gap,
            orientation_cos_min=self.orientation_cos_min,
            support_min=self.bridge_support_min,
            score_min=self.bridge_score_min,
        )
        skeleton = _prune_small_components(
            skeleton,
            score_map=score_map,
            support_map=support,
            min_pixels=self.min_component_pixels,
            min_mean_score=self.min_component_mean_score,
            min_mean_support=self.min_component_mean_support,
            min_peak_score=self.min_component_peak_score,
        )

        neighbors = _neighbor_count(skeleton)
        endpoint_mask = skeleton & (neighbors == 1)
        junction_mask = skeleton & (neighbors >= 3)
        component_paths = _component_paths_from_skeleton(skeleton)
        return DecodedCenterlineGraph(
            skeleton=skeleton,
            score_map=score_map,
            candidate_mask=np.asarray(candidate_mask, dtype=bool),
            strong_seed_mask=np.asarray(strong_seed_mask, dtype=bool),
            component_paths=component_paths,
            endpoint_mask=endpoint_mask,
            junction_mask=junction_mask,
        )
