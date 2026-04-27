import numpy as np
import numba as nb
from typing import List, Tuple

from src.core import FiberSegment


@nb.njit(cache=True)
def _compute_targets_3d_numba(edt_map, vector_map, starts, ends, max_dist):
    num_segs = starts.shape[0]
    shape_x, shape_y, shape_z = edt_map.shape

    for s in range(num_segs):
        start = starts[s]
        end = ends[s]

        ab_x = end[0] - start[0]
        ab_y = end[1] - start[1]
        ab_z = end[2] - start[2]

        ab_norm = np.sqrt(ab_x**2 + ab_y**2 + ab_z**2)
        if ab_norm < 1e-8:
            continue

        dir_x, dir_y, dir_z = ab_x / ab_norm, ab_y / ab_norm, ab_z / ab_norm

        min_x = max(0, int(np.floor(min(start[0], end[0]) - max_dist)))
        max_x = min(shape_x, int(np.ceil(max(start[0], end[0]) + max_dist)))
        min_y = max(0, int(np.floor(min(start[1], end[1]) - max_dist)))
        max_y = min(shape_y, int(np.ceil(max(start[1], end[1]) + max_dist)))
        min_z = max(0, int(np.floor(min(start[2], end[2]) - max_dist)))
        max_z = min(shape_z, int(np.ceil(max(start[2], end[2]) + max_dist)))

        ab_dot_max = max(ab_norm**2, 1e-8)

        for i in range(min_x, max_x):
            for j in range(min_y, max_y):
                for k in range(min_z, max_z):
                    ap_x = i - start[0]
                    ap_y = j - start[1]
                    ap_z = k - start[2]

                    t = (ap_x * ab_x + ap_y * ab_y + ap_z * ab_z) / ab_dot_max
                    t = max(0.0, min(1.0, t))

                    c_x = start[0] + t * ab_x
                    c_y = start[1] + t * ab_y
                    c_z = start[2] + t * ab_z

                    d = np.sqrt((i - c_x) ** 2 + (j - c_y) ** 2 + (k - c_z) ** 2)

                    if d < edt_map[i, j, k]:
                        edt_map[i, j, k] = d
                        vector_map[0, i, j, k] = dir_x
                        vector_map[1, i, j, k] = dir_y
                        vector_map[2, i, j, k] = dir_z


@nb.njit(cache=True)
def _compute_targets_2d_numba(edt_map, vector_map, starts, ends, max_dist):
    num_segs = starts.shape[0]
    shape_x, shape_y = edt_map.shape

    for s in range(num_segs):
        start = starts[s]
        end = ends[s]

        ab_x = end[0] - start[0]
        ab_y = end[1] - start[1]

        ab_norm = np.sqrt(ab_x**2 + ab_y**2)
        if ab_norm < 1e-8:
            continue

        dir_x, dir_y = ab_x / ab_norm, ab_y / ab_norm

        min_x = max(0, int(np.floor(min(start[0], end[0]) - max_dist)))
        max_x = min(shape_x, int(np.ceil(max(start[0], end[0]) + max_dist)))
        min_y = max(0, int(np.floor(min(start[1], end[1]) - max_dist)))
        max_y = min(shape_y, int(np.ceil(max(start[1], end[1]) + max_dist)))

        ab_dot_max = max(ab_norm**2, 1e-8)

        for i in range(min_x, max_x):
            for j in range(min_y, max_y):
                ap_x = i - start[0]
                ap_y = j - start[1]

                t = (ap_x * ab_x + ap_y * ab_y) / ab_dot_max
                t = max(0.0, min(1.0, t))

                c_x = start[0] + t * ab_x
                c_y = start[1] + t * ab_y

                d = np.sqrt((i - c_x) ** 2 + (j - c_y) ** 2)

                if d < edt_map[i, j]:
                    edt_map[i, j] = d
                    vector_map[0, i, j] = dir_x
                    vector_map[1, i, j] = dir_y


class TargetFieldGenerator:
    def __init__(self, grid_shape: Tuple[int, ...], max_distance: float = 5.0):
        self.grid_shape = grid_shape
        self.max_distance = max_distance
        self.dims = len(grid_shape)

        if self.dims not in (2, 3):
            raise ValueError("TargetFieldGenerator only supports 2D or 3D grids.")

    def generate(self, segments: List[FiberSegment]) -> Tuple[np.ndarray, np.ndarray]:
        edt_map = np.full(self.grid_shape, self.max_distance, dtype=np.float64)
        vector_map = np.zeros((self.dims,) + self.grid_shape, dtype=np.float64)

        if segments:
            starts = np.array([s.start for s in segments], dtype=np.float64)
            ends = np.array([s.end for s in segments], dtype=np.float64)

            if self.dims == 2:
                _compute_targets_2d_numba(edt_map, vector_map, starts, ends, self.max_distance)
            else:
                _compute_targets_3d_numba(edt_map, vector_map, starts, ends, self.max_distance)

        edt_normalized = 1.0 - (edt_map / self.max_distance)
        edt_normalized[edt_normalized < 0] = 0

        return edt_normalized, vector_map


class WeightedVisibilityTargetGenerator:
    def __init__(self, grid_shape: Tuple[int, int], base_sigma: float = 1.0):
        if len(grid_shape) != 2:
            raise ValueError("WeightedVisibilityTargetGenerator only supports 2D grids.")
        self.grid_shape = grid_shape
        self.base_sigma = base_sigma

    @staticmethod
    def _point_to_segment_distance(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        ab = b - a
        ap = p - a
        t = np.sum(ap * ab, axis=-1) / np.maximum(np.dot(ab, ab), 1e-8)
        t_clamped = np.clip(t, 0.0, 1.0)
        closest_point = a + t_clamped[..., np.newaxis] * ab
        return np.linalg.norm(p - closest_point, axis=-1)

    def generate(self, segments: List[FiberSegment], weights: np.ndarray) -> np.ndarray:
        visibility_map = np.zeros(self.grid_shape, dtype=np.float64)

        if not segments:
            return visibility_map

        if len(segments) != len(weights):
            raise ValueError("segments and weights must have the same length.")

        for segment, weight in zip(segments, weights):
            if weight <= 0.0:
                continue

            sigma = self.base_sigma * segment.thickness_mult
            cutoff = 4.0 * sigma

            mins = np.floor(np.minimum(segment.start, segment.end) - cutoff).astype(int)
            maxs = np.ceil(np.maximum(segment.start, segment.end) + cutoff).astype(int)

            mins = np.maximum(0, mins)
            maxs = np.minimum(np.array(self.grid_shape), maxs)

            if np.any(mins >= maxs):
                continue

            slices = tuple(slice(mins[d], maxs[d]) for d in range(2))
            ranges = [np.arange(mins[d], maxs[d]) for d in range(2)]
            local_grid_points = np.stack(np.meshgrid(*ranges, indexing="ij"), axis=-1)

            dist = self._point_to_segment_distance(local_grid_points, segment.start, segment.end)
            segment_density = np.exp(-(dist**2) / (2 * sigma**2)) * float(weight)
            visibility_map[slices] = np.maximum(visibility_map[slices], segment_density)

        return np.clip(visibility_map, 0.0, 1.0)


@nb.njit(cache=True)
def _compute_structural_targets_2d_numba(
    centerline_map,
    best_score_map,
    vector_map,
    radius_map,
    starts,
    ends,
    sigmas,
    centerline_sigma,
    radius_normalizer,
):
    num_segs = starts.shape[0]
    shape_x, shape_y = centerline_map.shape
    cutoff = max(2.5 * centerline_sigma, 1.5)
    sigma2 = max(centerline_sigma * centerline_sigma, 1e-8)

    for s in range(num_segs):
        start = starts[s]
        end = ends[s]

        ab_x = end[0] - start[0]
        ab_y = end[1] - start[1]

        ab_norm = np.sqrt(ab_x**2 + ab_y**2)
        if ab_norm < 1e-8:
            continue

        dir_x, dir_y = ab_x / ab_norm, ab_y / ab_norm
        sigma = sigmas[s]
        radius_value = min(1.0, sigma / max(radius_normalizer, 1e-8))

        min_x = max(0, int(np.floor(min(start[0], end[0]) - cutoff)))
        max_x = min(shape_x, int(np.ceil(max(start[0], end[0]) + cutoff)))
        min_y = max(0, int(np.floor(min(start[1], end[1]) - cutoff)))
        max_y = min(shape_y, int(np.ceil(max(start[1], end[1]) + cutoff)))

        ab_dot_max = max(ab_norm**2, 1e-8)

        for i in range(min_x, max_x):
            for j in range(min_y, max_y):
                ap_x = i - start[0]
                ap_y = j - start[1]

                t = (ap_x * ab_x + ap_y * ab_y) / ab_dot_max
                t = max(0.0, min(1.0, t))

                c_x = start[0] + t * ab_x
                c_y = start[1] + t * ab_y

                d2 = (i - c_x) ** 2 + (j - c_y) ** 2
                if d2 > cutoff * cutoff:
                    continue

                centerline_value = np.exp(-0.5 * d2 / sigma2)
                if centerline_value > centerline_map[i, j]:
                    centerline_map[i, j] = centerline_value

                if centerline_value > best_score_map[i, j]:
                    best_score_map[i, j] = centerline_value
                    vector_map[0, i, j] = dir_x
                    vector_map[1, i, j] = dir_y
                    radius_map[i, j] = radius_value


class StructuralTargetGenerator2D:
    def __init__(
        self,
        grid_shape: Tuple[int, int],
        base_sigma: float = 1.0,
        centerline_sigma: float = 0.65,
        radius_normalizer: float = 1.0,
    ):
        if len(grid_shape) != 2:
            raise ValueError("StructuralTargetGenerator2D only supports 2D grids.")
        self.grid_shape = grid_shape
        self.base_sigma = float(base_sigma)
        self.centerline_sigma = float(centerline_sigma)
        self.radius_normalizer = float(radius_normalizer)

    def generate(self, segments: List[FiberSegment]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        centerline_map = np.zeros(self.grid_shape, dtype=np.float64)
        best_score_map = np.zeros(self.grid_shape, dtype=np.float64)
        vector_map = np.zeros((2,) + self.grid_shape, dtype=np.float64)
        radius_map = np.zeros(self.grid_shape, dtype=np.float64)

        if not segments:
            return centerline_map, vector_map, radius_map

        starts = np.array([s.start for s in segments], dtype=np.float64)
        ends = np.array([s.end for s in segments], dtype=np.float64)
        sigmas = np.array(
            [max(0.1, self.base_sigma * float(s.thickness_mult)) for s in segments],
            dtype=np.float64,
        )

        _compute_structural_targets_2d_numba(
            centerline_map,
            best_score_map,
            vector_map,
            radius_map,
            starts,
            ends,
            sigmas,
            self.centerline_sigma,
            self.radius_normalizer,
        )
        return np.clip(centerline_map, 0.0, 1.0), vector_map, np.clip(radius_map, 0.0, 1.0)
