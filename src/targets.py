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
            
        dir_x, dir_y, dir_z = ab_x/ab_norm, ab_y/ab_norm, ab_z/ab_norm

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

                    t = (ap_x*ab_x + ap_y*ab_y + ap_z*ab_z) / ab_dot_max
                    t = max(0.0, min(1.0, t))

                    c_x = start[0] + t * ab_x
                    c_y = start[1] + t * ab_y
                    c_z = start[2] + t * ab_z

                    d = np.sqrt((i - c_x)**2 + (j - c_y)**2 + (k - c_z)**2)

                    if d < edt_map[i, j, k]:
                        edt_map[i, j, k] = d
                        vector_map[0, i, j, k] = dir_x
                        vector_map[1, i, j, k] = dir_y
                        vector_map[2, i, j, k] = dir_z

class TargetFieldGenerator:
    def __init__(self, grid_shape: Tuple[int, ...], max_distance: float = 5.0):
        self.grid_shape = grid_shape
        self.max_distance = max_distance

    def generate(self, segments: List[FiberSegment]) -> Tuple[np.ndarray, np.ndarray]:
        edt_map = np.full(self.grid_shape, self.max_distance, dtype=np.float64)
        vector_map = np.zeros((3,) + self.grid_shape, dtype=np.float64)
        
        starts = np.array([s.start for s in segments], dtype=np.float64)
        ends = np.array([s.end for s in segments], dtype=np.float64)
        
        _compute_targets_3d_numba(edt_map, vector_map, starts, ends, self.max_distance)
        
        # Normalize EDT to [0, 1] for stable regression, where 1 is exact centerline
        edt_normalized = 1.0 - (edt_map / self.max_distance)
        edt_normalized[edt_normalized < 0] = 0
        
        return edt_normalized, vector_map