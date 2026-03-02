import numpy as np
from typing import List, Tuple
from src.core import FiberSegment
import numba as nb
import scipy.ndimage as ndi


@nb.njit(cache=True)
def _render_3d_numba(density_map, starts, ends, thicknesses, base_sigma):
    """
    Highly optimized JIT rasterization kernel. 
    Bypasses python object overhead and dynamic memory allocation.
    """
    num_segs = starts.shape[0]
    shape_x, shape_y, shape_z = density_map.shape

    for s in range(num_segs):
        start = starts[s]
        end = ends[s]
        sigma = base_sigma * thicknesses[s]
        cutoff = 4.0 * sigma

        # Bounding Box calculations natively in C
        min_x = int(np.floor(min(start[0], end[0]) - cutoff))
        max_x = int(np.ceil(max(start[0], end[0]) + cutoff))
        min_y = int(np.floor(min(start[1], end[1]) - cutoff))
        max_y = int(np.ceil(max(start[1], end[1]) + cutoff))
        min_z = int(np.floor(min(start[2], end[2]) - cutoff))
        max_z = int(np.ceil(max(start[2], end[2]) + cutoff))

        min_x, max_x = max(0, min_x), min(shape_x, max_x)
        min_y, max_y = max(0, min_y), min(shape_y, max_y)
        min_z, max_z = max(0, min_z), min(shape_z, max_z)

        if min_x >= max_x or min_y >= max_y or min_z >= max_z:
            continue

        ab_x = end[0] - start[0]
        ab_y = end[1] - start[1]
        ab_z = end[2] - start[2]
        ab_dot = ab_x*ab_x + ab_y*ab_y + ab_z*ab_z
        ab_dot_max = max(ab_dot, 1e-8)
        sigma2_inv = -1.0 / (2.0 * sigma * sigma)

        for i in range(min_x, max_x):
            for j in range(min_y, max_y):
                for k in range(min_z, max_z):
                    ap_x = i - start[0]
                    ap_y = j - start[1]
                    ap_z = k - start[2]

                    # Scalar dot product and clamp
                    t = (ap_x*ab_x + ap_y*ab_y + ap_z*ab_z) / ab_dot_max
                    if t < 0.0: t = 0.0
                    elif t > 1.0: t = 1.0

                    c_x = start[0] + t * ab_x
                    c_y = start[1] + t * ab_y
                    c_z = start[2] + t * ab_z

                    # Squared distance
                    d2 = (i - c_x)**2 + (j - c_y)**2 + (k - c_z)**2
                    density = np.exp(d2 * sigma2_inv)

                    # Update tensor safely
                    if density > density_map[i, j, k]:
                        density_map[i, j, k] = density


class NDimRasterizer:
    def __init__(self, grid_shape: Tuple[int, ...], base_sigma: float):
        self.grid_shape = grid_shape
        self.dims = len(grid_shape)
        self.base_sigma = base_sigma

    def _point_to_segment_distance(self, p: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        ab = b - a
        ap = p - a
        t = np.sum(ap * ab, axis=-1) / np.maximum(np.dot(ab, ab), 1e-8)
        t_clamped = np.clip(t, 0.0, 1.0)
        closest_point = a + t_clamped[..., np.newaxis] * ab
        return np.linalg.norm(p - closest_point, axis=-1)

    def render(self, segments: List[FiberSegment]) -> np.ndarray:
        density_map = np.zeros(self.grid_shape, dtype=np.float64)
        
        # Prepare filtered matrices to pass into the Numba engine
        half_grid = np.array(self.grid_shape) / 2.0
        
        valid_starts, valid_ends, valid_thicks = [], [], []
        for seg in segments:
            # Strip periodic wrap-around glitches
            if np.any(np.abs(seg.end - seg.start) > half_grid):
                continue
            valid_starts.append(seg.start)
            valid_ends.append(seg.end)
            valid_thicks.append(seg.thickness_mult)
            
        if not valid_starts:
            return density_map

        # --- Numba Accelerated Path (For 3D) ---
        if self.dims == 3:
            # Cast python lists to contiguous numpy arrays
            starts_arr = np.array(valid_starts, dtype=np.float64)
            ends_arr = np.array(valid_ends, dtype=np.float64)
            thicks_arr = np.array(valid_thicks, dtype=np.float64)
            
            _render_3d_numba(density_map, starts_arr, ends_arr, thicks_arr, self.base_sigma)
            return density_map

        # --- Slower NumPy Slice Path (For arbitrary N-D) ---
        for i in range(len(valid_starts)):
            start, end, thick = valid_starts[i], valid_ends[i], valid_thicks[i]
            sigma = self.base_sigma * thick
            cutoff = 4.0 * sigma 
            
            mins = np.floor(np.minimum(start, end) - cutoff).astype(int)
            maxs = np.ceil(np.maximum(start, end) + cutoff).astype(int)
            
            mins = np.maximum(0, mins)
            maxs = np.minimum(np.array(self.grid_shape), maxs)
            
            if np.any(mins >= maxs):
                continue
                
            slices = tuple(slice(mins[d], maxs[d]) for d in range(self.dims))
            ranges = [np.arange(mins[d], maxs[d]) for d in range(self.dims)]
            local_grid_points = np.stack(np.meshgrid(*ranges, indexing='ij'), axis=-1)
            
            dist = self._point_to_segment_distance(local_grid_points, start, end)
            density = np.exp(-(dist**2) / (2 * sigma**2))
            
            density_map[slices] = np.maximum(density_map[slices], density)

        return density_map
    

class EmpiricalRasterizer:
    """Synthesizes empirical microscopy artifacts: Anisotropy, Noise, Debris, and Gaps."""
    def __init__(self, bounds, base_sigma=1.0, z_anisotropy=3.0, noise_level=0.1, debris_count=30, gap_prob=0.08):
        self.bounds = bounds
        self.base_sigma = base_sigma
        self.z_anisotropy = z_anisotropy  
        self.noise_level = noise_level    
        self.debris_count = debris_count  
        self.gap_prob = gap_prob          

    def render(self, segments):
        np.random.seed(42)
        valid_idx = np.random.rand(len(segments)) > self.gap_prob
        valid_segments = [seg for i, seg in enumerate(segments) if valid_idx[i]]

        base_rasterizer = NDimRasterizer(self.bounds, self.base_sigma)
        volume = base_rasterizer.render(valid_segments)

        # CORRECTED: Only stretch the Z-axis. Leave the already-rendered XY plane sharp.
        psf_sigma = (self.base_sigma * self.z_anisotropy, 
                     self.base_sigma * 0.1, 
                     self.base_sigma * 0.1)
        
        volume = ndi.gaussian_filter(volume, sigma=psf_sigma)
        if volume.max() > 0:
            volume /= volume.max()

        # CORRECTED: Debris is constrained to a realistic 2.5x multiplier of the base fiber
        debris_sigma = self.base_sigma * 2.5
        for _ in range(self.debris_count):
            z, y, x = [np.random.randint(0, b) for b in self.bounds]
            blob = np.zeros(self.bounds)
            blob[z, y, x] = 1.0
            blob = ndi.gaussian_filter(blob, sigma=debris_sigma)
            if blob.max() > 0:
                volume += (blob / blob.max()) * 0.85 

        noise = np.random.poisson(lam=1.0, size=self.bounds) * self.noise_level
        volume += noise
        
        return np.clip(volume, 0, 1)