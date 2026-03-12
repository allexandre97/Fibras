import numpy as np
import scipy.ndimage as ndi
from scipy.interpolate import RegularGridInterpolator

class StreamlineTracker:
    def __init__(self, step_size=0.5, min_edt=0.15, max_steps=5000):
        self.step_size = step_size
        self.min_edt = min_edt
        self.max_steps = max_steps

    def track(self, edt_map, vector_map):
        dim = edt_map.ndim
        is_2d = (dim == 2)
        
        # 1. Seed Generation: Find local cores (maxima) in the EDT
        valid_mask = edt_map > self.min_edt
        local_max = ndi.maximum_filter(edt_map, size=3) == edt_map
        seeds = np.argwhere(local_max & valid_mask).astype(float)
        
        # 2. Setup Continuous Space Interpolators
        grid_axes = [np.arange(s) for s in edt_map.shape]
        edt_interp = RegularGridInterpolator(grid_axes, edt_map, bounds_error=False, fill_value=0.0)
        
        # Map channels to the last dimension for SciPy interpolation
        vec_transposed = np.transpose(vector_map, (1, 2, 0)) if is_2d else np.transpose(vector_map, (1, 2, 3, 0))
        vec_interp = RegularGridInterpolator(grid_axes, vec_transposed, bounds_error=False, fill_value=0.0)
        
        streamlines = []
        
        for seed in seeds:
            # Integrate in both the positive and negative directions from the seed
            for direction in [1, -1]:
                path = [seed.copy()]
                current_pos = seed.copy()
                
                v_init = vec_interp(current_pos)[0]
                norm = np.linalg.norm(v_init)
                if norm < 1e-6: continue
                
                current_vec = (v_init / norm) * direction
                
                for _ in range(self.max_steps):
                    # Map mathematical [Vx, Vy, Vz] arrays to spatial [Z, Y, X] coordinates
                    if is_2d:
                        step = np.array([current_vec[1], current_vec[0]]) 
                    else:
                        step = np.array([current_vec[2], current_vec[1], current_vec[0]])
                        
                    next_pos = current_pos + step * self.step_size
                    
                    # Abort if the streamline exits the biological fiber volume
                    edt_val = edt_interp(next_pos)[0]
                    if edt_val < self.min_edt: break
                        
                    next_vec = vec_interp(next_pos)[0]
                    norm = np.linalg.norm(next_vec)
                    if norm < 1e-6: break
                    next_vec = next_vec / norm
                    
                    # Sign-Agnostic Momentum Resolution
                    # If the network arbitrarily flipped the vector polarity, flip it back to maintain momentum
                    if np.dot(current_vec, next_vec) < 0:
                        next_vec = -next_vec
                        
                    path.append(next_pos.copy())
                    current_pos = next_pos
                    current_vec = next_vec
                    
                if len(path) > 1:
                    streamlines.append(np.array(path))
                    
        return streamlines

    def to_binary_skeleton(self, streamlines, shape):
        """Burns the floating-point streamlines into a discrete binary TIFF mask."""
        skeleton = np.zeros(shape, dtype=np.uint8)
        for path in streamlines:
            coords = np.round(path).astype(int)
            for i in range(len(shape)):
                coords[:, i] = np.clip(coords[:, i], 0, shape[i] - 1)
            
            if len(shape) == 2:
                skeleton[coords[:, 0], coords[:, 1]] = 1
            else:
                skeleton[coords[:, 0], coords[:, 1], coords[:, 2]] = 1
        return skeleton
