import numpy as np
from dataclasses import dataclass
from typing import List, Tuple

# ==========================================
# Core Data Structures
# ==========================================

@dataclass
class FiberSegment:
    start: np.ndarray  # [N] dimension array
    end: np.ndarray    # [N] dimension array
    thickness_mult: float = 1.0

class BaseGenerator:
    def generate(self) -> List[FiberSegment]:
        raise NotImplementedError

# ==========================================
# Boundary Management System
# ==========================================

class BoundaryCondition:
    def __init__(self, bounds: Tuple[float, ...]):
        self.bounds = np.array(bounds, dtype=float)
        self.dims = len(self.bounds)

    def _get_intersection(self, pos: np.ndarray, vec: np.ndarray) -> Tuple[float, int]:
        t_min = float('inf')
        hit_dim = -1
        
        for i in range(self.dims):
            if vec[i] > 1e-8:
                t = (self.bounds[i] - pos[i]) / vec[i]
                if t < t_min:
                    t_min = t
                    hit_dim = i
            elif vec[i] < -1e-8:
                t = (0.0 - pos[i]) / vec[i]
                if t < t_min:
                    t_min = t
                    hit_dim = i
                    
        return t_min, hit_dim

    def apply_step(self, pos: np.ndarray, vec: np.ndarray, thickness: float) -> Tuple[List[FiberSegment], np.ndarray, np.ndarray, bool]:
        raise NotImplementedError

class DissipativeBoundary(BoundaryCondition):
    def apply_step(self, pos: np.ndarray, vec: np.ndarray, thickness: float):
        t, _ = self._get_intersection(pos, vec)
        if t >= 1.0:
            end_pos = pos + vec
            return [FiberSegment(pos.copy(), end_pos.copy(), thickness)], end_pos, vec / np.linalg.norm(vec), True
        else:
            hit_pos = pos + vec * t
            return [FiberSegment(pos.copy(), hit_pos.copy(), thickness)], hit_pos, np.zeros_like(vec), False

class ReflectiveBoundary(BoundaryCondition):
    def apply_step(self, pos: np.ndarray, vec: np.ndarray, thickness: float):
        segments = []
        curr_pos = pos.copy()
        curr_vec = vec.copy()
        
        for _ in range(10): 
            t, hit_dim = self._get_intersection(curr_pos, curr_vec)
            if t >= 1.0:
                end_pos = curr_pos + curr_vec
                segments.append(FiberSegment(curr_pos.copy(), end_pos.copy(), thickness))
                heading = curr_vec / np.linalg.norm(curr_vec)
                return segments, end_pos, heading, True
            
            hit_pos = curr_pos + curr_vec * t
            segments.append(FiberSegment(curr_pos.copy(), hit_pos.copy(), thickness))
            
            curr_vec = curr_vec * (1.0 - t) 
            curr_vec[hit_dim] *= -1.0       
            
            curr_pos = hit_pos
            curr_pos[hit_dim] += np.sign(curr_vec[hit_dim]) * 1e-6
            
        return segments, curr_pos, curr_vec / np.linalg.norm(curr_vec), False

class PeriodicBoundary(BoundaryCondition):
    def apply_step(self, pos: np.ndarray, vec: np.ndarray, thickness: float):
        segments = []
        curr_pos = pos.copy()
        curr_vec = vec.copy()
        
        for _ in range(10):
            t, hit_dim = self._get_intersection(curr_pos, curr_vec)
            if t >= 1.0:
                end_pos = curr_pos + curr_vec
                segments.append(FiberSegment(curr_pos.copy(), end_pos.copy(), thickness))
                heading = curr_vec / np.linalg.norm(curr_vec)
                return segments, end_pos, heading, True
            
            hit_pos = curr_pos + curr_vec * t
            segments.append(FiberSegment(curr_pos.copy(), hit_pos.copy(), thickness))
            
            curr_pos = hit_pos.copy()
            if curr_vec[hit_dim] > 0:
                curr_pos[hit_dim] = 0.0
            else:
                curr_pos[hit_dim] = self.bounds[hit_dim]
                
            curr_vec = curr_vec * (1.0 - t)
            curr_pos[hit_dim] += np.sign(curr_vec[hit_dim]) * 1e-6
            
        return segments, curr_pos, curr_vec / np.linalg.norm(curr_vec), False
