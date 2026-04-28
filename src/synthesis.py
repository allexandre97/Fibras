from typing import List, Optional, Tuple
import math
import numpy as np
from src.core import BaseGenerator, FiberSegment, BoundaryCondition


class RandomWalkGenerator(BaseGenerator):
    def __init__(
        self,
        start_pos: Tuple[float, ...],
        num_steps: int,
        step_length: float,
        max_turn_angle: float,
        boundary: BoundaryCondition,
        initial_direction: Optional[np.ndarray] = None,
        orthogonal_scale: Optional[np.ndarray] = None,
    ):
        self.start_pos = np.array(start_pos, dtype=float)
        self.dims = self.start_pos.shape[0]
        self.num_steps = num_steps
        self.step_length = step_length
        self.max_turn_angle = max_turn_angle
        self.boundary = boundary
        self.initial_direction = None if initial_direction is None else np.array(initial_direction, dtype=float)
        self.orthogonal_scale = None if orthogonal_scale is None else np.array(orthogonal_scale, dtype=float)

    def _get_random_orthogonal_vector(self, v: np.ndarray) -> np.ndarray:
        r = np.random.normal(size=self.dims)
        if self.orthogonal_scale is not None:
            r *= self.orthogonal_scale
        u = r - (np.dot(r, v) / np.dot(v, v)) * v
        norm = np.linalg.norm(u)
        if norm > 1e-8:
            return u / norm

        fallback = np.random.normal(size=self.dims)
        if self.orthogonal_scale is not None:
            fallback *= self.orthogonal_scale
        fallback = fallback - (np.dot(fallback, v) / np.dot(v, v)) * v
        fallback_norm = np.linalg.norm(fallback)
        return fallback / fallback_norm if fallback_norm > 1e-8 else v.copy()

    def generate(self) -> List[FiberSegment]:
        segments = []
        current_pos = self.start_pos.copy()
        
        if self.initial_direction is None:
            current_dir = np.random.normal(size=self.dims)
        else:
            current_dir = self.initial_direction.copy()
        current_dir_norm = np.linalg.norm(current_dir)
        if current_dir_norm < 1e-8:
            current_dir = np.random.normal(size=self.dims)
            current_dir_norm = np.linalg.norm(current_dir)
        current_dir /= current_dir_norm

        is_alive = True
        steps_taken = 0

        while is_alive and steps_taken < self.num_steps:
            ortho_axis = self._get_random_orthogonal_vector(current_dir)
            angle = np.random.uniform(-self.max_turn_angle, self.max_turn_angle)
            
            next_dir = current_dir * math.cos(angle) + ortho_axis * math.sin(angle)
            next_dir /= np.linalg.norm(next_dir)

            step_vector = next_dir * self.step_length
            
            new_segments, current_pos, current_dir, is_alive = self.boundary.apply_step(
                current_pos, step_vector, thickness=1.0
            )
            
            segments.extend(new_segments)
            steps_taken += 1

        return segments
