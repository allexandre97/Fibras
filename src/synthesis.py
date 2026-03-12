from typing import List, Optional, Tuple
import math
import numpy as np
from scipy.spatial import cKDTree
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

class SCNode:
    def __init__(self, pos: np.ndarray, parent=None, depth: int = 0):
        self.pos = pos
        self.parent = parent
        self.depth = depth

class SpaceColonizationGenerator(BaseGenerator):
    def __init__(self, attractors: np.ndarray, root_pos: Tuple[float, ...], 
                 step_length: float, attraction_distance: float, kill_distance: float,
                 bounds: Tuple[float, ...], periodic: bool = False,
                 max_iterations: int = 1000, thickness_decay: float = 0.95):
        self.attractors = np.copy(attractors) 
        self.nodes = [SCNode(np.array(root_pos, dtype=float))]
        self.step_length = step_length
        self.attraction_distance = attraction_distance
        self.kill_distance = kill_distance
        self.bounds = np.array(bounds, dtype=float)
        self.dims = len(bounds)
        self.periodic = periodic
        self.max_iterations = max_iterations
        self.thickness_decay = thickness_decay

    def generate(self) -> List[FiberSegment]:
        active_attractors = self.attractors.copy()
        
        for _ in range(self.max_iterations):
            if len(active_attractors) == 0:
                break
                
            node_positions = np.array([n.pos for n in self.nodes])
            
            if self.periodic:
                tree = cKDTree(node_positions, boxsize=self.bounds)
            else:
                tree = cKDTree(node_positions)
                
            distances, indices = tree.query(active_attractors, distance_upper_bound=self.attraction_distance)
            
            # --- Vectorized Influence & Culling (Replaces nested loops) ---
            
            # 1. Determine Attractors to Keep
            survivor_mask = distances >= self.kill_distance
            next_attractors = active_attractors[survivor_mask]
            
            # 2. Extract Valid Influences
            valid_mask = (distances >= self.kill_distance) & (distances < self.attraction_distance)
            valid_attractors = active_attractors[valid_mask]
            valid_indices = indices[valid_mask]
            
            if len(valid_attractors) > 0:
                # Calculate direction vectors
                directions = valid_attractors - node_positions[valid_indices]
                if self.periodic:
                    directions = directions - self.bounds * np.round(directions / self.bounds)
                    
                norms = np.linalg.norm(directions, axis=1, keepdims=True)
                nonzero_mask = norms[:, 0] > 0
                
                if np.any(nonzero_mask):
                    directions = directions[nonzero_mask] / norms[nonzero_mask]
                    target_indices = valid_indices[nonzero_mask]
                    
                    # Accumulate vectors per node automatically using np.add.at
                    node_influences = np.zeros((len(self.nodes), self.dims))
                    np.add.at(node_influences, target_indices, directions)
                    
                    # Spawn new nodes only for nodes that received influence
                    influenced_idx = np.where(np.any(node_influences != 0, axis=1))[0]
                    new_nodes = []
                    for idx in influenced_idx:
                        node = self.nodes[idx]
                        avg_dir = node_influences[idx]
                        avg_norm = np.linalg.norm(avg_dir)
                        if avg_norm > 0:
                            avg_dir /= avg_norm
                            new_pos = node.pos + avg_dir * self.step_length
                            if self.periodic:
                                new_pos = new_pos % self.bounds
                            new_nodes.append(SCNode(new_pos, parent=node, depth=node.depth + 1))
                            
                    self.nodes.extend(new_nodes)
                    
            active_attractors = next_attractors

        segments = []
        for node in self.nodes:
            if node.parent is not None:
                thickness = self.thickness_decay ** node.depth
                segments.append(FiberSegment(node.parent.pos.copy(), node.pos.copy(), thickness))
                
        return segments

class CompositeGenerator(BaseGenerator):
    def __init__(self, generators: List[BaseGenerator]):
        self.generators = generators

    def generate(self) -> List[FiberSegment]:
        all_segments = []
        for gen in self.generators:
            all_segments.extend(gen.generate())
        return all_segments
