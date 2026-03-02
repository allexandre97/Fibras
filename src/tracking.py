import numpy as np
import networkx as nx
from scipy.ndimage import map_coordinates

class StreamlineTracker:
    def __init__(self, step_size: float = 0.5, edt_threshold: float = 0.5, max_steps: int = 1000):
        self.step_size = step_size
        self.edt_threshold = edt_threshold
        self.max_steps = max_steps

    def _sample_field(self, field, pos):
        # Sample continuously from the 3D grid
        return map_coordinates(field, [[pos[0]], [pos[1]], [pos[2]]], order=1, mode='constant', cval=0.0)[0]

    def _track_direction(self, start_pos, edt_map, vx, vy, vz, direction=1):
        path = [start_pos]
        curr_pos = np.copy(start_pos)
        
        for _ in range(self.max_steps):
            v_x = self._sample_field(vx, curr_pos)
            v_y = self._sample_field(vy, curr_pos)
            v_z = self._sample_field(vz, curr_pos)
            
            vec = np.array([v_x, v_y, v_z])
            norm = np.linalg.norm(vec)
            if norm < 1e-5: break
                
            vec = (vec / norm) * direction
            next_pos = curr_pos + vec * self.step_size
            
            # Check stopping condition (exited the fiber body)
            if self._sample_field(edt_map, next_pos) < self.edt_threshold:
                break
                
            path.append(next_pos)
            curr_pos = next_pos
            
        return path

    def extract_graph(self, edt_pred: np.ndarray, vector_pred: np.ndarray) -> nx.Graph:
        """
        edt_pred: (Z, Y, X)
        vector_pred: (3, Z, Y, X)
        """
        vx, vy, vz = vector_pred[0], vector_pred[1], vector_pred[2]
        
        # 1. Seeding: Find strong local maxima in the inverted EDT prediction
        from skimage.feature import peak_local_max
        seeds = peak_local_max(edt_pred, min_distance=2, threshold_abs=self.edt_threshold)
        
        graph = nx.Graph()
        node_idx = 0
        
        # 2. Integration
        # To avoid redundant tracking, mask out areas already tracked
        visited_mask = np.zeros_like(edt_pred, dtype=bool)
        
        for seed in seeds:
            if visited_mask[tuple(seed)]: continue
                
            # Track forwards and backwards along the vector flow
            path_fwd = self._track_direction(seed.astype(float), edt_pred, vx, vy, vz, direction=1)
            path_bwd = self._track_direction(seed.astype(float), edt_pred, vx, vy, vz, direction=-1)
            
            # Combine paths and add to graph
            full_path = path_bwd[::-1][:-1] + path_fwd
            
            prev_node = None
            for p in full_path:
                idx_tuple = tuple(np.round(p).astype(int))
                if not (0 <= idx_tuple[0] < edt_pred.shape[0] and 
                        0 <= idx_tuple[1] < edt_pred.shape[1] and 
                        0 <= idx_tuple[2] < edt_pred.shape[2]):
                    continue
                    
                visited_mask[idx_tuple] = True
                
                graph.add_node(node_idx, pos=p)
                if prev_node is not None:
                    graph.add_edge(prev_node, node_idx)
                prev_node = node_idx
                node_idx += 1
                
        return graph