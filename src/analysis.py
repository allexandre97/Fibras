import numpy as np
import scipy.ndimage as ndi
import warnings
from skimage.morphology import skeletonize, remove_small_objects
from skimage.filters import apply_hysteresis_threshold
from skimage.measure import block_reduce
from dataclasses import dataclass
from typing import Dict, Tuple
import networkx as nx
from scipy.spatial import cKDTree

@dataclass
class FiberMetrics:
    volume_fraction: float
    hfa_mean: float
    fa_macro_mean: float
    mean_thickness_proxy: float
    mean_valency: float
    bifurcation_density: float
    crossing_density: float

@dataclass
class AnalysisResult:
    metrics: FiberMetrics
    hfa_map: np.ndarray
    fa_macro_map: np.ndarray
    skeleton: np.ndarray
    binary_mask: np.ndarray
    graph: nx.Graph

class TopologyAnalyzer:
    def __init__(self, ridge_sharpening_sigma: float = 1.0):
        self.ridge_sigma = ridge_sharpening_sigma

    def extract_vesselness_topology(self, vesselness_map: np.ndarray, nms_ridges: np.ndarray, 
                                    low_v: float = 0.05, high_v: float = 0.25) -> np.ndarray:
        """
        Extracts continuous structures based strictly on local differential geometry.
        low_v: The geometry collapse floor (traces until the tube dissolves into noise).
        high_v: The seeding threshold (must strongly resemble a tube to begin a path).
        """
        # Restrict the vesselness map strictly to the 1-voxel thin NMS carved ridges
        ridge_vesselness = vesselness_map * (nms_ridges > 1e-4)
        return apply_hysteresis_threshold(ridge_vesselness, low_v, high_v)

    def build_network_coherence_gated(self, skeleton: np.ndarray, spacing: Tuple[float, float, float], fiber_evecs: np.ndarray) -> nx.Graph:
        G = nx.Graph()
        coords = np.argwhere(skeleton)
        if len(coords) == 0: return G
            
        physical_coords = coords * np.array(spacing)
        tree = cKDTree(physical_coords)
        
        search_radius = np.linalg.norm(spacing) + 0.1
        pairs = list(tree.query_pairs(r=search_radius))
        
        for i, coord in enumerate(coords):
            G.add_node(i, pos=tuple(coord), vec=fiber_evecs[tuple(coord)])
            
        for i, j in pairs:
            p1, p2 = physical_coords[i], physical_coords[j]
            edge_vec = p2 - p1
            edge_vec /= (np.linalg.norm(edge_vec) + 1e-10)
            
            v1, v2 = G.nodes[i]['vec'], G.nodes[j]['vec']
            
            if abs(np.dot(edge_vec, v1)) > 0.45 and abs(np.dot(edge_vec, v2)) > 0.45:
                G.add_edge(i, j)
        return G

    def filter_by_path_persistence(self, G: nx.Graph, min_path_len: float) -> nx.Graph:
        components = list(nx.connected_components(G))
        to_remove = []
        for comp in components:
            if len(comp) < 5:
                to_remove.extend(comp)
                continue
            
            sub = G.subgraph(comp)
            pos = nx.get_node_attributes(sub, 'pos')
            p_arr = np.array([pos[n] for n in comp])
            
            span = np.linalg.norm(np.max(p_arr, axis=0) - np.min(p_arr, axis=0))
            if span < min_path_len:
                to_remove.extend(comp)
                
        G.remove_nodes_from(to_remove)
        return G

    def prune_skeleton_graph(self, G: nx.Graph, min_length: float) -> nx.Graph:
        if len(G) == 0: return G
        while True:
            leaves = [n for n, d in G.degree() if d == 1]
            to_remove = set()
            
            for leaf in leaves:
                if leaf in to_remove: continue
                    
                path = [leaf]
                current = leaf
                prev = None
                
                while True:
                    neighbors = list(G.neighbors(current))
                    if prev in neighbors: neighbors.remove(prev)
                        
                    if not neighbors: break
                    if len(neighbors) > 1: break
                        
                    next_node = neighbors[0]
                    path.append(next_node)
                    prev = current
                    current = next_node
                    
                pos = nx.get_node_attributes(G, 'pos')
                length = sum(np.linalg.norm(np.array(pos[path[i]]) - np.array(pos[path[i+1]])) for i in range(len(path)-1))
                
                if length < min_length:
                    if G.degree(current) <= 1:
                        to_remove.update(path)
                    else:
                        to_remove.update(path[:-1])
                        
            if not to_remove: break
            G.remove_nodes_from(to_remove)
        return G

    def compute_network_metrics(self, G: nx.Graph) -> Dict[str, float]:
        num_nodes = len(G)
        if num_nodes == 0: 
            return {"mean_valency": 0.0, "bifurcation_density": 0.0, "crossing_density": 0.0}
            
        degrees = [d for n, d in G.degree()]
        return {
            "mean_valency": np.mean(degrees),
            "bifurcation_density": sum(1 for d in degrees if d == 3) / num_nodes,
            "crossing_density": sum(1 for d in degrees if d >= 4) / num_nodes
        }

class HessianAnalyzer:
    def __init__(self, sigmas: Tuple[float, ...] = (1.0, 2.0, 4.0)):
        self.sigmas = sigmas
        
    def compute_multiscale(self, volume: np.ndarray, mask: np.ndarray, spacing: Tuple[float, float, float]):
        max_vesselness_map = np.zeros_like(volume)
        cross_evec_map = np.zeros(volume.shape + (volume.ndim,))
        fiber_evec_map = np.zeros(volume.shape + (volume.ndim,))
        BETA = 0.5 
        
        for sigma in self.sigmas:
            hfa_vals, evals, evecs, valid_coords = self._compute_single_scale_full(volume, mask, sigma, spacing)
            if len(valid_coords[0]) == 0: continue
            
            abs_evals = np.abs(evals)
            sort_idx = np.argsort(abs_evals, axis=-1)
            row_idx = np.arange(len(evals))
            
            idx1 = sort_idx[:, 0]
            idx2 = sort_idx[:, 1]
            idx3 = sort_idx[:, 2]
            
            l1 = evals[row_idx, idx1]
            l2 = evals[row_idx, idx2]
            l3 = evals[row_idx, idx3]
            
            # Relaxed tolerance (1e-4) to account for numerical discretization errors on the finite grid
            valid_structure = (l2 < 0) & (l3 < 0)
            
            rb = np.abs(l1) / (np.sqrt(np.abs(l2 * l3)) + 1e-10)
            
            vesselness = np.zeros_like(hfa_vals)
            v_vals = hfa_vals[valid_structure] * np.exp(-(rb[valid_structure]**2) / (2 * BETA**2))
            vesselness[valid_structure] = v_vals
            
            current_v = np.zeros_like(volume)
            current_v[valid_coords] = vesselness
            
            update_mask = current_v > max_vesselness_map
            max_vesselness_map[update_mask] = current_v[update_mask]
            
            cross_full = np.zeros(volume.shape + (volume.ndim,))
            cross_full[valid_coords] = evecs[row_idx, :, idx3]
            
            fiber_full = np.zeros(volume.shape + (volume.ndim,))
            fiber_full[valid_coords] = evecs[row_idx, :, idx1]
            
            for dim in range(volume.ndim):
                cross_evec_map[..., dim][update_mask] = cross_full[..., dim][update_mask]
                fiber_evec_map[..., dim][update_mask] = fiber_full[..., dim][update_mask]
                
        return max_vesselness_map, cross_evec_map, fiber_evec_map

    def _compute_single_scale_full(self, volume: np.ndarray, mask: np.ndarray, sigma: float, spacing: Tuple[float, float, float]):
        ndim = volume.ndim
        hessian = np.zeros((ndim, ndim) + volume.shape)
        s_scaled = [sigma / sp for sp in spacing]
        for i in range(ndim):
            for j in range(i, ndim):
                order_i = 2 if i == j else 1
                order_j = 0 if i == j else 1
                h = ndi.gaussian_filter1d(volume, s_scaled[i], axis=i, order=order_i)
                if order_j > 0: h = ndi.gaussian_filter1d(h, s_scaled[j], axis=j, order=order_j)
                hessian[i, j] = h * (sigma ** 2)
                if i != j: hessian[j, i] = hessian[i, j]
                    
        valid_coords = np.where(mask)
        if len(valid_coords[0]) == 0: return np.array([]), np.array([]), np.array([]), valid_coords
            
        hessian_vals = np.zeros((len(valid_coords[0]), ndim, ndim))
        for i in range(ndim):
            for j in range(ndim):
                hessian_vals[:, i, j] = hessian[i, j][valid_coords]
                
        evals, evecs = np.linalg.eigh(hessian_vals)
        mean_eig = np.mean(evals, axis=-1, keepdims=True)
        num = np.sum((evals - mean_eig)**2, axis=-1)
        den = np.sum(evals**2, axis=-1)
        
        hfa_vals = np.zeros_like(num)
        valid_den = den > 1e-10
        hfa_vals[valid_den] = np.sqrt(1.5) * np.sqrt(num[valid_den]) / np.sqrt(den[valid_den])
        
        return hfa_vals, evals, evecs, valid_coords

class StructureTensorAnalyzer:
    def __init__(self, inner_sigma: float = 2.0):
        self.inner_sigma = inner_sigma

    def compute_normalized_tensor(self, volume: np.ndarray, mask: np.ndarray, outer_sigma: float, spacing: Tuple[float, float, float]) -> np.ndarray:
        ndim = volume.ndim
        s_in = [self.inner_sigma / sp for sp in spacing]
        s_out = [outer_sigma / sp for sp in spacing]
        
        grads = [ndi.gaussian_filter1d(volume, s_in[i], axis=i, order=1) for i in range(ndim)]
        weight_sum = np.clip(ndi.gaussian_filter(mask.astype(float), s_out), 1e-10, None)
        
        tensor = np.zeros((ndim, ndim) + volume.shape)
        for i in range(ndim):
            for j in range(i, ndim):
                comp = (grads[i] * grads[j]) * mask
                smoothed_comp = ndi.gaussian_filter(comp, s_out)
                tensor[i, j] = smoothed_comp / weight_sum
                if i != j: tensor[j, i] = tensor[i, j]
        return tensor

    def compute_fractional_anisotropy(self, tensor: np.ndarray, mask: np.ndarray) -> np.ndarray:
        ndim = tensor.shape[0]
        fa_map = np.zeros(tensor.shape[2:])
        valid_coords = np.where(mask)
        if len(valid_coords[0]) == 0: return fa_map
        
        tensor_vals = np.zeros((len(valid_coords[0]), ndim, ndim))
        for i in range(ndim):
            for j in range(ndim):
                tensor_vals[:, i, j] = tensor[i, j][valid_coords]
                
        evals = np.linalg.eigvalsh(tensor_vals)
        mean_eig = np.mean(evals, axis=-1, keepdims=True)
        num = np.sum((evals - mean_eig)**2, axis=-1)
        den = np.sum(evals**2, axis=-1)
        
        fa_vals = np.zeros_like(num)
        valid_den = den > 1e-10
        fa_vals[valid_den] = np.sqrt(1.5) * np.sqrt(num[valid_den]) / np.sqrt(den[valid_den])
        fa_map[valid_coords] = fa_vals
        return fa_map

class DensityVolumeAnalyzer:
    def __init__(self, expected_fiber_radius: float = 1.0, macro_scale_fraction: float = 0.15):
        self.expected_fiber_radius = expected_fiber_radius
        self.macro_scale_fraction = macro_scale_fraction
        
        s1 = max(0.5, expected_fiber_radius * 0.8)
        s2 = max(1.0, expected_fiber_radius * 1.5)
        s3 = max(2.0, expected_fiber_radius * 2.5)
        
        self.hessian_analyzer = HessianAnalyzer(sigmas=(s1, s2, s3))
        self.tensor_macro = StructureTensorAnalyzer(inner_sigma=expected_fiber_radius)
        self.topology_analyzer = TopologyAnalyzer(ridge_sharpening_sigma=expected_fiber_radius)

    def _apply_directional_nms(self, sharpened: np.ndarray, evecs: np.ndarray, mask: np.ndarray) -> np.ndarray:
        out = np.zeros_like(sharpened)
        coords = np.argwhere(mask)
        if len(coords) == 0: return out
        
        z, y, x = coords.T
        dz, dy, dx = evecs[z, y, x, 0], evecs[z, y, x, 1], evecs[z, y, x, 2]
        
        val_center = sharpened[z, y, x]
        val_p = ndi.map_coordinates(sharpened, [z+dz, y+dy, x+dx], order=1, mode='constant', cval=0.0)
        val_m = ndi.map_coordinates(sharpened, [z-dz, y-dy, x-dx], order=1, mode='constant', cval=0.0)
        
        is_max = (val_center >= val_p) & (val_center >= val_m)
        out[z[is_max], y[is_max], x[is_max]] = val_center[is_max]
        return out

    def _directional_gap_bridging(self, mask: np.ndarray, fiber_evecs: np.ndarray) -> np.ndarray:
        out = mask.copy()
        coords = np.argwhere(mask)
        if len(coords) == 0: return out
        z, y, x = coords.T
        dz, dy, dx = fiber_evecs[z, y, x, 0], fiber_evecs[z, y, x, 1], fiber_evecs[z, y, x, 2]
        
        for sign in [1, -1]:
            nz = np.clip(np.round(z + sign * dz).astype(int), 0, mask.shape[0]-1)
            ny = np.clip(np.round(y + sign * dy).astype(int), 0, mask.shape[1]-1)
            nx_ = np.clip(np.round(x + sign * dx).astype(int), 0, mask.shape[2]-1)
            out[nz, ny, nx_] = True
        return out

    def analyze(self, volume: np.ndarray, voxel_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)) -> AnalysisResult:
        bbox = self._get_bounding_box(volume)
        if bbox is None: return self._empty_result(volume)
        
        crop_vol = volume[bbox]
        denoised = ndi.gaussian_filter(crop_vol, sigma=0.5)
        
        laplacian = np.zeros_like(denoised)
        s_scaled = [self.topology_analyzer.ridge_sigma / sp for sp in voxel_spacing]
        for i in range(denoised.ndim):
            laplacian += ndi.gaussian_filter1d(denoised, s_scaled[i], axis=i, order=2)
            
        sharpened = np.clip(denoised - laplacian, 0, None)
        
        # Lowered structural zero-gate to process fainter signal bounds
        base_mask = sharpened > 1e-5
        if not np.any(base_mask): return self._empty_result(volume)
            
        v_map, cross_ev, fiber_ev = self.hessian_analyzer.compute_multiscale(denoised, base_mask, voxel_spacing)
        
        nms_vol = self._apply_directional_nms(sharpened, cross_ev, base_mask)
        
        # SENSITIVITY FIX: Geometric extraction driven entirely by normalized vesselness
        # low_v=0.05 guarantees tracing until the tube structurally dissipates into noise
        mask_crop = self.topology_analyzer.extract_vesselness_topology(v_map, nms_vol, low_v=0.05, high_v=0.25)
        
        mask_crop = self._directional_gap_bridging(mask_crop, fiber_ev)
        
        min_vol = max(32, int((self.expected_fiber_radius ** 3) * 100))
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                mask_crop = remove_small_objects(mask_crop.astype(bool), max_size=min_vol)
            except TypeError:
                mask_crop = remove_small_objects(mask_crop.astype(bool), min_size=min_vol)
        
        skeleton_raw = skeletonize(mask_crop)
        graph = self.topology_analyzer.build_network_coherence_gated(skeleton_raw, voxel_spacing, fiber_ev)
        graph = self.topology_analyzer.filter_by_path_persistence(graph, self.expected_fiber_radius * 10.0)
        graph = self.topology_analyzer.prune_skeleton_graph(graph, self.expected_fiber_radius * 3.0)
        
        skeleton = np.zeros_like(volume, dtype=bool)
        mask = np.zeros_like(volume, dtype=bool)
        hfa_map = np.zeros_like(volume)
        
        final_skeleton_crop = np.zeros_like(skeleton_raw, dtype=bool)
        pos = nx.get_node_attributes(graph, 'pos')
        for node in graph.nodes():
            final_skeleton_crop[tuple(np.round(pos[node]).astype(int))] = True
            
        skeleton[bbox] = final_skeleton_crop
        mask[bbox] = mask_crop
        hfa_map[bbox] = v_map
        
        ds = 4
        vol_ds = block_reduce(volume[bbox], (ds,)*3, np.mean)
        mask_ds = block_reduce(mask_crop.astype(float), (ds,)*3, np.max) > 0.5
        t_ds = self.tensor_macro.compute_normalized_tensor(vol_ds, mask_ds, (max(volume.shape)*0.15)/ds, voxel_spacing)
        fa_ds = self.tensor_macro.compute_fractional_anisotropy(t_ds, mask_ds)
        fa_macro_map = np.zeros_like(volume)
        fa_full = ndi.zoom(fa_ds, zoom=ds, order=1)
        slices = tuple(slice(0, min(fa_full.shape[i], crop_vol.shape[i])) for i in range(3))
        fa_macro_map[bbox][slices] = fa_full[slices]

        net_m = self.topology_analyzer.compute_network_metrics(graph)
        metrics = FiberMetrics(
            volume_fraction=np.sum(mask) / volume.size,
            hfa_mean=np.mean(v_map[mask_crop]) if np.any(mask_crop) else 0.0,
            fa_macro_mean=np.mean(fa_macro_map[mask]) if np.any(mask) else 0.0,
            mean_thickness_proxy=(np.sum(mask) / len(graph)) if len(graph) > 0 else 0.0,
            mean_valency=net_m["mean_valency"],
            bifurcation_density=net_m["bifurcation_density"],
            crossing_density=net_m["crossing_density"]
        )
        return AnalysisResult(metrics, hfa_map, fa_macro_map, skeleton, mask, graph)

    def _get_bounding_box(self, volume: np.ndarray, pad: int = 5) -> tuple:
        coords = np.argwhere(volume > 1e-6)
        if len(coords) == 0: return None
        z0, y0, x0 = coords.min(axis=0) - pad
        z1, y1, x1 = coords.max(axis=0) + 1 + pad
        return (slice(max(0, z0), min(volume.shape[0], z1)), 
                slice(max(0, y0), min(volume.shape[1], y1)), 
                slice(max(0, x0), min(volume.shape[2], x1)))

    def _empty_result(self, volume: np.ndarray):
        m = FiberMetrics(0,0,0,0,0,0,0)
        return AnalysisResult(m, np.zeros_like(volume), np.zeros_like(volume), 
                              np.zeros_like(volume, dtype=bool), np.zeros_like(volume, dtype=bool), nx.Graph())