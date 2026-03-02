#%%
import time
import numpy as np

# Assuming the proposed modular architecture:
from src.synthesis import SpaceColonizationGenerator, CompositeGenerator
from src.rasterization import NDimRasterizer
from src.analysis import StructureTensorAnalyzer, TopologyAnalyzer, DensityVolumeAnalyzer
from src.visualization import VolumetricVisualizer, MIPVisualizer

#%%

def generate_test_volume(N: int = 128) -> np.ndarray:
    """Synthesizes a 3D density volume for testing."""
    print("--- 1. Data Synthesis ---")
    np.random.seed(42)
    grid_bounds = (N, N, N)
    
    random_attractors = np.random.uniform(low=0, high=N, size=(2**12, 3))
    sc_gen = SpaceColonizationGenerator(
        attractors=random_attractors,
        root_pos=(N//2, N//2, N//2),
        step_length=1.0,            
        attraction_distance=30.0,   
        kill_distance=3.0, 
        bounds=grid_bounds, 
        periodic=False, 
        max_iterations=1500,        
        thickness_decay=0.99        
    )
    
    t0 = time.time()
    composite = CompositeGenerator([sc_gen])
    skeleton = composite.generate()
    print(f"Generated {len(skeleton)} fiber segments in {time.time() - t0:.2f}s")
    
    t1 = time.time()
    rasterizer = NDimRasterizer(grid_shape=grid_bounds, base_sigma=2.0)
    density_volume = rasterizer.render(skeleton)
    print(f"Rasterized {N}x{N}x{N} volume in {time.time() - t1:.2f}s\n")
    
    return density_volume

def showcase_analysis(volume: np.ndarray):
    print("--- 2. Pipeline Analysis ---")
    
    # Initialize standalone analyzers to extract intermediate maps
    tensor_analyzer = StructureTensorAnalyzer(inner_sigma=1.0, outer_sigma=3.0)
    topology_analyzer = TopologyAnalyzer()
    
    # 1. Topological Extraction
    t0 = time.time()
    skeleton_map, binary_mask = topology_analyzer.extract_skeleton(volume)
    print(f"Extracted topological skeleton in {time.time() - t0:.2f}s")
    
    # 2. Graph Building
    t1 = time.time()
    graph = topology_analyzer.build_network(skeleton_map)
    print(f"Constructed spatial graph ({len(graph.nodes)} nodes, {len(graph.edges)} edges) in {time.time() - t1:.2f}s")
    
    # 3. Structure Tensor Analysis
    t2 = time.time()
    tensor = tensor_analyzer.compute_tensor(volume)
    fa_map = tensor_analyzer.compute_fractional_anisotropy(tensor, binary_mask)
    print(f"Computed structure tensor & FA map in {time.time() - t2:.2f}s\n")

    print("--- 3. Orchestrator Metrics ---")
    # Execute the unified orchestrator
    orchestrator = DensityVolumeAnalyzer(inner_sigma=1.0, outer_sigma=3.0)
    metrics = orchestrator.analyze(volume)
    
    print(f"Volume Fraction:            {metrics.volume_fraction:.4f}")
    print(f"Mean Thickness Proxy:       {metrics.mean_thickness_proxy:.4f} voxels/skeleton-unit")
    print(f"Fractional Anisotropy (FA): {metrics.fractional_anisotropy_mean:.4f} ± {metrics.fractional_anisotropy_std:.4f}")
    print(f"Network Mean Valency:       {metrics.mean_valency:.4f}")
    print(f"Network Branching Density:  {metrics.branching_density:.4f} branches/node\n")

    print("--- 4. Visualizing 2D Analysis Projections ---")
    # Note: This will block execution until the matplotlib window is closed
    MIPVisualizer.show_analysis_maps(volume, skeleton_map, fa_map)

    print("--- 5. Visualizing 3D Volumetric Data ---")
    vol_vis = VolumetricVisualizer(volume)
    vol_vis.show_volume(vmin=0.1, vmax=np.max(volume))

#%%
if __name__ == "__main__":
    density_volume = generate_test_volume(N=128)
    showcase_analysis(density_volume)
# %%
