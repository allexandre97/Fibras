import numpy as np
import time
import networkx as nx

from scipy.ndimage import label, generate_binary_structure
from src.synthesis import SpaceColonizationGenerator
from src.rasterization import NDimRasterizer
from src.analysis import DensityVolumeAnalyzer
from src.visualization import AdvancedVisualizer

def generate_complex_bundle(N=512):
    """Generates a dense, branched volume with dynamically scaled constraints."""
    print(f"Synthesizing {N}^3 volume...")
    grid_bounds = (N, N, N)
    np.random.seed(42)
    
    #Number of attrators scaled to volum size
    num_attractors = int(N**3 * 0.0005)
    attractors = np.random.uniform(0, N, size=(num_attractors, 3))
    
    gen = SpaceColonizationGenerator(
        attractors, 
        root_pos=(N//2, N//2, N//2), 
        # Dynamic step length based on volume size
        step_length=max(0.5, N * 0.01),
        #Dynamic attraction distance
        attraction_distance=max(20.0, N * 0.05),
        # Dynamic kill distance         
        kill_distance=max(2.0, N * 0.005),
        bounds=grid_bounds, 
        # Dynamic max iterations based on volume size
        max_iterations=int(N * 20),
        thickness_decay=0.99
    )
    
    skeleton = gen.generate()
    rasterizer = NDimRasterizer(grid_bounds, base_sigma=1.5)
    volume = rasterizer.render(skeleton)
    return volume

def run_advanced_showcase():
    # 1. Prepare Data
    N = 128
    volume = generate_complex_bundle(N)
    
    # 2. Perform Analysis
    print("Performing multi-scale analysis...")
    analyzer = DensityVolumeAnalyzer(expected_fiber_radius=1.5, macro_scale_fraction=0.15)
    
    t0 = time.time()
    result = analyzer.analyze(volume) 
    print(f"Analysis complete in {time.time() - t0:.2f}s")
    
    # 3. Print Summary Metrics
    m = result.metrics
    print("\n--- Fiber Metrics Summary ---")
    print(f"Volume Fraction:    {m.volume_fraction*100:.4f}%")
    print(f"HFA (Tubularity):   {m.hfa_mean:.4f}")
    print(f"Macro FA (Align):   {m.fa_macro_mean:.4f}")
    print(f"Bifurcation %:      {m.bifurcation_density*100:.2f}%")
    print(f"Crossing %:         {m.crossing_density*100:.2f}%")
    print(f"Mean Valency:       {m.mean_valency:.4f}\n")

    # --- CORRECTED DIAGNOSTIC BLOCK ---
    print("--- Topology Continuity Diagnostics ---")
    
    # Generate a 3x3x3 structure of all 1s (26-connectivity for 3D diagonals)
    s_3d = generate_binary_structure(3, 3)
    
    # Apply the 26-connected structure to both the mask and the skeleton
    mask_labels, num_mask_features = label(result.binary_mask, structure=s_3d)
    skeleton_labels, num_skel_features = label(result.skeleton, structure=s_3d)
    
    print(f"Number of disconnected Mask chunks:     {num_mask_features}")
    print(f"Number of disconnected Skeleton chunks: {num_skel_features}")

    num_graph_components = nx.number_connected_components(result.graph)
    print(f"Number of disconnected Graph components: {num_graph_components}")
    
    if num_skel_features > num_mask_features * 1.5:
        print(">> WARNING: Severe skeleton fragmentation detected. <<")
    else:
        print(">> SUCCESS: Skeleton continuity matches mask continuity. <<")
    print("---------------------------------------\n")

    # --- EXPORT DEBUG DATA ---
    debug_filename = "fiber_debug_data.npz"
    print(f"Exporting raw 3D arrays to {debug_filename} for external debugging...")
    np.savez_compressed(
        debug_filename, 
        volume=volume, 
        mask=result.binary_mask, 
        skeleton=result.skeleton
    )
    print("Export complete.\n")

    # 4. Showcase Interactive Multi-Channel Slicing (Napari)
    print("Launching Napari Interactive Viewer...")
    print("Use the layer toggles to resolve magnitudes (HFA/Macro FA) in space.")
    AdvancedVisualizer.show_interactive_napari(volume, result)

if __name__ == "__main__":
    run_advanced_showcase()