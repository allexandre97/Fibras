import os
import time
import numpy as np
import matplotlib.pyplot as plt
from src.synthesis import SpaceColonizationGenerator, RandomWalkGenerator
from src.core import ReflectiveBoundary
from src.rasterization import EmpiricalRasterizer
from src.analysis import DensityVolumeAnalyzer

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def save_mip_projections(volume: np.ndarray, skeleton: np.ndarray, title: str, filename: str):
    """Saves a Maximum Intensity Projection (MIP) of the 3D volume and skeleton."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"Phenotype: {title} (Confocal/STED Simulation)", fontsize=16)

    # 1. Density Volume MIP (Z-axis)
    vol_mip = np.max(volume, axis=0)
    ax0 = axes[0]
    im0 = ax0.imshow(vol_mip, cmap='magma', origin='lower')
    ax0.set_title("Raw Empirical Density (MIP)\n[Calibrated Noise & Debris]")
    ax0.axis('off')
    fig.colorbar(im0, ax=ax0, fraction=0.046, pad=0.04)

    # 2. Topological Skeleton MIP (Z-axis)
    skel_mip = np.max(skeleton, axis=0)
    ax1 = axes[1]
    ax1.imshow(skel_mip, cmap='gray', origin='lower')
    ax1.set_title("Filtered Skeleton (MIP)\n[HFA Gated & Gap Closed]")
    ax1.axis('off')

    # 3. Overlay (Skeleton on Density)
    ax2 = axes[2]
    ax2.imshow(vol_mip, cmap='magma', origin='lower')
    skel_overlay = np.ma.masked_where(skel_mip == 0, skel_mip)
    ax2.imshow(skel_overlay, cmap='hsv', alpha=0.7, origin='lower')
    ax2.set_title("Structural Overlay")
    ax2.axis('off')

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  -> Saved visualization to {filename}")

def generate_phenotype(name: str, N: int):
    """Generates specific structural phenotypes using adaptive, scale-invariant parameters."""
    grid_bounds = (N, N, N)
    np.random.seed(42)
    
    dynamic_step = max(1.0, N * 0.012)
    dynamic_kill = max(2.0, N * 0.015)
    dynamic_iters = int(N * 75)
    
    if name == "Highly Branched":
        num_attractors = int(N**3 * 0.0015)
        attractors = np.random.uniform(N * 0.1, N * 0.9, size=(num_attractors, 3))
        gen = SpaceColonizationGenerator(
            attractors, root_pos=(N//2, N//2, N//2), step_length=dynamic_step,
            attraction_distance=max(20.0, N * 0.25), kill_distance=dynamic_kill, 
            bounds=grid_bounds, max_iterations=dynamic_iters, thickness_decay=0.99
        )
        return gen.generate()

    elif name == "Directional":
        num_attractors = int(N**2 * 0.15)
        cyl_radius = max(10, N * 0.12)
        attractors = np.zeros((num_attractors, 3))
        attractors[:, 0] = np.random.uniform(N//2 - cyl_radius, N//2 + cyl_radius, num_attractors)
        attractors[:, 1] = np.random.uniform(N//2 - cyl_radius, N//2 + cyl_radius, num_attractors)
        attractors[:, 2] = np.random.uniform(N * 0.2, N * 0.9, num_attractors)
        gen = SpaceColonizationGenerator(
            attractors, root_pos=(N//2, N//2, N*0.1), step_length=dynamic_step,
            attraction_distance=max(30.0, N * 0.35), kill_distance=dynamic_kill, 
            bounds=grid_bounds, max_iterations=dynamic_iters, thickness_decay=0.99
        )
        return gen.generate()

    elif name == "Random Tangle":
        dynamic_steps = int(N * 25)
        boundary = ReflectiveBoundary(grid_bounds)
        gen = RandomWalkGenerator(
            start_pos=(N//2, N//2, N//2), num_steps=dynamic_steps, step_length=dynamic_step,
            max_turn_angle=1.0, boundary=boundary
        )
        return gen.generate()
    
    raise ValueError("Unknown phenotype")

def generate_latex_table(results: dict) -> str:
    """Formats the extracted metrics into a LaTeX table."""
    latex = [
        "\\begin{table}[h!]",
        "\\centering",
        "\\caption{Empirical Morphometric Analysis (Confocal/STED Simulation)}",
        "\\label{tab:empirical_fiber_metrics}",
        "\\begin{tabular}{l | c c c c c c}",
        "\\hline",
        "\\textbf{Phenotype} & \\textbf{Vol \\%} & \\textbf{HFA} & \\textbf{Macro FA} & \\textbf{Valency} & \\textbf{Bifurc \\%} & \\textbf{Cross \\%} \\\\",
        "\\hline"
    ]
    
    for name, m in results.items():
        row = (f"{name} & {m.volume_fraction*100:.3f} & {m.hfa_mean:.3f} & "
               f"{m.fa_macro_mean:.3f} & {m.mean_valency:.3f} & "
               f"{m.bifurcation_density*100:.2f} & {m.crossing_density*100:.2f} \\\\")
        latex.append(row)
        
    latex.extend([
        "\\hline",
        "\\end{tabular}",
        "\\vspace{1ex}",
        "\\raggedright \\small{\\textit{Note:} Analyzed with anisotropic voxel spacing (Z=2.0, Y=1.0, X=1.0) simulating high-NA confocal/STED PSF. Structural extraction applied HFA-gating to exclude biological debris and morphological closing to bridge staining dropouts.}",
        "\\end{table}"
    ])
    
    return "\n".join(latex)

def run_showcase():
    N = 64
    phenotypes = ["Highly Branched", "Directional", "Random Tangle"]
    output_dir = "showcase_output"
    ensure_dir(output_dir)
    
    dynamic_sigma = max(0.8, N * 0.008)
    
    # Adjusted to High-NA objective geometry (Z is 2x wider than XY)
    empirical_spacing = (2.0, 1.0, 1.0)
    
    analyzer = DensityVolumeAnalyzer(expected_fiber_radius=dynamic_sigma, macro_scale_fraction=0.15)
    results = {}

    print(f"Starting Empirical Project Showcase (Grid Size: {N}^3)")
    print(f"Optical PSF Anisotropy: {empirical_spacing}")
    print(f"Dynamic Fiber Sigma: {dynamic_sigma:.2f}")
    print("-" * 65)

    for p_name in phenotypes:
        print(f"Processing: {p_name}")
        t0 = time.time()
        
        skeleton_pts = generate_phenotype(p_name, N)
        
        # Calibrated artifacts for realistic, high-quality empirical data
        rasterizer = EmpiricalRasterizer(
            bounds=(N, N, N), 
            base_sigma=dynamic_sigma, 
            z_anisotropy=empirical_spacing[0], 
            noise_level=0.03,                 # Reduced to 3% Poisson background
            debris_count=int(N**3 * 0.00001), # Reduced to ~20 blobs for N=128
            gap_prob=0.025                    # Reduced to 2.5% photobleaching rate
        )
        volume = rasterizer.render(skeleton_pts)
        
        result = analyzer.analyze(volume, voxel_spacing=empirical_spacing)
        results[p_name] = result.metrics
        
        img_path = os.path.join(output_dir, f"{p_name.replace(' ', '_')}_Empirical_MIP.png")
        save_mip_projections(volume, result.skeleton, p_name, img_path)
        
        print(f"  -> Completed in {time.time() - t0:.2f}s\n")

    print("-" * 65)
    print("LaTeX Table Output for Documentation:")
    print("-" * 65)
    latex_code = generate_latex_table(results)
    print(latex_code)
    print("-" * 65)
    
    tex_path = os.path.join(output_dir, "empirical_results_table.tex")
    with open(tex_path, "w") as f:
        f.write(latex_code)
    print(f"Saved LaTeX table code to {tex_path}")

if __name__ == "__main__":
    run_showcase()