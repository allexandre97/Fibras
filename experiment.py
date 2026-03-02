import time
import numpy as np
from src.core import ReflectiveBoundary
from src.synthesis import SpaceColonizationGenerator, RandomWalkGenerator
from src.rasterization import NDimRasterizer
from src.analysis import DensityVolumeAnalyzer

def generate_phenotype(phenotype: str, seed: int, N: int):
    np.random.seed(seed)
    grid_bounds = (N, N, N)
    
    if phenotype == "Baseline":
        attractors = np.random.uniform(0, N, size=(1500, 3))
        gen = SpaceColonizationGenerator(
            attractors, root_pos=(N//2, N//2, N//2), step_length=1.0, 
            attraction_distance=25.0, kill_distance=2.0, bounds=grid_bounds, 
            max_iterations=1000, thickness_decay=0.98
        )
        return gen.generate(), 1.5

    elif phenotype == "Thick":
        # Same topology parameters as Baseline, but rendered with double thickness
        attractors = np.random.uniform(0, N, size=(1500, 3))
        gen = SpaceColonizationGenerator(
            attractors, root_pos=(N//2, N//2, N//2), step_length=1.0, 
            attraction_distance=25.0, kill_distance=2.0, bounds=grid_bounds, 
            max_iterations=1000, thickness_decay=0.98
        )
        return gen.generate(), 3.0  

    elif phenotype == "Highly Branched":
        # Denser attractors force continuous topological splitting
        attractors = np.random.uniform(0, N, size=(4000, 3))
        gen = SpaceColonizationGenerator(
            attractors, root_pos=(N//2, N//2, N//2), step_length=1.0, 
            attraction_distance=20.0, kill_distance=1.5, bounds=grid_bounds, 
            max_iterations=1500, thickness_decay=0.99 
        )
        return gen.generate(), 1.5

    elif phenotype == "Directional":
        # Attractors confined to a narrow vertical cylinder to force parallel growth
        attractors = np.zeros((2000, 3))
        attractors[:, 0] = np.random.uniform(N//2 - 10, N//2 + 10, 2000)
        attractors[:, 1] = np.random.uniform(N//2 - 10, N//2 + 10, 2000)
        attractors[:, 2] = np.random.uniform(0, N, 2000)
        
        gen = SpaceColonizationGenerator(
            attractors, root_pos=(N//2, N//2, 0), step_length=1.0, 
            attraction_distance=25.0, kill_distance=2.0, bounds=grid_bounds, 
            max_iterations=1500, thickness_decay=0.99
        )
        return gen.generate(), 1.5

    elif phenotype == "Random Tangle":
        # Continuous mathematical string with zero true topological branches
        boundary = ReflectiveBoundary(grid_bounds)
        gen = RandomWalkGenerator(
            start_pos=(N//2, N//2, N//2), num_steps=2000, step_length=1.5, 
            max_turn_angle=1.2, boundary=boundary
        )
        return gen.generate(), 1.5

def run_robust_experiment(num_trials: int = 3, N: int = 256):
    phenotypes = ["Baseline", "Thick", "Highly Branched", "Directional", "Random Tangle"]
    
    # We initialize the analyzer with an expected base radius.
    # The pipeline must mathematically accommodate the 'Thick' bundle (rendered at 3.0)
    # despite being told to expect 1.5, proving the multi-scale pyramid works.
    analyzer = DensityVolumeAnalyzer(expected_fiber_radius=1.5, macro_scale_fraction=0.15)
    
    results = {p: {
        "Vol_Frac": [], "Thick_Proxy": [], "HFA": [], "Macro_FA": [],
        "Valency": [], "Bifurcations": [], "Crossings": []
    } for p in phenotypes}

    print(f"Running {num_trials} statistical trials per phenotype on a {N}x{N}x{N} grid...\n" + "-"*135)

    for trial in range(num_trials):
        seed = 42 + trial
        print(f"Executing Trial {trial + 1}/{num_trials} (Seed: {seed})")
        
        for p in phenotypes:
            t_start = time.time()
            skeleton, base_sigma = generate_phenotype(p, seed, N)
            
            # 1. Rasterize
            rasterizer = NDimRasterizer((N, N, N), base_sigma=base_sigma)
            volume = rasterizer.render(skeleton)
            
            # 2. Analyze
            metrics = analyzer.analyze(volume)
            
            # 3. Store Results
            results[p]["Vol_Frac"].append(metrics.volume_fraction)
            results[p]["Thick_Proxy"].append(metrics.mean_thickness_proxy)
            results[p]["HFA"].append(metrics.hfa_mean)
            results[p]["Macro_FA"].append(metrics.fa_macro_mean)
            results[p]["Valency"].append(metrics.mean_valency)
            results[p]["Bifurcations"].append(metrics.bifurcation_density)
            results[p]["Crossings"].append(metrics.crossing_density)
            
            print(f"  -> {p:<16} processed in {time.time() - t_start:>5.1f}s")

    # ==========================================
    # Print Comparative Table
    # ==========================================
    print("\nStatistical Results (Mean ± Std Dev)")
    print("-" * 145)
    headers = (f"{'Phenotype':<17} | {'Vol(%)':<12} | {'Thick':<12} | "
               f"{'HFA':<13} | {'Macro FA':<13} | {'Valency':<12} | {'Bifurc(%)':<12} | {'Cross(%)':<12}")
    print(headers)
    print("-" * 145)

    for p in phenotypes:
        vf_m, vf_s = np.mean(results[p]['Vol_Frac'])*100, np.std(results[p]['Vol_Frac'])*100
        tp_m, tp_s = np.mean(results[p]['Thick_Proxy']), np.std(results[p]['Thick_Proxy'])
        
        hfa_m, hfa_s = np.mean(results[p]['HFA']), np.std(results[p]['HFA'])
        mafa_m, mafa_s = np.mean(results[p]['Macro_FA']), np.std(results[p]['Macro_FA'])
        
        va_m, va_s = np.mean(results[p]['Valency']), np.std(results[p]['Valency'])
        
        # Multiply junctions by 100 to show as a percentage of total nodes
        bf_m, bf_s = np.mean(results[p]['Bifurcations'])*100, np.std(results[p]['Bifurcations'])*100
        cr_m, cr_s = np.mean(results[p]['Crossings'])*100, np.std(results[p]['Crossings'])*100

        row = (f"{p:<17} | {vf_m:>5.3f}±{vf_s:<4.3f} | {tp_m:>5.2f}±{tp_s:<4.2f} | "
               f"{hfa_m:>5.3f}±{hfa_s:<5.3f} | {mafa_m:>5.3f}±{mafa_s:<5.3f} | "
               f"{va_m:>5.3f}±{va_s:<4.3f} | {bf_m:>5.2f}±{bf_s:<4.2f} | {cr_m:>5.2f}±{cr_s:<4.2f}")
        print(row)

if __name__ == "__main__":
    # Running at 256 tests the extreme sparsity thresholding and dynamic integration scales.
    # Expect each trial to take roughly ~30-60 seconds depending on the CPU, as 
    # multi-scale 3D Hessian operations are mathematically heavy.
    run_robust_experiment(num_trials=3, N=256)