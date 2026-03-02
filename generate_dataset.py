import os
import argparse
import numpy as np
import torch
import concurrent.futures
from functools import partial

from src.synthesis import SpaceColonizationGenerator, RandomWalkGenerator
from src.core import ReflectiveBoundary
from src.rasterization import EmpiricalRasterizer
from src.targets import TargetFieldGenerator

PHENOTYPES = ["Highly Branched", "Directional", "Random Tangle"]

def _generate_flexible_segments(phenotype: str, seed: int, bounds: tuple):
    """
    Dimension-agnostic synthesis logic.
    bounds is expected in (Z, Y, X) format.
    """
    np.random.seed(seed)
    
    Z, Y, X = bounds
    is_2d = (Z == 1)
    max_dim = max(bounds)
    
    # Scale dynamics dynamically to the largest spatial dimension
    dynamic_step = max(1.0, max_dim * 0.012)
    dynamic_kill = max(2.0, max_dim * 0.015)
    
    if phenotype == "Highly Branched":
        num_attractors = int((X * Y * max(1, Z)) * 0.0015)
        attractors = np.zeros((num_attractors, 3))
        
        attractors[:, 0] = np.random.uniform(X * 0.1, X * 0.9, num_attractors)
        attractors[:, 1] = np.random.uniform(Y * 0.1, Y * 0.9, num_attractors)
        attractors[:, 2] = 0.5 if is_2d else np.random.uniform(Z * 0.1, Z * 0.9, num_attractors)
        
        root = (X//2, Y//2, 0.5) if is_2d else (X//2, Y//2, Z//2)
        
        gen = SpaceColonizationGenerator(
            attractors, root_pos=root, step_length=dynamic_step,
            attraction_distance=max(20.0, max_dim * 0.25), kill_distance=dynamic_kill, 
            bounds=bounds, max_iterations=int(max_dim * 75), thickness_decay=0.99
        )
        return gen.generate()

    elif phenotype == "Directional":
        num_attractors = int((X * Y * max(1, Z)) * 0.0005)
        cyl_radius = max(10, max_dim * 0.12)
        attractors = np.zeros((num_attractors, 3))
        
        attractors[:, 0] = np.random.uniform(X//2 - cyl_radius, X//2 + cyl_radius, num_attractors)
        attractors[:, 1] = np.random.uniform(Y//2 - cyl_radius, Y//2 + cyl_radius, num_attractors)
        attractors[:, 2] = 0.5 if is_2d else np.random.uniform(Z * 0.2, Z * 0.9, num_attractors)
        
        root = (X//2, Y//2, 0.5) if is_2d else (X//2, Y//2, Z * 0.1)
        
        gen = SpaceColonizationGenerator(
            attractors, root_pos=root, step_length=dynamic_step,
            attraction_distance=max(30.0, max_dim * 0.35), kill_distance=dynamic_kill, 
            bounds=bounds, max_iterations=int(max_dim * 75), thickness_decay=0.99
        )
        return gen.generate()

    elif phenotype == "Random Tangle":
        boundary = ReflectiveBoundary(bounds)
        root = (X//2, Y//2, 0.5) if is_2d else (X//2, Y//2, Z//2)
        
        gen = RandomWalkGenerator(
            start_pos=root, num_steps=int(max_dim * 25), step_length=dynamic_step,
            max_turn_angle=1.0, boundary=boundary
        )
        return gen.generate()

def process_single_sample(idx: int, seed_offset: int, bounds: tuple, output_dir: str):
    seed = seed_offset + idx
    np.random.seed(seed)
    phenotype = np.random.choice(PHENOTYPES)
    
    # 1. Synthesize exact mathematical geometry constrained to dimensions
    segments = _generate_flexible_segments(phenotype, seed, bounds)
    
    # 2. Domain Randomization (Sim2Real mapping)
    is_2d = (bounds[0] == 1)
    
    # 2D STED has no Z-anisotropy. 3D Confocal suffers from elongated PSFs.
    sim_z_aniso = 1.0 if is_2d else np.random.uniform(1.0, 4.0)
    sim_noise = np.random.uniform(0.01, 0.15)
    sim_debris = np.random.randint(5, 45)
    sim_gaps = np.random.uniform(0.0, 0.12)
    
    rasterizer = EmpiricalRasterizer(
        bounds=bounds, 
        base_sigma=1.0,
        z_anisotropy=sim_z_aniso,
        noise_level=sim_noise,
        debris_count=sim_debris,
        gap_prob=sim_gaps
    )
    
    target_gen = TargetFieldGenerator(bounds, max_distance=5.0)
    
    # 3. Render
    volume = rasterizer.render(segments)
    edt_target, vector_target = target_gen.generate(segments)
    
    # 4. Tensor formatting
    volume_tensor = torch.tensor(np.expand_dims(volume, axis=0), dtype=torch.float32)
    edt_tensor = torch.tensor(np.expand_dims(edt_target, axis=0), dtype=torch.float32)
    vec_tensor = torch.tensor(vector_target, dtype=torch.float32)
    
    targets_tensor = torch.cat([edt_tensor, vec_tensor], dim=0)
    
    file_path = os.path.join(output_dir, f"sample_{seed}.pt")
    torch.save({'volume': volume_tensor, 'targets': targets_tensor}, file_path)
    
    return seed

def build_dataset_split(split_name: str, size: int, seed_offset: int, bounds: tuple, base_dir: str, workers: int):
    split_dir = os.path.join(base_dir, split_name)
    os.makedirs(split_dir, exist_ok=True)
    
    print(f"Building '{split_name}' split ({size} samples | Topology: {bounds}) at {split_dir}...")
    
    worker_func = partial(process_single_sample, seed_offset=seed_offset, bounds=bounds, output_dir=split_dir)
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(worker_func, i): i for i in range(size)}
        
        completed = 0
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
                completed += 1
                if completed % max(1, (size // 10)) == 0:
                    print(f"  [{completed}/{size}] Samples processed.")
            except Exception as e:
                print(f"Error generating sample {futures[future]}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Offline Generator for Flexible Synthetic Fiber Datasets")
    parser.add_argument('--output_dir', type=str, required=True, help="Base directory to save the dataset")
    parser.add_argument('--bounds', type=int, nargs='+', default=[64, 64, 64], 
                        help="Spatial resolution: '64 64 64' for 3D, '1 256 256' for 2D STED.")
    parser.add_argument('--train_size', type=int, default=2000, help="Number of training samples")
    parser.add_argument('--val_size', type=int, default=400, help="Number of validation samples")
    parser.add_argument('--test_size', type=int, default=400, help="Number of test samples")
    parser.add_argument('--workers', type=int, default=os.cpu_count(), help="Number of CPU cores to utilize")
    args = parser.parse_args()

    # 1. Dimensionality Standardization
    dims = len(args.bounds)
    if dims == 2:
        args.bounds = [1, args.bounds[0], args.bounds[1]]
    elif dims != 3:
        raise ValueError("Bounds must be either 2 (Y, X) or 3 (Z, Y, X) integers.")
        
    bounds_tuple = tuple(args.bounds)

    # 2. Striding Validation (U-Net Max Pooling constraints)
    for b in bounds_tuple:
        if b > 1 and b % 4 != 0:
            raise ValueError(f"Spatial boundaries > 1 must be divisible by 4. Got {bounds_tuple}")

    os.makedirs(args.output_dir, exist_ok=True)

    # 3. Deterministic Splitting
    build_dataset_split("train", args.train_size, seed_offset=0, 
                        bounds=bounds_tuple, base_dir=args.output_dir, workers=args.workers)
    
    build_dataset_split("val", args.val_size, seed_offset=args.train_size, 
                        bounds=bounds_tuple, base_dir=args.output_dir, workers=args.workers)
    
    build_dataset_split("test", args.test_size, seed_offset=(args.train_size + args.val_size), 
                        bounds=bounds_tuple, base_dir=args.output_dir, workers=args.workers)
    
    print(f"\nDataset fully synthesized at: {args.output_dir}")