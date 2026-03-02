import os
import argparse
import numpy as np
import torch
import concurrent.futures
from functools import partial

from src.synthesis import SpaceColonizationGenerator, RandomWalkGenerator
from src.core import ReflectiveBoundary
from src.rasterization import NDimRasterizer
from src.targets import TargetFieldGenerator

PHENOTYPES = ["Highly Branched", "Directional", "Random Tangle"]

def _generate_segments(phenotype: str, seed: int, N: int):
    """Core synthesis logic decoupled for multiprocessing."""
    np.random.seed(seed)
    grid_bounds = (N, N, N)
    
    dynamic_step = max(1.0, N * 0.012)
    dynamic_kill = max(2.0, N * 0.015)
    
    if phenotype == "Highly Branched":
        attractors = np.random.uniform(N * 0.1, N * 0.9, size=(int(N**3 * 0.0015), 3))
        gen = SpaceColonizationGenerator(
            attractors, root_pos=(N//2, N//2, N//2), step_length=dynamic_step,
            attraction_distance=max(20.0, N * 0.25), kill_distance=dynamic_kill, 
            bounds=grid_bounds, max_iterations=int(N * 75), thickness_decay=0.99
        )
        return gen.generate()

    elif phenotype == "Directional":
        num_attractors = int(N**2 * 0.15)
        cyl_radius = max(10, N * 0.12)
        attractors = np.zeros((num_attractors, 3))
        attractors[:, 0] = np.random.uniform(N//2 - cyl_radius, N//2 + cyl_radius, num_attractors)
        attractors[:, 1] = np.random.uniform(N//2 - cyl_radius, N//2 + cyl_radius, num_attractors)
        attractors[:, 2] = np.random.uniform(N * 0.2, N * 0.9, num_attractors)
        gen = SpaceColonizationGenerator(
            attractors, root_pos=(N//2, N//2, N*0.1), step_length=dynamic_step,
            attraction_distance=max(30.0, N * 0.35), kill_distance=dynamic_kill, 
            bounds=grid_bounds, max_iterations=int(N * 75), thickness_decay=0.99
        )
        return gen.generate()

    elif phenotype == "Random Tangle":
        boundary = ReflectiveBoundary(grid_bounds)
        gen = RandomWalkGenerator(
            start_pos=(N//2, N//2, N//2), num_steps=int(N * 25), step_length=dynamic_step,
            max_turn_angle=1.0, boundary=boundary
        )
        return gen.generate()

def process_single_sample(idx: int, seed_offset: int, grid_size: int, output_dir: str):
    """Generates, rasterizes, and saves a single paired volume/target to disk."""
    seed = seed_offset + idx
    np.random.seed(seed)
    phenotype = np.random.choice(PHENOTYPES)
    
    # 1. Synthesize explicit geometry
    segments = _generate_segments(phenotype, seed, grid_size)
    
    # 2. Rasterize Empirical Density & Mathematical Targets
    rasterizer = NDimRasterizer((grid_size, grid_size, grid_size), base_sigma=1.0)
    target_gen = TargetFieldGenerator((grid_size, grid_size, grid_size), max_distance=5.0)
    
    volume = rasterizer.render(segments)
    edt_target, vector_target = target_gen.generate(segments)
    
    # 3. Format as standard C-H-W-D PyTorch tensors
    volume_tensor = torch.tensor(np.expand_dims(volume, axis=0), dtype=torch.float32)
    edt_tensor = torch.tensor(np.expand_dims(edt_target, axis=0), dtype=torch.float32)
    vec_tensor = torch.tensor(vector_target, dtype=torch.float32)
    
    targets_tensor = torch.cat([edt_tensor, vec_tensor], dim=0)
    
    # 4. Serialize to disk
    file_path = os.path.join(output_dir, f"sample_{seed}.pt")
    torch.save({'volume': volume_tensor, 'targets': targets_tensor}, file_path)
    
    return seed

def build_dataset_split(split_name: str, size: int, seed_offset: int, grid_size: int, base_dir: str, workers: int):
    split_dir = os.path.join(base_dir, split_name)
    os.makedirs(split_dir, exist_ok=True)
    
    print(f"Building '{split_name}' split ({size} samples) at {split_dir}...")
    
    worker_func = partial(process_single_sample, seed_offset=seed_offset, grid_size=grid_size, output_dir=split_dir)
    
    # Execute across isolated processes to bypass Python's GIL
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
    parser = argparse.ArgumentParser(description="Offline Generator for Synthetic Fiber Datasets")
    parser.add_argument('--output_dir', type=str, required=True, help="Base directory to save the dataset")
    parser.add_argument('--grid_size', type=int, default=64, help="Spatial resolution of the 3D grid (e.g., 64 for 64x64x64)")
    parser.add_argument('--train_size', type=int, default=2000, help="Number of training samples")
    parser.add_argument('--val_size', type=int, default=400, help="Number of validation samples")
    parser.add_argument('--test_size', type=int, default=400, help="Number of test samples")
    parser.add_argument('--workers', type=int, default=os.cpu_count(), help="Number of CPU cores to utilize")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Strictly disjoint seed offsets to prevent data leakage between sets
    build_dataset_split("train", args.train_size, seed_offset=0, 
                        grid_size=args.grid_size, base_dir=args.output_dir, workers=args.workers)
    
    build_dataset_split("val", args.val_size, seed_offset=args.train_size, 
                        grid_size=args.grid_size, base_dir=args.output_dir, workers=args.workers)
    
    build_dataset_split("test", args.test_size, seed_offset=(args.train_size + args.val_size), 
                        grid_size=args.grid_size, base_dir=args.output_dir, workers=args.workers)
    
    print(f"\nDataset fully synthesized at: {args.output_dir}")