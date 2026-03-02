import os
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import wandb  # <-- W&B Import

from src.synthesis import SpaceColonizationGenerator, RandomWalkGenerator
from src.core import ReflectiveBoundary
from src.rasterization import NDimRasterizer
from src.targets import TargetFieldGenerator
from src.model import CVFUNet

class SyntheticFiberDataset(Dataset):
    def __init__(self, size: int, grid_size: int, seed_offset: int, mode: str = 'train'):
        self.size = size
        self.grid_size = grid_size
        self.seed_offset = seed_offset
        self.phenotypes = ["Highly Branched", "Directional", "Random Tangle"]
        self.rasterizer = NDimRasterizer((grid_size, grid_size, grid_size), base_sigma=1.0)
        self.target_gen = TargetFieldGenerator((grid_size, grid_size, grid_size), max_distance=5.0)

    def __len__(self):
        return self.size

    def _generate_segments(self, phenotype: str, seed: int, N: int):
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

    def __getitem__(self, idx):
        seed = self.seed_offset + idx
        np.random.seed(seed)
        phenotype = np.random.choice(self.phenotypes)
        
        segments = self._generate_segments(phenotype, seed, self.grid_size)
        
        volume = self.rasterizer.render(segments)
        edt_target, vector_target = self.target_gen.generate(segments)
        
        volume = np.expand_dims(volume, axis=0)
        edt_target = np.expand_dims(edt_target, axis=0)
        targets = np.concatenate([edt_target, vector_target], axis=0)
        
        return torch.tensor(volume, dtype=torch.float32), torch.tensor(targets, dtype=torch.float32)

class MaskedVectorLoss(nn.Module):
    def __init__(self, vector_weight: float = 1.0):
        super().__init__()
        self.mse = nn.MSELoss(reduction='none')
        self.vector_weight = vector_weight

    def forward(self, pred, target):
        pred_edt, pred_vec = pred[:, 0:1], pred[:, 1:4]
        targ_edt, targ_vec = target[:, 0:1], target[:, 1:4]

        loss_edt = self.mse(pred_edt, targ_edt).mean()

        mask = (targ_edt > 0.0).float()
        loss_vec_raw = self.mse(pred_vec, targ_vec) * mask
        
        mask_sum = mask.sum() + 1e-8 
        loss_vec = loss_vec_raw.sum() / mask_sum

        return loss_edt + self.vector_weight * loss_vec

def train_model(gpus: str):
    if gpus:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpus
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_gpus_available = torch.cuda.device_count()
    
    print(f"Executing on: {device}")
    if device.type == 'cuda':
        print(f"Allocated GPUs: {num_gpus_available} (Logical IDs: {list(range(num_gpus_available))})")

    grid_size = 64 
    base_batch_size_per_gpu = 4
    batch_size = base_batch_size_per_gpu * max(1, num_gpus_available)
    epochs = 50
    lr = 1e-4
    vector_loss_weight = 5.0

    # <-- W&B Initialization -->
    wandb.init(
        project="fibras-cvfunet",
        config={
            "grid_size": grid_size,
            "global_batch_size": batch_size,
            "base_batch_size_per_gpu": base_batch_size_per_gpu,
            "epochs": epochs,
            "learning_rate": lr,
            "vector_loss_weight": vector_loss_weight,
            "num_gpus": num_gpus_available
        }
    )

    train_ds = SyntheticFiberDataset(size=2000, grid_size=grid_size, seed_offset=0, mode='train')
    val_ds   = SyntheticFiberDataset(size=400, grid_size=grid_size, seed_offset=2000, mode='val')

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4)

    model = CVFUNet(in_channels=1, base_filters=16)

    if num_gpus_available > 1:
        model = nn.DataParallel(model)
        
    model = model.to(device)
    
    criterion = MaskedVectorLoss(vector_weight=vector_loss_weight)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    
    # Watch model gradients and parameters with wandb
    wandb.watch(model, criterion, log="all", log_freq=20)
    
    best_val_loss = float('inf')
    os.makedirs("weights", exist_ok=True)

    print("\nStarting Training Loop...")
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        t0 = time.time()
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            if num_gpus_available > 1:
                loss = loss.mean()
                
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            if batch_idx % max(1, (len(train_loader)//5)) == 0:
                print(f"Epoch [{epoch+1}/{epochs}] Batch [{batch_idx}/{len(train_loader)}] Loss: {loss.item():.4f}")

        train_loss /= len(train_loader)
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                if num_gpus_available > 1:
                    loss = loss.mean()
                    
                val_loss += loss.item()
                
        val_loss /= len(val_loader)
        t_elapsed = time.time() - t0
        
        # <-- W&B Metric Logging -->
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "epoch_time_seconds": t_elapsed
        })
        
        print(f"-> Epoch {epoch+1} Summary: Train: {train_loss:.4f} | Val: {val_loss:.4f} | Time: {t_elapsed:.1f}s")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            state_dict = model.module.state_dict() if num_gpus_available > 1 else model.state_dict()
            save_path = f"weights/cvfunet_best.pth"
            torch.save(state_dict, save_path)
            
            # <-- W&B Model Artifact Saving -->
            wandb.save(save_path)
            print("   [Model Checkpoint Saved & Synced to W&B]")

    wandb.finish()
    print("Training Complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CVFUNet on Synthetic Fibers")
    parser.add_argument('--gpus', type=str, default="0", 
                        help='Comma-separated string of GPU IDs (e.g., "0,1" or "1,3"). Pass an empty string "" to run on CPU.')
    args = parser.parse_args()
    
    train_model(args.gpus)