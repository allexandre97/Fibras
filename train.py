import os
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import wandb

from src.model import FlexibleCVFUNet

class PrecomputedFiberDataset(Dataset):
    def __init__(self, data_dir: str, dim: int):
        self.data_dir = data_dir
        self.dim = dim
        self.files = [f for f in os.listdir(data_dir) if f.endswith('.pt')]
        if len(self.files) == 0:
            raise FileNotFoundError(f"No .pt files found in {data_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = os.path.join(self.data_dir, self.files[idx])
        # Force CPU mapping during IO to avoid cuda context locks
        data = torch.load(file_path, weights_only=True, map_location='cpu')
        vol, targets = data['volume'], data['targets']
        
        if self.dim == 2:
            # Volume shape is (1, 1, Y, X). Squeeze to (1, Y, X).
            vol = vol.squeeze(1)
            # Targets shape is (4, 1, Y, X). Extract EDT, Vx, and Vy channels only.
            targets = targets[[0, 1, 2], 0, :, :]
            
        return vol, targets

class MaskedVectorLoss(nn.Module):
    def __init__(self, vector_weight: float = 1.0, dim: int = 3):
        super().__init__()
        self.mse = nn.MSELoss(reduction='none')
        self.vector_weight = vector_weight
        self.dim = dim

    def forward(self, pred, target):
        pred_edt, pred_vec = pred[:, 0:1], pred[:, 1:1+self.dim]
        targ_edt, targ_vec = target[:, 0:1], target[:, 1:1+self.dim]

        loss_edt = self.mse(pred_edt, targ_edt).mean()

        mask = (targ_edt > 0.0).float()
        loss_vec_raw = self.mse(pred_vec, targ_vec) * mask
        
        mask_sum = mask.sum() + 1e-8 
        loss_vec = loss_vec_raw.sum() / mask_sum

        return loss_edt + self.vector_weight * loss_vec

def train_model(gpus: str, data_dir: str, dim: int):
    if gpus:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpus
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_gpus_available = torch.cuda.device_count()

    base_batch_size_per_gpu = 4
    batch_size = base_batch_size_per_gpu * max(1, num_gpus_available)
    epochs = 50
    lr = 1e-4
    vector_loss_weight = 5.0

    wandb.init(
        project="fibras-cvfunet",
        config={
            "dimensionality": dim,
            "global_batch_size": batch_size,
            "base_batch_size_per_gpu": base_batch_size_per_gpu,
            "epochs": epochs,
            "learning_rate": lr,
            "vector_loss_weight": vector_loss_weight,
            "num_gpus": num_gpus_available,
            "data_dir": data_dir
        }
    )

    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")
    
    train_ds = PrecomputedFiberDataset(train_dir, dim=dim)
    val_ds   = PrecomputedFiberDataset(val_dir, dim=dim)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, 
                              num_workers=8, pin_memory=True, prefetch_factor=4, persistent_workers=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, 
                            num_workers=8, pin_memory=True, prefetch_factor=4, persistent_workers=True)

    # Initialize requested dimensionality
    model = FlexibleCVFUNet(in_channels=1, base_filters=16, dim=dim)

    if num_gpus_available > 1:
        model = nn.DataParallel(model)
        
    model = model.to(device)
    
    criterion = MaskedVectorLoss(vector_weight=vector_loss_weight, dim=dim)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None

    wandb.watch(model, criterion, log="all", log_freq=20)
    best_val_loss = float('inf')
    os.makedirs("weights", exist_ok=True)

    print(f"\nStarting {dim}D Training Loop...")
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        t0 = time.time()
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)
            
            if scaler:
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    if num_gpus_available > 1: loss = loss.mean()
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                if num_gpus_available > 1: loss = loss.mean()
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
                inputs = inputs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                
                if scaler:
                    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                        outputs = model(inputs)
                        loss = criterion(outputs, targets)
                        if num_gpus_available > 1: loss = loss.mean()
                else:
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    if num_gpus_available > 1: loss = loss.mean()
                        
                val_loss += loss.item()
                
        val_loss /= len(val_loader)
        t_elapsed = time.time() - t0
        
        wandb.log({"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss, "epoch_time_seconds": t_elapsed})
        print(f"-> Epoch {epoch+1} Summary: Train: {train_loss:.4f} | Val: {val_loss:.4f} | Time: {t_elapsed:.1f}s")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            state_dict = model.module.state_dict() if num_gpus_available > 1 else model.state_dict()
            save_path = f"weights/cvfunet_{dim}d_best.pth"
            torch.save(state_dict, save_path)
            wandb.save(save_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CVFUNet")
    parser.add_argument('--gpus', type=str, default="0")
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--dim', type=int, choices=[2, 3], required=True, 
                        help="Specify 2 for STED models or 3 for Confocal models")
    args = parser.parse_args()
    
    train_model(args.gpus, args.data_dir, args.dim)