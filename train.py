import os
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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
        data = torch.load(file_path, weights_only=True, map_location='cpu')
        vol, targets = data['volume'], data['targets']
        
        if self.dim == 2:
            vol = vol.squeeze(1)
            targets = targets[:, 0, :, :]
            
        return vol, targets

class MaskedVectorLoss(nn.Module):
    def __init__(self, vector_weight: float = 1.0, visibility_weight: float = 0.35, dim: int = 3):
        super().__init__()
        self.mse = nn.MSELoss(reduction='none')
        self.vector_weight = vector_weight
        self.visibility_weight = visibility_weight
        self.dim = dim

    def compute_components(self, pred, target):
        pred_edt, pred_vec = pred[:, 0:1], pred[:, 1:1+self.dim]
        targ_edt, targ_vec = target[:, 0:1], target[:, 1:1+self.dim]

        # 1. Standard EDT Regression
        loss_edt = self.mse(pred_edt, targ_edt).mean()

        # 2. Sign-Agnostic Vector Regression (Symmetric MSE)
        mask = (targ_edt > 0.0).float()
        
        # Calculate squared errors for both orientations, averaged across channels to maintain scale
        err_pos = torch.sum((pred_vec - targ_vec)**2, dim=1, keepdim=True) / self.dim
        err_neg = torch.sum((pred_vec + targ_vec)**2, dim=1, keepdim=True) / self.dim
        
        # Backpropagate strictly through the orientation that yields the lowest error
        loss_vec_raw = torch.min(err_pos, err_neg) * mask
        
        mask_sum = mask.sum() + 1e-8 
        loss_vec = loss_vec_raw.sum() / mask_sum

        components = {
            "edt": loss_edt,
            "vector": loss_vec,
        }

        if self.dim == 2:
            pred_visibility = pred[:, 3:4]
            targ_visibility = target[:, 3:4]
            components["visibility"] = F.binary_cross_entropy_with_logits(pred_visibility, targ_visibility)

        return components

    def forward(self, pred, target):
        components = self.compute_components(pred, target)
        total_loss = components["edt"] + self.vector_weight * components["vector"]
        if self.dim == 2:
            total_loss = total_loss + self.visibility_weight * components["visibility"]
        return total_loss

def train_model(args):
    if args.gpus:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_gpus_available = torch.cuda.device_count()

    batch_size = args.base_batch_size * max(1, num_gpus_available)

    wandb.init(
        project="fibras-cvfunet-final",
        config=vars(args)
    )

    train_dir = os.path.join(args.data_dir, "train")
    val_dir = os.path.join(args.data_dir, "val")
    
    train_ds = PrecomputedFiberDataset(train_dir, dim=args.dim)
    val_ds   = PrecomputedFiberDataset(val_dir, dim=args.dim)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, 
                              num_workers=8, pin_memory=True, prefetch_factor=4, persistent_workers=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, 
                            num_workers=8, pin_memory=True, prefetch_factor=4, persistent_workers=True)

    model = FlexibleCVFUNet(in_channels=1, base_filters=args.base_filters, dim=args.dim)

    if num_gpus_available > 1:
        model = nn.DataParallel(model)
        
    model = model.to(device)
    
    criterion = MaskedVectorLoss(
        vector_weight=args.vector_loss_weight,
        visibility_weight=args.visibility_loss_weight,
        dim=args.dim,
    )
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None

    best_val_loss = float('inf')
    os.makedirs("weights", exist_ok=True)

    print(f"\nStarting Final {args.dim}D Training Loop...")
    for epoch in range(args.epochs):
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
            save_path = f"weights/cvfunet_{args.dim}d_final.pth"
            torch.save(state_dict, save_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Final Training for CVFUNet")
    parser.add_argument('--gpus', type=str, default="0")
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--dim', type=int, choices=[2, 3], required=True)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--base_batch_size', type=int, default=4)
    # Swept Parameters
    parser.add_argument('--base_filters', type=int, required=True)
    parser.add_argument('--learning_rate', type=float, required=True)
    parser.add_argument('--weight_decay', type=float, required=True)
    parser.add_argument('--vector_loss_weight', type=float, required=True)
    parser.add_argument('--visibility_loss_weight', type=float, default=0.35)
    
    args = parser.parse_args()
    train_model(args)
