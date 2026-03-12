import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import wandb

from src.model import FlexibleCVFUNet
from train import PrecomputedFiberDataset, MaskedVectorLoss

def train_sweep():
    wandb.init()
    config = wandb.config

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_gpus_available = torch.cuda.device_count()

    base_batch_size_per_gpu = 4
    batch_size = base_batch_size_per_gpu * max(1, num_gpus_available)
    epochs = 40  

    train_dir = os.path.join(GLOBAL_DATA_DIR, "train")
    val_dir = os.path.join(GLOBAL_DATA_DIR, "val")
    
    train_ds = PrecomputedFiberDataset(train_dir, dim=GLOBAL_DIM)
    val_ds   = PrecomputedFiberDataset(val_dir, dim=GLOBAL_DIM)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, 
                              num_workers=8, pin_memory=True, prefetch_factor=4, persistent_workers=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, 
                            num_workers=8, pin_memory=True, prefetch_factor=4, persistent_workers=True)

    model = FlexibleCVFUNet(in_channels=1, base_filters=config.base_filters, dim=GLOBAL_DIM)

    if num_gpus_available > 1:
        model = nn.DataParallel(model)
        
    model = model.to(device)
    
    criterion = MaskedVectorLoss(
        vector_weight=config.vector_loss_weight,
        visibility_weight=getattr(config, "visibility_loss_weight", 0.35),
        dim=GLOBAL_DIM,
        vector_mask_floor=getattr(config, "vector_mask_floor", 0.05),
        loss_visibility_floor=getattr(config, "loss_visibility_floor", 0.25),
    )
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None

    # --- Early Stopping Tracking Variables ---
    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0
    min_epochs_before_stop = 10 

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        for inputs, targets in train_loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)
            
            if scaler:
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    if num_gpus_available > 1:
                        loss = loss.mean()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                if num_gpus_available > 1:
                    loss = loss.mean()
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
                        if num_gpus_available > 1:
                            loss = loss.mean()
                else:
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    if num_gpus_available > 1:
                        loss = loss.mean()
                        
                val_loss += loss.item()
                
        val_loss /= len(val_loader)
        
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss
        })

        # --- Intra-Run Early Stopping Logic ---
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch >= min_epochs_before_stop and patience_counter >= patience:
            print(f"Intra-run early stopping triggered at epoch {epoch+1}. No improvement for {patience} epochs.")
            break

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="W&B Sweep for FlexibleCVFUNet")
    parser.add_argument('--gpus', type=str, default="0")
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--dim', type=int, choices=[2, 3], required=True)
    parser.add_argument('--sweep_count', type=int, default=20)
    args = parser.parse_args()
    
    if args.gpus:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        
    global GLOBAL_DATA_DIR
    GLOBAL_DATA_DIR = args.data_dir
    
    global GLOBAL_DIM
    GLOBAL_DIM = args.dim

    sweep_config = {
        'method': 'bayes',
        'metric': {
            'name': 'val_loss',
            'goal': 'minimize'   
        },
        # --- Inter-Run Early Stopping (Hyperband) ---
        'early_terminate': {
            'type': 'hyperband',
            'min_iter': 10,   # First evaluation at epoch 10
            'eta': 2          # Bracket halving rate
        },
        'parameters': {
            'learning_rate': {
                'distribution': 'log_uniform_values',
                'min': 1e-5,
                'max': 1e-3
            },
            'weight_decay': {
                'distribution': 'log_uniform_values',
                'min': 1e-6,
                'max': 1e-2
            },
            'vector_loss_weight': {
                'distribution': 'uniform',
                'min': 1.0,
                'max': 10.0
            },
            'visibility_loss_weight': {
                'distribution': 'uniform',
                'min': 0.10,
                'max': 0.70
            },
            'vector_mask_floor': {
                'distribution': 'uniform',
                'min': 0.00,
                'max': 0.20
            },
            'loss_visibility_floor': {
                'distribution': 'uniform',
                'min': 0.05,
                'max': 0.45
            },
            'base_filters': {
                'values': [8, 16, 32]
            }
        }
    }

    project_name = f"fibras-cvfunet-{args.dim}d-sweep"
    sweep_id = wandb.sweep(sweep_config, project=project_name)
    
    wandb.agent(sweep_id, train_sweep, count=args.sweep_count)
