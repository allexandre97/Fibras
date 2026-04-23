import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import wandb

from src.model import STEDResUNet2D
from train import PrecomputedFiberDataset, StedFieldLoss2D, _add_metric_batch, _average_metrics, _empty_metric_totals

def train_sweep():
    wandb.init()
    config = wandb.config

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_gpus_available = torch.cuda.device_count()

    base_batch_size_per_gpu = 2
    batch_size = base_batch_size_per_gpu * max(1, num_gpus_available)
    epochs = 60

    train_dir = os.path.join(GLOBAL_DATA_DIR, "train")
    val_dir = os.path.join(GLOBAL_DATA_DIR, "val")
    
    train_ds = PrecomputedFiberDataset(train_dir, dim=2, crop_size=GLOBAL_CROP_SIZE, random_crop=True)
    val_ds = PrecomputedFiberDataset(val_dir, dim=2, crop_size=GLOBAL_CROP_SIZE, random_crop=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, 
                              num_workers=8, pin_memory=True, prefetch_factor=4, persistent_workers=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, 
                            num_workers=8, pin_memory=True, prefetch_factor=4, persistent_workers=True)

    model = STEDResUNet2D(in_channels=1, base_filters=config.base_filters)

    if num_gpus_available > 1:
        model = nn.DataParallel(model)
        
    model = model.to(device)
    
    criterion = StedFieldLoss2D(
        orientation_weight=config.orientation_loss_weight,
        visibility_weight=getattr(config, "visibility_loss_weight", 0.35),
        orientation_mask_floor=getattr(config, "orientation_mask_floor", 0.05),
        loss_visibility_floor=getattr(config, "loss_visibility_floor", 0.25),
        skeleton_weight=getattr(config, "skeleton_score_weight", 0.25),
    )
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None

    # --- Early Stopping Tracking Variables ---
    best_val_score = float('inf')
    patience = 5
    patience_counter = 0
    min_epochs_before_stop = 10 

    for epoch in range(epochs):
        model.train()
        train_totals = _empty_metric_totals()
        
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
                    components = criterion.compute_components(outputs, targets)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                if num_gpus_available > 1:
                    loss = loss.mean()
                components = criterion.compute_components(outputs, targets)
                loss.backward()
                optimizer.step()
            
            _add_metric_batch(train_totals, criterion, loss, components)

        train_metrics = _average_metrics(train_totals, len(train_loader))
        
        model.eval()
        val_totals = _empty_metric_totals()
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
                        components = criterion.compute_components(outputs, targets)
                else:
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    if num_gpus_available > 1:
                        loss = loss.mean()
                    components = criterion.compute_components(outputs, targets)
                
                _add_metric_batch(val_totals, criterion, loss, components)

        val_metrics = _average_metrics(val_totals, len(val_loader))
        
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_metrics["loss"],
            "train_score": train_metrics["score"],
            "val_loss": val_metrics["loss"],
            "val_score": val_metrics["score"],
            "val_edt": val_metrics["edt"],
            "val_orientation": val_metrics["orientation"],
            "val_visibility": val_metrics["visibility"],
            "val_skeleton_proxy": val_metrics["skeleton_proxy"],
        })

        # --- Intra-Run Early Stopping Logic ---
        if val_metrics["score"] < best_val_score:
            best_val_score = val_metrics["score"]
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch >= min_epochs_before_stop and patience_counter >= patience:
            print(f"Intra-run early stopping triggered at epoch {epoch+1}. No improvement for {patience} epochs.")
            break

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="W&B Sweep for 2D STED ResUNet")
    parser.add_argument('--gpus', type=str, default="0")
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--dim', type=int, choices=[2], default=2)
    parser.add_argument('--sweep_count', type=int, default=20)
    parser.add_argument('--crop_size', type=int, default=512)
    args = parser.parse_args()
    
    if args.gpus:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        
    global GLOBAL_DATA_DIR
    GLOBAL_DATA_DIR = args.data_dir
    
    global GLOBAL_CROP_SIZE
    GLOBAL_CROP_SIZE = args.crop_size

    sweep_config = {
        'method': 'bayes',
        'metric': {
            'name': 'val_score',
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
            'orientation_loss_weight': {
                'distribution': 'uniform',
                'min': 0.5,
                'max': 3.0
            },
            'visibility_loss_weight': {
                'distribution': 'uniform',
                'min': 0.10,
                'max': 0.70
            },
            'orientation_mask_floor': {
                'distribution': 'uniform',
                'min': 0.00,
                'max': 0.20
            },
            'loss_visibility_floor': {
                'distribution': 'uniform',
                'min': 0.05,
                'max': 0.45
            },
            'skeleton_score_weight': {
                'values': [0.1, 0.25, 0.5]
            },
            'base_filters': {
                'values': [24, 32, 48]
            }
        }
    }

    project_name = "fibras-sted-resunet2d-sweep"
    sweep_id = wandb.sweep(sweep_config, project=project_name)
    
    wandb.agent(sweep_id, train_sweep, count=args.sweep_count)
