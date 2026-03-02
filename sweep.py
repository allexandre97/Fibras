import os
import time
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import wandb

from src.model import CVFUNet
from train import PrecomputedFiberDataset, MaskedVectorLoss

def train_sweep():
    # wandb.init() without arguments expects the sweep agent to populate wandb.config
    wandb.init()
    config = wandb.config

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_gpus_available = torch.cuda.device_count()

    base_batch_size_per_gpu = 4
    batch_size = base_batch_size_per_gpu * max(1, num_gpus_available)
    epochs = 40  # Reduced max epochs for sweep efficiency

    # Data loaders
    train_dir = os.path.join(GLOBAL_DATA_DIR, "train")
    val_dir = os.path.join(GLOBAL_DATA_DIR, "val")
    
    train_ds = PrecomputedFiberDataset(train_dir)
    val_ds   = PrecomputedFiberDataset(val_dir)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, 
                              num_workers=8, pin_memory=True, prefetch_factor=4, persistent_workers=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, 
                            num_workers=8, pin_memory=True, prefetch_factor=4, persistent_workers=True)

    # Initialize model dynamically based on sweep configuration
    model = CVFUNet(in_channels=1, base_filters=config.base_filters)

    if num_gpus_available > 1:
        model = nn.DataParallel(model)
        
    model = model.to(device)
    
    criterion = MaskedVectorLoss(vector_weight=config.vector_loss_weight)
    
    # Optimizer utilizing swept learning rate and weight decay
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None

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
        
        # Validation
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
        
        # Log strictly to wandb for the sweep agent to evaluate
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss
        })

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="W&B Sweep for CVFUNet")
    parser.add_argument('--gpus', type=str, default="0")
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--sweep_count', type=int, default=20, help="Number of models to train")
    args = parser.parse_args()
    
    if args.gpus:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        
    # Expose data directory to the parameter-less sweep function
    global GLOBAL_DATA_DIR
    GLOBAL_DATA_DIR = args.data_dir

    # Define the Bayesian optimization search space
    sweep_config = {
        'method': 'bayes',  # Bayesian optimization calculates probability models to find optimal params
        'metric': {
            'name': 'val_loss',
            'goal': 'minimize'   
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
            'base_filters': {
                'values': [8, 16, 32]
            }
        }
    }

    # Initialize the sweep on the W&B servers
    sweep_id = wandb.sweep(sweep_config, project="fibras-cvfunet-sweep")
    
    # Execute the agent locally
    wandb.agent(sweep_id, train_sweep, count=args.sweep_count)