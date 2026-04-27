import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import wandb

from src.model import STEDResUNet2D
from train import PrecomputedFiberDataset, StedFieldLoss2D, _add_metric_batch, _average_metrics, _empty_metric_totals


GLOBAL_MULTI_GPU = False

def train_sweep():
    wandb.init()
    config = wandb.config

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_gpus_available = torch.cuda.device_count()

    base_batch_size_per_gpu = 4
    use_data_parallel = bool(GLOBAL_MULTI_GPU and num_gpus_available > 1)
    batch_size = base_batch_size_per_gpu * max(1, num_gpus_available) if use_data_parallel else base_batch_size_per_gpu
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

    if use_data_parallel:
        model = nn.DataParallel(model)
        
    model = model.to(device)
    
    criterion = StedFieldLoss2D(
        orientation_weight=config.orientation_loss_weight,
        visibility_weight=getattr(config, "visibility_loss_weight", 0.35),
        orientation_mask_floor=getattr(config, "orientation_mask_floor", 0.05),
        loss_visibility_floor=getattr(config, "loss_visibility_floor", 0.25),
        score_centerline_weight=getattr(
            config,
            "score_centerline_weight",
            getattr(config, "skeleton_score_weight", 0.25),
        ),
        train_centerline_weight=getattr(config, "train_centerline_weight", 1.0),
        centerline_warmup_epochs=getattr(config, "centerline_warmup_epochs", 1),
        centerline_warmup_start_factor=getattr(config, "centerline_warmup_start_factor", 0.5),
        radius_weight=getattr(config, "radius_loss_weight", 0.15),
        centerline_threshold=getattr(config, "centerline_threshold", 0.5),
        score_stability_weight=getattr(config, "score_stability_weight", 0.2),
        stability_margin_weight=getattr(config, "stability_margin_weight", 0.2),
    )
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None

    # --- Early Stopping Tracking Variables ---
    best_val_score = float('inf')
    patience = 6
    patience_counter = 0
    min_epochs_before_stop = 15

    for epoch in range(epochs):
        active_train_centerline_weight = criterion.set_epoch(epoch)
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
                    if use_data_parallel:
                        loss = loss.mean()
                    components = criterion.compute_components(outputs, targets)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                if use_data_parallel:
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
                        if use_data_parallel:
                            loss = loss.mean()
                        components = criterion.compute_components(outputs, targets)
                else:
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    if use_data_parallel:
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
            "val_centerline": val_metrics["centerline"],
            "val_centerline_focal": val_metrics["centerline_focal"],
            "val_centerline_dice": val_metrics["centerline_dice"],
            "val_cldice": val_metrics["cldice"],
            "val_orientation": val_metrics["orientation"],
            "val_traceability": val_metrics["traceability"],
            "val_radius": val_metrics["radius"],
            "val_threshold_sensitivity": val_metrics["threshold_sensitivity"],
            "train_centerline_weight": active_train_centerline_weight,
            "score_centerline_weight": criterion.score_centerline_weight,
            "score_stability_weight": criterion.score_stability_weight,
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
    parser.add_argument('--project', type=str, default="fibras-sted-resunet2d-sweep-v2")
    parser.add_argument('--multi_gpu', action='store_true', help="Enable nn.DataParallel across all visible GPUs.")
    parser.add_argument('--nccl_p2p_disable', action='store_true', help="Set NCCL_P2P_DISABLE=1 when using --multi_gpu.")
    parser.add_argument('--nccl_ib_disable', action='store_true', help="Set NCCL_IB_DISABLE=1 when using --multi_gpu.")
    parser.add_argument('--nccl_debug', type=str, choices=['INFO', 'WARN'], default="", help="Set NCCL_DEBUG when using --multi_gpu.")
    args = parser.parse_args()
    
    if args.gpus:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    if args.multi_gpu:
        if args.nccl_p2p_disable:
            os.environ["NCCL_P2P_DISABLE"] = "1"
        if args.nccl_ib_disable:
            os.environ["NCCL_IB_DISABLE"] = "1"
        if args.nccl_debug:
            os.environ["NCCL_DEBUG"] = args.nccl_debug
        
    global GLOBAL_DATA_DIR
    GLOBAL_DATA_DIR = args.data_dir
    
    global GLOBAL_CROP_SIZE
    GLOBAL_CROP_SIZE = args.crop_size

    GLOBAL_MULTI_GPU = bool(args.multi_gpu)

    sweep_config = {
        'method': 'bayes',
        'metric': {
            'name': 'val_score',
            'goal': 'minimize'
        },
        # The centerline term now exists in both the train loss and the selection
        # score. Sweep the train weight, but keep the score weight fixed so runs
        # are compared under one consistent model-selection criterion.
        'early_terminate': {
            'type': 'hyperband',
            'min_iter': 15,
            'eta': 2
        },
        'parameters': {
            'learning_rate': {
                'distribution': 'log_uniform_values',
                'min': 2e-5,
                'max': 1e-4
            },
            'weight_decay': {
                'distribution': 'log_uniform_values',
                'min': 1e-6,
                'max': 5e-4
            },
            'orientation_loss_weight': {
                'distribution': 'uniform',
                'min': 0.5,
                'max': 1.4
            },
            'visibility_loss_weight': {
                'distribution': 'uniform',
                'min': 0.25,
                'max': 0.55
            },
            'orientation_mask_floor': {
                'distribution': 'uniform',
                'min': 0.10,
                'max': 0.25
            },
            'loss_visibility_floor': {
                'distribution': 'uniform',
                'min': 0.05,
                'max': 0.35
            },
            'radius_loss_weight': {
                'distribution': 'uniform',
                'min': 0.05,
                'max': 0.30
            },
            'train_centerline_weight': {
                'distribution': 'uniform',
                'min': 0.8,
                'max': 1.4
            },
            'score_centerline_weight': {
                'value': 1.0
            },
            'centerline_warmup_epochs': {
                'value': 0
            },
            'centerline_warmup_start_factor': {
                'value': 1.0
            },
            'centerline_threshold': {
                'value': 0.5
            },
            'stability_margin_weight': {
                'value': 0.2
            },
            'score_stability_weight': {
                'value': 0.2
            },
            'base_filters': {
                'values': [32, 40, 48]
            }
        }
    }

    sweep_id = wandb.sweep(sweep_config, project=args.project)
    
    wandb.agent(sweep_id, train_sweep, count=args.sweep_count)
