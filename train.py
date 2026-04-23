import os
import time
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import wandb

from src.model import STEDResUNet2D
from src.sted import normalize_orientation_torch

class PrecomputedFiberDataset(Dataset):
    def __init__(self, data_dir: str, dim: int = 2, crop_size: int = 0, random_crop: bool = False):
        self.data_dir = data_dir
        self.dim = dim
        self.crop_size = int(crop_size or 0)
        self.random_crop = bool(random_crop)
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
            vol, targets = self._crop_2d(vol, targets)
            
        return vol, targets

    def _crop_2d(self, vol, targets):
        if self.crop_size <= 0:
            return vol, targets

        h, w = vol.shape[-2:]
        crop_h = min(self.crop_size, h)
        crop_w = min(self.crop_size, w)
        if crop_h == h and crop_w == w:
            return vol, targets

        if self.random_crop:
            y0 = int(torch.randint(0, h - crop_h + 1, (1,)).item())
            x0 = int(torch.randint(0, w - crop_w + 1, (1,)).item())
        else:
            y0 = (h - crop_h) // 2
            x0 = (w - crop_w) // 2

        return vol[:, y0:y0 + crop_h, x0:x0 + crop_w], targets[:, y0:y0 + crop_h, x0:x0 + crop_w]

class MaskedVectorLoss(nn.Module):
    def __init__(
        self,
        vector_weight: float = 1.0,
        visibility_weight: float = 0.35,
        dim: int = 3,
        vector_mask_floor: float = 0.05,
        loss_visibility_floor: float = 0.25,
    ):
        super().__init__()
        self.mse = nn.MSELoss(reduction='none')
        self.vector_weight = vector_weight
        self.visibility_weight = visibility_weight
        self.dim = dim
        if vector_mask_floor < 0.0:
            raise ValueError("vector_mask_floor must be non-negative.")
        if loss_visibility_floor < 0.0 or loss_visibility_floor > 1.0:
            raise ValueError("loss_visibility_floor must be in the interval [0, 1].")
        self.vector_mask_floor = vector_mask_floor
        self.loss_visibility_floor = loss_visibility_floor

    def compute_components(self, pred, target):
        pred_edt, pred_vec = pred[:, 0:1], pred[:, 1:1+self.dim]
        targ_edt, targ_vec = target[:, 0:1], target[:, 1:1+self.dim]
        vis_conf = None

        if self.dim == 2:
            targ_visibility = target[:, 3:4]
            vis_conf = torch.clamp(targ_visibility, self.loss_visibility_floor, 1.0)

        # 1. EDT Regression (visibility-weighted in 2D STED)
        edt_err = self.mse(pred_edt, targ_edt)
        if vis_conf is None:
            loss_edt = edt_err.mean()
        else:
            vis_sum = vis_conf.sum() + 1e-8
            loss_edt = (edt_err * vis_conf).sum() / vis_sum

        # 2. Sign-Agnostic Vector Regression (Symmetric MSE)
        mask_conf = torch.clamp(targ_edt, 0.0, 1.0)
        mask = (mask_conf > self.vector_mask_floor).float() * mask_conf
        if vis_conf is not None:
            mask = mask * vis_conf
        
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
            components["visibility"] = F.binary_cross_entropy_with_logits(pred_visibility, targ_visibility)

        return components

    def forward(self, pred, target):
        components = self.compute_components(pred, target)
        total_loss = components["edt"] + self.vector_weight * components["vector"]
        if self.dim == 2:
            total_loss = total_loss + self.visibility_weight * components["visibility"]
        return total_loss


class StedFieldLoss2D(nn.Module):
    def __init__(
        self,
        orientation_weight: float = 1.0,
        visibility_weight: float = 0.35,
        orientation_mask_floor: float = 0.05,
        loss_visibility_floor: float = 0.25,
        fixed_orientation_weight: float = 1.0,
        fixed_visibility_weight: float = 0.35,
        skeleton_weight: float = 0.25,
    ):
        super().__init__()
        if orientation_mask_floor < 0.0:
            raise ValueError("orientation_mask_floor must be non-negative.")
        if loss_visibility_floor < 0.0 or loss_visibility_floor > 1.0:
            raise ValueError("loss_visibility_floor must be in the interval [0, 1].")
        self.orientation_weight = float(orientation_weight)
        self.visibility_weight = float(visibility_weight)
        self.orientation_mask_floor = float(orientation_mask_floor)
        self.loss_visibility_floor = float(loss_visibility_floor)
        self.fixed_orientation_weight = float(fixed_orientation_weight)
        self.fixed_visibility_weight = float(fixed_visibility_weight)
        self.skeleton_weight = float(skeleton_weight)
        self.mse = nn.MSELoss(reduction="none")

    def compute_components(self, pred, target):
        pred_edt = pred[:, 0:1]
        pred_orientation = pred[:, 1:3]
        pred_visibility = pred[:, 3:4]

        target_edt = target[:, 0:1]
        target_orientation = normalize_orientation_torch(target[:, 1:3])
        target_visibility = target[:, 3:4]
        visibility_conf = torch.clamp(target_visibility, self.loss_visibility_floor, 1.0)

        edt_err = self.mse(pred_edt, target_edt)
        edt_loss = (edt_err * visibility_conf).sum() / (visibility_conf.sum() + 1e-8)

        pred_orientation = normalize_orientation_torch(pred_orientation)
        orientation_dot = torch.sum(pred_orientation * target_orientation, dim=1, keepdim=True)
        orientation_err = 1.0 - torch.clamp(orientation_dot, -1.0, 1.0)
        edt_conf = torch.clamp(target_edt, 0.0, 1.0)
        orientation_mask = (edt_conf > self.orientation_mask_floor).to(pred.dtype) * edt_conf * visibility_conf
        orientation_loss = (orientation_err * orientation_mask).sum() / (orientation_mask.sum() + 1e-8)

        visibility_loss = F.binary_cross_entropy_with_logits(pred_visibility, target_visibility)
        pred_centerline = (torch.clamp(pred_edt, 0.0, 1.0) > 0.85).to(pred.dtype)
        target_centerline = (target_edt > 0.85).to(pred.dtype)
        skeleton_proxy = torch.abs(pred_centerline.mean(dim=(-2, -1)) - target_centerline.mean(dim=(-2, -1))).mean()

        return {
            "edt": edt_loss,
            "orientation": orientation_loss,
            "visibility": visibility_loss,
            "skeleton_proxy": skeleton_proxy,
        }

    def fixed_score(self, components):
        return (
            components["edt"]
            + self.fixed_orientation_weight * components["orientation"]
            + self.fixed_visibility_weight * components["visibility"]
            + self.skeleton_weight * components["skeleton_proxy"]
        )

    def forward(self, pred, target):
        components = self.compute_components(pred, target)
        return (
            components["edt"]
            + self.orientation_weight * components["orientation"]
            + self.visibility_weight * components["visibility"]
        )


def _empty_metric_totals():
    return {
        "loss": 0.0,
        "score": 0.0,
        "edt": 0.0,
        "orientation": 0.0,
        "visibility": 0.0,
        "skeleton_proxy": 0.0,
    }


def _add_metric_batch(totals, criterion, loss, components):
    totals["loss"] += float(loss.detach().cpu())
    totals["score"] += float(criterion.fixed_score(components).detach().cpu())
    for key in ("edt", "orientation", "visibility", "skeleton_proxy"):
        totals[key] += float(components[key].detach().cpu())


def _average_metrics(totals, n_batches):
    return {key: value / max(n_batches, 1) for key, value in totals.items()}


def _data_loader_kwargs(num_workers: int):
    kwargs = {
        "num_workers": num_workers,
        "pin_memory": True,
    }
    if num_workers > 0:
        kwargs["prefetch_factor"] = 4
        kwargs["persistent_workers"] = True
    return kwargs

def train_model(args):
    if args.dim != 2:
        raise ValueError("The upgraded STED training path is 2D only. Use --dim 2.")

    if args.gpus:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_gpus_available = torch.cuda.device_count()

    batch_size = args.base_batch_size * max(1, num_gpus_available)

    wandb_run = None
    if not args.no_wandb:
        wandb_run = wandb.init(
            project="fibras-sted-resunet2d",
            config=vars(args)
        )

    train_dir = os.path.join(args.data_dir, "train")
    val_dir = os.path.join(args.data_dir, "val")
    
    train_ds = PrecomputedFiberDataset(train_dir, dim=2, crop_size=args.crop_size, random_crop=True)
    val_ds = PrecomputedFiberDataset(val_dir, dim=2, crop_size=args.val_crop_size or args.crop_size, random_crop=False)

    loader_kwargs = _data_loader_kwargs(args.num_workers)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, **loader_kwargs)

    model = STEDResUNet2D(in_channels=1, base_filters=args.base_filters)

    if num_gpus_available > 1:
        model = nn.DataParallel(model)
        
    model = model.to(device)
    
    criterion = StedFieldLoss2D(
        orientation_weight=args.orientation_loss_weight,
        visibility_weight=args.visibility_loss_weight,
        orientation_mask_floor=args.orientation_mask_floor,
        loss_visibility_floor=args.loss_visibility_floor,
        skeleton_weight=args.skeleton_score_weight,
    )
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None

    best_val_score = float('inf')
    os.makedirs("weights", exist_ok=True)

    print("\nStarting 2D STED ResUNet Training Loop...")
    for epoch in range(args.epochs):
        model.train()
        train_totals = _empty_metric_totals()
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
                    components = criterion.compute_components(outputs, targets)
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                if num_gpus_available > 1: loss = loss.mean()
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
                        if num_gpus_available > 1: loss = loss.mean()
                        components = criterion.compute_components(outputs, targets)
                else:
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    if num_gpus_available > 1: loss = loss.mean()
                    components = criterion.compute_components(outputs, targets)
                
                _add_metric_batch(val_totals, criterion, loss, components)

        val_metrics = _average_metrics(val_totals, len(val_loader))
        t_elapsed = time.time() - t0
        
        log_data = {
            "epoch": epoch + 1,
            "train_loss": train_metrics["loss"],
            "train_score": train_metrics["score"],
            "val_loss": val_metrics["loss"],
            "val_score": val_metrics["score"],
            "val_edt": val_metrics["edt"],
            "val_orientation": val_metrics["orientation"],
            "val_visibility": val_metrics["visibility"],
            "val_skeleton_proxy": val_metrics["skeleton_proxy"],
            "epoch_time_seconds": t_elapsed,
        }
        if wandb_run is not None:
            wandb.log(log_data)
        print(
            f"-> Epoch {epoch+1} Summary: Train: {train_metrics['loss']:.4f} | "
            f"Val Score: {val_metrics['score']:.4f} | Val Loss: {val_metrics['loss']:.4f} | "
            f"Time: {t_elapsed:.1f}s"
        )

        if val_metrics["score"] < best_val_score:
            best_val_score = val_metrics["score"]
            state_dict = model.module.state_dict() if num_gpus_available > 1 else model.state_dict()
            save_path = "weights/sted_resunet2d_final.pth"
            torch.save(state_dict, save_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Final Training for 2D STED ResUNet")
    parser.add_argument('--gpus', type=str, default="0")
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--dim', type=int, choices=[2], default=2)
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--base_batch_size', type=int, default=4)
    parser.add_argument('--crop_size', type=int, default=512)
    parser.add_argument('--val_crop_size', type=int, default=0)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--base_filters', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--orientation_loss_weight', type=float, default=1.0)
    parser.add_argument('--visibility_loss_weight', type=float, default=0.35)
    parser.add_argument('--orientation_mask_floor', type=float, default=0.05)
    parser.add_argument('--loss_visibility_floor', type=float, default=0.25)
    parser.add_argument('--skeleton_score_weight', type=float, default=0.25)
    parser.add_argument('--no_wandb', action='store_true')
    
    args = parser.parse_args()
    train_model(args)
