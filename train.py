import os
import time
import argparse
import sys
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
        target_schema = data.get("target_schema", "legacy")
        
        if self.dim == 2:
            if target_schema != "structural_v2" or targets.shape[0] != 6:
                raise ValueError(
                    "This training path expects 2D samples with target_schema='structural_v2' "
                    "and 6 target channels (centerline, orientation, traceability, radius, bundle count). "
                    "Regenerate the dataset with the upgraded structural target pipeline."
                )
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

class StedFieldLoss2D(nn.Module):
    def __init__(
        self,
        orientation_weight: float = 1.0,
        visibility_weight: float = 0.35,
        orientation_mask_floor: float = 0.15,
        loss_visibility_floor: float = 0.25,
        fixed_orientation_weight: float = 1.0,
        fixed_visibility_weight: float = 0.35,
        skeleton_weight: float = 1.0,
        train_centerline_weight: float = 1.0,
        score_centerline_weight: float | None = None,
        centerline_warmup_epochs: int = 0,
        centerline_warmup_start_factor: float = 1.0,
        radius_weight: float = 0.15,
        bundle_count_weight: float = 0.15,
        centerline_threshold: float = 0.5,
        centerline_focal_weight: float = 1.0,
        centerline_dice_weight: float = 1.0,
        centerline_cldice_weight: float = 0.5,
        stability_margin_weight: float = 0.2,
        score_stability_weight: float = 0.2,
    ):
        super().__init__()
        if orientation_mask_floor < 0.0:
            raise ValueError("orientation_mask_floor must be non-negative.")
        if loss_visibility_floor < 0.0 or loss_visibility_floor > 1.0:
            raise ValueError("loss_visibility_floor must be in the interval [0, 1].")
        if centerline_warmup_epochs < 0:
            raise ValueError("centerline_warmup_epochs must be non-negative.")
        if centerline_warmup_start_factor < 0.0 or centerline_warmup_start_factor > 1.0:
            raise ValueError("centerline_warmup_start_factor must be in the interval [0, 1].")
        if centerline_threshold <= 0.0 or centerline_threshold >= 1.0:
            raise ValueError("centerline_threshold must be in the interval (0, 1).")
        self.orientation_weight = float(orientation_weight)
        self.traceability_weight = float(visibility_weight)
        self.centerline_support_floor = float(orientation_mask_floor)
        self.traceability_floor = float(loss_visibility_floor)
        self.fixed_orientation_weight = float(fixed_orientation_weight)
        self.fixed_traceability_weight = float(fixed_visibility_weight)
        self.score_centerline_weight = float(
            skeleton_weight if score_centerline_weight is None else score_centerline_weight
        )
        self.train_centerline_weight = float(train_centerline_weight)
        self.centerline_warmup_epochs = int(centerline_warmup_epochs)
        self.centerline_warmup_start_factor = float(centerline_warmup_start_factor)
        self._current_train_centerline_weight = self.train_centerline_weight
        self.radius_weight = float(radius_weight)
        self.bundle_count_weight = float(bundle_count_weight)
        self.centerline_threshold = float(centerline_threshold)
        self.centerline_focal_weight = float(centerline_focal_weight)
        self.centerline_dice_weight = float(centerline_dice_weight)
        self.centerline_cldice_weight = float(centerline_cldice_weight)
        self.stability_margin_weight = float(stability_margin_weight)
        self.score_stability_weight = float(score_stability_weight)
        self.centerline_eps = 1e-6
        self.centerline_pos_margin = 0.75
        self.centerline_neg_margin = 0.25
        self.centerline_sensitivity_band = 0.10
        self.cldice_iterations = 8
        self.mse = nn.MSELoss(reduction="none")

    @staticmethod
    def _binary_focal_with_logits(logits, targets, alpha: float = 0.75, gamma: float = 2.0):
        prob = torch.sigmoid(logits)
        ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p_t = prob * targets + (1.0 - prob) * (1.0 - targets)
        alpha_t = alpha * targets + (1.0 - alpha) * (1.0 - targets)
        return (alpha_t * torch.pow(1.0 - p_t, gamma) * ce).mean()

    def _soft_dice_loss(self, prob, target):
        reduce_dims = tuple(range(1, prob.ndim))
        intersection = (prob * target).sum(dim=reduce_dims)
        denom = prob.sum(dim=reduce_dims) + target.sum(dim=reduce_dims)
        dice = (2.0 * intersection + self.centerline_eps) / (denom + self.centerline_eps)
        return (1.0 - dice).mean()

    @staticmethod
    def _soft_erode(img):
        p1 = -F.max_pool2d(-img, kernel_size=(3, 1), stride=1, padding=(1, 0))
        p2 = -F.max_pool2d(-img, kernel_size=(1, 3), stride=1, padding=(0, 1))
        return torch.min(p1, p2)

    def _soft_open(self, img):
        return F.max_pool2d(self._soft_erode(img), kernel_size=3, stride=1, padding=1)

    def _soft_skeletonize(self, img):
        img = torch.clamp(img, 0.0, 1.0)
        opened = self._soft_open(img)
        skeleton = F.relu(img - opened)
        current = img
        for _ in range(self.cldice_iterations - 1):
            current = self._soft_erode(current)
            opened = self._soft_open(current)
            delta = F.relu(current - opened)
            skeleton = skeleton + F.relu(delta - skeleton * delta)
        return torch.clamp(skeleton, 0.0, 1.0)

    def _cldice_loss(self, prob, target):
        skel_prob = self._soft_skeletonize(prob)
        skel_target = self._soft_skeletonize(target)
        reduce_dims = tuple(range(1, prob.ndim))
        topology_precision = (skel_prob * target).sum(dim=reduce_dims) / (skel_prob.sum(dim=reduce_dims) + self.centerline_eps)
        topology_sensitivity = (skel_target * prob).sum(dim=reduce_dims) / (skel_target.sum(dim=reduce_dims) + self.centerline_eps)
        cldice = (2.0 * topology_precision * topology_sensitivity + self.centerline_eps) / (
            topology_precision + topology_sensitivity + self.centerline_eps
        )
        return (1.0 - cldice).mean()

    def _stability_margin_loss(self, prob, target):
        pos_violation = F.relu(self.centerline_pos_margin - prob) * target
        neg_violation = F.relu(prob - self.centerline_neg_margin) * (1.0 - target)
        return (pos_violation + neg_violation).mean()

    def _threshold_sensitivity(self, prob, target):
        target_binary = (target > 0.5).to(prob.dtype)
        thresholds = [
            max(0.05, self.centerline_threshold - self.centerline_sensitivity_band),
            self.centerline_threshold,
            min(0.95, self.centerline_threshold + self.centerline_sensitivity_band),
        ]
        dice_losses = []
        reduce_dims = tuple(range(1, prob.ndim))
        for threshold in thresholds:
            pred_binary = (prob > threshold).to(prob.dtype)
            intersection = (pred_binary * target_binary).sum(dim=reduce_dims)
            denom = pred_binary.sum(dim=reduce_dims) + target_binary.sum(dim=reduce_dims)
            dice = (2.0 * intersection + self.centerline_eps) / (denom + self.centerline_eps)
            dice_losses.append(1.0 - dice)
        stacked = torch.stack(dice_losses, dim=0)
        return (stacked.max(dim=0).values - stacked.min(dim=0).values).mean()

    def _junction_mask(self, target_centerline):
        binary = (target_centerline > 0.5).to(target_centerline.dtype)
        kernel = torch.ones((1, 1, 3, 3), dtype=target_centerline.dtype, device=target_centerline.device)
        neighbors = F.conv2d(binary, kernel, padding=1) - binary
        junction = (neighbors >= 3.0).to(target_centerline.dtype)
        return F.max_pool2d(junction, kernel_size=3, stride=1, padding=1)

    def _compute_train_centerline_weight(self, epoch: int) -> float:
        if self.centerline_warmup_epochs <= 1:
            return self.train_centerline_weight
        clamped_epoch = min(max(int(epoch), 0), self.centerline_warmup_epochs - 1)
        progress = clamped_epoch / max(self.centerline_warmup_epochs - 1, 1)
        factor = self.centerline_warmup_start_factor + (1.0 - self.centerline_warmup_start_factor) * progress
        return self.train_centerline_weight * factor

    def set_epoch(self, epoch: int) -> float:
        self._current_train_centerline_weight = self._compute_train_centerline_weight(epoch)
        return self._current_train_centerline_weight

    def current_train_centerline_weight(self) -> float:
        return self._current_train_centerline_weight

    def compute_components(self, pred, target):
        pred_centerline_logits = pred[:, 0:1]
        pred_orientation = pred[:, 1:3]
        pred_traceability_logits = pred[:, 3:4]
        pred_radius = torch.sigmoid(pred[:, 4:5])
        pred_bundle_count = torch.sigmoid(pred[:, 5:6])

        target_centerline = target[:, 0:1]
        target_orientation = normalize_orientation_torch(target[:, 1:3])
        target_traceability = target[:, 3:4]
        target_radius = target[:, 4:5]
        target_bundle_count = target[:, 5:6]

        centerline_prob = torch.sigmoid(pred_centerline_logits)
        traceability_conf = torch.clamp(target_traceability, self.traceability_floor, 1.0)
        junction_mask = self._junction_mask(target_centerline)

        centerline_focal = self._binary_focal_with_logits(pred_centerline_logits, target_centerline)
        centerline_dice = self._soft_dice_loss(centerline_prob, target_centerline)
        cldice = self._cldice_loss(centerline_prob, target_centerline)
        centerline_loss = (
            self.centerline_focal_weight * centerline_focal
            + self.centerline_dice_weight * centerline_dice
            + self.centerline_cldice_weight * cldice
        )

        pred_orientation = normalize_orientation_torch(pred_orientation)
        orientation_dot = torch.sum(pred_orientation * target_orientation, dim=1, keepdim=True)
        orientation_err = 1.0 - torch.clamp(orientation_dot, -1.0, 1.0)
        centerline_conf = torch.clamp(target_centerline, 0.0, 1.0)
        orientation_mask = (
            (centerline_conf > self.centerline_support_floor).to(pred.dtype)
            * centerline_conf
            * traceability_conf
            * (1.0 - junction_mask)
        )
        orientation_loss = (orientation_err * orientation_mask).sum() / (orientation_mask.sum() + 1e-8)

        traceability_loss = F.binary_cross_entropy_with_logits(pred_traceability_logits, target_traceability)
        radius_mask = (centerline_conf > self.centerline_support_floor).to(pred.dtype)
        radius_err = F.smooth_l1_loss(pred_radius, target_radius, reduction="none")
        radius_loss = (radius_err * radius_mask).sum() / (radius_mask.sum() + 1e-8)
        bundle_count_err = F.smooth_l1_loss(pred_bundle_count, target_bundle_count, reduction="none")
        bundle_count_loss = (bundle_count_err * radius_mask).sum() / (radius_mask.sum() + 1e-8)
        stability_margin = self._stability_margin_loss(centerline_prob, target_centerline)
        threshold_sensitivity = self._threshold_sensitivity(centerline_prob, target_centerline)

        return {
            "centerline": centerline_loss,
            "centerline_focal": centerline_focal,
            "centerline_dice": centerline_dice,
            "cldice": cldice,
            "orientation": orientation_loss,
            "traceability": traceability_loss,
            "radius": radius_loss,
            "bundle_count": bundle_count_loss,
            "stability_margin": stability_margin,
            "threshold_sensitivity": threshold_sensitivity,
        }

    def fixed_score(self, components):
        return (
            self.score_centerline_weight * components["centerline"]
            + self.fixed_orientation_weight * components["orientation"]
            + self.fixed_traceability_weight * components["traceability"]
            + self.radius_weight * components["radius"]
            + self.bundle_count_weight * components["bundle_count"]
            + self.score_stability_weight * components["threshold_sensitivity"]
        )

    def forward(self, pred, target):
        components = self.compute_components(pred, target)
        return (
            self.current_train_centerline_weight() * components["centerline"]
            + self.orientation_weight * components["orientation"]
            + self.traceability_weight * components["traceability"]
            + self.radius_weight * components["radius"]
            + self.bundle_count_weight * components["bundle_count"]
            + self.stability_margin_weight * components["stability_margin"]
        )


def _empty_metric_totals():
    return {
        "loss": 0.0,
        "score": 0.0,
        "centerline": 0.0,
        "centerline_focal": 0.0,
        "centerline_dice": 0.0,
        "cldice": 0.0,
        "orientation": 0.0,
        "traceability": 0.0,
        "radius": 0.0,
        "bundle_count": 0.0,
        "stability_margin": 0.0,
        "threshold_sensitivity": 0.0,
    }


def _add_metric_batch(totals, criterion, loss, components):
    totals["loss"] += float(loss.detach().cpu())
    totals["score"] += float(criterion.fixed_score(components).detach().cpu())
    for key in (
        "centerline",
        "centerline_focal",
        "centerline_dice",
        "cldice",
        "orientation",
        "traceability",
        "radius",
        "bundle_count",
        "stability_margin",
        "threshold_sensitivity",
    ):
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

    if args.multi_gpu:
        if args.nccl_p2p_disable:
            os.environ["NCCL_P2P_DISABLE"] = "1"
        if args.nccl_ib_disable:
            os.environ["NCCL_IB_DISABLE"] = "1"
        if args.nccl_debug:
            os.environ["NCCL_DEBUG"] = args.nccl_debug

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_gpus_available = torch.cuda.device_count()
    use_data_parallel = bool(args.multi_gpu and num_gpus_available > 1)
    batch_size = args.base_batch_size * max(1, num_gpus_available) if use_data_parallel else args.base_batch_size

    if device.type == "cuda" and num_gpus_available > 1 and not use_data_parallel:
        print(
            f"Detected {num_gpus_available} visible GPUs, but --multi_gpu is disabled. "
            "Training on a single GPU to avoid NCCL/DataParallel issues."
        )

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

    if use_data_parallel:
        model = nn.DataParallel(model)
        
    model = model.to(device)
    
    criterion = StedFieldLoss2D(
        orientation_weight=args.orientation_loss_weight,
        visibility_weight=args.visibility_loss_weight,
        orientation_mask_floor=args.orientation_mask_floor,
        loss_visibility_floor=args.loss_visibility_floor,
        score_centerline_weight=args.score_centerline_weight,
        train_centerline_weight=args.train_centerline_weight,
        centerline_warmup_epochs=args.centerline_warmup_epochs,
        centerline_warmup_start_factor=args.centerline_warmup_start_factor,
        radius_weight=args.radius_loss_weight,
        bundle_count_weight=args.bundle_count_loss_weight,
        centerline_threshold=args.centerline_threshold,
        score_stability_weight=args.score_stability_weight,
        stability_margin_weight=args.stability_margin_weight,
    )
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None

    best_val_score = float('inf')
    save_path = args.save_path
    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    print("\nStarting 2D STED ResUNet Training Loop...")
    for epoch in range(args.epochs):
        active_train_centerline_weight = criterion.set_epoch(epoch)
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
                    if use_data_parallel: loss = loss.mean()
                    components = criterion.compute_components(outputs, targets)
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                if use_data_parallel: loss = loss.mean()
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
                        if use_data_parallel: loss = loss.mean()
                        components = criterion.compute_components(outputs, targets)
                else:
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    if use_data_parallel: loss = loss.mean()
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
            "val_centerline": val_metrics["centerline"],
            "val_centerline_focal": val_metrics["centerline_focal"],
            "val_centerline_dice": val_metrics["centerline_dice"],
            "val_cldice": val_metrics["cldice"],
            "val_orientation": val_metrics["orientation"],
            "val_traceability": val_metrics["traceability"],
            "val_radius": val_metrics["radius"],
            "val_bundle_count": val_metrics["bundle_count"],
            "val_threshold_sensitivity": val_metrics["threshold_sensitivity"],
            "train_centerline_weight": active_train_centerline_weight,
            "score_centerline_weight": criterion.score_centerline_weight,
            "score_stability_weight": criterion.score_stability_weight,
            "epoch_time_seconds": t_elapsed,
        }
        if wandb_run is not None:
            wandb.log(log_data)
        print(
            f"-> Epoch {epoch+1} Summary: Train: {train_metrics['loss']:.4f} | "
            f"Val Score: {val_metrics['score']:.4f} | Val Loss: {val_metrics['loss']:.4f} | "
            f"Centerline W: {active_train_centerline_weight:.4f} | "
            f"Bundle: {val_metrics['bundle_count']:.4f} | "
            f"Sens: {val_metrics['threshold_sensitivity']:.4f} | Time: {t_elapsed:.1f}s"
        )

        if val_metrics["score"] < best_val_score:
            best_val_score = val_metrics["score"]
            state_dict = model.module.state_dict() if use_data_parallel else model.state_dict()
            torch.save(state_dict, save_path)

def add_train_arguments(parser: argparse.ArgumentParser) -> None:
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
    parser.add_argument('--visibility_loss_weight', '--traceability_loss_weight', dest='visibility_loss_weight', type=float, default=0.35)
    parser.add_argument('--orientation_mask_floor', '--centerline_support_floor', dest='orientation_mask_floor', type=float, default=0.15)
    parser.add_argument('--loss_visibility_floor', type=float, default=0.25)
    parser.add_argument('--radius_loss_weight', type=float, default=0.15)
    parser.add_argument('--bundle_count_loss_weight', type=float, default=0.15)
    parser.add_argument('--train_centerline_weight', type=float, default=1.0)
    parser.add_argument('--score_centerline_weight', '--skeleton_score_weight', dest='score_centerline_weight', type=float, default=1.0)
    parser.add_argument('--centerline_warmup_epochs', type=int, default=0)
    parser.add_argument('--centerline_warmup_start_factor', type=float, default=1.0)
    parser.add_argument('--centerline_threshold', type=float, default=0.5)
    parser.add_argument('--stability_margin_weight', type=float, default=0.2)
    parser.add_argument('--score_stability_weight', type=float, default=0.2)
    parser.add_argument('--save_path', type=str, default='weights/sted_resunet2d_final.pth')
    parser.add_argument('--multi_gpu', action='store_true', help="Enable nn.DataParallel across all visible GPUs.")
    parser.add_argument('--nccl_p2p_disable', action='store_true', help="Set NCCL_P2P_DISABLE=1 when using --multi_gpu.")
    parser.add_argument('--nccl_ib_disable', action='store_true', help="Set NCCL_IB_DISABLE=1 when using --multi_gpu.")
    parser.add_argument('--nccl_debug', type=str, choices=['INFO', 'WARN'], default="", help="Set NCCL_DEBUG when using --multi_gpu.")
    parser.add_argument('--no_wandb', action='store_true')


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="2D STED ResUNet training and evaluation.")
    subparsers = parser.add_subparsers(dest="mode")

    fit_parser = subparsers.add_parser("fit", help="Train STEDResUNet2D")
    add_train_arguments(fit_parser)

    from src.evaluation import add_evaluate_arguments

    eval_parser = subparsers.add_parser("evaluate", help="Evaluate STEDResUNet2D on the test split")
    add_evaluate_arguments(eval_parser)
    return parser


def main(argv=None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    parser = build_parser()

    # Preserve the old training invocation style: python train.py --data_dir ...
    if argv and argv[0] not in {"fit", "evaluate", "-h", "--help"}:
        argv = ["fit", *argv]

    args = parser.parse_args(argv)
    if args.mode == "fit":
        train_model(args)
    elif args.mode == "evaluate":
        from src.evaluation import evaluate_model

        evaluate_model(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
