import os
import time
import argparse
import sys
import random
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import wandb

from src.model import DEFAULT_ASPP_DILATIONS, LEGACY_ASPP_DILATIONS, STEDResUNet2D, normalize_aspp_dilations
from src.sted import normalize_orientation_torch


def _torch_load(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Load a .pt record while staying compatible with older PyTorch versions."""
    try:
        return torch.load(path, weights_only=True, map_location="cpu")
    except TypeError:
        return torch.load(path, map_location="cpu")


def format_aspp_dilations(aspp_dilations) -> str:
    return ",".join(str(value) for value in normalize_aspp_dilations(aspp_dilations))


def parse_aspp_dilations(value) -> tuple[int, ...]:
    if value is None:
        return DEFAULT_ASPP_DILATIONS
    if isinstance(value, str):
        parts = [part.strip() for part in value.split(",")]
        if not parts or any(part == "" for part in parts):
            raise ValueError("ASPP dilations must be a comma-separated list of positive integers.")
        try:
            return normalize_aspp_dilations(tuple(int(part) for part in parts))
        except ValueError as error:
            raise ValueError("ASPP dilations must be a comma-separated list of positive integers.") from error
    return normalize_aspp_dilations(value)


def checkpoint_aspp_dilations(checkpoint, override=None) -> tuple[int, ...]:
    if override not in (None, ""):
        return parse_aspp_dilations(override)
    if isinstance(checkpoint, dict):
        config = checkpoint.get("config")
        if isinstance(config, dict) and "aspp_dilations" in config:
            return parse_aspp_dilations(config["aspp_dilations"])
    return LEGACY_ASPP_DILATIONS


class PrecomputedFiberDataset(Dataset):
    """Strict structural_v2 2D STED dataset used by standalone and sweep training."""

    def __init__(
        self,
        data_dir: str | os.PathLike[str],
        dim: int = 2,
        crop_size: int = 0,
        random_crop: bool = False,
        augment_geometric: bool = False,
        augment_intensity: bool = False,
    ):
        if dim != 2:
            raise ValueError("The upgraded STED dataset path is 2D only. Use dim=2.")
        self.data_dir = Path(data_dir)
        self.dim = dim
        self.crop_size = int(crop_size or 0)
        self.random_crop = bool(random_crop)
        self.augment_geometric = bool(augment_geometric)
        self.augment_intensity = bool(augment_intensity)
        self.files = sorted(self.data_dir.glob("*.pt"))
        if not self.files:
            raise FileNotFoundError(f"No .pt files found in {self.data_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        data = _torch_load(path)
        if "volume" not in data or "targets" not in data:
            raise KeyError(f"{path} is missing required keys 'volume' and/or 'targets'.")

        vol = data["volume"].float()
        targets = data["targets"].float()
        target_schema = data.get("target_schema", "legacy")

        if target_schema != "structural_v2":
            raise ValueError(
                f"{path} has target_schema={target_schema!r}. Bundle-count training requires "
                "target_schema='structural_v2'. Regenerate the dataset with the upgraded generator."
            )

        if targets.ndim == 4 and targets.shape[1] == 1:
            targets = targets[:, 0, :, :]
        elif targets.ndim != 3:
            raise ValueError(f"{path} has unsupported target tensor shape {tuple(targets.shape)}.")

        if targets.shape[0] != 6:
            raise ValueError(
                f"{path} has {targets.shape[0]} target channels. Expected 6 channels: "
                "centerline, cos(2theta), sin(2theta), traceability, radius, bundle_count."
            )

        if vol.ndim == 4 and vol.shape[1] == 1:
            vol = vol.squeeze(1)
        elif vol.ndim == 2:
            vol = vol.unsqueeze(0)
        elif vol.ndim != 3:
            raise ValueError(f"{path} has unsupported volume tensor shape {tuple(vol.shape)}.")

        vol, targets = self._crop_2d(vol, targets)

        if self.augment_geometric:
            vol, targets = self._augment_geometric(vol, targets)
        if self.augment_intensity:
            vol = self._augment_intensity(vol)

        return vol.contiguous(), targets.contiguous()

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

    @staticmethod
    def _augment_geometric(vol: torch.Tensor, targets: torch.Tensor):
        # Orientation is encoded as cos(2theta), sin(2theta). Horizontal and
        # vertical flips preserve cos(2theta) and negate sin(2theta). Odd 90 deg
        # rotations negate both channels.
        if torch.rand(()) < 0.5:
            vol = torch.flip(vol, dims=(-1,))
            targets = torch.flip(targets, dims=(-1,))
            targets[2] = -targets[2]

        if torch.rand(()) < 0.5:
            vol = torch.flip(vol, dims=(-2,))
            targets = torch.flip(targets, dims=(-2,))
            targets[2] = -targets[2]

        k = int(torch.randint(0, 4, (1,)).item())
        if k:
            vol = torch.rot90(vol, k=k, dims=(-2, -1))
            targets = torch.rot90(targets, k=k, dims=(-2, -1))
            if k % 2 == 1:
                targets[1:3] = -targets[1:3]

        return vol, targets

    @staticmethod
    def _augment_intensity(vol: torch.Tensor):
        scale = 0.90 + 0.20 * torch.rand((), dtype=vol.dtype)
        offset = -0.04 + 0.08 * torch.rand((), dtype=vol.dtype)
        noise_std = 0.00 + 0.015 * torch.rand((), dtype=vol.dtype)
        vol = vol * scale + offset
        if noise_std > 0:
            vol = vol + torch.randn_like(vol) * noise_std
        return torch.clamp(vol, 0.0, 1.0)

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
        fixed_score_radius_weight: float = 0.25,
        fixed_score_bundle_count_weight: float = 0.30,
        fixed_score_threshold_sensitivity_weight: float = 0.20,
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
        self.fixed_score_radius_weight = float(fixed_score_radius_weight)
        self.fixed_score_bundle_count_weight = float(fixed_score_bundle_count_weight)
        self.fixed_score_threshold_sensitivity_weight = float(fixed_score_threshold_sensitivity_weight)
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
            + self.fixed_score_radius_weight * components["radius"]
            + self.fixed_score_bundle_count_weight * components["bundle_count"]
            + self.fixed_score_threshold_sensitivity_weight * components["threshold_sensitivity"]
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


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _loader_kwargs(num_workers: int, device: torch.device):
    kwargs = {
        "num_workers": int(num_workers),
        "pin_memory": device.type == "cuda",
    }
    if num_workers > 0:
        kwargs["prefetch_factor"] = 4
        kwargs["persistent_workers"] = True
    return kwargs


def _data_loader_kwargs(num_workers: int):
    return _loader_kwargs(num_workers, torch.device("cuda" if torch.cuda.is_available() else "cpu"))


def _amp_settings(device: torch.device, amp_dtype: str):
    if device.type != "cuda" or amp_dtype == "off":
        return None, None
    if amp_dtype == "bf16":
        return torch.bfloat16, None
    if amp_dtype == "fp16":
        return torch.float16, torch.amp.GradScaler("cuda")
    raise ValueError(f"Unsupported amp dtype: {amp_dtype}")


def _autocast_context(device: torch.device, dtype):
    if device.type == "cuda" and dtype is not None:
        return torch.amp.autocast("cuda", dtype=dtype)
    return nullcontext()


def _validate_dataset(data_dir: str, check_samples: int) -> None:
    for split in ("train", "val"):
        split_dir = Path(data_dir) / split
        files = sorted(split_dir.glob("*.pt"))
        if not files:
            raise FileNotFoundError(f"No .pt files found in {split_dir}")

        n_check = min(int(check_samples), len(files))
        for path in files[:n_check]:
            record = _torch_load(path)
            schema = record.get("target_schema", "legacy")
            if schema != "structural_v2":
                raise ValueError(f"{path} has target_schema={schema!r}; expected 'structural_v2'.")
            targets = record.get("targets")
            if targets is None:
                raise KeyError(f"{path} is missing the 'targets' tensor.")
            if targets.shape[0] != 6:
                raise ValueError(f"{path} has target shape {tuple(targets.shape)}; expected 6 target channels.")


def _make_criterion(config) -> StedFieldLoss2D:
    return StedFieldLoss2D(
        orientation_weight=float(config.orientation_loss_weight),
        visibility_weight=float(config.visibility_loss_weight),
        orientation_mask_floor=float(config.orientation_mask_floor),
        loss_visibility_floor=float(config.loss_visibility_floor),
        fixed_orientation_weight=float(config.fixed_score_orientation_weight),
        fixed_visibility_weight=float(config.fixed_score_traceability_weight),
        score_centerline_weight=float(config.score_centerline_weight),
        train_centerline_weight=float(config.train_centerline_weight),
        centerline_warmup_epochs=int(config.centerline_warmup_epochs),
        centerline_warmup_start_factor=float(config.centerline_warmup_start_factor),
        radius_weight=float(config.radius_loss_weight),
        bundle_count_weight=float(config.bundle_count_loss_weight),
        centerline_threshold=float(config.centerline_threshold),
        score_stability_weight=float(config.score_stability_weight),
        stability_margin_weight=float(config.stability_margin_weight),
        fixed_score_radius_weight=float(config.fixed_score_radius_weight),
        fixed_score_bundle_count_weight=float(config.fixed_score_bundle_count_weight),
        fixed_score_threshold_sensitivity_weight=float(config.fixed_score_threshold_sensitivity_weight),
    )


def _state_dict(model: nn.Module, use_data_parallel: bool):
    return model.module.state_dict() if use_data_parallel else model.state_dict()


def _checkpoint_payload(
    model: nn.Module,
    use_data_parallel: bool,
    epoch: int,
    best_val_score: float,
    config: dict[str, Any],
):
    return {
        "model_state_dict": _state_dict(model, use_data_parallel),
        "epoch": int(epoch),
        "best_val_score": float(best_val_score),
        "config": dict(config),
    }


def extract_model_state_dict(checkpoint):
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        checkpoint = checkpoint["model_state_dict"]
    if not isinstance(checkpoint, dict):
        raise TypeError("Checkpoint must be a state_dict or contain a 'model_state_dict' entry.")
    return {k[7:] if k.startswith("module.") else k: v for k, v in checkpoint.items()}


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

    _set_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_gpus_available = torch.cuda.device_count()
    use_data_parallel = bool(args.multi_gpu and num_gpus_available > 1)
    batch_size = args.base_batch_size * max(1, num_gpus_available) if use_data_parallel else args.base_batch_size

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    if device.type == "cuda" and num_gpus_available > 1 and not use_data_parallel:
        print(
            f"Detected {num_gpus_available} visible GPUs, but --multi_gpu is disabled. "
            "Training on a single GPU to avoid NCCL/DataParallel issues."
        )

    aspp_dilations = parse_aspp_dilations(args.aspp_dilations)
    aspp_dilations_config = format_aspp_dilations(aspp_dilations)
    _validate_dataset(args.data_dir, args.check_samples)

    static_config = {
        "data_dir": args.data_dir,
        "crop_size": args.crop_size,
        "val_crop_size": args.val_crop_size or args.crop_size,
        "epochs": args.epochs,
        "base_batch_size": args.base_batch_size,
        "effective_batch_size": batch_size,
        "num_workers": args.num_workers,
        "amp_dtype": args.amp_dtype,
        "grad_clip_norm": args.grad_clip_norm,
        "multi_gpu": args.multi_gpu,
        "augment_geometric": args.augment_geometric,
        "augment_intensity": args.augment_intensity,
        "seed": args.seed,
        "aspp_dilations": aspp_dilations_config,
    }

    wandb_run = None
    if not args.no_wandb:
        wandb_run = wandb.init(
            project=args.project,
            config=vars(args)
        )
        wandb.config.update(static_config, allow_val_change=True)

    train_dir = os.path.join(args.data_dir, "train")
    val_dir = os.path.join(args.data_dir, "val")

    train_ds = PrecomputedFiberDataset(
        train_dir,
        dim=2,
        crop_size=args.crop_size,
        random_crop=True,
        augment_geometric=args.augment_geometric,
        augment_intensity=args.augment_intensity,
    )
    val_ds = PrecomputedFiberDataset(
        val_dir,
        dim=2,
        crop_size=args.val_crop_size or args.crop_size,
        random_crop=False,
        augment_geometric=False,
        augment_intensity=False,
    )

    loader_kwargs = _loader_kwargs(args.num_workers, device)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, **loader_kwargs)

    model = STEDResUNet2D(in_channels=1, base_filters=args.base_filters, aspp_dilations=aspp_dilations)

    if use_data_parallel:
        model = nn.DataParallel(model)
        
    model = model.to(device)
    
    criterion = _make_criterion(args)
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    autocast_dtype, scaler = _amp_settings(device, args.amp_dtype)

    best_val_score = float('inf')
    best_epoch = 0
    patience_counter = 0
    patience = int(args.early_stop_patience)
    min_epochs_before_stop = int(args.min_epochs_before_stop)
    save_path = args.save_path
    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    print(
        "\nStarting 2D STED ResUNet Training Loop... "
        f"device={device}, batch_size={batch_size}, train={len(train_ds)}, val={len(val_ds)}, "
        f"base_filters={int(args.base_filters)}, aspp_dilations={aspp_dilations_config}"
    )
    for epoch in range(args.epochs):
        active_train_centerline_weight = criterion.set_epoch(epoch)
        model.train()
        train_totals = _empty_metric_totals()
        t0 = time.time()
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)

            with _autocast_context(device, autocast_dtype):
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                if use_data_parallel:
                    loss = loss.mean()
                components = criterion.compute_components(outputs, targets)

            if scaler is not None:
                scaler.scale(loss).backward()
                if args.grad_clip_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if args.grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
                optimizer.step()
            
            _add_metric_batch(train_totals, criterion, loss, components)

        train_metrics = _average_metrics(train_totals, len(train_loader))
        
        model.eval()
        val_totals = _empty_metric_totals()
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)

                with _autocast_context(device, autocast_dtype):
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    if use_data_parallel:
                        loss = loss.mean()
                    components = criterion.compute_components(outputs, targets)
                
                _add_metric_batch(val_totals, criterion, loss, components)

        val_metrics = _average_metrics(val_totals, len(val_loader))
        t_elapsed = time.time() - t0

        improved = val_metrics["score"] < best_val_score
        if improved:
            best_val_score = val_metrics["score"]
            best_epoch = epoch + 1
            patience_counter = 0
            torch.save(
                _checkpoint_payload(
                    model,
                    use_data_parallel,
                    epoch + 1,
                    best_val_score,
                    {**vars(args), **static_config},
                ),
                save_path,
            )
        else:
            patience_counter += 1

        log_data = {
            "epoch": epoch + 1,
            "epoch_time_seconds": t_elapsed,
            "train_loss": train_metrics["loss"],
            "train_score": train_metrics["score"],
            "train_centerline": train_metrics["centerline"],
            "train_centerline_focal": train_metrics["centerline_focal"],
            "train_centerline_dice": train_metrics["centerline_dice"],
            "train_cldice": train_metrics["cldice"],
            "train_orientation": train_metrics["orientation"],
            "train_traceability": train_metrics["traceability"],
            "train_radius": train_metrics["radius"],
            "train_bundle_count": train_metrics["bundle_count"],
            "train_threshold_sensitivity": train_metrics["threshold_sensitivity"],
            "val_loss": val_metrics["loss"],
            "val_score": val_metrics["score"],
            "best_val_score": best_val_score,
            "best_epoch": best_epoch,
            "val_centerline": val_metrics["centerline"],
            "val_centerline_focal": val_metrics["centerline_focal"],
            "val_centerline_dice": val_metrics["centerline_dice"],
            "val_cldice": val_metrics["cldice"],
            "val_orientation": val_metrics["orientation"],
            "val_traceability": val_metrics["traceability"],
            "val_radius": val_metrics["radius"],
            "val_bundle_count": val_metrics["bundle_count"],
            "val_threshold_sensitivity": val_metrics["threshold_sensitivity"],
            "active_train_centerline_weight": active_train_centerline_weight,
            "train_radius_loss_weight": criterion.radius_weight,
            "train_bundle_count_loss_weight": criterion.bundle_count_weight,
            "score_centerline_weight": criterion.score_centerline_weight,
            "score_orientation_weight": criterion.fixed_orientation_weight,
            "score_traceability_weight": criterion.fixed_traceability_weight,
            "score_radius_weight": criterion.fixed_score_radius_weight,
            "score_bundle_count_weight": criterion.fixed_score_bundle_count_weight,
            "score_threshold_sensitivity_weight": criterion.fixed_score_threshold_sensitivity_weight,
        }
        if wandb_run is not None:
            wandb.log(log_data)
        print(
            f"Epoch {epoch + 1:03d}/{args.epochs} | "
            f"val_score={val_metrics['score']:.5f} | best={best_val_score:.5f} @ {best_epoch} | "
            f"val_bundle={val_metrics['bundle_count']:.5f} | val_radius={val_metrics['radius']:.5f} | "
            f"time={t_elapsed:.1f}s"
        )

        if patience > 0 and epoch + 1 >= min_epochs_before_stop and patience_counter >= patience:
            print(
                f"Early stopping at epoch {epoch + 1}: no best_val_score improvement "
                f"for {patience} epochs after minimum {min_epochs_before_stop} epochs."
            )
            break

def add_train_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('--gpus', type=str, default="0")
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--dim', type=int, choices=[2], default=2)
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--base_batch_size', type=int, default=4)
    parser.add_argument('--crop_size', type=int, default=512)
    parser.add_argument('--val_crop_size', type=int, default=0)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--project', type=str, default="fibras-sted-resunet2d")
    parser.add_argument('--base_filters', type=int, default=32)
    parser.add_argument('--aspp_dilations', type=str, default=format_aspp_dilations(DEFAULT_ASPP_DILATIONS))
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--amp_dtype', type=str, choices=['bf16', 'fp16', 'off'], default='bf16')
    parser.add_argument('--grad_clip_norm', type=float, default=1.0)
    parser.add_argument('--check_samples', type=int, default=16)
    parser.add_argument('--augment_geometric', action='store_true')
    parser.add_argument('--augment_intensity', action='store_true')
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--orientation_loss_weight', type=float, default=1.0)
    parser.add_argument('--visibility_loss_weight', '--traceability_loss_weight', dest='visibility_loss_weight', type=float, default=0.35)
    parser.add_argument('--orientation_mask_floor', '--centerline_support_floor', dest='orientation_mask_floor', type=float, default=0.15)
    parser.add_argument('--loss_visibility_floor', type=float, default=0.25)
    parser.add_argument('--radius_loss_weight', type=float, default=0.15)
    parser.add_argument('--bundle_count_loss_weight', type=float, default=0.15)
    parser.add_argument('--train_centerline_weight', type=float, default=1.0)
    parser.add_argument('--score_centerline_weight', '--skeleton_score_weight', dest='score_centerline_weight', type=float, default=1.0)
    parser.add_argument('--fixed_score_orientation_weight', type=float, default=1.0)
    parser.add_argument('--fixed_score_traceability_weight', type=float, default=0.35)
    parser.add_argument('--fixed_score_radius_weight', type=float, default=0.25)
    parser.add_argument('--fixed_score_bundle_count_weight', type=float, default=0.30)
    parser.add_argument('--fixed_score_threshold_sensitivity_weight', type=float, default=0.20)
    parser.add_argument('--centerline_warmup_epochs', type=int, default=3)
    parser.add_argument('--centerline_warmup_start_factor', type=float, default=0.5)
    parser.add_argument('--centerline_threshold', type=float, default=0.5)
    parser.add_argument('--stability_margin_weight', type=float, default=0.2)
    parser.add_argument('--score_stability_weight', type=float, default=0.2)
    parser.add_argument('--early_stop_patience', type=int, default=10)
    parser.add_argument('--min_epochs_before_stop', type=int, default=25)
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
