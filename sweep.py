"""W&B hyperparameter sweep for the 2D STED ResUNet.

This sweep script is intended for the upgraded synthetic STED datasets generated as
``target_schema='structural_v2'``.  In particular, it refuses to train on old
5-channel structural_v1 samples because those do not contain real bundle-count
labels.
"""

from __future__ import annotations

import argparse
import glob
import os
import random
import sys
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import wandb

from src.model import STEDResUNet2D
from train import StedFieldLoss2D, _add_metric_batch, _average_metrics, _empty_metric_totals


@dataclass
class StaticSweepArgs:
    data_dir: str
    crop_size: int
    val_crop_size: int
    epochs: int
    base_batch_size: int
    num_workers: int
    amp_dtype: str
    grad_clip_norm: float
    multi_gpu: bool
    check_samples: int
    augment_geometric: bool
    augment_intensity: bool
    save_best: bool
    save_dir: str
    seed: int


GLOBAL_ARGS: StaticSweepArgs | None = None


def _torch_load(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Load a .pt record while staying compatible with older PyTorch versions."""
    try:
        return torch.load(path, weights_only=True, map_location="cpu")
    except TypeError:
        return torch.load(path, map_location="cpu")


class StrictStructuralV2Dataset(Dataset):
    """Precomputed 2D STED dataset with strict structural_v2 validation.

    The original training dataset class has a backwards-compatibility path that
    fabricates a bundle-count target for 5-channel structural_v1 samples.  That
    is undesirable for the current project because bundle count is a real target.
    This dataset therefore hard-fails unless every sample has exactly six target
    channels.
    """

    def __init__(
        self,
        data_dir: str | os.PathLike[str],
        crop_size: int = 0,
        random_crop: bool = False,
        augment_geometric: bool = False,
        augment_intensity: bool = False,
    ):
        self.data_dir = Path(data_dir)
        self.crop_size = int(crop_size or 0)
        self.random_crop = bool(random_crop)
        self.augment_geometric = bool(augment_geometric)
        self.augment_intensity = bool(augment_intensity)
        self.files = sorted(self.data_dir.glob("*.pt"))
        if not self.files:
            raise FileNotFoundError(f"No .pt files found in {self.data_dir}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        path = self.files[idx]
        record = _torch_load(path)
        if "volume" not in record or "targets" not in record:
            raise KeyError(f"{path} is missing required keys 'volume' and/or 'targets'.")

        target_schema = record.get("target_schema", "legacy")
        if target_schema != "structural_v2":
            raise ValueError(
                f"{path} has target_schema={target_schema!r}. Bundle-count training requires "
                "target_schema='structural_v2'. Regenerate the dataset with the upgraded generator."
            )

        vol = record["volume"].float()
        targets = record["targets"].float()

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

    def _crop_2d(self, vol: torch.Tensor, targets: torch.Tensor):
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

        return vol[:, y0 : y0 + crop_h, x0 : x0 + crop_w], targets[:, y0 : y0 + crop_h, x0 : x0 + crop_w]

    @staticmethod
    def _augment_geometric(vol: torch.Tensor, targets: torch.Tensor):
        # Orientation is encoded as cos(2theta), sin(2theta).  Horizontal and
        # vertical flips preserve cos(2theta) and negate sin(2theta).  Odd 90°
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
        # Keep this conservative: the synthetic generator already tries to match
        # real intensity/noise statistics.  This only prevents overfitting to a
        # fixed normalization.
        scale = 0.90 + 0.20 * torch.rand((), dtype=vol.dtype)
        offset = -0.04 + 0.08 * torch.rand((), dtype=vol.dtype)
        noise_std = 0.00 + 0.015 * torch.rand((), dtype=vol.dtype)
        vol = vol * scale + offset
        if noise_std > 0:
            vol = vol + torch.randn_like(vol) * noise_std
        return torch.clamp(vol, 0.0, 1.0)


class SweepStedFieldLoss2D(StedFieldLoss2D):
    """Loss with a model-selection score decoupled from swept train weights."""

    def __init__(
        self,
        *args,
        fixed_score_radius_weight: float = 0.25,
        fixed_score_bundle_count_weight: float = 0.30,
        fixed_score_threshold_sensitivity_weight: float = 0.20,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.fixed_score_radius_weight = float(fixed_score_radius_weight)
        self.fixed_score_bundle_count_weight = float(fixed_score_bundle_count_weight)
        self.fixed_score_threshold_sensitivity_weight = float(fixed_score_threshold_sensitivity_weight)

    def fixed_score(self, components):
        return (
            self.score_centerline_weight * components["centerline"]
            + self.fixed_orientation_weight * components["orientation"]
            + self.fixed_traceability_weight * components["traceability"]
            + self.fixed_score_radius_weight * components["radius"]
            + self.fixed_score_bundle_count_weight * components["bundle_count"]
            + self.fixed_score_threshold_sensitivity_weight * components["threshold_sensitivity"]
        )


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _loader_kwargs(num_workers: int, device: torch.device) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "num_workers": int(num_workers),
        "pin_memory": device.type == "cuda",
    }
    if num_workers > 0:
        kwargs["prefetch_factor"] = 4
        kwargs["persistent_workers"] = True
    return kwargs


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


def _make_criterion(config) -> SweepStedFieldLoss2D:
    return SweepStedFieldLoss2D(
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


def _save_checkpoint(
    model: nn.Module,
    use_data_parallel: bool,
    save_dir: str,
    run_id: str,
    epoch: int,
    best_val_score: float,
    config: dict[str, Any],
) -> None:
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"sweep_{run_id}_best.pth")
    torch.save(
        {
            "model_state_dict": _state_dict(model, use_data_parallel),
            "epoch": int(epoch),
            "best_val_score": float(best_val_score),
            "config": dict(config),
        },
        path,
    )


def train_sweep() -> None:
    if GLOBAL_ARGS is None:
        raise RuntimeError("GLOBAL_ARGS was not initialized.")
    args = GLOBAL_ARGS

    run = wandb.init()
    config = wandb.config
    _set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_gpus_available = torch.cuda.device_count()
    use_data_parallel = bool(args.multi_gpu and num_gpus_available > 1)
    batch_size = args.base_batch_size * max(1, num_gpus_available) if use_data_parallel else args.base_batch_size

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

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
    }
    wandb.config.update(static_config, allow_val_change=True)

    train_dir = os.path.join(args.data_dir, "train")
    val_dir = os.path.join(args.data_dir, "val")
    train_ds = StrictStructuralV2Dataset(
        train_dir,
        crop_size=args.crop_size,
        random_crop=True,
        augment_geometric=args.augment_geometric,
        augment_intensity=args.augment_intensity,
    )
    val_ds = StrictStructuralV2Dataset(
        val_dir,
        crop_size=args.val_crop_size or args.crop_size,
        random_crop=False,
        augment_geometric=False,
        augment_intensity=False,
    )

    loader_kwargs = _loader_kwargs(args.num_workers, device)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, **loader_kwargs)

    model = STEDResUNet2D(in_channels=1, base_filters=int(config.base_filters))
    if use_data_parallel:
        model = nn.DataParallel(model)
    model = model.to(device)

    criterion = _make_criterion(config)
    optimizer = optim.AdamW(model.parameters(), lr=float(config.learning_rate), weight_decay=float(config.weight_decay))
    autocast_dtype, scaler = _amp_settings(device, args.amp_dtype)

    best_val_score = float("inf")
    best_epoch = 0
    patience_counter = 0
    patience = int(config.early_stop_patience)
    min_epochs_before_stop = int(config.min_epochs_before_stop)

    print(
        f"Run {run.id if run is not None else '<no-run>'}: "
        f"device={device}, batch_size={batch_size}, train={len(train_ds)}, val={len(val_ds)}, "
        f"base_filters={int(config.base_filters)}"
    )

    for epoch in range(args.epochs):
        active_train_centerline_weight = criterion.set_epoch(epoch)
        model.train()
        train_totals = _empty_metric_totals()
        t0 = time.time()

        for inputs, targets in train_loader:
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
        epoch_time = time.time() - t0

        improved = val_metrics["score"] < best_val_score
        if improved:
            best_val_score = val_metrics["score"]
            best_epoch = epoch + 1
            patience_counter = 0
            if args.save_best:
                _save_checkpoint(
                    model,
                    use_data_parallel,
                    args.save_dir,
                    run.id if run is not None else "local",
                    epoch + 1,
                    best_val_score,
                    dict(wandb.config),
                )
        else:
            patience_counter += 1

        log_data = {
            "epoch": epoch + 1,
            "epoch_time_seconds": epoch_time,
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
        wandb.log(log_data)

        print(
            f"Epoch {epoch + 1:03d}/{args.epochs} | "
            f"val_score={val_metrics['score']:.5f} | best={best_val_score:.5f} @ {best_epoch} | "
            f"val_bundle={val_metrics['bundle_count']:.5f} | val_radius={val_metrics['radius']:.5f} | "
            f"time={epoch_time:.1f}s"
        )

        if epoch + 1 >= min_epochs_before_stop and patience_counter >= patience:
            print(
                f"Early stopping at epoch {epoch + 1}: no best_val_score improvement "
                f"for {patience} epochs after minimum {min_epochs_before_stop} epochs."
            )
            break


def _build_sweep_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "method": "bayes",
        "metric": {
            "name": "best_val_score",
            "goal": "minimize",
        },
        "early_terminate": {
            "type": "hyperband",
            "min_iter": 25,
            "eta": 2,
        },
        "parameters": {
            # Best first-sweep value was ~9.7e-5, close to the old upper bound.
            # Expand upward, but keep the lower end near the useful region.
            "learning_rate": {
                "distribution": "log_uniform_values",
                "min": 6e-5,
                "max": 2.0e-4,
            },

            # Best value was ~1.35e-4, fairly high.
            # Search around that, allowing stronger regularization.
            "weight_decay": {
                "distribution": "log_uniform_values",
                "min": 5e-5,
                "max": 4e-4,
            },

            # Best value was ~0.92, well inside the old range.
            # Narrow around the useful region.
            "orientation_loss_weight": {
                "distribution": "uniform",
                "min": 0.75,
                "max": 1.15,
            },

            # Best value was ~0.57, close to the old upper bound.
            # Expand upward.
            "visibility_loss_weight": {
                "distribution": "uniform",
                "min": 0.45,
                "max": 0.85,
            },

            # Best value was ~0.072, toward the lower end.
            # Keep the floor low so orientation supervision is not too narrowly masked.
            "orientation_mask_floor": {
                "distribution": "uniform",
                "min": 0.04,
                "max": 0.11,
            },

            # Best value was ~0.116, comfortably inside this narrower range.
            "loss_visibility_floor": {
                "distribution": "uniform",
                "min": 0.07,
                "max": 0.18,
            },

            # Best value was ~0.365, near the old upper bound.
            # Expand upward because radius is clearly important for the thick-fiber dataset.
            "radius_loss_weight": {
                "distribution": "uniform",
                "min": 0.25,
                "max": 0.60,
            },

            # Best value was ~0.221.
            # Keep moderate range, but still allow higher values in case bundle count was underweighted.
            "bundle_count_loss_weight": {
                "distribution": "uniform",
                "min": 0.15,
                "max": 0.50,
            },

            # Best value was ~1.17, near the old upper bound.
            # Expand upward.
            "train_centerline_weight": {
                "distribution": "uniform",
                "min": 1.00,
                "max": 1.45,
            },

            # Keep these fixed for this sweep.
            # Do not add extra degrees of freedom until the main loss-weight region is stable.
            "centerline_warmup_epochs": {"value": 3},
            "centerline_warmup_start_factor": {"value": 0.50},
            "centerline_threshold": {"value": 0.50},
            "stability_margin_weight": {"value": 0.20},

            # Fixed validation/model-selection score weights.
            # These should remain fixed while sweeping training-loss weights.
            "score_centerline_weight": {"value": 1.00},
            "fixed_score_orientation_weight": {"value": 1.00},
            "fixed_score_traceability_weight": {"value": 0.35},
            "fixed_score_radius_weight": {"value": 0.25},
            "fixed_score_bundle_count_weight": {"value": 0.30},
            "fixed_score_threshold_sensitivity_weight": {"value": 0.20},

            # Kept for compatibility with train.py naming and for logging.
            "score_stability_weight": {"value": 0.20},

            "early_stop_patience": {"value": 10},
            "min_epochs_before_stop": {"value": 25},

            # For the second sweep, I would usually restrict this to [40, 48].
            # If args.base_filters_values is already [40, 48], keep this.
            "base_filters": {
                "values": args.base_filters_values,
            },
        },
    }


def add_train_sweep_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--gpus", type=str, default="0")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--dim", type=int, choices=[2], default=2)
    parser.add_argument("--sweep_count", type=int, default=20)
    parser.add_argument("--crop_size", type=int, default=512)
    parser.add_argument("--val_crop_size", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--base_batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--project", type=str, default="fibras-sted-resunet2d-sweep-v3")
    parser.add_argument("--base_filters_values", type=int, nargs="+", default=[32, 40])
    parser.add_argument("--amp_dtype", type=str, choices=["bf16", "fp16", "off"], default="bf16")
    parser.add_argument("--grad_clip_norm", type=float, default=1.0)
    parser.add_argument("--check_samples", type=int, default=16)
    parser.add_argument("--augment_geometric", action="store_true")
    parser.add_argument("--augment_intensity", action="store_true")
    parser.add_argument("--save_best", action="store_true")
    parser.add_argument("--save_dir", type=str, default="weights/sweeps")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--multi_gpu", action="store_true", help="Enable nn.DataParallel across all visible GPUs.")
    parser.add_argument("--nccl_p2p_disable", action="store_true", help="Set NCCL_P2P_DISABLE=1 when using --multi_gpu.")
    parser.add_argument("--nccl_ib_disable", action="store_true", help="Set NCCL_IB_DISABLE=1 when using --multi_gpu.")
    parser.add_argument("--nccl_debug", type=str, choices=["INFO", "WARN"], default="", help="Set NCCL_DEBUG when using --multi_gpu.")


def parse_args(argv=None) -> argparse.Namespace:
    argv = sys.argv[1:] if argv is None else argv
    parser = argparse.ArgumentParser(description="W&B sweep tools for structural_v2 2D STED ResUNet training")
    subparsers = parser.add_subparsers(dest="mode")

    train_parser = subparsers.add_parser("train", help="Launch a W&B training sweep")
    add_train_sweep_arguments(train_parser)

    from src.wandb_export import add_export_arguments

    export_parser = subparsers.add_parser("export-runs", help="Export W&B sweep runs to CSV/JSON")
    add_export_arguments(export_parser)

    if not argv:
        argv = ["train"]
    elif argv[0] not in {"train", "export-runs", "-h", "--help"}:
        argv = ["train", *argv]
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()

    if args.mode == "export-runs":
        from src.wandb_export import export_runs

        export_runs(args)
        return

    if args.mode not in {"train", None}:
        raise ValueError(f"Unknown sweep mode: {args.mode}")

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

    _validate_dataset(args.data_dir, args.check_samples)

    global GLOBAL_ARGS
    GLOBAL_ARGS = StaticSweepArgs(
        data_dir=args.data_dir,
        crop_size=args.crop_size,
        val_crop_size=args.val_crop_size,
        epochs=args.epochs,
        base_batch_size=args.base_batch_size,
        num_workers=args.num_workers,
        amp_dtype=args.amp_dtype,
        grad_clip_norm=args.grad_clip_norm,
        multi_gpu=bool(args.multi_gpu),
        check_samples=args.check_samples,
        augment_geometric=bool(args.augment_geometric),
        augment_intensity=bool(args.augment_intensity),
        save_best=bool(args.save_best),
        save_dir=args.save_dir,
        seed=args.seed,
    )

    sweep_config = _build_sweep_config(args)
    sweep_id = wandb.sweep(sweep_config, project=args.project)
    wandb.agent(sweep_id, train_sweep, count=args.sweep_count)


if __name__ == "__main__":
    main()
