"""W&B hyperparameter sweep for the 2D STED ResUNet.

This sweep script is intended for the upgraded synthetic STED datasets generated as
``target_schema='structural_v2'``.  In particular, it refuses to train on old
5-channel structural_v1 samples because those do not contain real bundle-count
labels.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import wandb

from src.model import PREDICTION_HEAD_TYPE, STEDResUNet2D
from train import (
    PrecomputedFiberDataset,
    _add_metric_batch,
    _amp_settings,
    _autocast_context,
    _average_metrics,
    _checkpoint_payload,
    _empty_metric_totals,
    _loader_kwargs,
    _make_criterion,
    _set_seed,
    _validate_dataset,
    format_aspp_dilations,
    parse_aspp_dilations,
)


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
    torch.save(_checkpoint_payload(model, use_data_parallel, epoch, best_val_score, config), path)


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
        "prediction_head_type": PREDICTION_HEAD_TYPE,
    }
    wandb.config.update(static_config, allow_val_change=True)
    aspp_dilations = parse_aspp_dilations(config.aspp_dilations)
    aspp_dilations_config = format_aspp_dilations(aspp_dilations)
    unet_depth = int(config.unet_depth)
    wandb.config.update(
        {"aspp_dilations": aspp_dilations_config, "unet_depth": unet_depth},
        allow_val_change=True,
    )

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

    model = STEDResUNet2D(
        in_channels=1,
        base_filters=int(config.base_filters),
        aspp_dilations=aspp_dilations,
        unet_depth=unet_depth,
    )
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
        f"base_filters={int(config.base_filters)}, unet_depth={unet_depth}, "
        f"aspp_dilations={aspp_dilations_config}"
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
                "min": 1.5e-4,
                "max": 3.0e-4,
            },

            # Best value was ~1.35e-4, fairly high.
            # Search around that, allowing stronger regularization.
            "weight_decay": {
                "distribution": "log_uniform_values",
                "min": 1e-5,
                "max": 1.5e-4,
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
                "max": 0.70,
            },

            # Best value was ~0.072, toward the lower end.
            # Keep the floor low so orientation supervision is not too narrowly masked.
            "orientation_mask_floor": {
                "distribution": "uniform",
                "min": 0.07,
                "max": 0.14,
            },

            # Best value was ~0.116, comfortably inside this narrower range.
            "loss_visibility_floor": {
                "distribution": "uniform",
                "min": 0.06,
                "max": 0.13,
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
                "min": 1.20,
                "max": 1.70,
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
            "aspp_dilations": {
                "values": args.aspp_dilation_values,
            },
            "unet_depth": {
                "values": args.unet_depth_values,
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
    parser.add_argument("--unet_depth_values", type=int, nargs="+", default=[3, 4])
    parser.add_argument("--aspp_dilation_values", type=str, nargs="+", default=["1,2,4", "1,2,3", "2,4,8"])
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
