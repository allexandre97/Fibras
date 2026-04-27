import os
import argparse
import torch
from torch.utils.data import DataLoader

from src.model import STEDResUNet2D
from train import PrecomputedFiberDataset, StedFieldLoss2D

def evaluate_model(args):
    if args.dim != 2:
        raise ValueError("The upgraded STED test path is 2D only. Use --dim 2.")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    test_dir = os.path.join(args.data_dir, "test")
    test_ds = PrecomputedFiberDataset(test_dir, dim=2, crop_size=args.crop_size, random_crop=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    model = STEDResUNet2D(in_channels=1, base_filters=args.base_filters)
    
    state_dict = torch.load(args.model_path, map_location=device, weights_only=True)
    new_state_dict = {k[7:] if k.startswith('module.') else k: v for k, v in state_dict.items()}
        
    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()

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
        centerline_threshold=args.centerline_threshold,
        score_stability_weight=args.score_stability_weight,
        stability_margin_weight=args.stability_margin_weight,
    )
    
    total_metrics = {
        "loss": 0.0,
        "score": 0.0,
        "centerline": 0.0,
        "centerline_focal": 0.0,
        "centerline_dice": 0.0,
        "cldice": 0.0,
        "orientation": 0.0,
        "traceability": 0.0,
        "radius": 0.0,
        "threshold_sensitivity": 0.0,
    }
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            pred = model(inputs)
            
            # Use criterion to get individual components
            components = criterion.compute_components(pred, targets)
            batch_total_loss = criterion(pred, targets)
            
            total_metrics["loss"] += batch_total_loss.item()
            total_metrics["score"] += criterion.fixed_score(components).item()
            total_metrics["centerline"] += components["centerline"].item()
            total_metrics["centerline_focal"] += components["centerline_focal"].item()
            total_metrics["centerline_dice"] += components["centerline_dice"].item()
            total_metrics["cldice"] += components["cldice"].item()
            total_metrics["orientation"] += components["orientation"].item()
            total_metrics["traceability"] += components["traceability"].item()
            total_metrics["radius"] += components["radius"].item()
            total_metrics["threshold_sensitivity"] += components["threshold_sensitivity"].item()

    n_batches = len(test_loader)
    print(f"\n--- Unseen Test Set Evaluation ({args.dim}D Model) ---")
    print(f"Target Checkpoint: {args.model_path}")
    print(f"Average Total Loss: {total_metrics['loss'] / n_batches:.4f}")
    print(f"Average Fixed Score: {total_metrics['score'] / n_batches:.4f}")
    print(f"Average Centerline Loss: {total_metrics['centerline'] / n_batches:.4f}")
    print(f"Average Centerline Focal: {total_metrics['centerline_focal'] / n_batches:.4f}")
    print(f"Average Centerline Dice: {total_metrics['centerline_dice'] / n_batches:.4f}")
    print(f"Average clDice: {total_metrics['cldice'] / n_batches:.4f}")
    print(f"Average Orientation: {total_metrics['orientation'] / n_batches:.4f}")
    print(f"Average Traceability: {total_metrics['traceability'] / n_batches:.4f}")
    print(f"Average Radius: {total_metrics['radius'] / n_batches:.4f}")
    print(f"Average Threshold Sensitivity: {total_metrics['threshold_sensitivity'] / n_batches:.4f}")
    print("---------------------------------------------")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate CVFUNet on Test Set")
    parser.add_argument('--model_path', type=str, required=True, help="Path to the .pth file")
    parser.add_argument('--data_dir', type=str, required=True, help="Path to base dataset folder")
    parser.add_argument('--dim', type=int, choices=[2], default=2)
    parser.add_argument('--base_filters', type=int, default=32)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--crop_size', type=int, default=0)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--orientation_loss_weight', type=float, default=1.0)
    parser.add_argument('--visibility_loss_weight', '--traceability_loss_weight', dest='visibility_loss_weight', type=float, default=0.35)
    parser.add_argument('--orientation_mask_floor', '--centerline_support_floor', dest='orientation_mask_floor', type=float, default=0.15)
    parser.add_argument('--loss_visibility_floor', type=float, default=0.25)
    parser.add_argument('--radius_loss_weight', type=float, default=0.15)
    parser.add_argument('--train_centerline_weight', type=float, default=1.0)
    parser.add_argument('--score_centerline_weight', '--skeleton_score_weight', dest='score_centerline_weight', type=float, default=1.0)
    parser.add_argument('--centerline_warmup_epochs', type=int, default=0)
    parser.add_argument('--centerline_warmup_start_factor', type=float, default=1.0)
    parser.add_argument('--centerline_threshold', type=float, default=0.5)
    parser.add_argument('--stability_margin_weight', type=float, default=0.2)
    parser.add_argument('--score_stability_weight', type=float, default=0.2)
    
    args = parser.parse_args()
    evaluate_model(args)
