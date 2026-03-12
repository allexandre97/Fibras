import os
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.model import FlexibleCVFUNet
from train import PrecomputedFiberDataset

def evaluate_model(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    test_dir = os.path.join(args.data_dir, "test")
    test_ds = PrecomputedFiberDataset(test_dir, dim=args.dim)
    test_loader = DataLoader(test_ds, batch_size=4, shuffle=False, num_workers=4, pin_memory=True)

    model = FlexibleCVFUNet(in_channels=1, base_filters=args.base_filters, dim=args.dim)
    
    state_dict = torch.load(args.model_path, map_location=device, weights_only=True)
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v
        
    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()

    mse = torch.nn.MSELoss(reduction='none')
    
    total_loss = 0.0
    total_edt_loss = 0.0
    total_vec_loss = 0.0
    total_visibility_loss = 0.0
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            pred = model(inputs)
            
            pred_edt, pred_vec = pred[:, 0:1], pred[:, 1:1+args.dim]
            targ_edt, targ_vec = targets[:, 0:1], targets[:, 1:1+args.dim]
            
            # 1. EDT Evaluation
            loss_edt = mse(pred_edt, targ_edt).mean().item()
            
            # 2. Sign-Agnostic Vector Evaluation
            mask = (targ_edt > 0.0).float()
            
            # Calculate errors for both polarities
            err_pos = torch.sum((pred_vec - targ_vec)**2, dim=1, keepdim=True) / args.dim
            err_neg = torch.sum((pred_vec + targ_vec)**2, dim=1, keepdim=True) / args.dim
            
            # Take the minimum error
            loss_vec_raw = torch.min(err_pos, err_neg) * mask
            
            mask_sum = mask.sum() + 1e-8
            loss_vec = (loss_vec_raw.sum() / mask_sum).item()
            
            total_edt_loss += loss_edt
            total_vec_loss += loss_vec
            total_batch_loss = loss_edt + (args.vector_loss_weight * loss_vec)

            if args.dim == 2:
                pred_visibility = pred[:, 3:4]
                targ_visibility = targets[:, 3:4]
                loss_visibility = F.binary_cross_entropy_with_logits(pred_visibility, targ_visibility).item()
                total_visibility_loss += loss_visibility
                total_batch_loss += args.visibility_loss_weight * loss_visibility

            total_loss += total_batch_loss

    n_batches = len(test_loader)
    print(f"\n--- Unseen Test Set Evaluation ({args.dim}D Model) ---")
    print(f"Target Checkpoint: {args.model_path}")
    print(f"Average Total Loss: {total_loss / n_batches:.4f}")
    print(f"Average EDT MSE:    {total_edt_loss / n_batches:.4f}")
    print(f"Average Vector MSE: {total_vec_loss / n_batches:.4f}")
    if args.dim == 2:
        print(f"Average Visibility: {total_visibility_loss / n_batches:.4f}")
    print("---------------------------------------------")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate CVFUNet on Test Set")
    parser.add_argument('--model_path', type=str, required=True, help="Path to the .pth file")
    parser.add_argument('--data_dir', type=str, required=True, help="Path to base dataset folder")
    parser.add_argument('--dim', type=int, required=True)
    parser.add_argument('--base_filters', type=int, required=True)
    parser.add_argument('--vector_loss_weight', type=float, default=1.175) # Updated to your sweep optimal
    parser.add_argument('--visibility_loss_weight', type=float, default=0.35)
    
    args = parser.parse_args()
    evaluate_model(args)
