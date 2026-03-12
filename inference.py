import os
import argparse
import torch
import torch.nn.functional as F
import numpy as np
import tifffile
from types import SimpleNamespace

from src.model import FlexibleCVFUNet
from src.tracking import StreamlineTracker
from src.visualization import AdvancedVisualizer

def pad_to_multiple(tensor, multiple=4):
    shape = tensor.shape[2:]
    pad = []
    for s in reversed(shape):
        rem = s % multiple
        pad.extend([0, (multiple - rem) % multiple])
    return F.pad(tensor, pad, mode='reflect'), shape

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Executing inference on: {device}")
    
    # 1. Initialize and Load Weights
    model = FlexibleCVFUNet(in_channels=1, base_filters=args.base_filters, dim=args.dim)
    state_dict = torch.load(args.model_path, map_location=device, weights_only=True)
    
    new_state_dict = {k[7:] if k.startswith('module.') else k: v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()
    
    # 2. I/O Handling
    if args.image_path.endswith('.pt'):
        data = torch.load(args.image_path, map_location='cpu', weights_only=True)
        img = data['volume'].squeeze(0).numpy()
        if args.dim == 2 and img.shape[0] == 1:
            img = img.squeeze(0)
    elif args.image_path.endswith('.npy'):
        img = np.load(args.image_path)
    elif args.image_path.endswith(('.tif', '.tiff')):
        img = tifffile.imread(args.image_path)
    else:
        raise ValueError("Unsupported format. Use .tif, .npy, or .pt")
        
    original_shape = img.shape
    
    # Robust Percentile Normalization (Ignores hot pixels)
    if img.max() > 1.0 or img.min() < 0.0:
        p_low, p_high = np.percentile(img, (0.5, 99.9))
        img_norm = np.clip((img - p_low) / (p_high - p_low + 1e-8), 0, 1)
    else:
        img_norm = img
        
    # 3. Dynamic Downsampling
    img_tensor = torch.tensor(img_norm, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    
    if args.downsample > 1.0:
        mode = 'bilinear' if args.dim == 2 else 'trilinear'
        img_tensor = F.interpolate(img_tensor, scale_factor=1.0/args.downsample, mode=mode, align_corners=False)
        print(f"Downsampled tensor from {original_shape} to {img_tensor.shape[2:]}")
        
    img_tensor = img_tensor.to(device)
    tensor, orig_spatial_dims = pad_to_multiple(img_tensor, multiple=4)
    
    # 4. Network Forward Pass
    print("Processing structural tensors...")
    with torch.no_grad():
        with torch.amp.autocast('cuda', dtype=torch.bfloat16) if device.type == 'cuda' else torch.no_grad():
            pred = model(tensor)
            
    pred = pred.squeeze(0).cpu().float().numpy()
    
    # 5. Pad Cropping
    if args.dim == 2:
        H, W = orig_spatial_dims
        edt_map = pred[0, :H, :W]
        vector_map = pred[1:, :H, :W]
    else:
        Z, Y, X = orig_spatial_dims
        edt_map = pred[0, :Z, :Y, :X]
        vector_map = pred[1:, :Z, :Y, :X]
        
    # 6. Tractography in Downsampled Space
    print("Initiating structural tractography...")
    tracker = StreamlineTracker(step_size=0.5, min_edt=args.min_edt)
    streamlines = tracker.track(edt_map, vector_map)
    print(f"Successfully traced {len(streamlines)} fiber segments.")
    
    # 7. Coordinate Rescaling & High-Res Burning
    if args.downsample > 1.0:
        streamlines = [path * args.downsample for path in streamlines]
        
    # Burn into the ORIGINAL high-resolution shape
    skeleton = tracker.to_binary_skeleton(streamlines, original_shape)
    
    out_path = args.image_path.rsplit('.', 1)[0] + '_skeleton.tif'
    tifffile.imwrite(out_path, skeleton * 255)
    print(f"Skeleton graph exported to: {out_path}")

    # 8. Visualization
    if args.visualize:
        # Scale EDT mask up for visualization overlay matching
        vis_mask = (edt_map > args.min_edt).astype(np.float32)
        vis_mask_tensor = torch.tensor(vis_mask).unsqueeze(0).unsqueeze(0)
        mode = 'bilinear' if args.dim == 2 else 'nearest'
        vis_mask_up = F.interpolate(vis_mask_tensor, size=original_shape, mode=mode).squeeze().numpy()
        
        mock_result = SimpleNamespace(
            binary_mask=(vis_mask_up > 0.5).astype(np.uint8),
            skeleton=skeleton,
            hfa_map=np.zeros(original_shape),
            fa_macro_map=np.zeros(original_shape)
        )
        
        print("Launching Napari AdvancedVisualizer...")
        AdvancedVisualizer.show_interactive_napari(img, mock_result)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--image_path', type=str, required=True)
    parser.add_argument('--dim', type=int, choices=[2, 3], required=True)
    parser.add_argument('--base_filters', type=int, default=16)
    parser.add_argument('--min_edt', type=float, default=0.15)
    parser.add_argument('--downsample', type=float, default=1.0, help="Factor to downsample the image before FCN.")
    parser.add_argument('--visualize', action='store_true')
    args = parser.parse_args()
    main(args)
