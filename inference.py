import os
import argparse
import torch
import numpy as np
import tifffile
from types import SimpleNamespace

from src.inference_utils import normalize_image_percentile, predict_tiled_2d
from src.model import STEDResUNet2D
from src.sted import orientation_confidence_np, orientation_to_vector_map_np
from src.tracking import StreamlineTracker
from src.visualization import AdvancedVisualizer

def main(args):
    if args.dim != 2:
        raise ValueError("The upgraded STED inference path is 2D only. Use --dim 2.")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Executing inference on: {device}")
    
    # 1. Initialize and Load Weights
    model = STEDResUNet2D(in_channels=1, base_filters=args.base_filters)
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
        img = np.squeeze(img)
        if img.ndim != 2:
            print(f"Warning: Image has shape {img.shape}. Reducing to 2D slice [0, ...]")
            img = img[0]
    else:
        raise ValueError("Unsupported format. Use .tif, .npy, or .pt")
        
    original_shape = img.shape
    img_norm = normalize_image_percentile(img)
    if args.downsample > 1.0:
        print("Warning: --downsample is ignored by the 2D STED tiled inference path.")
    
    # 3. Full-resolution tiled forward pass
    print(
        f"Processing {original_shape} image with tile_size={args.tile_size}, "
        f"tile_overlap={args.tile_overlap}"
    )
    pred = predict_tiled_2d(
        model,
        img_norm,
        device=device,
        tile_size=args.tile_size,
        overlap=args.tile_overlap,
        output_channels=4,
        multiple=16,
        use_amp=not args.no_amp,
    )
    
    # 4. Field extraction
    edt_map = pred[0]
    orientation_map = pred[1:3]
    vector_map = orientation_to_vector_map_np(orientation_map)
    orientation_confidence = orientation_confidence_np(orientation_map)
    visibility_logits = np.clip(pred[3], -20.0, 20.0)
    visibility_map = 1.0 / (1.0 + np.exp(-visibility_logits))
    edt_for_tracking = edt_map * np.clip(visibility_map, args.visibility_floor, 1.0)
        
    # 5. Tractography in native-resolution space
    print("Initiating structural tractography...")
    from skimage.morphology import skeletonize
    seed_mask = skeletonize((edt_for_tracking > args.min_edt).astype(np.uint8))
    tracker = StreamlineTracker(step_size=0.5, min_edt=args.min_edt)
    streamlines = tracker.track(edt_for_tracking, vector_map, seed_mask=seed_mask)
    print(f"Successfully traced {len(streamlines)} fiber segments.")
    
    skeleton = tracker.to_binary_skeleton(streamlines, original_shape)
    
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        stem = os.path.splitext(os.path.basename(args.image_path))[0]
        base_path = os.path.join(args.output_dir, stem)
    else:
        base_path = args.image_path.rsplit('.', 1)[0]

    tifffile.imwrite(base_path + '_skeleton.tif', skeleton * 255)
    tifffile.imwrite(base_path + '_pred_edt.tif', (np.clip(edt_map, 0, 1) * 255).astype(np.uint8))
    tifffile.imwrite(base_path + '_pred_vis.tif', (visibility_map * 255).astype(np.uint8))
    tifffile.imwrite(base_path + '_pred_orient_conf.tif', (orientation_confidence * 255).astype(np.uint8))
    print(f"Skeleton graph exported to: {base_path}_skeleton.tif")

    # 8. Visualization
    if args.visualize:
        # Scale EDT mask up for visualization overlay matching
        vis_mask = (edt_for_tracking > args.min_edt).astype(np.float32)
        
        mock_result = SimpleNamespace(
            binary_mask=(vis_mask > 0.5).astype(np.uint8),
            skeleton=skeleton,
            hfa_map=edt_map,
            fa_macro_map=orientation_confidence,
        )
        
        print("Launching Napari AdvancedVisualizer...")
        AdvancedVisualizer.show_interactive_napari(img, mock_result)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--image_path', type=str, required=True)
    parser.add_argument('--dim', type=int, choices=[2], default=2)
    parser.add_argument('--base_filters', type=int, default=32)
    parser.add_argument('--min_edt', type=float, default=0.15)
    parser.add_argument('--visibility_floor', type=float, default=0.25)
    parser.add_argument('--tile_size', type=int, default=512)
    parser.add_argument('--tile_overlap', type=int, default=128)
    parser.add_argument('--downsample', type=float, default=1.0, help="Factor to downsample the image before FCN.")
    parser.add_argument('--output_dir', type=str, default="", help="Optional directory for prediction outputs.")
    parser.add_argument('--no_amp', action='store_true')
    parser.add_argument('--visualize', action='store_true')
    args = parser.parse_args()
    main(args)
