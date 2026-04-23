import os
import argparse
import numpy as np
import tifffile
import re

from src.inference_utils import normalize_image_percentile, predict_tiled_2d
from src.model import STEDResUNet2D
from src.sted import orientation_confidence_np, orientation_to_vector_map_np
from src.tracking import StreamlineTracker
from src.visualization import StedSynthesisVisualizer

def _sample_file_sort_key(path: str):
    match = re.search(r"(\d+)", os.path.basename(path))
    if match:
        return int(match.group(1))
    return os.path.basename(path)

def _resolve_dataset_file(data_dir: str, split: str, index: int, random_sample: bool):
    split_dir = os.path.join(data_dir, split)
    sample_dir = split_dir if os.path.isdir(split_dir) else data_dir

    files = [
        os.path.join(sample_dir, fname)
        for fname in os.listdir(sample_dir)
        if fname.endswith(".pt")
    ]
    files.sort(key=_sample_file_sort_key)
    if len(files) == 0:
        raise FileNotFoundError(f"No .pt files found in directory: {sample_dir}")

    if random_sample:
        idx = int(np.random.randint(0, len(files)))
    else:
        if index < 0 or index >= len(files):
            raise IndexError(
                f"Requested --index={index}, but dataset has {len(files)} files in {sample_dir}."
            )
        idx = index

    return files[idx], idx, len(files), sample_dir

def _extract_sample_arrays(data):
    volume = data["volume"].detach().cpu().numpy()
    targets = data["targets"].detach().cpu().numpy()

    is_2d = volume.ndim == 4 and volume.shape[0] == 1 and volume.shape[1] == 1
    if is_2d:
        image = volume[0, 0]
        edt = targets[0, 0]
        vector = targets[1:3, 0]
        visibility = targets[3, 0] if targets.shape[0] > 3 else None
        return {
            "is_2d": True,
            "image": image,
            "edt": edt,
            "vector": vector,
            "visibility": visibility,
        }

    is_3d = volume.ndim == 4 and volume.shape[0] == 1
    if is_3d:
        image = volume[0]
        edt = targets[0]
        vector = targets[1:4]
        return {
            "is_2d": False,
            "image": image,
            "edt": edt,
            "vector": vector,
            "visibility": None,
        }

    raise ValueError(
        f"Unsupported tensor format: volume shape={volume.shape}, targets shape={targets.shape}"
    )

def run_inference(args):
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
        # Squeeze extra dimensions (e.g., (1, H, W) or (H, W, 1)) for 2D inference
        if args.dim == 2:
            img = np.squeeze(img)
            if img.ndim != 2:
                 # If it's still not 2D, it might be a multi-channel or Z-stack. 
                 # Take the first slice/channel as a fallback.
                 print(f"Warning: Image has shape {img.shape}. Reducing to 2D slice [0, ...]")
                 img = img[0]
    else:
        raise ValueError("Unsupported format. Use .tif, .npy, or .pt")
        
    original_shape = img.shape
    img_norm = normalize_image_percentile(img)
    
    # 3. Full-resolution tiled forward pass
    if args.downsample > 1.0:
        print("Warning: --downsample is ignored by the 2D STED tiled inference path.")
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

    print(f"Predicted EDT range: [{edt_map.min():.4f}, {edt_map.max():.4f}]")
    print(f"Predicted Visibility range: [{visibility_map.min():.4f}, {visibility_map.max():.4f}]")
    print(f"Predicted Orientation Confidence range: [{orientation_confidence.min():.4f}, {orientation_confidence.max():.4f}]")
        
    # 6. Ridge Sharpening & Tractography
    print(f"Initiating structural tractography (min_edt={args.min_edt})...")
    
    # NEW: Skeletonize the EDT mask to collapse wide fibers into a single seeding line
    from skimage.morphology import skeletonize
    mask = (edt_for_tracking > args.min_edt).astype(np.uint8)
    thinned_mask = skeletonize(mask)
    
    # We pass the thinned mask to the tracker to use as the seed source
    # while keeping the full EDT map for smooth continuous interpolation
    tracker = StreamlineTracker(step_size=0.5, min_edt=args.min_edt)
    streamlines = tracker.track(edt_for_tracking, vector_map, seed_mask=thinned_mask)
    print(f"Successfully traced {len(streamlines)} fiber segments.")
    
    skeleton = tracker.to_binary_skeleton(streamlines, original_shape)
    
    # Save skeleton and diagnostic maps if requested
    if args.save_skeleton:
        base_path = args.image_path.rsplit('.', 1)[0]
        tifffile.imwrite(base_path + '_skeleton.tif', (skeleton * 255).astype(np.uint8))
        tifffile.imwrite(base_path + '_pred_edt.tif', (np.clip(edt_map, 0, 1) * 255).astype(np.uint8))
        tifffile.imwrite(base_path + '_pred_vis.tif', (visibility_map * 255).astype(np.uint8))
        tifffile.imwrite(base_path + '_pred_orient_conf.tif', (orientation_confidence * 255).astype(np.uint8))
        print(f"Results exported to: {base_path}_*.tif")

    # 8. Visualization
    if args.visualize:
        import napari
        viewer = napari.Viewer(title=f"Fibras Inference - {args.dim}D")
        viewer.add_image(img, name="Raw Image", colormap="magma")
        viewer.add_image(edt_map, name="Predicted EDT", colormap="viridis", visible=False, opacity=0.7)
        viewer.add_image(visibility_map, name="Predicted Visibility", colormap="inferno", visible=False)
        viewer.add_image(orientation_confidence, name="Predicted Orientation Confidence", colormap="cyan", visible=False)
        viewer.add_labels(skeleton.astype(int), name="Inferred Skeleton")
        
        print("Launching Napari...")
        napari.run()

def show_dataset_sample(args):
    import torch
    import napari

    if args.file:
        pt_path = args.file
        sample_name = os.path.basename(pt_path)
    else:
        pt_path, idx, total, sample_dir = _resolve_dataset_file(
            data_dir=args.data_dir,
            split=args.split,
            index=args.index,
            random_sample=args.random_sample,
        )
        sample_name = f"{os.path.basename(pt_path)} ({idx+1}/{total})"

    print(f"Loading sample: {pt_path}")
    data = torch.load(pt_path, map_location="cpu", weights_only=True)
    sample = _extract_sample_arrays(data)
    
    image = sample["image"]
    edt = sample["edt"]
    vector = sample["vector"]
    visibility = sample["visibility"]
    is_2d = sample["is_2d"]
    
    viewer = napari.Viewer(title=f"Fibras Dataset Viewer - {sample_name}")
    viewer.add_image(image, name="Synthetic Image", colormap="magma")
    viewer.add_image(edt, name="GT EDT", colormap="viridis", visible=False, opacity=0.7)
    
    if visibility is not None:
        viewer.add_image(visibility, name="GT Visibility", colormap="inferno", visible=False)
        
    viewer.add_labels((edt > 0.15).astype(int), name="Annotation Mask", visible=False)
    viewer.add_labels((edt > 0.85).astype(int), name="GT Centerlines", visible=True)
    
    napari.run()

def show_sted_debug(args):
    from generate_dataset import build_sted_debug_sample
    
    debug_data = build_sted_debug_sample(
        tuple(args.bounds),
        synth_depth=args.synth_depth,
        label_slab_thickness=args.label_slab_thickness,
        label_slab_scale=args.label_slab_scale,
        annotation_weight_floor=args.annotation_weight_floor,
        soft_skeleton_alpha=args.soft_skeleton_alpha,
        visibility_weight_floor=args.visibility_weight_floor,
        seed=args.seed,
    )
    StedSynthesisVisualizer.show_sted_debug_summary(
        debug_data,
        save_path=args.save,
        show=not args.no_show,
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified Fibras Visualization & Inference Tool")
    subparsers = parser.add_subparsers(dest="mode", help="Operating mode")

    # Inference Mode
    inf_parser = subparsers.add_parser("inference", help="Run model inference on images")
    inf_parser.add_argument('--model_path', type=str, required=True)
    inf_parser.add_argument('--image_path', type=str, required=True)
    inf_parser.add_argument('--dim', type=int, choices=[2], default=2)
    inf_parser.add_argument('--base_filters', type=int, default=32)
    inf_parser.add_argument('--min_edt', type=float, default=0.15)
    inf_parser.add_argument('--visibility_floor', type=float, default=0.25)
    inf_parser.add_argument('--tile_size', type=int, default=512)
    inf_parser.add_argument('--tile_overlap', type=int, default=128)
    inf_parser.add_argument('--downsample', type=float, default=1.0)
    inf_parser.add_argument('--no_amp', action='store_true')
    inf_parser.add_argument('--visualize', action='store_true', default=True)
    inf_parser.add_argument('--save_skeleton', action='store_true')

    # Dataset Mode
    ds_parser = subparsers.add_parser("dataset", help="Visualize synthetic dataset samples")
    ds_parser.add_argument("--file", type=str, help="Specific .pt file")
    ds_parser.add_argument("--data_dir", type=str, help="Dataset root directory")
    ds_parser.add_argument("--split", type=str, choices=["train", "val", "test"], default="train")
    ds_parser.add_argument("--index", type=int, default=0)
    ds_parser.add_argument("--random_sample", action="store_true")
    ds_parser.add_argument("--visibility_floor", type=float, default=0.25)

    # STED Debug Mode
    debug_parser = subparsers.add_parser("sted-debug", help="Debug 3D-to-2D STED synthesis")
    debug_parser.add_argument("--bounds", type=int, nargs=2, default=[64, 64])
    debug_parser.add_argument("--synth_depth", type=int, default=16)
    debug_parser.add_argument("--label_slab_thickness", type=float, default=None)
    debug_parser.add_argument("--label_slab_scale", type=float, default=1.3)
    debug_parser.add_argument("--annotation_weight_floor", type=float, default=0.25)
    debug_parser.add_argument("--soft_skeleton_alpha", type=float, default=0.35)
    debug_parser.add_argument("--visibility_weight_floor", type=float, default=0.03)
    debug_parser.add_argument("--seed", type=int, default=None)
    debug_parser.add_argument("--save", type=str, default=None)
    debug_parser.add_argument("--no_show", action="store_true")

    args = parser.parse_args()

    if args.mode == "inference":
        run_inference(args)
    elif args.mode == "dataset":
        show_dataset_sample(args)
    elif args.mode == "sted-debug":
        show_sted_debug(args)
    else:
        parser.print_help()
