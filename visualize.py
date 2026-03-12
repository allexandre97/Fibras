import os
import argparse
import torch
import numpy as np

def show_synthetic_data(pt_path: str):
    try:
        import napari
    except ImportError:
        print("Napari not installed. Please run: pip install napari[all]")
        return

    print(f"Loading tensor from: {pt_path}")
    
    # Force CPU mapping to prevent unnecessary VRAM allocation during visualization
    data = torch.load(pt_path, map_location='cpu', weights_only=True)
    
    # Extract the synthetic microscopy volume (Shape: 1, Z, Y, X)
    volume = data['volume'].squeeze(0).numpy()
    
    # Extract the EDT target to act as a ground-truth centerline overlay
    edt = data['targets'][0].numpy()
    
    # Dimension resolution
    is_2d = volume.shape[0] == 1
    dim_str = "2D STED" if is_2d else "3D Confocal"
    
    if is_2d:
        # Strip the Z-axis for strict 2D planar rendering in Napari
        volume = volume.squeeze(0)
        edt = edt.squeeze(0)
        
    print(f"Render Mode: {dim_str} | Array Shape: {volume.shape}")

    # Initialize the Viewer
    viewer = napari.Viewer(title=f"Fibras Dataset Viewer - {dim_str}")
    
    # 1. Main Synthetic Microscopy Layer
    viewer.add_image(
        volume, 
        name='Synthetic Microscopy Data', 
        colormap='magma', 
        blending='additive'
    )
    
    # 2. Ground Truth Centerline Layer (Hidden by default)
    # The EDT is normalized to [0, 1] where 1.0 is the exact core of the fiber.
    centerline_mask = edt > 0.85
    viewer.add_labels(
        centerline_mask.astype(int), 
        name='Ground Truth Centerlines', 
        visible=False,
        opacity=0.7
    )

    napari.run()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize Synthetic Microscopy Datasets")
    parser.add_argument('--file', type=str, required=True, help="Path to a specific .pt dataset file")
    args = parser.parse_args()
    
    if not os.path.exists(args.file):
        raise FileNotFoundError(f"Dataset file not found: {args.file}")
        
    show_synthetic_data(args.file)
