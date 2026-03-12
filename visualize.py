import argparse
import os

def show_synthetic_data(pt_path: str):
    import torch

    try:
        import napari
    except ImportError:
        print("Napari not installed. Please run: pip install napari[all]")
        return

    print(f"Loading tensor from: {pt_path}")

    data = torch.load(pt_path, map_location="cpu", weights_only=True)
    volume = data["volume"].squeeze(0).numpy()
    edt = data["targets"][0].numpy()
    visibility = data["targets"][3].numpy() if data["targets"].shape[0] > 3 else None

    is_2d = volume.shape[0] == 1
    dim_str = "2D STED" if is_2d else "3D Confocal"

    if is_2d:
        volume = volume.squeeze(0)
        edt = edt.squeeze(0)
        if visibility is not None:
            visibility = visibility.squeeze(0)

    print(f"Render Mode: {dim_str} | Array Shape: {volume.shape}")

    viewer = napari.Viewer(title=f"Fibras Dataset Viewer - {dim_str}")
    viewer.add_image(
        volume,
        name="Synthetic Microscopy Data",
        colormap="magma",
        blending="additive",
    )

    centerline_mask = edt > 0.85
    viewer.add_labels(
        centerline_mask.astype(int),
        name="Ground Truth Centerlines",
        visible=False,
        opacity=0.7,
    )

    if is_2d and visibility is not None:
        viewer.add_image(
            visibility,
            name="Visibility Target",
            colormap="viridis",
            visible=False,
            opacity=0.65,
        )

    napari.run()


def show_sted_debug(
    bounds,
    synth_depth,
    label_slab_thickness,
    label_slab_scale=1.3,
    annotation_weight_floor=0.25,
    soft_skeleton_alpha=0.35,
    seed=None,
    save_path=None,
    show=True,
):
    from generate_dataset import build_sted_debug_sample
    from src.visualization import StedSynthesisVisualizer

    debug_data = build_sted_debug_sample(
        tuple(bounds),
        synth_depth=synth_depth,
        label_slab_thickness=label_slab_thickness,
        label_slab_scale=label_slab_scale,
        annotation_weight_floor=annotation_weight_floor,
        soft_skeleton_alpha=soft_skeleton_alpha,
        seed=seed,
    )
    StedSynthesisVisualizer.show_sted_debug_summary(
        debug_data,
        save_path=save_path,
        show=show,
    )

    print(
        "Generated STED debug sample "
        f"(bounds={debug_data['bounds']}, slice_center={debug_data['slice_center']:.2f}, "
        f"projected_segments={debug_data['projected_segment_count']}, "
        f"label_slab_scale={debug_data.get('label_slab_scale', label_slab_scale):.2f}, "
        f"label_slab_thickness={debug_data['label_slab_thickness']:.2f}, "
        f"annotation_weight_floor={debug_data.get('annotation_weight_floor', annotation_weight_floor):.2f}, "
        f"soft_skeleton_alpha={debug_data.get('soft_skeleton_alpha', soft_skeleton_alpha):.2f}, "
        f"noise_level={debug_data['noise_level']:.4f}, "
        f"noise_n={debug_data['noise_level_normalized']:.4f}, "
        f"monomer_regime={debug_data.get('monomer_regime', 'n/a')}, "
        f"monomer_amp={debug_data.get('monomer_amplitude', 0.0):.4f})."
    )
    if save_path is not None:
        print(f"Saved debug summary to: {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize saved samples or inspect STED synthesis internals")
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--file", type=str, help="Path to a specific .pt dataset file")
    mode_group.add_argument(
        "--sted-debug",
        action="store_true",
        help="Synthesize one 2D STED sample and show the intermediate 3D-to-2D formation stages.",
    )

    parser.add_argument("--bounds", type=int, nargs=2, default=[64, 64], help="XY bounds for --sted-debug mode")
    parser.add_argument("--synth_depth", type=int, default=16, help="Internal z depth for --sted-debug mode")
    parser.add_argument(
        "--label_slab_thickness",
        type=float,
        default=None,
        help="Optional override for the focus-localization slab in voxels for --sted-debug mode",
    )
    parser.add_argument(
        "--label_slab_scale",
        type=float,
        default=1.3,
        help="Scale applied to optical depth of field when --label_slab_thickness is not provided.",
    )
    parser.add_argument(
        "--annotation_weight_floor",
        type=float,
        default=0.25,
        help="Axial-weight floor that defines the broader soft-annotation band in --sted-debug mode.",
    )
    parser.add_argument(
        "--soft_skeleton_alpha",
        type=float,
        default=0.35,
        help="Soft out-of-focus blend strength for EDT/vector targets in --sted-debug mode.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Optional RNG seed for --sted-debug mode")
    parser.add_argument("--save", type=str, default=None, help="Optional path to save the STED debug summary figure")
    parser.add_argument("--no-show", action="store_true", help="Do not open the STED debug figure interactively")
    args = parser.parse_args()

    if args.file:
        if not os.path.exists(args.file):
            raise FileNotFoundError(f"Dataset file not found: {args.file}")
        show_synthetic_data(args.file)
    else:
        if args.label_slab_scale <= 0.0:
            raise ValueError("--label_slab_scale must be greater than 0.")
        if args.annotation_weight_floor <= 0.0 or args.annotation_weight_floor > 1.0:
            raise ValueError("--annotation_weight_floor must be in the interval (0, 1].")
        if args.soft_skeleton_alpha < 0.0:
            raise ValueError("--soft_skeleton_alpha must be greater than or equal to 0.")
        show_sted_debug(
            bounds=args.bounds,
            synth_depth=args.synth_depth,
            label_slab_thickness=args.label_slab_thickness,
            label_slab_scale=args.label_slab_scale,
            annotation_weight_floor=args.annotation_weight_floor,
            soft_skeleton_alpha=args.soft_skeleton_alpha,
            seed=args.seed,
            save_path=args.save,
            show=not args.no_show,
        )
