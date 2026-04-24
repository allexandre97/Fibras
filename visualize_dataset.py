import argparse
import os
import re

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np


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


def _direction_rgb(vector, edt, double_angle: bool = False):
    """Encode 2D direction/orientation as hue and EDT as brightness."""
    if double_angle:
        angle = (0.5 * np.arctan2(vector[1], vector[0])) % np.pi
    else:
        angle = np.arctan2(vector[1], vector[0]) % np.pi
    hue = angle / np.pi
    brightness = np.clip(edt, 0, 1)
    hsv = np.stack([hue, np.ones_like(hue), brightness], axis=-1)
    from matplotlib.colors import hsv_to_rgb
    return hsv_to_rgb(hsv)


def export_sample_png(pt_path: str, out_path: str, visibility_floor: float = 0.25):
    import torch

    data = torch.load(pt_path, map_location="cpu", weights_only=True)
    sample = _extract_sample_arrays(data)
    image = sample["image"]
    edt = sample["edt"]
    vector = sample["vector"]
    visibility = sample["visibility"]

    is_2d = sample["is_2d"]
    if not is_2d:
        mid = image.shape[0] // 2
        image = image[mid]
        edt = edt[mid]
        vector = vector[:, mid]

    direction_rgb = _direction_rgb(vector[:2], edt, double_angle=is_2d)

    ncols = 4 if (is_2d and visibility is not None) else 3
    fig, axes = plt.subplots(1, ncols, figsize=(4 * ncols, 4), dpi=150)

    axes[0].imshow(image, cmap="magma", vmin=0, vmax=max(image.max(), 1e-8))
    axes[0].set_title("Image")
    axes[1].imshow(edt, cmap="viridis", vmin=0, vmax=max(edt.max(), 1e-8))
    axes[1].set_title("EDT")
    axes[2].imshow(direction_rgb)
    axes[2].set_title("Vector Direction")

    if ncols == 4:
        axes[3].imshow(visibility, cmap="inferno", vmin=0, vmax=1)
        axes[3].set_title("Visibility")

    for ax in axes:
        ax.axis("off")

    fig.suptitle(os.path.basename(pt_path), fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def export_dataset_pngs(
    data_dir: str,
    export_dir: str,
    split: str = "train",
    num_samples: int = 0,
    visibility_floor: float = 0.25,
):
    split_dir = os.path.join(data_dir, split)
    sample_dir = split_dir if os.path.isdir(split_dir) else data_dir

    files = [
        os.path.join(sample_dir, f)
        for f in os.listdir(sample_dir)
        if f.endswith(".pt")
    ]
    files.sort(key=_sample_file_sort_key)
    if len(files) == 0:
        raise FileNotFoundError(f"No .pt files found in {sample_dir}")

    if num_samples > 0:
        files = files[:num_samples]

    os.makedirs(export_dir, exist_ok=True)
    print(f"Exporting {len(files)} samples from {sample_dir} to {export_dir}")

    for i, pt_path in enumerate(files):
        name = os.path.splitext(os.path.basename(pt_path))[0]
        out_path = os.path.join(export_dir, f"{name}.png")
        export_sample_png(pt_path, out_path, visibility_floor=visibility_floor)
        print(f"  [{i + 1}/{len(files)}] {out_path}")

    print("Done.")


def show_synthetic_data(pt_path: str, visibility_floor: float = 0.25):
    import torch

    try:
        import napari
    except ImportError:
        print("Napari not installed. Please run: pip install napari[all]")
        return

    print(f"Loading tensor from: {pt_path}")

    data = torch.load(pt_path, map_location="cpu", weights_only=True)
    sample = _extract_sample_arrays(data)
    image = sample["image"]
    edt = sample["edt"]
    vector = sample["vector"]
    visibility = sample["visibility"]
    is_2d = sample["is_2d"]
    dim_str = "2D STED" if is_2d else "3D Confocal"
    print(f"Render Mode: {dim_str} | Array Shape: {image.shape}")
    print(
        f"Image min/max={float(image.min()):.4f}/{float(image.max()):.4f} | "
        f"EDT min/max={float(edt.min()):.4f}/{float(edt.max()):.4f}"
    )
    if visibility is not None:
        print(f"Visibility min/max={float(visibility.min()):.4f}/{float(visibility.max()):.4f}")

    viewer = napari.Viewer(title=f"Fibras Dataset Viewer - {dim_str} - {os.path.basename(pt_path)}")
    viewer.add_image(
        image,
        name="Synthetic Microscopy Data",
        colormap="magma",
        blending="additive",
    )

    viewer.add_image(
        edt,
        name="EDT Target",
        colormap="viridis",
        visible=False,
        opacity=0.75,
    )

    if is_2d:
        direction_rgb = _direction_rgb(vector[:2], edt, double_angle=True)
        viewer.add_image(
            direction_rgb,
            name="Orientation Direction",
            rgb=True,
            visible=False,
            opacity=0.75,
        )
    else:
        direction_mid = _direction_rgb(vector[:2, vector.shape[1] // 2], edt[edt.shape[0] // 2])
        viewer.add_image(
            direction_mid,
            name="Vector Direction (mid-Z)",
            rgb=True,
            visible=False,
            opacity=0.75,
        )

    viewer.add_labels(
        (edt > 0.15).astype(int),
        name="Annotation Mask (EDT > 0.15)",
        visible=False,
        opacity=0.5,
    )
    viewer.add_labels(
        (edt > 0.85).astype(int),
        name="Ground Truth Centerlines (EDT > 0.85)",
        visible=False,
        opacity=0.7,
    )

    if is_2d and visibility is not None:
        viewer.add_image(
            visibility,
            name="Visibility Target",
            colormap="inferno",
            visible=False,
            opacity=0.7,
        )
        viewer.add_labels(
            (visibility > 0.25).astype(int),
            name="Visibility > 0.25",
            visible=False,
            opacity=0.5,
        )
        viewer.add_labels(
            (visibility > 0.50).astype(int),
            name="Visibility > 0.50",
            visible=False,
            opacity=0.5,
        )
        viewer.add_image(
            edt * np.clip(visibility, visibility_floor, 1.0),
            name=f"EDT x Visibility (floor={visibility_floor:.2f})",
            colormap="magma",
            visible=False,
            opacity=0.7,
        )

    napari.run()


def show_dataset_sample(
    data_dir: str,
    split: str = "train",
    index: int = 0,
    random_sample: bool = False,
    visibility_floor: float = 0.25,
):
    sample_file, idx, total, sample_dir = _resolve_dataset_file(
        data_dir=data_dir,
        split=split,
        index=index,
        random_sample=random_sample,
    )
    print(
        f"Dataset directory: {sample_dir} | Selected sample {idx + 1}/{total}: "
        f"{os.path.basename(sample_file)}"
    )
    show_synthetic_data(sample_file, visibility_floor=visibility_floor)


def show_sted_debug(
    bounds,
    synth_depth,
    label_slab_thickness,
    label_slab_scale=1.3,
    annotation_weight_floor=0.25,
    soft_skeleton_alpha=0.35,
    visibility_weight_floor=0.03,
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
        visibility_weight_floor=visibility_weight_floor,
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
        f"visibility_weight_floor={debug_data.get('visibility_weight_floor', visibility_weight_floor):.2f}, "
        f"haze_regime={debug_data.get('haze_regime', 'n/a')}, "
        f"noise_level={debug_data['noise_level']:.4f}, "
        f"noise_n={debug_data['noise_level_normalized']:.4f}, "
        f"monomer_regime={debug_data.get('monomer_regime', 'n/a')}, "
        f"monomer_amp={debug_data.get('monomer_amplitude', 0.0):.4f})."
    )
    if save_path is not None:
        print(f"Saved debug summary to: {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize generated samples with Napari or inspect STED synthesis internals")
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--file", type=str, help="Path to a specific .pt dataset file")
    mode_group.add_argument("--data-dir", type=str, help="Dataset root or split directory containing .pt files")
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
    parser.add_argument(
        "--visibility_weight_floor",
        type=float,
        default=0.03,
        help="Minimum axial visibility weight for 2D STED visibility targets in --sted-debug mode.",
    )
    parser.add_argument("--split", type=str, choices=["train", "val", "test"], default="train", help="Split used with --data-dir")
    parser.add_argument("--index", type=int, default=0, help="Sample index used with --data-dir")
    parser.add_argument("--random-sample", action="store_true", help="Randomly pick a sample when using --data-dir")
    parser.add_argument(
        "--visibility_floor",
        type=float,
        default=0.25,
        help="Floor used in EDT x visibility inspection overlay.",
    )
    parser.add_argument("--export-dir", type=str, default=None, help="Export samples as PNG files to this directory (works with --file or --data-dir)")
    parser.add_argument("--num-samples", type=int, default=0, help="Number of samples to export (0 = all). Used with --data-dir --export-dir")
    parser.add_argument("--seed", type=int, default=None, help="Optional RNG seed for --sted-debug mode")
    parser.add_argument("--save", type=str, default=None, help="Optional path to save the STED debug summary figure")
    parser.add_argument("--no-show", action="store_true", help="Do not open the STED debug figure interactively")
    args = parser.parse_args()

    if args.file:
        if not os.path.exists(args.file):
            raise FileNotFoundError(f"Dataset file not found: {args.file}")
        if args.export_dir:
            os.makedirs(args.export_dir, exist_ok=True)
            name = os.path.splitext(os.path.basename(args.file))[0]
            out_path = os.path.join(args.export_dir, f"{name}.png")
            export_sample_png(args.file, out_path, visibility_floor=args.visibility_floor)
            print(f"Exported: {out_path}")
        else:
            show_synthetic_data(args.file, visibility_floor=args.visibility_floor)
    elif args.data_dir:
        if not os.path.exists(args.data_dir):
            raise FileNotFoundError(f"Dataset directory not found: {args.data_dir}")
        if args.export_dir:
            export_dataset_pngs(
                data_dir=args.data_dir,
                export_dir=args.export_dir,
                split=args.split,
                num_samples=args.num_samples,
                visibility_floor=args.visibility_floor,
            )
        else:
            show_dataset_sample(
                data_dir=args.data_dir,
                split=args.split,
                index=args.index,
                random_sample=args.random_sample,
                visibility_floor=args.visibility_floor,
            )
    else:
        if args.label_slab_scale <= 0.0:
            raise ValueError("--label_slab_scale must be greater than 0.")
        if args.annotation_weight_floor <= 0.0 or args.annotation_weight_floor > 1.0:
            raise ValueError("--annotation_weight_floor must be in the interval (0, 1].")
        if args.soft_skeleton_alpha < 0.0:
            raise ValueError("--soft_skeleton_alpha must be greater than or equal to 0.")
        if args.visibility_weight_floor <= 0.0 or args.visibility_weight_floor > 1.0:
            raise ValueError("--visibility_weight_floor must be in the interval (0, 1].")
        show_sted_debug(
            bounds=args.bounds,
            synth_depth=args.synth_depth,
            label_slab_thickness=args.label_slab_thickness,
            label_slab_scale=args.label_slab_scale,
            annotation_weight_floor=args.annotation_weight_floor,
            soft_skeleton_alpha=args.soft_skeleton_alpha,
            visibility_weight_floor=args.visibility_weight_floor,
            seed=args.seed,
            save_path=args.save,
            show=not args.no_show,
        )
