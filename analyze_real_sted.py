import argparse
import csv
import json
import os
from typing import Dict, List

import numpy as np
import tifffile

from src.sted_calibration import (
    DEFAULT_PATCH_SIZE,
    build_calibration_profile,
    compute_image_metrics,
    iter_patches,
    parse_sted_filename,
)


def _discover_tiffs(real_dir: str) -> List[str]:
    paths = []
    for root, _, files in os.walk(real_dir):
        for name in files:
            if name.lower().endswith((".tif", ".tiff")):
                paths.append(os.path.join(root, name))
    return sorted(paths)


def _write_csv(rows: List[Dict[str, object]], path: str) -> None:
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _save_qa_contact_sheet(rows: List[Dict[str, object]], image_cache: Dict[str, np.ndarray], output_path: str, max_images: int) -> None:
    if max_images <= 0 or not rows:
        return

    import matplotlib.pyplot as plt

    image_rows = [row for row in rows if row.get("row_type") == "image"]
    if not image_rows:
        return

    sparse = sorted(image_rows, key=lambda row: float(row.get("foreground_fraction", 0.0)))[: max_images // 2]
    dense = sorted(image_rows, key=lambda row: float(row.get("foreground_fraction", 0.0)), reverse=True)[: max_images - len(sparse)]
    selected = sparse + dense

    fig, axes = plt.subplots(len(selected), 3, figsize=(9, max(3, 2.4 * len(selected))))
    if len(selected) == 1:
        axes = np.asarray([axes])

    for row_axes, row in zip(axes, selected):
        source = str(row["source"])
        image = image_cache[source]
        threshold = float(row["foreground_threshold"])
        mask = image > threshold

        row_axes[0].imshow(image, cmap="gray")
        row_axes[0].set_title(os.path.basename(source)[:42])
        row_axes[1].imshow(mask, cmap="gray")
        row_axes[1].set_title(f"fg={float(row['foreground_fraction']):.4f}")
        row_axes[2].hist(image.ravel(), bins=80, color="black")
        row_axes[2].axvline(threshold, color="red", linewidth=1)
        row_axes[2].set_title(f"p99={float(row['p99']):.3f}")
        for axis in row_axes:
            axis.set_xticks([])
            axis.set_yticks([])

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def analyze_real_sted(args) -> None:
    os.makedirs(args.output_dir, exist_ok=True)
    tiff_paths = _discover_tiffs(args.real_dir)
    if args.max_files > 0:
        tiff_paths = tiff_paths[: args.max_files]
    if not tiff_paths:
        raise FileNotFoundError(f"No TIFF files found under {args.real_dir}")

    rows: List[Dict[str, object]] = []
    image_cache: Dict[str, np.ndarray] = {}
    for index, path in enumerate(tiff_paths, start=1):
        raw = tifffile.imread(path)
        image = np.squeeze(raw)
        if image.ndim != 2:
            raise ValueError(f"{path} did not resolve to a 2D image after squeeze: {image.shape}")

        normalized = image.astype(np.float64)
        if np.issubdtype(image.dtype, np.integer):
            normalized /= float(np.iinfo(image.dtype).max)
        else:
            normalized /= max(float(np.nanpercentile(normalized, 99.9)), 1.0)
        normalized = np.clip(normalized, 0.0, 1.0)
        image_cache[path] = normalized

        metadata = parse_sted_filename(path)
        image_row = compute_image_metrics(
            image,
            source=path,
            row_type="image",
            min_component_area=args.min_component_area,
            metadata=metadata,
        )
        rows.append(image_row)

        patch_index = 0
        for y0, x0, patch in iter_patches(image, args.patch_size, stride=args.patch_stride):
            patch_row = compute_image_metrics(
                patch,
                source=path,
                row_type="patch",
                patch_index=patch_index,
                patch_origin=(y0, x0),
                min_component_area=args.min_component_area,
                metadata=metadata,
            )
            rows.append(patch_row)
            patch_index += 1

        if index % max(1, len(tiff_paths) // 10) == 0:
            print(f"[{index}/{len(tiff_paths)}] analyzed {os.path.basename(path)}")

    stats_path = os.path.join(args.output_dir, "sted_real_stats.csv")
    profile_path = os.path.join(args.output_dir, "sted_real_profile.json")
    qa_path = os.path.join(args.output_dir, "sted_real_qa.png")

    _write_csv(rows, stats_path)
    profile = build_calibration_profile(rows, source_dir=args.real_dir, patch_size=args.patch_size)
    with open(profile_path, "w", encoding="utf-8") as handle:
        json.dump(profile, handle, indent=2, sort_keys=True)

    _save_qa_contact_sheet(rows, image_cache, qa_path, max_images=args.qa_samples)

    global_metrics = profile["global"]["metrics"]
    print(f"Wrote {stats_path}")
    print(f"Wrote {profile_path}")
    if args.qa_samples > 0:
        print(f"Wrote {qa_path}")
    print("Calibration summary:")
    print(f"  images={profile['image_count']} patches={profile['patch_count']}")
    print(f"  foreground q050={global_metrics['foreground_fraction']['q050']:.6f}")
    print(f"  skeleton   q050={global_metrics['skeleton_fraction']['q050']:.6f}")
    print(f"  p99        q050={global_metrics['p99']['q050']:.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze real 2D STED TIFFs and build a synthesizer calibration profile.")
    parser.add_argument("--real_dir", type=str, required=True, help="Directory containing real STED TIFF images.")
    parser.add_argument("--output_dir", type=str, default="reports/sted_real")
    parser.add_argument("--patch_size", type=int, default=DEFAULT_PATCH_SIZE)
    parser.add_argument("--patch_stride", type=int, default=None)
    parser.add_argument("--min_component_area", type=int, default=8)
    parser.add_argument("--max_files", type=int, default=0)
    parser.add_argument("--qa_samples", type=int, default=8)
    analyze_real_sted(parser.parse_args())
