import argparse
import csv
import os
from typing import Dict, List

import numpy as np
import torch

from src.sted_calibration import COMPARISON_METRICS, compute_image_metrics, load_calibration_profile


def _tensor_to_image(volume) -> np.ndarray:
    if torch.is_tensor(volume):
        arr = volume.detach().cpu().numpy()
    else:
        arr = np.asarray(volume)

    arr = np.squeeze(arr)
    if arr.ndim != 2:
        raise ValueError(f"Expected synthetic 2D image after squeeze, got shape {arr.shape}.")
    return np.asarray(arr, dtype=np.float64)


def _discover_samples(data_dir: str, split: str) -> List[str]:
    split_dir = os.path.join(data_dir, split)
    if not os.path.isdir(split_dir):
        raise FileNotFoundError(f"Split directory not found: {split_dir}")
    return sorted(os.path.join(split_dir, name) for name in os.listdir(split_dir) if name.endswith(".pt"))


def _write_csv(rows: List[Dict[str, object]], path: str) -> None:
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _summary_value(profile: Dict[str, object], metric: str, quantile_key: str) -> float:
    return float(profile["global"]["metrics"][metric][quantile_key])


def compare(args) -> None:
    profile = load_calibration_profile(args.profile)
    sample_paths = _discover_samples(args.data_dir, args.split)
    if args.max_samples > 0:
        sample_paths = sample_paths[: args.max_samples]
    if not sample_paths:
        raise FileNotFoundError(f"No .pt samples found in {os.path.join(args.data_dir, args.split)}")

    rows: List[Dict[str, object]] = []
    for index, path in enumerate(sample_paths, start=1):
        record = torch.load(path, map_location="cpu", weights_only=False)
        image = _tensor_to_image(record["volume"])
        rows.append(compute_image_metrics(image, source=path, row_type="synthetic"))
        if index % max(1, len(sample_paths) // 10) == 0:
            print(f"[{index}/{len(sample_paths)}] measured {os.path.basename(path)}")

    if args.output_csv:
        os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
        _write_csv(rows, args.output_csv)

    print("\nSynthetic vs real profile:")
    failures = []
    for metric in COMPARISON_METRICS:
        if metric not in profile["global"]["metrics"]:
            continue
        synthetic_values = np.asarray([float(row[metric]) for row in rows if metric in row], dtype=np.float64)
        if synthetic_values.size == 0:
            continue

        syn_median = float(np.median(synthetic_values))
        real_q10 = _summary_value(profile, metric, "q010")
        real_q25 = _summary_value(profile, metric, "q025")
        real_q50 = _summary_value(profile, metric, "q050")
        real_q75 = _summary_value(profile, metric, "q075")
        real_q90 = _summary_value(profile, metric, "q090")
        status = "PASS" if real_q10 <= syn_median <= real_q90 else "WARN"
        if status != "PASS":
            failures.append(metric)
        print(
            f"  {metric:22s} syn_q50={syn_median:.6f} "
            f"real_q25/q50/q75={real_q25:.6f}/{real_q50:.6f}/{real_q75:.6f} {status}"
        )

    if failures:
        print(f"\nMetrics outside real q10-q90 band: {', '.join(failures)}")
    else:
        print("\nAll compared synthetic medians are inside the real q10-q90 band.")


def add_compare_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--profile", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--output_csv", type=str, default="")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare generated synthetic STED samples against a real-data profile.")
    add_compare_arguments(parser)
    compare(parser.parse_args())
