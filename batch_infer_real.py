import argparse
import csv
import json
import os
import random
from typing import Dict, List, Sequence

import numpy as np

from src.real_inference import (
    add_inference_arguments,
    build_output_paths,
    load_image_for_inference,
    load_optional_profile,
    load_saved_output_arrays,
    load_sted_model,
    load_summary_json,
    outputs_exist,
    render_preview_panel,
    resolve_output_base,
    run_real_image_inference,
    save_inference_outputs,
    save_summary_json,
    summarize_inference_result,
)
from src.sted_calibration import parse_sted_filename


def _discover_tiff_files(input_dir: str, recursive: bool) -> List[str]:
    files = []
    if recursive:
        walker = os.walk(input_dir)
    else:
        walker = [(input_dir, [], os.listdir(input_dir))]

    for root, _, names in walker:
        for name in names:
            if name.lower().endswith((".tif", ".tiff")):
                files.append(os.path.join(root, name))
    files.sort()
    return files


def _write_manifest(rows: Sequence[Dict[str, object]], path: str) -> str:
    fieldnames = sorted({key for row in rows for key in row.keys()})
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


def _numeric_values(rows: Sequence[Dict[str, object]], key: str) -> List[float]:
    values = []
    for row in rows:
        value = row.get(key)
        if isinstance(value, (int, float, np.integer, np.floating)):
            values.append(float(value))
    return values


def _metric_summary(rows: Sequence[Dict[str, object]], key: str) -> Dict[str, float]:
    values = _numeric_values(rows, key)
    if not values:
        return {"mean": 0.0, "median": 0.0, "p90": 0.0}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p90": float(np.percentile(arr, 90.0)),
    }


def _build_run_summary(rows: Sequence[Dict[str, object]]) -> Dict[str, object]:
    success_rows = [row for row in rows if row.get("status") == "ok"]
    failed_rows = [row for row in rows if row.get("status") != "ok"]

    summary = {
        "image_count": len(rows),
        "success_count": len(success_rows),
        "error_count": len(failed_rows),
        "cache_hit_count": int(sum(int(row.get("cache_hit", 0)) for row in rows)),
        "overall": {},
        "groups": {"condition": {}, "div": {}},
        "flagged_examples": {},
    }

    for key in (
        "streamline_count",
        "skeleton_fraction",
        "raw_skeleton_contrast",
        "pred_edt_p99",
        "pred_vis_p99",
        "input_profile_oob_count",
        "total_seconds",
    ):
        summary["overall"][key] = _metric_summary(success_rows, key)

    for group_key in ("condition", "div"):
        values = sorted({row.get(group_key) for row in success_rows if row.get(group_key) not in (None, "", -1)})
        for value in values:
            group_rows = [row for row in success_rows if row.get(group_key) == value]
            summary["groups"][group_key][str(value)] = {
                "count": len(group_rows),
                "streamline_count": _metric_summary(group_rows, "streamline_count"),
                "skeleton_fraction": _metric_summary(group_rows, "skeleton_fraction"),
                "raw_skeleton_contrast": _metric_summary(group_rows, "raw_skeleton_contrast"),
                "pred_vis_p99": _metric_summary(group_rows, "pred_vis_p99"),
            }

    def _top_examples(key: str, largest: bool, limit: int = 5) -> List[Dict[str, object]]:
        ordered = [
            row for row in success_rows
            if isinstance(row.get(key), (int, float, np.integer, np.floating))
        ]
        ordered.sort(key=lambda row: float(row[key]), reverse=largest)
        return [
            {
                "source": row["source"],
                key: float(row[key]),
                "condition": row.get("condition", "unknown"),
                "div": row.get("div", -1),
            }
            for row in ordered[:limit]
        ]

    summary["flagged_examples"] = {
        "densest_predictions": _top_examples("skeleton_fraction", largest=True),
        "emptiest_predictions": _top_examples("skeleton_fraction", largest=False),
        "lowest_support": _top_examples("raw_skeleton_contrast", largest=False),
        "highest_profile_oob": _top_examples("input_profile_oob_count", largest=True),
    }
    return summary


def _write_json(data: Dict[str, object], path: str) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
    return path


def _unique_append(selected: List[Dict[str, object]], row: Dict[str, object], seen: set) -> None:
    source = row.get("source")
    if source in seen:
        return
    selected.append(row)
    seen.add(source)


def _pick_extreme(rows: Sequence[Dict[str, object]], key: str, largest: bool) -> Dict[str, object]:
    candidates = [row for row in rows if isinstance(row.get(key), (int, float, np.integer, np.floating))]
    if not candidates:
        return {}
    return sorted(candidates, key=lambda row: float(row[key]), reverse=largest)[0]


def _select_preview_rows(rows: Sequence[Dict[str, object]], preview_count: int, preview_seed: int) -> List[Dict[str, object]]:
    success_rows = [row for row in rows if row.get("status") == "ok"]
    if preview_count <= 0 or not success_rows:
        return []

    selected: List[Dict[str, object]] = []
    seen = set()

    for key, largest in (
        ("skeleton_fraction", False),
        ("skeleton_fraction", True),
        ("raw_skeleton_contrast", False),
        ("pred_vis_p99", False),
        ("pred_edt_p99", True),
        ("input_profile_oob_count", True),
    ):
        row = _pick_extreme(success_rows, key, largest)
        if row:
            _unique_append(selected, row, seen)

    conditions = sorted({row.get("condition") for row in success_rows if row.get("condition") not in (None, "", "unknown")})
    for condition in conditions:
        group_rows = [row for row in success_rows if row.get("condition") == condition]
        if not group_rows:
            continue
        median_target = float(np.median([float(row["skeleton_fraction"]) for row in group_rows]))
        representative = min(group_rows, key=lambda row: abs(float(row["skeleton_fraction"]) - median_target))
        _unique_append(selected, representative, seen)

    if len(selected) < preview_count:
        rng = random.Random(preview_seed)
        remaining = [row for row in success_rows if row.get("source") not in seen]
        rng.shuffle(remaining)
        for row in remaining:
            _unique_append(selected, row, seen)
            if len(selected) >= preview_count:
                break

    return selected[:preview_count]


def _error_row(image_path: str, output_paths: Dict[str, str], error: Exception) -> Dict[str, object]:
    metadata = parse_sted_filename(image_path)
    row = {
        "source": image_path,
        "name": metadata["name"],
        "condition": metadata["condition"],
        "div": metadata["div"] if metadata["div"] is not None else -1,
        "replicate": metadata["replicate"],
        "series": metadata["series"] if metadata["series"] is not None else -1,
        "status": "error",
        "error_message": str(error),
    }
    for key, value in output_paths.items():
        row[f"output_{key}"] = value
    return row


def main(args):
    if args.dim != 2:
        raise ValueError("The real-data batch inference path is 2D only. Use --dim 2.")
    if args.downsample > 1.0:
        print("Warning: --downsample is ignored by the 2D STED tiled inference path.")

    image_paths = _discover_tiff_files(args.input_dir, recursive=args.recursive)
    if args.max_images > 0:
        image_paths = image_paths[: args.max_images]
    if not image_paths:
        raise FileNotFoundError(f"No .tif/.tiff files found in {args.input_dir}")

    print(f"Discovered {len(image_paths)} real TIFF images.")
    model, device = load_sted_model(
        model_path=args.model_path,
        base_filters=args.base_filters,
        device_spec=args.device,
    )
    print(f"Executing batch inference on: {device}")
    calibration_profile = load_optional_profile(args.profile)

    rows: List[Dict[str, object]] = []
    for index, image_path in enumerate(image_paths, start=1):
        base_path = resolve_output_base(image_path, output_dir=args.output_dir, input_root=args.input_dir)
        output_paths = build_output_paths(base_path)

        if not args.force and outputs_exist(base_path):
            row = load_summary_json(output_paths["summary"])
            row["cache_hit"] = 1
            rows.append(row)
            print(f"[{index}/{len(image_paths)}] cached {os.path.basename(image_path)}")
            continue

        try:
            image = load_image_for_inference(image_path, dim=args.dim)
            result = run_real_image_inference(
                model,
                image=image,
                image_path=image_path,
                device=device,
                tile_size=args.tile_size,
                tile_overlap=args.tile_overlap,
                min_edt=args.min_edt,
                visibility_floor=args.visibility_floor,
                use_amp=not args.no_amp,
            )
            saved_paths = save_inference_outputs(result, base_path)
            row = summarize_inference_result(
                result,
                output_paths=saved_paths,
                calibration_profile=calibration_profile,
            )
            row["cache_hit"] = 0
            save_summary_json(row, saved_paths["summary"])
            rows.append(row)
            print(
                f"[{index}/{len(image_paths)}] traced {os.path.basename(image_path)} "
                f"| streamlines={row['streamline_count']} "
                f"| skeleton_fraction={float(row['skeleton_fraction']):.6f}"
            )
        except Exception as error:
            row = _error_row(image_path, output_paths, error)
            row["cache_hit"] = 0
            save_summary_json(row, output_paths["summary"])
            rows.append(row)
            print(f"[{index}/{len(image_paths)}] ERROR {os.path.basename(image_path)}: {error}")

    manifest_path = _write_manifest(rows, os.path.join(args.output_dir, args.manifest_name))
    summary_path = _write_json(_build_run_summary(rows), os.path.join(args.output_dir, args.run_summary_name))

    preview_rows = _select_preview_rows(rows, preview_count=args.preview_count, preview_seed=args.preview_seed)
    preview_dir = os.path.join(args.output_dir, args.preview_dirname)
    for preview_index, row in enumerate(preview_rows, start=1):
        base_path = resolve_output_base(row["source"], output_dir=args.output_dir, input_root=args.input_dir)
        arrays = load_saved_output_arrays(base_path)
        preview_name = f"{preview_index:02d}_{os.path.basename(base_path)}_preview.png"
        preview_path = os.path.join(preview_dir, preview_name)
        render_preview_panel(
            image=load_image_for_inference(row["source"], dim=args.dim),
            skeleton=arrays["skeleton"],
            edt_map=arrays["pred_edt"],
            visibility_map=arrays["pred_vis"],
            orientation_confidence=arrays["pred_orient_conf"],
            out_path=preview_path,
            title=os.path.basename(row["source"]),
            metrics=row,
        )
        row["output_preview"] = preview_path
        if row.get("output_summary"):
            save_summary_json(row, row["output_summary"])

    if preview_rows:
        _write_manifest(rows, manifest_path)

    success_rows = [row for row in rows if row.get("status") == "ok"]
    print(f"Saved manifest: {manifest_path}")
    print(f"Saved run summary: {summary_path}")
    print(
        f"Completed batch inference: success={len(success_rows)} "
        f"errors={len(rows) - len(success_rows)} "
        f"cache_hits={sum(int(row.get('cache_hit', 0)) for row in rows)}"
    )
    if success_rows:
        print(
            "Median outputs: "
            f"streamlines={np.median(_numeric_values(success_rows, 'streamline_count')):.1f} "
            f"skeleton_fraction={np.median(_numeric_values(success_rows, 'skeleton_fraction')):.6f} "
            f"contrast={np.median(_numeric_values(success_rows, 'raw_skeleton_contrast')):.3f}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run tiled inference on a directory of real 2D STED TIFF images.")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing real TIFF images.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory where inference artifacts will be written.")
    parser.add_argument("--recursive", action="store_true", help="Recursively discover TIFF files under --input_dir.")
    parser.add_argument("--profile", type=str, default="", help="Optional real-data calibration profile JSON for OOD context.")
    parser.add_argument("--manifest_name", type=str, default="manifest.csv")
    parser.add_argument("--run_summary_name", type=str, default="run_summary.json")
    parser.add_argument("--preview_dirname", type=str, default="qa_previews")
    parser.add_argument("--preview_count", type=int, default=16)
    parser.add_argument("--preview_seed", type=int, default=0)
    parser.add_argument("--max_images", type=int, default=0, help="Optional cap on the number of images to process.")
    parser.add_argument("--force", action="store_true", help="Re-run inference even when outputs and per-image summaries already exist.")
    add_inference_arguments(
        parser,
        include_image_path=False,
        include_output_dir=False,
        include_visualize=False,
        include_preview_options=False,
    )
    args = parser.parse_args()
    main(args)
