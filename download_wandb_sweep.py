import argparse
import csv
import json
import os

import wandb


def _is_scalar(value):
    return isinstance(value, (str, int, float, bool)) or value is None


def main():
    parser = argparse.ArgumentParser(description="Download a W&B sweep run table to CSV and JSON.")
    parser.add_argument("--entity", required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--sweep_id", required=True)
    parser.add_argument("--output_dir", default="reports/wandb")
    args = parser.parse_args()

    api = wandb.Api()
    sweep = api.sweep(f"{args.entity}/{args.project}/{args.sweep_id}")

    os.makedirs(args.output_dir, exist_ok=True)
    rows = []
    for run in sweep.runs:
        row = {
            "run_id": run.id,
            "run_name": run.name,
            "state": run.state,
            "created_at": str(run.created_at),
            "url": run.url,
        }

        for key, value in run.config.items():
            if key.startswith("_"):
                continue
            row[f"config.{key}"] = value if _is_scalar(value) else json.dumps(value, sort_keys=True)

        for key, value in dict(run.summary).items():
            if _is_scalar(value):
                row[f"summary.{key}"] = value

        rows.append(row)

    fieldnames = sorted({key for row in rows for key in row})
    csv_path = os.path.join(args.output_dir, f"sweep_{args.sweep_id}_runs.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    meta = {
        "sweep_id": sweep.id,
        "sweep_name": sweep.name,
        "entity": sweep.entity,
        "project": sweep.project,
        "url": sweep.url,
        "run_count": len(rows),
        "states": {state: sum(1 for row in rows if row["state"] == state) for state in sorted({row["state"] for row in rows})},
    }
    json_path = os.path.join(args.output_dir, f"sweep_{args.sweep_id}_meta.json")
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2, sort_keys=True)

    print(csv_path)
    print(json_path)
    print(json.dumps(meta, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
