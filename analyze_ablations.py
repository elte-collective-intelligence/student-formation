import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class RunRecord:
    # A W&B run record with its directory, config, and summary.
    run_dir: Path
    config: dict[str, Any]
    summary: dict[str, Any]


def _find_wandb_runs(root: Path) -> list[RunRecord]:
    records: list[RunRecord] = []

    # Find W&B run summary and config: wandb/run-*/files/{wandb-summary.json,config.yaml}
    for summary_path in root.rglob("wandb-summary.json"):
        if summary_path.name != "wandb-summary.json":
            continue
        files_dir = summary_path.parent
        config_path = files_dir / "config.yaml"
        if not config_path.exists():
            # Config can be missing; still allow summary-only.
            config: dict[str, Any] = {}
        else:
            config_raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
            config = _flatten_wandb_config(config_raw)

        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            # Ignore corrupted/partial files.
            continue

        records.append(
            RunRecord(run_dir=files_dir.parent, config=config, summary=summary)
        )

    return records


def _flatten_wandb_config(cfg: dict[str, Any]) -> dict[str, Any]:

    # Recursively flatten W&B config structure.
    def unpack(node: Any) -> Any:
        # W&B stores leaf values as {"value": <actual>}.
        if isinstance(node, dict) and set(node.keys()) == {"value"}:
            return node["value"]
        return node

    def rec(prefix: str, node: Any, out: dict[str, Any]) -> None:
        node = unpack(node)
        # Recursively flatten into dicts.
        if isinstance(node, dict):
            for k, v in node.items():
                rec(f"{prefix}.{k}" if prefix else str(k), v, out)
        else:
            out[prefix] = node

    out: dict[str, Any] = {}
    rec("", cfg, out)
    return out


def _to_float(value: Any) -> float | None:
    # Solve previous change with storing numbers as strings (due to UOM).
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        # Extract first float-looking number.
        m = (re.compile(r"-?\d+(?:\.\d+)?")).search(value)
        if not m:
            return None
        try:
            return float(m.group(0))
        except ValueError:
            return None
    return None


def _group_key(record: RunRecord, key: str) -> Any:
    # Prefer config, fall back to summary if needed.
    if key in record.config:
        return record.config.get(key)
    return record.summary.get(key)


def _parse_where_args(where_args: list[str]) -> dict[str, str]:
    # Convert filters from the CLI.
    filters: dict[str, str] = {}
    for raw in where_args:
        if "=" not in raw:
            raise ValueError(f"Invalid --where filter '{raw}'. Expected key=value.")
        key, value = raw.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            raise ValueError(f"Invalid --where filter '{raw}'. Key is empty.")
        filters[key] = value
    return filters


def _record_matches_filters(record: RunRecord, filters: dict[str, str]) -> bool:
    # Check each filter key/value pair.
    for key, expected in filters.items():
        actual = _group_key(record, key)
        if actual is None:
            return False
        if str(actual) != expected:
            return False
        
    return True


def _mean_std(values: list[float]) -> tuple[float, float] | tuple[None, None]:
    # Compute mean and sample standard deviation.
    if not values:
        return (None, None)
    
    mu = sum(values) / len(values)
    if len(values) == 1:
        return (mu, 0.0)
    
    var = sum((x - mu) ** 2 for x in values) / (len(values) - 1)
    return (mu, var**0.5)


def main() -> int:
    # CLI helper description
    ap = argparse.ArgumentParser(
        description=(
            "Summarize W&B (offline) runs for ablation reporting.\n"
        )
    )
    #Grouping/seed/filtering args
    ap.add_argument(
        "--group",
        required=True,
        help=(
            "Grouping W&B config e.g: env.assignment_method or algo.entropy_coef"
        ),
    )
    ap.add_argument(
        "--seed-key",
        default="base.seed",
        help="W&B config key for reproducible seed",
    )
    ap.add_argument(
        "--where",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help=(
            "Filters to aggregate only the needed runs e.g: --where env.assignment_method=greedy --where base.seed=0"
        ),
    )

    # Sweep filtering args
    sweep_group = ap.add_mutually_exclusive_group(required=True)
    sweep_group.add_argument(
        "--sweep-id",
        default=None,
        help=(
            "Aggregate only runs from this sweep id. "
            "Required unless --all-sweeps is set."
        ),
    )
    sweep_group.add_argument(
        "--all-sweeps",
        action="store_true",
        help="Aggregate across all sweeps (no sweep_id filtering).",
    )
    

    args = ap.parse_args()
    root = Path.cwd()
    # Define metrics to extract => potential filtering later
    metrics=[
        "Evaluation/Boundary_Error_Mean",
        "Evaluation/Boundary_Error_Max",
        "Evaluation/Collision_Rate_Pct",
        "Evaluation/Uniformity_Coefficient",
        "Reward/MeanRewardInBatch",
    ]

    runs = _find_wandb_runs(root)
    if not runs:
        print("No W&B runs found at the given root.")
        return 2

    # Apply filters
    try:
        filters = _parse_where_args(args.where)
    except ValueError as e:
        print(str(e))
        return 2

    # Sweep filtering for possibility of running both all runs aggregated or specific sweep.
    sweep_key = "base.sweep_id"
    if not args.all_sweeps:
        if sweep_key in filters and str(filters[sweep_key]) != str(args.sweep_id):
            print(f"Do not set {sweep_key} in --where; use --sweep-id instead.")
            return 2
        filters[sweep_key] = str(args.sweep_id)

    if filters:
        runs = [r for r in runs if _record_matches_filters(r, filters)]
        if not runs:
            print(f"No runs matched filters: {filters}")
            return 2


    # Extract run data
    rows: list[dict[str, Any]] = []
    for r in runs:
        group_val = _group_key(r, args.group)
        seed_val = _group_key(r, args.seed_key)

        row: dict[str, Any] = {
            "run_dir": str(r.run_dir),
            args.group: group_val,
            args.seed_key: seed_val,
        }
        for m in metrics:
            row[m] = r.summary.get(m)
        rows.append(row)

    # Store data to csv
    out_path = Path("runs.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Written run CSV: {out_path}")

    # Group and aggregate.
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        key = str(row.get(args.group))
        grouped.setdefault(key, []).append(row)

    print(f"Found {len(rows)} runs under: {root}\n\n")
    print("Summary of the ablation analysis:")

    # Print group stats
    grouped_items = sorted(grouped.items(), key=lambda item: item[0])
    for group_value, runs_in_group in grouped_items:
        seed_values = [str(run_row.get(args.seed_key)) for run_row in runs_in_group]
        seed_list = ", ".join(seed_values)
        print(
            f"\n- {args.group} = {group_value} "
            f"(n={len(runs_in_group)}, seeds={seed_list})"
        )
        
        # Process each metric
        for metric_name in metrics:
            metric_values = [
                _to_float(run_row.get(metric_name)) for run_row in runs_in_group
            ]
            metric_values = [v for v in metric_values if v is not None]

            mean, std = _mean_std(metric_values)
            if mean is None:
                print(f"The metric: {metric_name} is missing!")
            else:
                print(f"The value of metric (mean/std): {metric_name} is {mean:.4f} / {std:.4f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
