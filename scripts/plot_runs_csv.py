import argparse
import csv
import os
import statistics
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import matplotlib.pyplot as plt


@dataclass(frozen=True)
class MetricSpec:
    column: str
    title: str
    lower_is_better: bool


METRICS: list[MetricSpec] = [
    MetricSpec(
        column="Evaluation/Boundary_Error_Mean",
        title="Boundary error (mean) ↓", # down arrow = lower is better
        lower_is_better=True,
    ),
    MetricSpec(
        column="Evaluation/Collision_Rate_Pct",
        title="Collision rate (%) ↓", # down arrow = lower is better
        lower_is_better=True,
    ),
    MetricSpec(
        column="Evaluation/Uniformity_Coefficient",
        title="Uniformity (CV) ↓", # down arrow = lower is better
        lower_is_better=True,
    ),
    MetricSpec(
        column="Reward/MeanRewardInBatch",
        title="Training reward (mean) ↑", # up arrow = higher is better
        lower_is_better=False,
    ),
]


def _safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _sort_key(value: str):
    as_float = _safe_float(value)
    if as_float is None:
        return (1, value)
    return (0, as_float)


def _format_sci(value: float) -> str:
    # Convert to number
    text = f"{value:.0e}"
    mantissa, exp = text.split("e")
    return f"{mantissa}e{int(exp)}"


def _format_group_label(group_value: str) -> str:
    # Format group label nicely for plotting
    as_float = _safe_float(group_value)
    if as_float is None:
        return group_value
    if as_float == 0:
        return "0"
    if abs(as_float) < 1e-2:
        return _format_sci(as_float)
    return f"{as_float:g}"


def _ensure_outdir(path: str) -> None:
    # Check output dir
    outdir = os.path.dirname(os.path.abspath(path))
    if outdir:
        os.makedirs(outdir, exist_ok=True)


def _read_runs(csv_path: str) -> list[dict[str, str]]:
    # Read CSV rows
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("CSV has no header row")
        return list(reader)


def plot_ablation_from_runs_csv(
    *,
    csv_path: str,
    group_key: str,
    out_path: str,
    title: str | None,
    exclude_reward: bool,
) -> str:
    # Read CSV
    rows = _read_runs(csv_path)
    if not rows:
        raise ValueError(f"No rows found in {csv_path!r}")

    # Check group exists
    if group_key not in rows[0]:
        raise ValueError(
            f"Group column {group_key!r} not found in {csv_path!r}. "
            f"Available columns: {sorted(rows[0].keys())}"
        )

    # Determine metrics to plot
    columns_present = set(rows[0].keys())
    metrics = [m for m in METRICS if m.column in columns_present]
    if exclude_reward:
        metrics = [m for m in metrics if m.column != "Reward/MeanRewardInBatch"]
    if not metrics:
        raise ValueError(
            "None of the expected metric columns were found. "
            f"Expected one of: {[m.column for m in METRICS]}"
        )

    values_by_group: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

    # Collect values by group
    for row in rows:
        group_value = str(row.get(group_key, "")).strip()
        if not group_value:
            continue

        for metric in metrics:
            value = _safe_float(row.get(metric.column))
            if value is None:
                continue
            values_by_group[group_value][metric.column].append(value)

    if not values_by_group:
        raise ValueError(f"No usable rows for group {group_key!r} in {csv_path!r}")

    # Sort group values
    group_values = sorted(values_by_group.keys(), key=_sort_key)
    group_labels = [_format_group_label(g) for g in group_values]

    # Create subplots (2x2)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    axes_list = [axes[0][0], axes[0][1], axes[1][0], axes[1][1]]

    # Legend in top-right corner
    fig.text(
        0.99,
        0.985,
        "μ Mean\nσ Std",
        ha="right",
        va="top",
        fontsize=9,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85, "edgecolor": "0.7"},
    )

    # Plot each metric
    for ax, metric in zip(axes_list, metrics, strict=False):
        means: list[float] = []
        stds: list[float] = []

        for g in group_values:
            values = values_by_group[g][metric.column]
            if values:
                means.append(statistics.mean(values))
                stds.append(statistics.stdev(values) if len(values) > 1 else 0.0)
            else:
                means.append(float("nan"))
                stds.append(0.0)

        # Plot bar chart
        x = list(range(len(group_values)))
        ax.bar(x, means)
        ax.set_title(metric.title)
        ax.set_xticks(x)
        ax.set_xticklabels(group_labels, rotation=15, ha="right")
        ax.grid(True, axis="y", alpha=0.3)

        # Make the scale easier to read.
        finite_upper = [m + s for (m, s) in zip(means, stds, strict=False) if m == m]
        finite_lower = [m - s for (m, s) in zip(means, stds, strict=False) if m == m]
        if finite_upper and finite_lower:
            ymax = max(finite_upper)
            ymin = min(finite_lower)
            
            # Shrink y axis for differences
            span = ymax - ymin
            if span == 0:
                span = abs(ymax) if ymax != 0 else 1.0
            pad = 0.05 * span
            ax.set_ylim(ymin - pad, ymax + pad)


        # Annotate mean and std on bars
        for xi, (m, s) in enumerate(zip(means, stds, strict=False)):
            if m == m:  # not NaN
                label = f"μ={m:.3g}\nσ={s:.2g}"
                ax.text(
                    xi,
                    m,
                    label,
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

    # Show only needed axes
    for ax in axes_list[len(metrics) :]:
        ax.axis("off")

    if title is None:
        title = f"Ablation results grouped by {group_key}"

    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0.02, 1, 0.95))

    _ensure_outdir(out_path)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot ablation charts from runs.csv produced by analyze_ablations.py")
    parser.add_argument("--csv", dest="csv_path", default="runs.csv", help="Path to runs.csv")
    parser.add_argument("--group", dest="group_key", required=True, help="Grouping column, e.g. algo.lr")
    parser.add_argument(
        "--out",
        dest="out_path",
        default=None,
        help="Output PNG path. Default: docs/charts/ablation_<group>.png",
    )
    parser.add_argument("--title", dest="title", default=None, help="Figure title")
    parser.add_argument(
        "--exclude-reward",
        action="store_true",
        help="Exclude the training reward subplot (keeps only Evaluation/* metrics).",
    )

    args = parser.parse_args()

    out_path = args.out_path
    if out_path is None:
        safe_group = args.group_key.replace("/", "_").replace(" ", "_")
        out_path = os.path.join("docs", "charts", f"ablation_{safe_group}.png")

    written = plot_ablation_from_runs_csv(
        csv_path=args.csv_path,
        group_key=args.group_key,
        out_path=out_path,
        title=args.title,
        exclude_reward=args.exclude_reward,
    )

    print(f"Wrote: {written}")


if __name__ == "__main__":
    main()
