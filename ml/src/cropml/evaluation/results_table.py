"""
Aggregates per-run metrics (one row per model x method x target_domain
x seed) into the final results table: mean +/- std over seeds, per the
phase's 'done when' criterion.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass(frozen=True)
class RunResult:
    model: str              # e.g. "mobilenet_v3"
    method: str              # e.g. "softmax", "temperature_scaling", "mc_dropout", "deep_ensemble"
    target_domain: str       # e.g. "plantvillage_test", "plantdoc", "fieldplant"
    seed: int
    metrics: dict[str, float] = field(default_factory=dict)


def build_results_table(results: list[RunResult]) -> pd.DataFrame:
    """One row per RunResult - the raw, unaggregated table."""
    if not results:
        raise ValueError("results list is empty")
    rows = []
    for r in results:
        row = {"model": r.model, "method": r.method, "target_domain": r.target_domain, "seed": r.seed}
        row.update(r.metrics)
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate_over_seeds(
    raw_df: pd.DataFrame, metric_cols: list[str] | None = None
) -> pd.DataFrame:
    """Groups by (model, method, target_domain), computes mean/std of each
    metric across seeds. One row per (model, method, target_domain)."""
    group_cols = ["model", "method", "target_domain"]
    if metric_cols is None:
        metric_cols = [c for c in raw_df.columns if c not in (*group_cols, "seed")]

    grouped = raw_df.groupby(group_cols)
    agg = grouped[metric_cols].agg(["mean", "std", "count"])
    agg.columns = [f"{metric}_{stat}" for metric, stat in agg.columns]
    agg = agg.reset_index()
    return agg


def format_mean_std(agg_df: pd.DataFrame, metric: str, decimals: int = 3) -> pd.Series:
    """Formats one metric's mean +/- std as a display string per row.
    Single-seed rows (std is NaN) show '(n=1)' instead of a misleading
    '± nan' or a fabricated '± 0.000' that implies more seeds than exist."""
    mean_col, std_col, count_col = f"{metric}_mean", f"{metric}_std", f"{metric}_count"

    def _format_row(row: pd.Series) -> str:
        mean_val = row[mean_col]
        std_val = row[std_col]
        count_val = int(row[count_col])
        if pd.isna(std_val) or count_val < 2:
            return f"{mean_val:.{decimals}f} (n={count_val})"
        return f"{mean_val:.{decimals}f} \u00b1 {std_val:.{decimals}f}"

    return agg_df.apply(_format_row, axis=1)


def build_display_table(
    agg_df: pd.DataFrame, metrics: list[str], decimals: int = 3
) -> pd.DataFrame:
    """A human-readable table: one 'mean ± std (n)' string column per
    requested metric, alongside the model/method/target_domain identifiers."""
    display = agg_df[["model", "method", "target_domain"]].copy()
    for metric in metrics:
        display[metric] = format_mean_std(agg_df, metric, decimals=decimals)
    return display


def to_markdown_table(display_df: pd.DataFrame) -> str:
    """Manual markdown table formatter - avoids adding the `tabulate`
    dependency that pandas' built-in .to_markdown() requires."""
    columns = list(display_df.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for _, row in display_df.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in columns) + " |")
    return "\n".join(lines)




