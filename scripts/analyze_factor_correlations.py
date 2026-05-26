#!/usr/bin/env python3
"""Analyze factor redundancy using persisted daily snapshot bundles.

Inputs are CSV snapshot files produced by `run_cross_sectional_research.py`
(`*_daily_snapshots.csv`). The script computes static and rolling correlation
structures across factors and exports both machine-readable artifacts and plots.
"""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from predictor.research.errors import ResearchInputError


REQUIRED_SNAPSHOT_COLUMNS = {
    "date",
    "symbol",
    "factor_name",
    "raw_factor",
    "daily_ic",
}


def _try_import_matplotlib():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        return plt
    except ImportError:
        return None


def _pair_key(left: str, right: str) -> str:
    return f"{left}__{right}"


def _normalise_factor_filter(raw: str | None) -> set[str] | None:
    if not raw:
        return None
    names = {token.strip() for token in raw.split(",") if token.strip()}
    return names or None


def _load_snapshot_file(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = REQUIRED_SNAPSHOT_COLUMNS - set(frame.columns)
    if missing:
        raise ResearchInputError(
            f"snapshot file {path} missing columns: {sorted(missing)}"
        )
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame = frame.dropna(subset=["date", "symbol", "factor_name"])
    frame["symbol"] = frame["symbol"].astype(str)
    frame["factor_name"] = frame["factor_name"].astype(str)
    frame["raw_factor"] = pd.to_numeric(frame["raw_factor"], errors="coerce")
    frame["daily_ic"] = pd.to_numeric(frame["daily_ic"], errors="coerce")
    return frame


def load_factor_snapshots(
    *,
    input_dir: Path,
    factor_filter: set[str] | None,
) -> pd.DataFrame:
    """Load and concatenate snapshot files from an input directory."""
    files = sorted(input_dir.glob("*_daily_snapshots.csv"))
    if not files:
        raise ResearchInputError(
            f"no snapshot files found in {input_dir}; expected *_daily_snapshots.csv"
        )

    frames: List[pd.DataFrame] = []
    for file_path in files:
        frame = _load_snapshot_file(file_path)
        if factor_filter is not None:
            frame = frame[frame["factor_name"].isin(factor_filter)]
        if not frame.empty:
            frames.append(frame)

    if not frames:
        raise ResearchInputError("no snapshot rows remained after factor filtering")
    return pd.concat(frames, axis=0, ignore_index=True)


def build_factor_value_panel(snapshots: pd.DataFrame) -> pd.DataFrame:
    """Build (date, symbol) x factor panel of raw factor values."""
    panel = snapshots.pivot_table(
        index=["date", "symbol"],
        columns="factor_name",
        values="raw_factor",
        aggfunc="mean",
    )
    panel = panel.sort_index()
    if panel.empty or panel.shape[1] < 2:
        raise ResearchInputError("need at least 2 factors with valid raw values")
    return panel


def build_ic_panel(snapshots: pd.DataFrame) -> pd.DataFrame:
    """Build date x factor panel of daily IC values."""
    panel = snapshots.pivot_table(
        index="date",
        columns="factor_name",
        values="daily_ic",
        aggfunc="mean",
    ).sort_index()
    if panel.empty or panel.shape[1] < 2:
        raise ResearchInputError("need at least 2 factors with valid daily_ic values")
    return panel


def compute_static_correlations(
    factor_value_panel: pd.DataFrame,
    ic_panel: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compute static Pearson/Spearman factor and IC correlation matrices."""
    pearson = factor_value_panel.corr(method="pearson")
    spearman = factor_value_panel.corr(method="spearman")
    ic_corr = ic_panel.corr(method="pearson")
    return pearson, spearman, ic_corr


def _daily_pairwise_cross_sectional_corr(
    factor_value_panel: pd.DataFrame,
    *,
    method: str,
) -> pd.DataFrame:
    dates = sorted(factor_value_panel.index.get_level_values("date").unique())
    factors = list(factor_value_panel.columns)
    rows: List[dict[str, object]] = []

    for date in dates:
        cross_section = factor_value_panel.xs(date, level="date")
        row: dict[str, object] = {"date": pd.Timestamp(date)}

        for left, right in combinations(factors, 2):
            joined = cross_section[[left, right]].dropna()
            if len(joined) >= 3:
                left_std = float(joined[left].std(ddof=0))
                right_std = float(joined[right].std(ddof=0))
                if left_std > 0.0 and right_std > 0.0:
                    value = float(joined[left].corr(joined[right], method=method))
                else:
                    value = np.nan
            else:
                value = np.nan
            row[_pair_key(left, right)] = value

        rows.append(row)

    return pd.DataFrame(rows).set_index("date").sort_index()


def _rolling_average(panel: pd.DataFrame, window: int) -> pd.DataFrame:
    if window <= 1:
        raise ResearchInputError("rolling window must be > 1")
    return panel.rolling(window=window, min_periods=window).mean()


def _rolling_ic_correlation(ic_panel: pd.DataFrame, window: int) -> pd.DataFrame:
    factors = list(ic_panel.columns)
    rolling: Dict[str, pd.Series] = {}
    for left, right in combinations(factors, 2):
        rolling[_pair_key(left, right)] = ic_panel[left].rolling(window=window).corr(ic_panel[right])
    if not rolling:
        return pd.DataFrame(index=ic_panel.index)
    return pd.DataFrame(rolling, index=ic_panel.index)


def compute_rolling_correlations(
    factor_value_panel: pd.DataFrame,
    ic_panel: pd.DataFrame,
    *,
    window: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compute rolling Pearson/Spearman and rolling IC correlations."""
    daily_pearson = _daily_pairwise_cross_sectional_corr(factor_value_panel, method="pearson")
    daily_spearman = _daily_pairwise_cross_sectional_corr(factor_value_panel, method="spearman")

    rolling_pearson = _rolling_average(daily_pearson, window=window)
    rolling_spearman = _rolling_average(daily_spearman, window=window)
    rolling_ic = _rolling_ic_correlation(ic_panel, window=window)
    return rolling_pearson, rolling_spearman, rolling_ic


def _save_table(frame: pd.DataFrame, csv_path: Path, parquet_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(csv_path)
    try:
        import pyarrow  # noqa: F401
    except ImportError:
        return
    frame.to_parquet(parquet_path)


def _plot_heatmap(matrix: pd.DataFrame, *, title: str, output_path: Path) -> Path | None:
    plt = _try_import_matplotlib()
    if plt is None or matrix.empty:
        return None

    values = matrix.values.astype(float)
    values = np.nan_to_num(values, nan=0.0)
    vmax = float(np.max(np.abs(values))) if values.size else 1.0
    if vmax <= 0.0:
        vmax = 1.0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(values, cmap="coolwarm", vmin=-vmax, vmax=vmax)
    ax.set_title(title)
    ax.set_xticks(np.arange(len(matrix.columns)))
    ax.set_yticks(np.arange(len(matrix.index)))
    ax.set_xticklabels(matrix.columns, rotation=60, ha="right")
    ax.set_yticklabels(matrix.index)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=140)
    plt.close(fig)
    return output_path


def _plot_rolling_panel(panel: pd.DataFrame, *, title: str, output_path: Path) -> Path | None:
    plt = _try_import_matplotlib()
    if plt is None or panel.empty:
        return None

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 4))
    for col in panel.columns:
        ax.plot(panel.index, panel[col], linewidth=1.1, alpha=0.7)
    ax.axhline(0.0, color="black", linewidth=0.8, linestyle="--")
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Correlation")
    fig.tight_layout()
    fig.savefig(output_path, dpi=140)
    plt.close(fig)
    return output_path


def _extract_high_pairs(matrix: pd.DataFrame, *, threshold: float) -> List[dict[str, object]]:
    findings: List[dict[str, object]] = []
    cols = list(matrix.columns)
    for i, left in enumerate(cols):
        for right in cols[i + 1 :]:
            value = float(matrix.loc[left, right])
            if np.isfinite(value) and abs(value) >= threshold:
                findings.append(
                    {
                        "left": left,
                        "right": right,
                        "correlation": value,
                    }
                )
    findings.sort(key=lambda row: abs(float(row["correlation"])), reverse=True)
    return findings


def _extract_high_rolling_pairs(panel: pd.DataFrame, *, threshold: float) -> List[dict[str, object]]:
    findings: List[dict[str, object]] = []
    for pair_name in panel.columns:
        series = panel[pair_name].dropna()
        if series.empty:
            continue
        mean_abs = float(series.abs().mean())
        last = float(series.iloc[-1])
        if mean_abs >= threshold:
            left, right = pair_name.split("__", maxsplit=1)
            findings.append(
                {
                    "left": left,
                    "right": right,
                    "mean_abs_correlation": mean_abs,
                    "last_correlation": last,
                }
            )
    findings.sort(key=lambda row: float(row["mean_abs_correlation"]), reverse=True)
    return findings


def build_redundancy_report(
    *,
    static_pearson: pd.DataFrame,
    static_spearman: pd.DataFrame,
    ic_correlation: pd.DataFrame,
    rolling_pearson: pd.DataFrame,
    rolling_spearman: pd.DataFrame,
    rolling_ic: pd.DataFrame,
    threshold: float,
) -> dict[str, object]:
    """Build a machine-readable factor redundancy report."""
    return {
        "threshold": threshold,
        "static": {
            "pearson": _extract_high_pairs(static_pearson, threshold=threshold),
            "spearman": _extract_high_pairs(static_spearman, threshold=threshold),
            "ic": _extract_high_pairs(ic_correlation, threshold=threshold),
        },
        "rolling": {
            "pearson": _extract_high_rolling_pairs(rolling_pearson, threshold=threshold),
            "spearman": _extract_high_rolling_pairs(rolling_spearman, threshold=threshold),
            "ic": _extract_high_rolling_pairs(rolling_ic, threshold=threshold),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze redundancy across factor snapshot exports.")
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Directory containing *_daily_snapshots.csv files.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory for correlation artifacts (default: <input-dir>/factor_correlations).",
    )
    parser.add_argument(
        "--factors",
        type=str,
        help="Optional comma-separated factor filter.",
    )
    parser.add_argument(
        "--rolling-window",
        type=int,
        default=60,
        help="Rolling window for stability correlations (default: 60).",
    )
    parser.add_argument(
        "--redundancy-threshold",
        type=float,
        default=0.85,
        help="Absolute-correlation threshold for redundancy reporting.",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else (input_dir / "factor_correlations")
    factor_filter = _normalise_factor_filter(args.factors)

    snapshots = load_factor_snapshots(input_dir=input_dir, factor_filter=factor_filter)
    factor_value_panel = build_factor_value_panel(snapshots)
    ic_panel = build_ic_panel(snapshots)

    static_pearson, static_spearman, ic_correlation = compute_static_correlations(
        factor_value_panel,
        ic_panel,
    )
    rolling_pearson, rolling_spearman, rolling_ic = compute_rolling_correlations(
        factor_value_panel,
        ic_panel,
        window=args.rolling_window,
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    _save_table(
        static_pearson,
        output_dir / "static_pearson_correlation.csv",
        output_dir / "static_pearson_correlation.parquet",
    )
    _save_table(
        static_spearman,
        output_dir / "static_spearman_correlation.csv",
        output_dir / "static_spearman_correlation.parquet",
    )
    _save_table(
        ic_correlation,
        output_dir / "ic_correlation_matrix.csv",
        output_dir / "ic_correlation_matrix.parquet",
    )

    _save_table(
        rolling_pearson,
        output_dir / "rolling_pearson_correlation.csv",
        output_dir / "rolling_pearson_correlation.parquet",
    )
    _save_table(
        rolling_spearman,
        output_dir / "rolling_spearman_correlation.csv",
        output_dir / "rolling_spearman_correlation.parquet",
    )
    _save_table(
        rolling_ic,
        output_dir / "rolling_ic_correlation.csv",
        output_dir / "rolling_ic_correlation.parquet",
    )

    _plot_heatmap(
        static_pearson,
        title="Static Pearson Factor Correlation",
        output_path=output_dir / "static_pearson_correlation_heatmap.png",
    )
    _plot_heatmap(
        static_spearman,
        title="Static Spearman Factor Correlation",
        output_path=output_dir / "static_spearman_correlation_heatmap.png",
    )
    _plot_heatmap(
        ic_correlation,
        title="Static IC Correlation Matrix",
        output_path=output_dir / "ic_correlation_heatmap.png",
    )

    _plot_rolling_panel(
        rolling_pearson,
        title=f"Rolling Pearson Cross-Sectional Correlation ({args.rolling_window}D)",
        output_path=output_dir / "rolling_pearson_correlation.png",
    )
    _plot_rolling_panel(
        rolling_spearman,
        title=f"Rolling Spearman Cross-Sectional Correlation ({args.rolling_window}D)",
        output_path=output_dir / "rolling_spearman_correlation.png",
    )
    _plot_rolling_panel(
        rolling_ic,
        title=f"Rolling IC Correlation ({args.rolling_window}D)",
        output_path=output_dir / "rolling_ic_correlation.png",
    )

    report = build_redundancy_report(
        static_pearson=static_pearson,
        static_spearman=static_spearman,
        ic_correlation=ic_correlation,
        rolling_pearson=rolling_pearson,
        rolling_spearman=rolling_spearman,
        rolling_ic=rolling_ic,
        threshold=args.redundancy_threshold,
    )

    report_path = output_dir / "factor_redundancy_report.json"
    report_path.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
