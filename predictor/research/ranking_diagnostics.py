"""Visualization and export helpers for cross-sectional ranking diagnostics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from predictor.research.cross_sectional import CrossSectionalPermutationResult


def _try_import_matplotlib():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        return plt
    except ImportError:
        return None


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def plot_ranking_distribution(scores: pd.DataFrame, *, output_path: Path) -> Path | None:
    """Plot daily cross-sectional score dispersion."""
    plt = _try_import_matplotlib()
    if plt is None:
        return None
    if scores.empty:
        return None

    flattened = scores.replace([np.inf, -np.inf], np.nan).stack().dropna()
    if flattened.empty:
        return None

    _ensure_parent(output_path)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.hist(flattened.values, bins=40, color="#1f77b4", alpha=0.75)
    ax.set_title("Ranking Score Distribution")
    ax.set_xlabel("Score")
    ax.set_ylabel("Frequency")
    fig.tight_layout()
    fig.savefig(output_path, dpi=140)
    plt.close(fig)
    return output_path


def plot_factor_distributions(feature_panel: pd.DataFrame, *, output_path: Path) -> Path | None:
    """Plot per-factor score distributions from stacked feature panel."""
    plt = _try_import_matplotlib()
    if plt is None:
        return None
    if feature_panel.empty:
        return None

    _ensure_parent(output_path)
    cols = list(feature_panel.columns)
    n = len(cols)
    fig, axes = plt.subplots(n, 1, figsize=(9, max(3, 2 * n)), squeeze=False)
    for idx, col in enumerate(cols):
        series = feature_panel[col].replace([np.inf, -np.inf], np.nan).dropna()
        ax = axes[idx, 0]
        if not series.empty:
            ax.hist(series.values, bins=30, alpha=0.7, color="#2ca02c")
        ax.set_title(f"Factor Distribution: {col}")
        ax.set_xlabel("Value")
        ax.set_ylabel("Frequency")
    fig.tight_layout()
    fig.savefig(output_path, dpi=140)
    plt.close(fig)
    return output_path


def plot_prediction_vs_realized(
    scores: pd.DataFrame,
    targets: pd.DataFrame,
    *,
    output_path: Path,
) -> Path | None:
    """Scatter of predicted score vs realized next-day intraday return."""
    plt = _try_import_matplotlib()
    if plt is None:
        return None
    if scores.empty or targets.empty:
        return None

    aligned_dates = scores.index.intersection(targets.index)
    if aligned_dates.empty:
        return None

    s = scores.loc[aligned_dates].stack().rename("score")
    t = targets.loc[aligned_dates].stack().rename("target")
    merged = pd.concat([s, t], axis=1).dropna()
    if merged.empty:
        return None

    _ensure_parent(output_path)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(merged["score"], merged["target"], alpha=0.15, s=8)
    ax.set_title("Prediction vs Realized")
    ax.set_xlabel("Predicted Score")
    ax.set_ylabel("Realized Next-Day Intraday Return")
    fig.tight_layout()
    fig.savefig(output_path, dpi=140)
    plt.close(fig)
    return output_path


def plot_ranking_permutation_histogram(
    permutation_result: CrossSectionalPermutationResult,
    *,
    output_path: Path,
) -> Path | None:
    """Histogram of permutation IC null with observed IC marker."""
    plt = _try_import_matplotlib()
    if plt is None:
        return None

    null_ic = np.asarray(permutation_result.null_ic, dtype=float)
    if null_ic.size == 0:
        return None

    _ensure_parent(output_path)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(null_ic, bins=35, color="#7f7f7f", alpha=0.75)
    ax.axvline(permutation_result.observed_ic, color="#d62728", linewidth=2)
    ax.set_title("Permutation Null Distribution (IC)")
    ax.set_xlabel("Permuted IC")
    ax.set_ylabel("Frequency")
    fig.tight_layout()
    fig.savefig(output_path, dpi=140)
    plt.close(fig)
    return output_path


def plot_rolling_ic(ic_series: pd.Series, *, window: int = 20, output_path: Path) -> Path | None:
    """Rolling information coefficient stability plot."""
    plt = _try_import_matplotlib()
    if plt is None:
        return None
    if ic_series.empty or window <= 1:
        return None

    rolling = ic_series.rolling(window).mean()
    _ensure_parent(output_path)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(rolling.index, rolling.values, color="#9467bd")
    ax.axhline(0.0, color="black", linewidth=0.8, linestyle="--")
    ax.set_title("Rolling Information Coefficient")
    ax.set_xlabel("Date")
    ax.set_ylabel(f"{window}-Day Rolling IC")
    fig.tight_layout()
    fig.savefig(output_path, dpi=140)
    plt.close(fig)
    return output_path


def plot_cumulative_top_k_return(returns: pd.Series, *, output_path: Path) -> Path | None:
    """Plot cumulative top-k portfolio return."""
    plt = _try_import_matplotlib()
    if plt is None:
        return None
    if returns.empty:
        return None

    cumulative = (returns + 1.0).cumprod()
    _ensure_parent(output_path)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(cumulative.index, cumulative.values, color="#ff7f0e")
    ax.set_title("Cumulative Top-K Intraday Return")
    ax.set_xlabel("Date")
    ax.set_ylabel("Growth")
    fig.tight_layout()
    fig.savefig(output_path, dpi=140)
    plt.close(fig)
    return output_path


def plot_ranking_regime_overlay(
    returns: pd.Series,
    regimes: pd.Series,
    *,
    output_path: Path,
) -> Path | None:
    """Overlay cumulative returns with regime-colored backgrounds."""
    plt = _try_import_matplotlib()
    if plt is None:
        return None
    if returns.empty or regimes.empty:
        return None

    cumulative = (returns + 1.0).cumprod()
    aligned_regimes = regimes.reindex(cumulative.index).fillna("SIDEWAYS")

    _ensure_parent(output_path)
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(cumulative.index, cumulative.values, color="#1f77b4")

    regime_colors = {
        "HIGH_VOL": "#f4a261",
        "LOW_VOL": "#2a9d8f",
        "BULL": "#90be6d",
        "BEAR": "#f94144",
        "CRASH": "#6a4c93",
        "RECOVERY": "#f9c74f",
        "SIDEWAYS": "#adb5bd",
    }

    previous_label = None
    start_idx = None
    for idx, (date, label) in enumerate(aligned_regimes.items()):
        if previous_label is None:
            previous_label = label
            start_idx = date
            continue
        if label != previous_label:
            ax.axvspan(start_idx, date, color=regime_colors.get(previous_label, "#adb5bd"), alpha=0.10)
            previous_label = label
            start_idx = date
        if idx == len(aligned_regimes) - 1:
            ax.axvspan(start_idx, date, color=regime_colors.get(previous_label, "#adb5bd"), alpha=0.10)

    ax.set_title("Cumulative Top-K Return with Regime Overlay")
    ax.set_xlabel("Date")
    ax.set_ylabel("Growth")
    fig.tight_layout()
    fig.savefig(output_path, dpi=140)
    plt.close(fig)
    return output_path


def export_ranking_diagnostics_json(payload: Mapping[str, object], *, output_path: Path) -> Path:
    """Persist diagnostics metadata to JSON."""
    _ensure_parent(output_path)
    output_path.write_text(json.dumps(dict(payload), indent=2, default=str))
    return output_path


def export_ranking_diagnostics_parquet(frame: pd.DataFrame, *, output_path: Path) -> Path | None:
    """Persist diagnostics tabular data to parquet when pyarrow is available."""
    try:
        import pyarrow  # noqa: F401
    except ImportError:
        return None

    _ensure_parent(output_path)
    frame.to_parquet(output_path, index=False)
    return output_path
