"""Visual diagnostics for strategy research.

Responsibility: plotting and export only.
This module has zero strategy logic and zero validation logic.
It accepts typed result objects and produces files.

All plot functions:
- Use matplotlib with the non-interactive Agg backend (headless safe)
- Return the output Path on success, or None if matplotlib is absent
- Never raise on missing dependencies — emit a warning instead
- Accept output_path as a required keyword argument

All export functions:
- Return the output Path on success, None on failure
- JSON exports always available (no optional deps)
- Parquet exports require pyarrow (optional, warned if absent)

Public API
----------
plot_equity_curve
plot_drawdown_curve
plot_permutation_histogram
plot_trade_return_distribution
plot_rolling_sharpe
plot_regime_overlay
export_equity_to_json
export_metrics_to_parquet
"""

from __future__ import annotations

import json
import logging
import warnings
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from predictor.research.types import BacktestRun, PermutationTestResult, StrategyValidationReport

log = logging.getLogger(__name__)

_STYLE = {
    "bg": "#0f1117",
    "fg": "#e8eaf0",
    "grid": "#2a2d3a",
    "equity": "#4fc3f7",
    "benchmark": "#81c784",
    "drawdown": "#ef5350",
    "null": "#546e7a",
    "observed": "#ffd54f",
    "positive": "#66bb6a",
    "negative": "#ef5350",
    "rolling": "#ce93d8",
}

_REGIME_COLORS = {
    "BULL": "#1b5e20",
    "BEAR": "#b71c1c",
    "HIGH_VOL": "#e65100",
    "LOW_VOL": "#0d47a1",
    "CRASH": "#4a148c",
    "RECOVERY": "#827717",
    "SIDEWAYS": "#263238",
}


def _try_import_matplotlib():
    """Return (pyplot, matplotlib) or (None, None) with a warning."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt, matplotlib
    except ImportError:
        warnings.warn(
            "matplotlib is not installed. Install it with: pip install matplotlib\n"
            "All plot_* functions will return None until matplotlib is available.",
            stacklevel=3,
        )
        return None, None


def _ensure_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _apply_dark_style(fig, ax, title: str, xlabel: str, ylabel: str) -> None:
    """Apply consistent dark theme to a figure/axes pair."""
    fig.patch.set_facecolor(_STYLE["bg"])
    ax.set_facecolor(_STYLE["bg"])
    ax.set_title(title, color=_STYLE["fg"], fontsize=12, pad=12)
    ax.set_xlabel(xlabel, color=_STYLE["fg"], fontsize=9)
    ax.set_ylabel(ylabel, color=_STYLE["fg"], fontsize=9)
    ax.tick_params(colors=_STYLE["fg"])
    ax.spines["bottom"].set_color(_STYLE["grid"])
    ax.spines["left"].set_color(_STYLE["grid"])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, color=_STYLE["grid"], linewidth=0.5, linestyle="--", alpha=0.5)


# ---------------------------------------------------------------------------
# Equity curve
# ---------------------------------------------------------------------------


def plot_equity_curve(
    run: BacktestRun,
    *,
    benchmark_run: BacktestRun | None = None,
    output_path: Path,
) -> Path | None:
    """Plot strategy equity curve optionally overlaid with a benchmark.

    Parameters
    ----------
    run : BacktestRun
        Primary strategy backtest result.
    benchmark_run : BacktestRun | None
        Optional benchmark (e.g. buy_and_hold) to overlay.
    output_path : Path
        PNG file to write.

    Returns
    -------
    Path | None
        Path written, or None if matplotlib unavailable.
    """
    plt, _ = _try_import_matplotlib()
    if plt is None:
        return None

    _ensure_dir(output_path)
    fig, ax = plt.subplots(figsize=(12, 5))

    equity = run.equity_curve
    if equity.empty:
        log.warning("Equity curve is empty, skipping equity curve plot")
        return None

    ax.plot(equity.index, equity.values, color=_STYLE["equity"], linewidth=1.5, label=run.strategy_name)
    ax.axhline(1.0, color=_STYLE["grid"], linewidth=0.8, linestyle=":")

    if benchmark_run is not None:
        bench = benchmark_run.equity_curve
        ax.plot(bench.index, bench.values, color=_STYLE["benchmark"], linewidth=1.0,
                linestyle="--", label=benchmark_run.strategy_name, alpha=0.8)

    ax.legend(facecolor=_STYLE["bg"], edgecolor=_STYLE["grid"], labelcolor=_STYLE["fg"], fontsize=9)
    _apply_dark_style(fig, ax,
                      title=f"Equity Curve — {run.strategy_name}",
                      xlabel="Date", ylabel="Portfolio Value")

    sharpe = run.metrics.sharpe_ratio
    total_ret = run.metrics.total_return
    ax.text(0.01, 0.97, f"Sharpe: {sharpe:.3f}  |  Return: {total_ret:.1%}",
            transform=ax.transAxes, color=_STYLE["fg"], fontsize=8,
            verticalalignment="top")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=_STYLE["bg"])
    plt.close(fig)
    log.info("Saved equity curve to %s", output_path)
    return output_path


# ---------------------------------------------------------------------------
# Drawdown
# ---------------------------------------------------------------------------


def plot_drawdown_curve(
    run: BacktestRun,
    *,
    output_path: Path,
) -> Path | None:
    """Plot underwater (drawdown) curve.

    Returns
    -------
    Path | None
    """
    plt, _ = _try_import_matplotlib()
    if plt is None:
        return None

    _ensure_dir(output_path)
    equity = run.equity_curve
    if equity.empty:
        log.warning("Equity curve is empty, skipping drawdown plot")
        return None
    rolling_max = equity.cummax()
    drawdown = (equity / rolling_max) - 1.0

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.fill_between(drawdown.index, drawdown.values, 0, color=_STYLE["drawdown"], alpha=0.6)
    ax.plot(drawdown.index, drawdown.values, color=_STYLE["drawdown"], linewidth=0.8)
    ax.axhline(0, color=_STYLE["grid"], linewidth=0.6)

    max_dd = float(drawdown.min())
    ax.axhline(max_dd, color=_STYLE["observed"], linewidth=0.8, linestyle="--")
    ax.text(drawdown.index[0], max_dd * 1.05,
            f"Max DD: {max_dd:.1%}", color=_STYLE["observed"], fontsize=8)

    _apply_dark_style(fig, ax,
                      title=f"Drawdown — {run.strategy_name}",
                      xlabel="Date", ylabel="Drawdown")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=_STYLE["bg"])
    plt.close(fig)
    log.info("Saved drawdown curve to %s", output_path)
    return output_path


# ---------------------------------------------------------------------------
# Permutation histogram
# ---------------------------------------------------------------------------


def plot_permutation_histogram(
    perm_result: PermutationTestResult,
    *,
    strategy_name: str = "strategy",
    output_path: Path,
) -> Path | None:
    """Plot null distribution histogram with observed Sharpe marker.

    Returns
    -------
    Path | None
    """
    plt, _ = _try_import_matplotlib()
    if plt is None:
        return None

    _ensure_dir(output_path)
    null_dist = np.array(perm_result.null_distribution)
    observed = perm_result.observed_statistic
    p_value = perm_result.p_value

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(null_dist, bins=min(50, max(10, len(null_dist) // 5)),
            color=_STYLE["null"], alpha=0.75, edgecolor=_STYLE["bg"], label="Null distribution")
    ax.axvline(observed, color=_STYLE["observed"], linewidth=2.0, linestyle="-",
               label=f"Observed: {observed:.3f}")

    # 95th percentile marker
    pct95 = float(np.percentile(null_dist, 95))
    ax.axvline(pct95, color=_STYLE["equity"], linewidth=1.0, linestyle=":",
               label=f"95th pct: {pct95:.3f}")

    verdict = "PASS" if perm_result.passes else "FAIL"
    ax.text(0.98, 0.97,
            f"p = {p_value:.3f}  |  {verdict}",
            transform=ax.transAxes, ha="right", va="top",
            color=_STYLE["positive"] if perm_result.passes else _STYLE["negative"],
            fontsize=10, fontweight="bold")

    ax.legend(facecolor=_STYLE["bg"], edgecolor=_STYLE["grid"], labelcolor=_STYLE["fg"], fontsize=9)
    _apply_dark_style(fig, ax,
                      title=f"IS Permutation Null Distribution — {strategy_name}",
                      xlabel="Permuted Sharpe Ratio", ylabel="Frequency")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=_STYLE["bg"])
    plt.close(fig)
    log.info("Saved permutation histogram to %s", output_path)
    return output_path


# ---------------------------------------------------------------------------
# Trade return distribution
# ---------------------------------------------------------------------------


def plot_trade_return_distribution(
    run: BacktestRun,
    *,
    output_path: Path,
) -> Path | None:
    """Plot per-trade P&L distribution (bar-level returns where position != 0).

    Returns
    -------
    Path | None
    """
    plt, _ = _try_import_matplotlib()
    if plt is None:
        return None

    _ensure_dir(output_path)
    # Use bar returns weighted by position sign as trade-level P&L proxy
    trade_returns = run.returns[run.positions != 0]
    if trade_returns.empty:
        log.warning("No active-position bars to plot for %s", run.strategy_name)
        return None

    values = trade_returns.values
    pos_vals = values[values > 0]
    neg_vals = values[values < 0]

    fig, ax = plt.subplots(figsize=(9, 5))
    bins = min(60, max(10, len(values) // 5))

    if len(pos_vals):
        ax.hist(pos_vals, bins=bins, color=_STYLE["positive"], alpha=0.7, label="Wins")
    if len(neg_vals):
        ax.hist(neg_vals, bins=bins, color=_STYLE["negative"], alpha=0.7, label="Losses")
    ax.axvline(float(np.mean(values)), color=_STYLE["observed"], linewidth=1.5, linestyle="--",
               label=f"Mean: {float(np.mean(values)):.4f}")

    ax.legend(facecolor=_STYLE["bg"], edgecolor=_STYLE["grid"], labelcolor=_STYLE["fg"], fontsize=9)
    _apply_dark_style(fig, ax,
                      title=f"Trade Return Distribution — {run.strategy_name}",
                      xlabel="Bar Return (active position)", ylabel="Frequency")
    win_rate = len(pos_vals) / max(len(values), 1)
    ax.text(0.01, 0.97, f"Win rate: {win_rate:.1%}  |  n={len(values)} bars",
            transform=ax.transAxes, color=_STYLE["fg"], fontsize=8, va="top")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=_STYLE["bg"])
    plt.close(fig)
    log.info("Saved trade distribution to %s", output_path)
    return output_path


# ---------------------------------------------------------------------------
# Rolling Sharpe
# ---------------------------------------------------------------------------


def plot_rolling_sharpe(
    run: BacktestRun,
    *,
    window: int = 63,
    output_path: Path,
) -> Path | None:
    """Plot rolling Sharpe ratio.

    Parameters
    ----------
    window : int
        Rolling window in bars (default 63 ≈ 3 months daily).

    Returns
    -------
    Path | None
    """
    plt, _ = _try_import_matplotlib()
    if plt is None:
        return None

    _ensure_dir(output_path)
    returns = run.returns
    if returns.empty or len(returns) < window:
        log.warning("Not enough returns bars to plot rolling Sharpe")
        return None
    bars_per_year = run.bars_per_year if hasattr(run, "bars_per_year") else 252
    annualisation = float(bars_per_year) ** 0.5

    roll_mean = returns.rolling(window).mean()
    roll_std = returns.rolling(window).std(ddof=1)
    rolling_sharpe = (roll_mean / roll_std.replace(0, np.nan)) * annualisation

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(rolling_sharpe.index, rolling_sharpe.values,
            color=_STYLE["rolling"], linewidth=1.2, label=f"Rolling Sharpe ({window}b)")
    ax.axhline(0, color=_STYLE["grid"], linewidth=0.8)
    ax.fill_between(rolling_sharpe.index, rolling_sharpe.values, 0,
                    where=(rolling_sharpe > 0),
                    color=_STYLE["positive"], alpha=0.15)
    ax.fill_between(rolling_sharpe.index, rolling_sharpe.values, 0,
                    where=(rolling_sharpe < 0),
                    color=_STYLE["negative"], alpha=0.15)

    ax.legend(facecolor=_STYLE["bg"], edgecolor=_STYLE["grid"], labelcolor=_STYLE["fg"], fontsize=9)
    _apply_dark_style(fig, ax,
                      title=f"Rolling Sharpe ({window} bars) — {run.strategy_name}",
                      xlabel="Date", ylabel="Annualised Sharpe")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=_STYLE["bg"])
    plt.close(fig)
    log.info("Saved rolling Sharpe to %s", output_path)
    return output_path


# ---------------------------------------------------------------------------
# Regime overlay
# ---------------------------------------------------------------------------


def plot_regime_overlay(
    frame: pd.DataFrame,
    run: BacktestRun,
    *,
    regime_labels: pd.Series | None = None,
    output_path: Path,
) -> Path | None:
    """Plot equity curve with regime colour bands overlaid.

    Parameters
    ----------
    frame : pd.DataFrame
        OHLCV frame (used to compute regimes if regime_labels is None).
    run : BacktestRun
        Strategy backtest result.
    regime_labels : pd.Series | None
        Pre-computed regime labels. If None, computes via classify_regimes().
    output_path : Path

    Returns
    -------
    Path | None
    """
    plt, mpl = _try_import_matplotlib()
    if plt is None:
        return None

    if regime_labels is None:
        try:
            from predictor.research.regime import classify_regimes
            regime_labels = classify_regimes(frame)
        except Exception as exc:
            log.warning("Could not compute regime labels: %s", exc)
            return None

    _ensure_dir(output_path)
    fig, ax = plt.subplots(figsize=(13, 5))

    equity = run.equity_curve
    if equity.empty:
        log.warning("Equity curve is empty, skipping regime overlay plot")
        return None
    ax.plot(equity.index, equity.values, color=_STYLE["equity"], linewidth=1.5,
            label=run.strategy_name, zorder=5)

    # Shade background by regime
    regimes = regime_labels.reindex(equity.index, method="ffill")
    prev_regime = None
    start_date = None

    for date, regime in regimes.items():
        if regime != prev_regime:
            if prev_regime is not None and start_date is not None:
                color = _REGIME_COLORS.get(prev_regime, "#37474f")
                ax.axvspan(start_date, date, color=color, alpha=0.18, zorder=1)
            start_date = date
            prev_regime = regime

    if prev_regime is not None and start_date is not None:
        color = _REGIME_COLORS.get(prev_regime, "#37474f")
        ax.axvspan(start_date, equity.index[-1], color=color, alpha=0.18, zorder=1)

    # Legend patches for regimes
    patches = [
        mpl.patches.Patch(color=color, alpha=0.4, label=regime)
        for regime, color in _REGIME_COLORS.items()
        if regime in regimes.values
    ]
    first_legend = ax.legend(handles=patches, loc="lower right", fontsize=7,
                             facecolor=_STYLE["bg"], edgecolor=_STYLE["grid"], labelcolor=_STYLE["fg"])
    ax.add_artist(first_legend)
    ax.legend(facecolor=_STYLE["bg"], edgecolor=_STYLE["grid"], labelcolor=_STYLE["fg"], fontsize=9)

    _apply_dark_style(fig, ax,
                      title=f"Equity + Regime Overlay — {run.strategy_name}",
                      xlabel="Date", ylabel="Portfolio Value")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=_STYLE["bg"])
    plt.close(fig)
    log.info("Saved regime overlay to %s", output_path)
    return output_path


# ---------------------------------------------------------------------------
# JSON export
# ---------------------------------------------------------------------------


def export_equity_to_json(
    run: BacktestRun,
    *,
    output_path: Path,
) -> Path | None:
    """Export equity curve, returns, and positions to JSON.

    Always available — no optional dependencies.

    Returns
    -------
    Path | None
    """
    _ensure_dir(output_path)
    payload = {
        "strategy_name": run.strategy_name,
        "metrics": {
            "sharpe_ratio": run.metrics.sharpe_ratio,
            "total_return": run.metrics.total_return,
            "max_drawdown": run.metrics.max_drawdown,
            "trade_count": run.metrics.trade_count,
            "win_rate": run.metrics.win_rate,
            "profit_factor": run.metrics.profit_factor,
            "expectancy": run.metrics.expectancy,
            "avg_holding_period": run.metrics.avg_holding_period,
        },
        "equity_curve": {
            str(dt): float(val)
            for dt, val in run.equity_curve.items()
        },
        "returns": {
            str(dt): float(val)
            for dt, val in run.returns.items()
        },
        "positions": {
            str(dt): float(val)
            for dt, val in run.positions.items()
        },
    }
    output_path.write_text(json.dumps(payload, indent=2))
    log.info("Saved equity JSON to %s", output_path)
    return output_path


# ---------------------------------------------------------------------------
# Parquet export
# ---------------------------------------------------------------------------


def export_metrics_to_parquet(
    reports: Sequence[StrategyValidationReport],
    *,
    output_path: Path,
) -> Path | None:
    """Export all strategy metrics to a parquet file.

    Requires pyarrow. Returns None with a warning if absent.

    Returns
    -------
    Path | None
    """
    try:
        import pyarrow  # noqa: F401
    except ImportError:
        warnings.warn(
            "pyarrow is not installed. Install it with: pip install pyarrow\n"
            "export_metrics_to_parquet will return None until pyarrow is available.",
            stacklevel=2,
        )
        return None

    _ensure_dir(output_path)
    rows = []
    for report in reports:
        if report.in_sample is None:
            continue
        m = report.in_sample.metrics
        raw = report.resolved_raw_metrics
        validated = report.resolved_validated_metrics
        rows.append({
            "strategy_name": report.strategy_name,
            "is_valid": report.is_valid,
            "fail_reasons": "|".join(report.fail_reasons),
            "rejection_reason": report.rejection_reason or "",
            "sharpe_ratio": m.sharpe_ratio,
            "total_return": m.total_return,
            "max_drawdown": m.max_drawdown,
            "trade_count": m.trade_count,
            "win_rate": m.win_rate,
            "profit_factor": m.profit_factor,
            "expectancy": m.expectancy,
            "avg_holding_period": m.avg_holding_period,
            "raw_sharpe_ratio": raw.sharpe_ratio,
            "raw_total_return": raw.total_return,
            "raw_trade_count": raw.trade_count,
            "raw_expectancy": raw.expectancy,
            "validated_sharpe_ratio": validated.sharpe_ratio,
            "validated_total_return": validated.total_return,
            "validated_trade_count": validated.trade_count,
            "validated_expectancy": validated.expectancy,
            "is_permutation_p_value": report.in_sample_permutation.p_value,
            "walk_forward_permutation_p_value": report.walk_forward_permutation.p_value,
        })

    df = pd.DataFrame(rows)
    df.to_parquet(output_path, index=False)
    log.info("Saved metrics parquet to %s", output_path)
    return output_path


# ---------------------------------------------------------------------------
# Cross-sectional Diagnostic Plots
# ---------------------------------------------------------------------------


def plot_factor_cumulative_returns(
    factor_returns: pd.Series,
    *,
    benchmark_returns: Dict[str, pd.Series] | None = None,
    output_path: Path,
) -> Path | None:
    """Plot cumulative return overlay of top-k factor portfolio vs benchmarks."""
    plt, _ = _try_import_matplotlib()
    if plt is None:
        return None

    if factor_returns.empty:
        log.warning("Factor returns empty, skipping cumulative returns plot")
        return None

    _ensure_dir(output_path)
    fig, ax = plt.subplots(figsize=(12, 5))

    # Plot factor cumulative return
    cum_factor = (factor_returns + 1.0).cumprod()
    ax.plot(cum_factor.index, cum_factor.values, color=_STYLE["equity"], linewidth=1.8, label="Factor Portfolio")

    # Plot benchmarks
    colors = [_STYLE["benchmark"], _STYLE["observed"], "#ab47bc", "#26a69a"]
    if benchmark_returns:
        for idx, (name, rets) in enumerate(benchmark_returns.items()):
            common_idx = cum_factor.index.intersection(rets.index)
            if not common_idx.empty:
                cum_bench = (rets.loc[common_idx] + 1.0).cumprod()
                color = colors[idx % len(colors)]
                ax.plot(cum_bench.index, cum_bench.values, color=color, linewidth=1.0,
                        linestyle="--", label=name, alpha=0.8)

    ax.legend(facecolor=_STYLE["bg"], edgecolor=_STYLE["grid"], labelcolor=_STYLE["fg"], fontsize=9)
    _apply_dark_style(fig, ax,
                      title="Factor Portfolio vs Benchmarks — Cumulative Intraday Return",
                      xlabel="Date", ylabel="Cumulative Growth")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=_STYLE["bg"])
    plt.close(fig)
    log.info("Saved cumulative returns plot to %s", output_path)
    return output_path


def plot_ic_distribution(
    ic_series: pd.Series,
    *,
    output_path: Path,
) -> Path | None:
    """Plot daily Spearman Rank Correlation (Information Coefficient) distribution."""
    plt, _ = _try_import_matplotlib()
    if plt is None:
        return None

    if ic_series.empty:
        log.warning("IC series is empty, skipping distribution plot")
        return None

    _ensure_dir(output_path)
    fig, ax = plt.subplots(figsize=(10, 4))

    # Histogram of daily IC values
    ax.hist(ic_series.dropna().values, bins=30, color=_STYLE["equity"], alpha=0.6, edgecolor=_STYLE["grid"])
    
    mean_ic = float(ic_series.mean())
    ax.axvline(mean_ic, color=_STYLE["observed"], linewidth=1.5, linestyle="--")
    ax.text(mean_ic * 1.05, ax.get_ylim()[1] * 0.9, f"Mean IC: {mean_ic:.3f}", color=_STYLE["observed"], fontsize=9)

    _apply_dark_style(fig, ax,
                      title="Information Coefficient (Daily Spearman Rank Correlation) Distribution",
                      xlabel="Spearman IC", ylabel="Frequency")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=_STYLE["bg"])
    plt.close(fig)
    log.info("Saved IC distribution plot to %s", output_path)
    return output_path


def plot_factor_regime_performance(
    regime_ic: Dict[str, float],
    *,
    output_path: Path,
) -> Path | None:
    """Plot bar chart showing factor mean IC across different market regimes."""
    plt, _ = _try_import_matplotlib()
    if plt is None:
        return None

    if not regime_ic:
        log.warning("Regime IC map empty, skipping regime plot")
        return None

    _ensure_dir(output_path)
    fig, ax = plt.subplots(figsize=(8, 4))

    regimes = list(regime_ic.keys())
    values = list(regime_ic.values())

    colors = [_REGIME_COLORS.get(r, "#37474f") for r in regimes]
    ax.bar(regimes, values, color=colors, alpha=0.7, edgecolor=_STYLE["grid"])
    ax.axhline(0, color=_STYLE["fg"], linewidth=0.8, linestyle=":")

    _apply_dark_style(fig, ax,
                      title="Mean Factor Correlation (IC) by Market Regime",
                      xlabel="Market Regime", ylabel="Mean Spearman IC")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=_STYLE["bg"])
    plt.close(fig)
    log.info("Saved regime IC plot to %s", output_path)
    return output_path
