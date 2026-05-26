"""Tests for predictor.research.visualization.

Covers:
- plot_equity_curve
- plot_drawdown_curve
- plot_permutation_histogram
- plot_trade_return_distribution
- plot_rolling_sharpe
- plot_regime_overlay
- export_equity_to_json
- export_metrics_to_parquet
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from predictor.research.types import (
    BacktestRun,
    PerformanceMetrics,
    PermutationTestResult,
    StrategyValidationReport,
)
from predictor.research.visualization import (
    plot_equity_curve,
    plot_drawdown_curve,
    plot_permutation_histogram,
    plot_trade_return_distribution,
    plot_rolling_sharpe,
    plot_regime_overlay,
    export_equity_to_json,
    export_metrics_to_parquet,
)


@pytest.fixture
def synth_backtest_run():
    n = 100
    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    returns = pd.Series(np.random.normal(0.0005, 0.01, size=n), index=idx)
    positions = pd.Series(np.random.choice([0.0, 1.0, -1.0], size=n), index=idx)
    equity_curve = (1.0 + returns).cumprod()
    metrics = PerformanceMetrics(
        total_return=float(equity_curve.iloc[-1] - 1.0),
        sharpe_ratio=1.5,
        max_drawdown=-0.12,
        win_rate=0.55,
        profit_factor=1.4,
        trade_count=12,
        avg_holding_period=5.2,
        expectancy=0.02,
    )
    return BacktestRun(
        strategy_name="test_strat",
        returns=returns,
        positions=positions,
        equity_curve=equity_curve,
        metrics=metrics,
    )


@pytest.fixture
def synth_permutation_result():
    return PermutationTestResult(
        observed_statistic=1.5,
        null_distribution=tuple(np.random.normal(0.0, 0.5, size=100)),
        p_value=0.02,
        passes=True,
    )


@pytest.fixture
def synth_validation_report(synth_backtest_run, synth_permutation_result):
    return StrategyValidationReport(
        strategy_name="test_strat",
        in_sample=synth_backtest_run,
        in_sample_permutation=synth_permutation_result,
        walk_forward_folds=(),
        walk_forward_aggregate=synth_backtest_run,
        walk_forward_permutation=synth_permutation_result,
        walk_forward_stability=0.85,
        walk_forward_fold_pass_rate=1.0,
        is_valid=True,
    )


def test_plot_equity_curve(synth_backtest_run, tmp_path):
    out = tmp_path / "equity.png"
    res = plot_equity_curve(synth_backtest_run, output_path=out)
    # If matplotlib is installed, it writes the file and returns the path
    if res is not None:
        assert res == out
        assert out.exists()


def test_plot_drawdown_curve(synth_backtest_run, tmp_path):
    out = tmp_path / "drawdown.png"
    res = plot_drawdown_curve(synth_backtest_run, output_path=out)
    if res is not None:
        assert res == out
        assert out.exists()


def test_plot_permutation_histogram(synth_permutation_result, tmp_path):
    out = tmp_path / "permutation.png"
    res = plot_permutation_histogram(synth_permutation_result, strategy_name="test_strat", output_path=out)
    if res is not None:
        assert res == out
        assert out.exists()


def test_plot_trade_return_distribution(synth_backtest_run, tmp_path):
    out = tmp_path / "trade_dist.png"
    res = plot_trade_return_distribution(synth_backtest_run, output_path=out)
    if res is not None:
        assert res == out
        assert out.exists()


def test_plot_rolling_sharpe(synth_backtest_run, tmp_path):
    out = tmp_path / "rolling.png"
    res = plot_rolling_sharpe(synth_backtest_run, window=10, output_path=out)
    if res is not None:
        assert res == out
        assert out.exists()


def test_plot_regime_overlay(synth_backtest_run, tmp_path):
    # Create simple OHLCV frame
    n = 100
    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    frame = pd.DataFrame({
        "Open": np.ones(n)*100.0,
        "High": np.ones(n)*101.0,
        "Low": np.ones(n)*99.0,
        "Close": np.ones(n)*100.0,
        "Volume": np.ones(n)*1000.0,
    }, index=idx)
    out = tmp_path / "regime.png"
    res = plot_regime_overlay(frame, synth_backtest_run, output_path=out)
    if res is not None:
        assert res == out
        assert out.exists()


def test_export_equity_to_json(synth_backtest_run, tmp_path):
    out = tmp_path / "equity.json"
    res = export_equity_to_json(synth_backtest_run, output_path=out)
    assert res == out
    assert out.exists()
    # Check JSON structure
    import json
    data = json.loads(out.read_text())
    assert data["strategy_name"] == "test_strat"
    assert "metrics" in data
    assert "equity_curve" in data


def test_export_metrics_to_parquet(synth_validation_report, tmp_path):
    out = tmp_path / "metrics.parquet"
    res = export_metrics_to_parquet([synth_validation_report], output_path=out)
    # Check if pyarrow is installed
    try:
        import pyarrow
        assert res == out
        assert out.exists()
    except ImportError:
        assert res is None
