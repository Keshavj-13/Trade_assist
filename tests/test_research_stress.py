"""Tests for worst-case stress testing."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from predictor.research.errors import ResearchInputError
from predictor.research.strategies import DonchianBreakoutStrategy, MomentumStrategy
from predictor.research.stress import (
    StressTestConfig,
    detect_worst_case_slices,
    run_stress_tests,
    worst_stress_drawdown,
    worst_stress_sharpe,
)
from predictor.research.types import StressTestResult


def _make_frame_with_crash(rows: int = 300) -> pd.DataFrame:
    idx = pd.date_range("2018-01-02", periods=rows, freq="B")
    close = np.concatenate([
        100.0 + np.arange(150, dtype=float) * 0.4,  # bull
        160.0 - np.arange(150, dtype=float) * 0.5,  # crash
    ])[:rows]
    high = close + 0.5
    low = np.maximum(close - 0.5, 0.01)
    return pd.DataFrame(
        {
            "Open": close - 0.05,
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": 1_000_000.0,
        },
        index=idx,
    )


def _make_volatile_frame(rows: int = 300) -> pd.DataFrame:
    """Frame with a pronounced calm-to-volatile transition guaranteed to produce a vol spike slice."""
    idx = pd.date_range("2020-01-02", periods=rows, freq="B")
    rng = np.random.default_rng(42)
    half = rows // 2
    calm = rng.normal(0.0, 0.002, half)
    volatile = rng.normal(0.0, 0.05, rows - half)
    returns = np.concatenate([calm, volatile])
    close = 100.0 * np.cumprod(1.0 + np.clip(returns, -0.15, 0.15))
    high = close * 1.005
    low = close * 0.995
    return pd.DataFrame(
        {
            "Open": close * 0.999,
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": 1_000_000.0,
        },
        index=idx,
    )


def test_detect_worst_case_slices_always_includes_full_history() -> None:
    frame = _make_frame_with_crash()
    slices = detect_worst_case_slices(frame)
    assert "full_history" in slices
    assert len(slices["full_history"]) == len(frame)


def test_detect_worst_case_slices_finds_drawdown_slice() -> None:
    frame = _make_frame_with_crash()
    slices = detect_worst_case_slices(frame)
    assert "worst_drawdown" in slices
    dd_frame = slices["worst_drawdown"]
    # The slice should end lower than it starts
    assert float(dd_frame["Close"].iloc[-1]) < float(dd_frame["Close"].iloc[0])


def test_detect_worst_case_slices_finds_vol_spike() -> None:
    frame = _make_volatile_frame()
    slices = detect_worst_case_slices(frame)
    assert "worst_vol_spike" in slices


def test_detect_worst_case_slices_finds_largest_day_drop() -> None:
    frame = _make_volatile_frame()
    slices = detect_worst_case_slices(frame)
    assert "largest_day_drop" in slices


def test_detect_worst_case_slices_raises_on_empty_frame() -> None:
    with pytest.raises(ResearchInputError):
        detect_worst_case_slices(pd.DataFrame())


def test_detect_worst_case_slices_raises_on_missing_close() -> None:
    bad = pd.DataFrame({"Open": [1.0, 2.0]})
    with pytest.raises(ResearchInputError):
        detect_worst_case_slices(bad)


def test_run_stress_tests_returns_result_with_correct_strategy_name() -> None:
    frame = _make_frame_with_crash()
    strategy = DonchianBreakoutStrategy(lookback=20, name="donchian_breakout")
    config = StressTestConfig(bars_per_year=252, transaction_cost_bps=5.0, min_slice_bars=30)
    result = run_stress_tests(frame, strategy, config=config)

    assert isinstance(result, StressTestResult)
    assert result.strategy_name == "donchian_breakout"


def test_run_stress_tests_slices_have_correct_strategy_name() -> None:
    frame = _make_frame_with_crash()
    strategy = MomentumStrategy(lookback=20, name="momentum")
    config = StressTestConfig(min_slice_bars=30)
    result = run_stress_tests(frame, strategy, config=config)

    for s in result.slices:
        assert s.strategy_name == "momentum"
        assert s.bar_count >= config.min_slice_bars


def test_run_stress_tests_full_history_always_in_slices() -> None:
    frame = _make_frame_with_crash()
    strategy = DonchianBreakoutStrategy(lookback=20, name="donchian_breakout")
    config = StressTestConfig(min_slice_bars=30)
    result = run_stress_tests(frame, strategy, config=config)

    labels = {s.label for s in result.slices}
    assert "full_history" in labels


def test_run_stress_tests_metrics_are_finite() -> None:
    frame = _make_frame_with_crash()
    strategy = DonchianBreakoutStrategy(lookback=20, name="donchian_breakout")
    config = StressTestConfig(min_slice_bars=30)
    result = run_stress_tests(frame, strategy, config=config)

    import math
    for s in result.slices:
        assert math.isfinite(s.metrics.sharpe_ratio)
        assert math.isfinite(s.metrics.total_return)
        assert s.metrics.max_drawdown <= 0.0


def test_worst_stress_drawdown_returns_most_negative() -> None:
    frame = _make_frame_with_crash()
    strategy = DonchianBreakoutStrategy(lookback=20, name="donchian_breakout")
    config = StressTestConfig(min_slice_bars=30)
    result = run_stress_tests(frame, strategy, config=config)

    wdd = worst_stress_drawdown(result)
    assert wdd <= 0.0
    assert wdd == min(s.metrics.max_drawdown for s in result.slices)


def test_worst_stress_sharpe_returns_minimum() -> None:
    frame = _make_volatile_frame()
    strategy = MomentumStrategy(lookback=20, name="momentum")
    config = StressTestConfig(min_slice_bars=30)
    result = run_stress_tests(frame, strategy, config=config)

    ws = worst_stress_sharpe(result)
    if result.slices:
        assert ws == min(s.metrics.sharpe_ratio for s in result.slices)


def test_worst_stress_drawdown_empty_result_returns_zero() -> None:
    result = StressTestResult(strategy_name="x", slices=())
    assert worst_stress_drawdown(result) == 0.0
    assert worst_stress_sharpe(result) == 0.0


def test_stress_test_is_deterministic() -> None:
    frame = _make_frame_with_crash()
    strategy = DonchianBreakoutStrategy(lookback=20, name="donchian_breakout")
    config = StressTestConfig(min_slice_bars=30)
    r1 = run_stress_tests(frame, strategy, config=config)
    r2 = run_stress_tests(frame, strategy, config=config)

    assert len(r1.slices) == len(r2.slices)
    for s1, s2 in zip(r1.slices, r2.slices):
        assert s1.label == s2.label
        assert s1.metrics.sharpe_ratio == s2.metrics.sharpe_ratio
