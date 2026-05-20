"""Tests for strategy signal generation and shared backtest metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from predictor.research.backtest import backtest_strategy
from predictor.research.errors import ResearchInputError
from predictor.research.strategies import (
    DonchianBreakoutStrategy,
    MeanReversionStrategy,
    MomentumStrategy,
    MovingAverageCrossoverStrategy,
    RegimeSwitchingStrategy,
    VolatilityBreakoutStrategy,
    WeightedStrategyEnsemble,
)


def test_strategy_signals_are_aligned(real_ohlcv_frame: pd.DataFrame) -> None:
    strategy = MovingAverageCrossoverStrategy(short_window=20, long_window=60)
    positions = strategy.generate_positions(real_ohlcv_frame)

    assert len(positions) == len(real_ohlcv_frame)
    assert positions.index.equals(real_ohlcv_frame.index)
    assert set(np.unique(positions.values)).issubset({-1.0, 0.0, 1.0})


def test_backtest_produces_required_metrics(real_ohlcv_frame: pd.DataFrame) -> None:
    strategy = DonchianBreakoutStrategy(lookback=55)
    run = backtest_strategy(real_ohlcv_frame, strategy, bars_per_year=252)

    assert run.strategy_name == strategy.name
    assert len(run.returns) == len(real_ohlcv_frame)
    assert run.metrics.max_drawdown <= 0.0
    assert 0.0 <= run.metrics.win_rate <= 1.0
    assert np.isfinite(run.metrics.total_return)
    assert np.isfinite(run.metrics.sharpe_ratio)
    assert run.metrics.profit_factor >= 0.0


def test_weighted_ensemble_combines_base_strategies(real_ohlcv_frame: pd.DataFrame) -> None:
    trend = MomentumStrategy(lookback=20)
    mean_reversion = MeanReversionStrategy(window=20, zscore_threshold=1.0)
    vol_breakout = VolatilityBreakoutStrategy(lookback=20, atr_window=14, atr_multiplier=1.5)

    combo = WeightedStrategyEnsemble(
        name="hybrid_combo",
        components=(trend, mean_reversion, vol_breakout),
        weights=(0.5, 0.3, 0.2),
        threshold=0.15,
    )

    positions = combo.generate_positions(real_ohlcv_frame)
    assert len(positions) == len(real_ohlcv_frame)
    assert set(np.unique(positions.values)).issubset({-1.0, 0.0, 1.0})


def test_regime_switching_strategy_runs_in_backtest(real_ohlcv_frame: pd.DataFrame) -> None:
    strategy = RegimeSwitchingStrategy(volatility_window=20, trend_window=50)
    run = backtest_strategy(real_ohlcv_frame, strategy, bars_per_year=252)

    assert run.metrics.win_rate >= 0.0
    assert len(run.equity_curve) == len(real_ohlcv_frame)


def test_backtest_rejects_non_monotonic_input(real_ohlcv_frame: pd.DataFrame) -> None:
    scrambled = real_ohlcv_frame.sample(frac=1.0, random_state=7)

    with pytest.raises(ResearchInputError):
        backtest_strategy(scrambled, MomentumStrategy(lookback=20), bars_per_year=252)
