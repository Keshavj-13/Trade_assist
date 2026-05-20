"""Tests for new strategy families: TripleMA, Keltner, RSIMeanReversion."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from predictor.research.backtest import backtest_strategy
from predictor.research.errors import ResearchInputError
from predictor.research.library import (
    build_core_strategy_universe,
    build_literature_strategy_universe,
)
from predictor.research.strategies import (
    KeltnerBreakoutStrategy,
    RSIMeanReversionStrategy,
    TripleMAcrossoverStrategy,
)


def _make_frame(rows: int = 300, seed: int = 7) -> pd.DataFrame:
    idx = pd.date_range("2019-01-02", periods=rows, freq="B")
    rng = np.random.default_rng(seed)
    returns = rng.normal(0.0003, 0.012, rows)
    close = 100.0 * np.cumprod(1.0 + returns)
    high = close * (1.0 + rng.uniform(0.001, 0.005, rows))
    low = close * (1.0 - rng.uniform(0.001, 0.005, rows))
    return pd.DataFrame(
        {
            "Open": close * (1.0 + rng.uniform(-0.002, 0.002, rows)),
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": 1_000_000.0,
        },
        index=idx,
    )


# --- TripleMAcrossoverStrategy ---

def test_triple_ma_crossover_positions_are_valid(real_ohlcv_frame: pd.DataFrame) -> None:
    strategy = TripleMAcrossoverStrategy(fast=10, medium=30, slow=100)
    pos = strategy.generate_positions(real_ohlcv_frame)

    assert len(pos) == len(real_ohlcv_frame)
    assert pos.index.equals(real_ohlcv_frame.index)
    assert set(np.unique(pos.values)).issubset({-1.0, 0.0, 1.0})


def test_triple_ma_crossover_raises_on_wrong_window_order() -> None:
    with pytest.raises(ResearchInputError):
        TripleMAcrossoverStrategy(fast=50, medium=30, slow=100).generate_positions(
            _make_frame()
        )


def test_triple_ma_crossover_raises_on_invalid_windows() -> None:
    with pytest.raises(ResearchInputError):
        TripleMAcrossoverStrategy(fast=0, medium=30, slow=100).generate_positions(_make_frame())


def test_triple_ma_crossover_backtest_runs(real_ohlcv_frame: pd.DataFrame) -> None:
    strategy = TripleMAcrossoverStrategy(fast=10, medium=30, slow=100)
    run = backtest_strategy(real_ohlcv_frame, strategy, bars_per_year=252)

    assert run.strategy_name == strategy.name
    assert np.isfinite(run.metrics.sharpe_ratio)
    assert run.metrics.max_drawdown <= 0.0


# --- KeltnerBreakoutStrategy ---

def test_keltner_breakout_positions_are_valid(real_ohlcv_frame: pd.DataFrame) -> None:
    strategy = KeltnerBreakoutStrategy(ema_window=20, atr_window=14, atr_multiplier=2.0)
    pos = strategy.generate_positions(real_ohlcv_frame)

    assert len(pos) == len(real_ohlcv_frame)
    assert set(np.unique(pos.values)).issubset({-1.0, 0.0, 1.0})


def test_keltner_breakout_raises_on_invalid_multiplier() -> None:
    frame = _make_frame()
    with pytest.raises(ResearchInputError):
        KeltnerBreakoutStrategy(atr_multiplier=-1.0).generate_positions(frame)


def test_keltner_breakout_backtest_runs(real_ohlcv_frame: pd.DataFrame) -> None:
    strategy = KeltnerBreakoutStrategy(ema_window=20, atr_window=14)
    run = backtest_strategy(real_ohlcv_frame, strategy, bars_per_year=252)

    assert run.strategy_name == strategy.name
    assert 0.0 <= run.metrics.win_rate <= 1.0
    assert run.metrics.profit_factor >= 0.0


def test_keltner_is_distinct_from_donchian(real_ohlcv_frame: pd.DataFrame) -> None:
    """Keltner and Donchian must produce different position series."""
    from predictor.research.strategies import DonchianBreakoutStrategy

    keltner = KeltnerBreakoutStrategy(ema_window=20, atr_window=14, atr_multiplier=2.0)
    donchian = DonchianBreakoutStrategy(lookback=20)

    kp = keltner.generate_positions(real_ohlcv_frame)
    dp = donchian.generate_positions(real_ohlcv_frame)

    # They should agree directionally on some bars but differ overall
    assert not kp.equals(dp), "Keltner and Donchian produced identical positions"


# --- RSIMeanReversionStrategy ---

def test_rsi_reversion_positions_are_valid(real_ohlcv_frame: pd.DataFrame) -> None:
    strategy = RSIMeanReversionStrategy(rsi_window=14, oversold=30.0, overbought=70.0)
    pos = strategy.generate_positions(real_ohlcv_frame)

    assert len(pos) == len(real_ohlcv_frame)
    assert set(np.unique(pos.values)).issubset({-1.0, 0.0, 1.0})


def test_rsi_reversion_raises_on_invalid_thresholds() -> None:
    frame = _make_frame()
    with pytest.raises(ResearchInputError):
        RSIMeanReversionStrategy(oversold=80.0, overbought=30.0).generate_positions(frame)


def test_rsi_reversion_raises_on_invalid_neutral_band() -> None:
    frame = _make_frame()
    with pytest.raises(ResearchInputError):
        RSIMeanReversionStrategy(neutral_band=0.0).generate_positions(frame)


def test_rsi_reversion_backtest_runs(real_ohlcv_frame: pd.DataFrame) -> None:
    strategy = RSIMeanReversionStrategy(rsi_window=14)
    run = backtest_strategy(real_ohlcv_frame, strategy, bars_per_year=252)

    assert np.isfinite(run.metrics.total_return)
    assert run.metrics.max_drawdown <= 0.0


# --- library completeness ---

def test_literature_universe_contains_new_families() -> None:
    strategies = build_literature_strategy_universe()
    names = {s.name for s in strategies}

    assert "triple_ma_crossover" in names
    assert "keltner_breakout" in names
    assert "rsi_mean_reversion" in names
    assert "ma_crossover_slow" in names
    assert "donchian_breakout_fast" in names


def test_literature_universe_has_no_duplicate_names() -> None:
    strategies = build_literature_strategy_universe()
    names = [s.name for s in strategies]
    assert len(names) == len(set(names))


def test_core_universe_has_one_per_family() -> None:
    strategies = build_core_strategy_universe()
    names = {s.name for s in strategies}
    # Expect one of each family
    assert "donchian_breakout" in names
    assert "momentum" in names
    assert "mean_reversion" in names
    assert "rsi_mean_reversion" in names
    assert "keltner_breakout" in names
    assert "triple_ma_crossover" in names
