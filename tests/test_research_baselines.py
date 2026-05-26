"""Tests for predictor.research.baselines.

Covers:
- Documentation contract enforcement (_DocumentedStrategy)
- BuyAndHoldStrategy
- ShortAndHoldStrategy
- RandomEntryFixedHoldStrategy
- SimpleMABaseline
- VolatilityTargetedHoldStrategy
- build_baseline_universe
"""

from __future__ import annotations

import pandas as pd
import pytest

from predictor.research.baselines import (
    _DocumentedStrategy,
    BuyAndHoldStrategy,
    ShortAndHoldStrategy,
    RandomEntryFixedHoldStrategy,
    SimpleMABaseline,
    VolatilityTargetedHoldStrategy,
    build_baseline_universe,
)
from predictor.research.errors import ResearchInputError


def _make_synth_data(n: int = 100) -> pd.DataFrame:
    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    # Simple sine wave trend
    import numpy as np
    close = 100.0 + np.sin(np.linspace(0, 10, n)) * 10
    high = close + 1.0
    low = close - 1.0
    open_ = close
    volume = np.ones(n) * 1000
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=idx,
    )


def test_documentation_contract():
    """Verify that _DocumentedStrategy enforces non-empty fields."""
    with pytest.raises(TypeError, match="must be a non-empty string"):
        class BadStrategy(_DocumentedStrategy):
            pass

    # A valid one
    class GoodStrategy(_DocumentedStrategy):
        theoretical_basis: str = "test basis"
        expected_market_condition: str = "test condition"
        known_failure_modes: str = "test failures"

    good = GoodStrategy()
    assert good.theoretical_basis == "test basis"


def test_buy_and_hold():
    frame = _make_synth_data()
    strat = BuyAndHoldStrategy()
    positions = strat.generate_positions(frame)
    assert len(positions) == len(frame)
    # Buy and hold must be long (1.0) all the time
    assert (positions == 1.0).all()
    assert strat.theoretical_basis != ""


def test_short_and_hold():
    frame = _make_synth_data()
    strat = ShortAndHoldStrategy()
    positions = strat.generate_positions(frame)
    assert len(positions) == len(frame)
    assert (positions == -1.0).all()


def test_random_entry_fixed_hold():
    frame = _make_synth_data()
    # RandomEntryFixedHoldStrategy parameters: hold_bars, long_only, seed
    strat = RandomEntryFixedHoldStrategy(hold_bars=5, long_only=True, seed=42)
    positions = strat.generate_positions(frame)
    assert len(positions) == len(frame)
    assert set(positions.unique()).issubset({0.0, 1.0, -1.0})


def test_simple_ma_baseline():
    frame = _make_synth_data()
    # SimpleMABaseline parameters: short_window, long_window
    strat = SimpleMABaseline(short_window=10, long_window=30)
    positions = strat.generate_positions(frame)
    assert len(positions) == len(frame)
    assert set(positions.unique()).issubset({0.0, 1.0, -1.0})


def test_volatility_targeted_hold():
    frame = _make_synth_data()
    # VolatilityTargetedHoldStrategy parameters: vol_window, baseline_window
    strat = VolatilityTargetedHoldStrategy(vol_window=10, baseline_window=30)
    positions = strat.generate_positions(frame)
    assert len(positions) == len(frame)
    assert set(positions.unique()).issubset({0.0, 1.0})


def test_build_baseline_universe():
    universe = build_baseline_universe()
    assert len(universe) == 5
    names = [s.name for s in universe]
    assert "buy_and_hold" in names
    assert "short_and_hold" in names
    assert "random_entry_fixed_hold" in names
    assert "simple_ma_baseline" in names
    assert "volatility_targeted_hold" in names
