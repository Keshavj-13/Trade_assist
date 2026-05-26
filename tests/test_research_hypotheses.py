"""Tests for predictor.research.hypotheses.

Covers:
- MarketHypothesis documentation contract enforcement
- GapFadeStrategy
- VolatilityCompressionBreakoutStrategy
- VolatilityExpansionStrategy
- RegimeFilteredTrendStrategy
- build_hypothesis_universe
"""

from __future__ import annotations

import pandas as pd
import pytest

from predictor.research.errors import ResearchInputError
from predictor.research.hypotheses import (
    MarketHypothesis,
    GapFadeStrategy,
    VolatilityCompressionBreakoutStrategy,
    VolatilityExpansionStrategy,
    RegimeFilteredTrendStrategy,
    build_hypothesis_universe,
)


def _make_synth_data(n: int = 150) -> pd.DataFrame:
    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    import numpy as np
    # Construct trending, ranging, and gapping behavior
    rng = np.random.default_rng(42)
    close = 100.0 + np.cumsum(rng.normal(0.05, 1.0, size=n))
    high = close + rng.uniform(0.5, 2.0, size=n)
    low = close - rng.uniform(0.5, 2.0, size=n)
    # Give some days large gaps
    open_ = close.copy()
    for i in range(1, n, 10):
        open_[i] = close[i-1] * 1.02  # Gap up
    for i in range(5, n, 10):
        open_[i] = close[i-1] * 0.98  # Gap down
        
    volume = np.ones(n) * 1000
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=idx,
    )


def test_market_hypothesis_contract():
    """Verify that MarketHypothesis enforces the documentation rule."""
    class BadHypothesis(MarketHypothesis):
        pass

    with pytest.raises(ResearchInputError, match="must be a non-empty string"):
        BadHypothesis(
            name="bad",
            theoretical_basis="",
            expected_market_condition="any",
            known_failure_modes="any",
        )


def test_gap_fade_strategy():
    frame = _make_synth_data()
    strat = GapFadeStrategy(gap_threshold=0.01, hold_bars=3)
    positions = strat.generate_positions(frame)
    assert len(positions) == len(frame)
    assert set(positions.unique()).issubset({0.0, 1.0, -1.0})
    # Verify that a gap up leads to negative positions (short)
    open_px = frame["Open"]
    prev_close = frame["Close"].shift(1)
    gap = (open_px - prev_close) / prev_close
    for i in range(1, len(frame)):
        if gap.iloc[i] > 0.01:
            assert positions.iloc[i] == -1.0


def test_volatility_compression_breakout():
    frame = _make_synth_data()
    strat = VolatilityCompressionBreakoutStrategy(
        atr_window=10, atr_lookback=30, compression_pct=25.0, expansion_pct=60.0, exit_bars=5
    )
    positions = strat.generate_positions(frame)
    assert len(positions) == len(frame)
    assert set(positions.unique()).issubset({0.0, 1.0, -1.0})


def test_volatility_expansion():
    frame = _make_synth_data()
    strat = VolatilityExpansionStrategy(
        atr_window=10, expansion_multiplier=1.2, slow_atr_window=30, hold_bars=5
    )
    positions = strat.generate_positions(frame)
    assert len(positions) == len(frame)
    assert set(positions.unique()).issubset({0.0, 1.0, -1.0})


def test_regime_filtered_trend():
    frame = _make_synth_data()
    strat = RegimeFilteredTrendStrategy(
        entry_lookback=20, exit_lookback=10, active_regimes=("BULL", "LOW_VOL")
    )
    positions = strat.generate_positions(frame)
    assert len(positions) == len(frame)
    assert set(positions.unique()).issubset({0.0, 1.0})


def test_build_hypothesis_universe():
    universe = build_hypothesis_universe()
    assert len(universe) == 6
    for strat in universe:
        assert isinstance(strat, MarketHypothesis)
        assert strat.theoretical_basis != ""
        assert strat.expected_market_condition != ""
        assert strat.known_failure_modes != ""
