"""Unit tests for cross-sectional ranking factors and baselines."""

from __future__ import annotations

import pandas as pd
import pytest

from predictor.research.factors import (
    ATRExpansionFactor,
    MarketBenchmarkFactor,
    Momentum20Factor,
    OvernightGapFactor,
    PreviousDayLoserRebound,
    PreviousDayReturnFactor,
    PreviousDayWinnerContinuation,
    RandomRankingFactor,
    RankingFactor,
    RollingBetaAdjustedStrengthFactor,
    SectorRelativeStrengthFactor,
    ShortTermMomentumFactor,
    SectorMomentumFactor,
    VolatilityCompressionFactor,
    build_factor_universe,
)


def test_documented_factor_enforcement():
    """Verify that _DocumentedFactor enforces documentation strings on subclasses."""
    with pytest.raises(TypeError, match="must be a non-empty string"):
        # Fails because documentation attributes are missing/empty
        class BrokenFactor(RankingFactor):
            pass

        BrokenFactor(name="broken")


def test_factor_universe_instantiation():
    """Verify that build_factor_universe returns all expected factors with valid properties."""
    universe = build_factor_universe()
    assert len(universe) >= 16
    
    names = [f.name for f in universe]
    assert "random_ranking" in names
    assert "buy_and_hold_baseline" in names
    assert "equal_weight_selection" in names
    assert "simple_momentum_rank" in names
    assert "volatility_rank" in names
    assert "momentum_20" in names
    assert "previous_day_return" in names
    assert "short_term_momentum" in names
    assert "sector_relative_strength" in names
    assert "overnight_gap" in names
    assert "volatility_compression" in names
    assert "relative_volume" in names
    assert "atr_expansion" in names
    assert "rolling_beta_adjusted_strength" in names

    for factor in universe:
        assert isinstance(factor, RankingFactor)
        assert len(factor.theoretical_basis) > 0
        assert len(factor.expected_market_condition) > 0
        assert len(factor.known_failure_modes) > 0


def test_random_ranking_factor(make_ohlcv_frame):
    """Verify that RandomRankingFactor generates randomized aligned scores."""
    symbol_data = {
        "TCS": make_ohlcv_frame(rows=100),
        "INFY": make_ohlcv_frame(rows=100),
    }
    
    factor = RandomRankingFactor(name="test_random", seed=42)
    scores = factor.compute_scores(symbol_data)
    
    assert isinstance(scores, pd.DataFrame)
    assert set(scores.columns) == {"TCS", "INFY"}
    assert len(scores) == 100
    # Seed makes it deterministic
    assert scores.loc[scores.index[0], "TCS"] != scores.loc[scores.index[0], "INFY"]


def test_market_benchmark_factor(make_ohlcv_frame):
    """Verify that MarketBenchmarkFactor outputs constant scores for equal-weight market baseline."""
    symbol_data = {
        "TCS": make_ohlcv_frame(rows=50),
        "INFY": make_ohlcv_frame(rows=50),
    }
    
    factor = MarketBenchmarkFactor(name="market")
    scores = factor.compute_scores(symbol_data)
    
    assert (scores["TCS"] == 1.0).all()
    assert (scores["INFY"] == 1.0).all()


def test_naive_momentum_continuation(make_ohlcv_frame):
    """Verify that PreviousDayWinnerContinuation computes correct positive momentum scores."""
    # Let's create an uptrending frame for TCS and downtrending for INFY
    symbol_data = {
        "TCS": make_ohlcv_frame(rows=10, trend=1.0),   # Winner
        "INFY": make_ohlcv_frame(rows=10, trend=-1.0), # Loser
    }
    
    factor = PreviousDayWinnerContinuation(name="winner")
    scores = factor.compute_scores(symbol_data)
    
    # Yesterday's winner should have higher score
    assert scores.loc[scores.index[-1], "TCS"] > scores.loc[scores.index[-1], "INFY"]


def test_naive_mean_reversion_rebound(make_ohlcv_frame):
    """Verify that PreviousDayLoserRebound computes correct rebound scores."""
    symbol_data = {
        "TCS": make_ohlcv_frame(rows=10, trend=1.0),   # Winner
        "INFY": make_ohlcv_frame(rows=10, trend=-1.0), # Loser
    }
    
    factor = PreviousDayLoserRebound(name="loser")
    scores = factor.compute_scores(symbol_data)
    
    # Yesterday's loser should have higher score (rebound)
    assert scores.loc[scores.index[-1], "INFY"] > scores.loc[scores.index[-1], "TCS"]


def test_sector_momentum_factor(make_ohlcv_frame):
    """Verify that SectorMomentumFactor groups and averages sector returns correctly."""
    # TCS and INFY are both mapped to the IT sector.
    symbol_data = {
        "TCS": make_ohlcv_frame(rows=10, trend=2.0),
        "INFY": make_ohlcv_frame(rows=10, trend=2.0),
        "HDFCBANK": make_ohlcv_frame(rows=10, trend=-1.0),
    }
    
    factor = SectorMomentumFactor(name="sector_mom")
    scores = factor.compute_scores(symbol_data)
    
    # IT sector has strong positive momentum, FIN has negative.
    assert scores.loc[scores.index[-1], "TCS"] > scores.loc[scores.index[-1], "HDFCBANK"]
    # TCS and INFY should have identical sector scores because they share the IT sector.
    assert scores.loc[scores.index[-1], "TCS"] == scores.loc[scores.index[-1], "INFY"]


def test_candidate_factors(make_ohlcv_frame):
    """Verify candidate factors run with expected shapes and finite outputs."""
    symbol_data = {
        "TCS": make_ohlcv_frame(rows=50),
        "INFY": make_ohlcv_frame(rows=50),
        "HDFCBANK": make_ohlcv_frame(rows=50),
    }
    
    gap = OvernightGapFactor(name="gap")
    comp = VolatilityCompressionFactor(name="comp", window=5)
    exp = ATRExpansionFactor(name="exp", fast_window=3, slow_window=10)
    prev = PreviousDayReturnFactor(name="prev_day")
    mom = ShortTermMomentumFactor(name="short_mom", lookback=3)
    sector_rel = SectorRelativeStrengthFactor(name="sector_rel", lookback=3)
    beta_adj = RollingBetaAdjustedStrengthFactor(name="beta_adj", beta_window=10, return_lookback=3)
    mom20 = Momentum20Factor(name="mom20", lookback=20)
    
    for f in (gap, comp, exp, prev, mom, sector_rel, beta_adj, mom20):
        scores = f.compute_scores(symbol_data)
        assert isinstance(scores, pd.DataFrame)
        assert set(scores.columns) == {"TCS", "INFY", "HDFCBANK"}
        assert len(scores) == 50
