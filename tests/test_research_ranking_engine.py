"""Tests for cross-sectional ranking engine snapshots, ranking, and evaluation."""

from __future__ import annotations

import pandas as pd

from predictor.research.factors import (
    PreviousDayReturnFactor,
    RelativeVolumeFactor,
)
from predictor.research.ranking import (
    IntradayExecutionAssumptions,
    compute_next_day_returns,
)
from predictor.research.ranking_engine import (
    build_daily_feature_snapshots,
    generate_daily_rankings,
    rankings_to_score_panel,
    run_ranking_engine,
)


def _symbol_data(make_ohlcv_frame):
    return {
        "TCS": make_ohlcv_frame(rows=80, trend=0.12),
        "INFY": make_ohlcv_frame(rows=80, trend=0.08),
        "HDFCBANK": make_ohlcv_frame(rows=80, trend=0.04),
    }


def test_build_daily_feature_snapshots(make_ohlcv_frame):
    symbol_data = _symbol_data(make_ohlcv_frame)
    factors = {
        "previous_day_return": PreviousDayReturnFactor(name="previous_day_return"),
        "relative_volume": RelativeVolumeFactor(name="relative_volume", window=5),
    }
    snapshots = build_daily_feature_snapshots(symbol_data, factors)

    assert len(snapshots) > 0
    first = snapshots[0]
    assert "previous_day_return" in first.features.columns
    assert "relative_volume" in first.features.columns
    assert set(first.features.index) == {"TCS", "INFY", "HDFCBANK"}


def test_generate_daily_rankings_and_score_panel(make_ohlcv_frame):
    symbol_data = _symbol_data(make_ohlcv_frame)
    factors = {
        "previous_day_return": PreviousDayReturnFactor(name="previous_day_return"),
    }
    snapshots = build_daily_feature_snapshots(symbol_data, factors)
    rankings = generate_daily_rankings(snapshots, top_k=2)
    panel = rankings_to_score_panel(rankings)

    assert len(rankings) == len(snapshots)
    assert isinstance(panel, pd.DataFrame)
    assert len(panel.index) == len(rankings)
    assert all(len(r.selected_symbols) == 2 for r in rankings)


def test_run_ranking_engine_is_reproducible(make_ohlcv_frame):
    symbol_data = _symbol_data(make_ohlcv_frame)
    factors = {
        "previous_day_return": PreviousDayReturnFactor(name="previous_day_return"),
        "relative_volume": RelativeVolumeFactor(name="relative_volume", window=5),
    }
    targets = compute_next_day_returns(symbol_data)
    execution = IntradayExecutionAssumptions(
        transaction_cost_bps=5.0,
        entry_slippage_bps=2.0,
        exit_slippage_bps=2.0,
    )

    run_a = run_ranking_engine(
        symbol_data,
        factors,
        targets,
        top_k=2,
        execution=execution,
        factor_weights={"previous_day_return": 1.0, "relative_volume": 0.4},
    )
    run_b = run_ranking_engine(
        symbol_data,
        factors,
        targets,
        top_k=2,
        execution=execution,
        factor_weights={"previous_day_return": 1.0, "relative_volume": 0.4},
    )

    pd.testing.assert_series_equal(run_a.portfolio_returns, run_b.portfolio_returns)
    pd.testing.assert_frame_equal(run_a.score_panel, run_b.score_panel)
    assert run_a.metrics.turnover_adjusted_return <= run_a.metrics.annualised_return
