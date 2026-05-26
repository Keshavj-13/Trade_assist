"""Unit tests for cross-sectional ranking alignment and performance evaluation metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from predictor.research.ranking import (
    compute_daily_regimes,
    compute_next_day_returns,
    evaluate_ranking,
)


def test_compute_next_day_returns(make_ohlcv_frame):
    """Verify that compute_next_day_returns correctly aligns tomorrow's return to today's decision index."""
    # TCS tomorrow's open is 99, tomorrow's close is 100 => intraday return is (100 - 99)/99 = 1.01%
    tcs_df = pd.DataFrame(
        {
            "Open": [10.0, 99.0, 50.0],
            "High": [10.5, 100.5, 50.5],
            "Low": [9.5, 98.5, 49.5],
            "Close": [10.0, 100.0, 50.0],
            "Volume": [1000, 1000, 1000],
        },
        index=pd.date_range("2026-01-01", periods=3),
    )
    
    symbol_data = {"TCS": tcs_df}
    targets = compute_next_day_returns(symbol_data)
    
    assert isinstance(targets, pd.DataFrame)
    # The return of row 1 (date 2) is (100-99)/99 = 1/99 ≈ 0.0101
    # This return must be shifted back to row 0 (date 1)
    assert abs(targets.loc[targets.index[0], "TCS"] - 1.0 / 99.0) < 1e-6
    # Last row target should be NaN since there is no tomorrow
    assert pd.isna(targets.loc[targets.index[-1], "TCS"])


def test_compute_daily_regimes(make_ohlcv_frame):
    """Verify that compute_daily_regimes maps regimes across the universe timeline."""
    symbol_data = {
        "TCS": make_ohlcv_frame(rows=30),
        "INFY": make_ohlcv_frame(rows=30),
    }
    
    regimes = compute_daily_regimes(symbol_data)
    assert isinstance(regimes, pd.Series)
    assert len(regimes) == 30
    assert not regimes.empty
    assert set(regimes.unique()).issubset(
        {"HIGH_VOL", "LOW_VOL", "TRENDING", "SIDEWAYS", "CRISIS", "RECOVERY"}
    )


def test_evaluate_ranking():
    """Verify that evaluate_ranking accurately constructs portfolios, applies costs, and computes metrics."""
    # 5 symbols, 3 days
    dates = pd.date_range("2026-01-01", periods=3)
    symbols = ["A", "B", "C", "D", "E"]
    
    # Factor scores at day t
    scores = pd.DataFrame(
        [
            [1.0, 2.0, 3.0, 4.0, 5.0],  # Day 0: E and D are top 2
            [5.0, 4.0, 3.0, 2.0, 1.0],  # Day 1: A and B are top 2
            [1.0, 1.0, 1.0, 1.0, 1.0],  # Day 2: Equal
        ],
        index=dates,
        columns=symbols,
    )
    
    # Tomorrow's intraday returns (targets) aligned to day t
    targets = pd.DataFrame(
        [
            [0.01, 0.02, -0.01, 0.04, 0.06],  # Day 0: D is +4%, E is +6% => Top 2 average = +5%
            [0.05, -0.02, 0.01, 0.02, 0.03],  # Day 1: A is +5%, B is -2% => Top 2 average = +1.5%
            [0.01, 0.01, 0.01, 0.01, 0.01],
        ],
        index=dates,
        columns=symbols,
    )
    
    port_rets, metrics = evaluate_ranking(
        scores=scores,
        targets=targets,
        top_k=2,
        transaction_cost_bps=10.0,  # 10 bps = 0.001
    )
    
    assert isinstance(port_rets, pd.Series)
    assert len(port_rets) == 3
    
    # Day 0 Top 2 (D, E) returns: (0.04 + 0.06)/2 = 0.05 => 5%
    assert abs(port_rets.iloc[0] - 0.05) < 1e-6
    # Day 1 Top 2 (A, B) returns: (0.05 - 0.02)/2 = 0.015 => 1.5%
    assert abs(port_rets.iloc[1] - 0.015) < 1e-6

    # Verify ranking metrics
    assert metrics.mean_selected_return > 0
    # Day 0 top 2 are both positive => precision = 100%
    # Day 1 top 2 has 1 positive (A) => precision = 50%
    # Average precision = 75% (for day 0 and day 1)
    assert abs(metrics.precision_at_k - 0.75) < 0.1
    
    # Top 1 hit rate: Day 0 E is +6% (>0), Day 1 A is +5% (>0) => Hit Rate = 100%
    assert metrics.top_1_hit_rate == 1.0
    
    # Turnover from Day 0 (D, E) to Day 1 (A, B) is 100% (completely changed)
    assert metrics.mean_turnover > 0.0
    # Adjusted returns should be lower than raw due to turnover trans-costs
    assert metrics.turnover_adjusted_return < metrics.annualised_return
    assert metrics.top_k_mean_return == metrics.mean_selected_return
    assert metrics.rank_correlation == metrics.mean_ic
    assert metrics.information_coefficient == metrics.mean_ic
    assert metrics.average_selected_return == metrics.mean_selected_return
    assert metrics.intraday_drawdown == metrics.max_drawdown


def test_evaluate_ranking_slippage_cost_assumption():
    """Round-trip execution cost + slippage should reduce adjusted annualised return."""
    dates = pd.date_range("2026-01-01", periods=6)
    scores = pd.DataFrame(
        [[3.0, 2.0, 1.0]] * 6,
        index=dates,
        columns=["A", "B", "C"],
    )
    targets = pd.DataFrame(
        [[0.01, 0.005, 0.0]] * 6,
        index=dates,
        columns=["A", "B", "C"],
    )

    _, no_slippage = evaluate_ranking(
        scores=scores,
        targets=targets,
        top_k=1,
        transaction_cost_bps=5.0,
        slippage_bps=0.0,
    )
    _, with_slippage = evaluate_ranking(
        scores=scores,
        targets=targets,
        top_k=1,
        transaction_cost_bps=5.0,
        slippage_bps=10.0,
    )

    assert with_slippage.turnover_adjusted_return < no_slippage.turnover_adjusted_return
