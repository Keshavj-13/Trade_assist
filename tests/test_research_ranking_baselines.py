"""Tests for mandatory baseline ranking systems."""

from __future__ import annotations

import pandas as pd

from predictor.research.ranking_baselines import build_ranking_baseline_universe


def test_build_ranking_baseline_universe(make_ohlcv_frame):
    baselines = build_ranking_baseline_universe()
    names = [b.name for b in baselines]
    assert names == [
        "random_ranking",
        "buy_and_hold_baseline",
        "simple_momentum_rank",
        "volatility_rank",
        "equal_weight_selection",
    ]

    symbol_data = {
        "TCS": make_ohlcv_frame(rows=60),
        "INFY": make_ohlcv_frame(rows=60),
        "HDFCBANK": make_ohlcv_frame(rows=60),
    }
    for baseline in baselines:
        scores = baseline.compute_scores(symbol_data)
        assert isinstance(scores, pd.DataFrame)
        assert set(scores.columns) == {"TCS", "INFY", "HDFCBANK"}
        assert len(scores) == 60
