"""Tests for predictor.research.calibration.

Covers:
- FutureLeakStrategy (Oracle)
- PureRandomEntryStrategy (Null)
- MildPredictiveProcess (Weak signal)
- run_permutation_calibration
- print_calibration_report
"""

from __future__ import annotations

import pandas as pd
import pytest

from predictor.research.calibration import (
    FutureLeakStrategy,
    PureRandomEntryStrategy,
    MildPredictiveProcess,
    run_permutation_calibration,
    print_calibration_report,
)
from predictor.research.validation import ResearchValidationConfig


def _make_synth_data(n: int = 150) -> pd.DataFrame:
    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    import numpy as np
    # Trending series
    close = 100.0 + np.cumsum(np.random.default_rng(42).normal(0.1, 1.0, size=n))
    high = close + 1.0
    low = close - 1.0
    open_ = close
    volume = np.ones(n) * 1000
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=idx,
    )


def test_future_leak_strategy():
    frame = _make_synth_data()
    # FutureLeakStrategy parameters: threshold
    strat = FutureLeakStrategy(threshold=0.0)
    positions = strat.generate_positions(frame)
    assert len(positions) == len(frame)
    # Verify it entered in the direction of future returns (shifted by -1)
    future_return = frame["Close"].pct_change().shift(-1)
    for i in range(len(frame) - 1):
        fut_ret = future_return.iloc[i]
        pos = positions.iloc[i]
        if fut_ret > 0:
            assert pos >= 0
        elif fut_ret < 0:
            assert pos <= 0


def test_pure_random_entry_strategy():
    frame = _make_synth_data()
    strat = PureRandomEntryStrategy(seed=42)
    positions1 = strat.generate_positions(frame)
    positions2 = strat.generate_positions(frame)
    # Check reproducibility
    pd.testing.assert_series_equal(positions1, positions2)
    assert set(positions1.unique()).issubset({0.0, 1.0, -1.0})


def test_mild_predictive_process():
    frame = _make_synth_data()
    # MildPredictiveProcess parameters: signal_strength, seed
    strat = MildPredictiveProcess(signal_strength=0.15, seed=42)
    positions = strat.generate_positions(frame)
    assert len(positions) == len(frame)
    assert set(positions.unique()).issubset({0.0, 1.0, -1.0})


def test_run_permutation_calibration():
    frame = _make_synth_data()
    config = ResearchValidationConfig(permutation_iterations=10, random_seed=42)
    # Run with small n_trials for fast execution
    res = run_permutation_calibration(frame, n_trials=2, config=config)
    
    assert len(res) == 3
    oracle, random, mild = res
    
    assert oracle.strategy_type == "oracle"
    assert random.strategy_type == "random"
    assert mild.strategy_type == "mild"
    
    # Run print report to ensure no errors
    print_calibration_report(res)
