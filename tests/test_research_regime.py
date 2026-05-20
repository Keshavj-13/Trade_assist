"""Tests for market regime detection and slice extraction."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from predictor.research.errors import ResearchInputError
from predictor.research.regime import (
    ALL_REGIMES,
    classify_regimes,
    dominant_regimes,
    extract_regime_slices,
    regime_distribution,
)


def _make_trending_frame(rows: int = 300, uptrend: bool = True) -> pd.DataFrame:
    idx = pd.date_range("2020-01-02", periods=rows, freq="B")
    direction = 1.0 if uptrend else -1.0
    close = 100.0 + direction * np.arange(rows, dtype=float) * 0.3
    high = close + 0.5
    low = close - 0.5
    return pd.DataFrame(
        {"Open": close - 0.1, "High": high, "Low": low, "Close": close, "Volume": 1e6},
        index=idx,
    )


def _make_crash_frame(rows: int = 300) -> pd.DataFrame:
    idx = pd.date_range("2020-01-02", periods=rows, freq="B")
    # Peak then sharp crash
    close = np.concatenate([
        100.0 + np.arange(100, dtype=float) * 0.5,   # bull phase
        150.0 - np.arange(200, dtype=float) * 0.4,   # crash phase
    ])
    close = close[:rows]
    high = close + 0.5
    low = close - 0.5
    return pd.DataFrame(
        {"Open": close - 0.1, "High": high, "Low": low, "Close": close, "Volume": 1e6},
        index=idx,
    )


def _make_real_frame(path: str = "tests/fixtures/market_data/INFY_NS_1d.csv") -> pd.DataFrame:
    frame = pd.read_csv(path, parse_dates=["Date"]).set_index("Date")
    return frame[["Open", "High", "Low", "Close", "Volume"]].astype(float).sort_index()


def test_classify_regimes_returns_series_aligned_with_input() -> None:
    frame = _make_trending_frame(300)
    labels = classify_regimes(frame)

    assert isinstance(labels, pd.Series)
    assert len(labels) == len(frame)
    assert labels.index.equals(frame.index)


def test_classify_regimes_all_labels_are_valid_regimes() -> None:
    frame = _make_trending_frame(300)
    labels = classify_regimes(frame)
    assert set(labels.unique()).issubset(set(ALL_REGIMES))


def test_classify_regimes_detects_bull_in_strong_uptrend() -> None:
    frame = _make_trending_frame(300, uptrend=True)
    labels = classify_regimes(frame)
    # Most of the later bars should be BULL
    tail = labels.iloc[150:]
    bull_frac = (tail == "BULL").mean()
    assert bull_frac > 0.4, f"Expected mostly BULL, got {bull_frac:.2%}"


def test_classify_regimes_detects_crash_in_sharp_decline() -> None:
    frame = _make_crash_frame(300)
    labels = classify_regimes(frame)
    crash_count = (labels == "CRASH").sum()
    assert crash_count > 0, "Expected at least some CRASH bars"


def test_classify_regimes_raises_on_empty_frame() -> None:
    with pytest.raises(ResearchInputError):
        classify_regimes(pd.DataFrame())


def test_classify_regimes_raises_on_missing_close_column() -> None:
    bad = pd.DataFrame({"Open": [1.0, 2.0], "High": [1.1, 2.1]})
    with pytest.raises(ResearchInputError):
        classify_regimes(bad)


def test_regime_distribution_sums_to_one() -> None:
    frame = _make_trending_frame(300)
    dist = regime_distribution(frame)
    assert abs(sum(dist.values()) - 1.0) < 1e-9
    assert set(dist.keys()) == set(ALL_REGIMES)


def test_regime_distribution_non_negative() -> None:
    frame = _make_crash_frame(300)
    dist = regime_distribution(frame)
    assert all(v >= 0.0 for v in dist.values())


def test_extract_regime_slices_returns_list_of_dataframes() -> None:
    frame = _make_trending_frame(300)
    slices = extract_regime_slices(frame, "BULL", min_bars=20)
    assert isinstance(slices, list)
    for s in slices:
        assert isinstance(s, pd.DataFrame)
        assert len(s) >= 20


def test_extract_regime_slices_raises_on_invalid_regime() -> None:
    frame = _make_trending_frame(200)
    with pytest.raises(ResearchInputError):
        extract_regime_slices(frame, "INVALID_REGIME")


def test_extract_regime_slices_minimum_bars_filter() -> None:
    frame = _make_trending_frame(300)
    # With very large min_bars, few or no slices should pass
    slices_strict = extract_regime_slices(frame, "BULL", min_bars=500)
    assert len(slices_strict) == 0


def test_dominant_regimes_returns_top_n() -> None:
    frame = _make_trending_frame(300)
    top = dominant_regimes(frame, top_n=3)
    assert len(top) == 3
    assert all(r in ALL_REGIMES for r in top)


def test_dominant_regimes_raises_on_invalid_top_n() -> None:
    frame = _make_trending_frame(200)
    with pytest.raises(ResearchInputError):
        dominant_regimes(frame, top_n=0)


def test_classify_regimes_on_real_data_produces_multiple_regimes() -> None:
    frame = _make_real_frame()
    dist = regime_distribution(frame)
    regimes_present = sum(1 for v in dist.values() if v > 0.0)
    assert regimes_present >= 2, "Real data should span at least 2 regimes"
