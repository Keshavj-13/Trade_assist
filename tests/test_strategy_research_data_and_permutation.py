"""Tests for research data contracts, permutation behavior, and walk-forward splits."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from predictor.research.data import CSVHistoricalDataSource, validate_research_frame
from predictor.research.errors import ResearchDataError, ResearchInputError
from predictor.research.permutation import block_permute_ohlcv
from predictor.research.validation import build_walk_forward_splits


def test_csv_historical_source_loads_frame(load_real_ohlcv_frame) -> None:
    source = CSVHistoricalDataSource(directory="tests/fixtures/market_data")
    frame = source.fetch_ohlcv("INFY_NS_1d")

    assert isinstance(frame, pd.DataFrame)
    assert not frame.empty
    assert set(frame.columns) == {"Open", "High", "Low", "Close", "Volume"}


def test_csv_historical_source_is_case_insensitive() -> None:
    source = CSVHistoricalDataSource(directory="tests/fixtures/market_data")
    frame = source.fetch_ohlcv("INFY_NS_1D")
    assert not frame.empty


def test_csv_historical_source_raises_for_missing_symbol() -> None:
    source = CSVHistoricalDataSource(directory="tests/fixtures/market_data")

    with pytest.raises(ResearchDataError):
        source.fetch_ohlcv("DOES_NOT_EXIST")


def test_validate_research_frame_rejects_missing_columns(real_ohlcv_frame: pd.DataFrame) -> None:
    bad = real_ohlcv_frame.drop(columns=["Volume"])

    with pytest.raises(ResearchInputError):
        validate_research_frame(bad, symbol="INFY")


def test_block_permutation_is_deterministic(real_ohlcv_frame: pd.DataFrame) -> None:
    perm_a = block_permute_ohlcv(real_ohlcv_frame, block_size=20, seed=7)
    perm_b = block_permute_ohlcv(real_ohlcv_frame, block_size=20, seed=7)

    assert perm_a.equals(perm_b)
    assert len(perm_a) == len(real_ohlcv_frame)
    assert list(perm_a.columns) == list(real_ohlcv_frame.columns)
    assert not perm_a["Close"].equals(real_ohlcv_frame["Close"])


def test_block_permutation_preserves_distribution_shape(real_ohlcv_frame: pd.DataFrame) -> None:
    permuted = block_permute_ohlcv(real_ohlcv_frame, block_size=15, seed=99)
    original_returns = np.log(real_ohlcv_frame["Close"]).diff().dropna()
    permuted_returns = np.log(permuted["Close"]).diff().dropna()

    mean_gap = abs(float(original_returns.mean() - permuted_returns.mean()))
    std_gap = abs(float(original_returns.std() - permuted_returns.std()))
    rel_std_gap = std_gap / max(float(original_returns.std()), 1e-12)

    # Preserve broad distributional structure while altering sequence.
    assert mean_gap < 0.01
    assert rel_std_gap < 0.35


def test_walk_forward_splits_are_ordered_and_non_overlapping() -> None:
    splits = build_walk_forward_splits(
        total_rows=300,
        train_size=120,
        test_size=40,
        step_size=40,
    )

    assert len(splits) == 4
    assert splits[0].train_start == 0
    assert splits[0].train_end == 119
    assert splits[0].test_start == 120
    assert splits[0].test_end == 159
    for split in splits:
        assert split.train_start <= split.train_end < split.test_start <= split.test_end


def test_walk_forward_splits_reject_invalid_windows() -> None:
    with pytest.raises(ResearchInputError):
        build_walk_forward_splits(total_rows=60, train_size=50, test_size=20, step_size=10)
