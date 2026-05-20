"""Tests for parallel research harness correctness and determinism."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from predictor.research.data import CSVHistoricalDataSource, CachedDataSource
from predictor.research.errors import ResearchDataError
from predictor.research.harness import compare_strategies_across_symbols, run_research
from predictor.research.library import build_core_strategy_universe
from predictor.research.strategies import DonchianBreakoutStrategy, MomentumStrategy
from predictor.research.types import ResearchRunResult
from predictor.research.validation import ResearchValidationConfig


_FIXTURE_DIR = "tests/fixtures/market_data"
_TEST_SYMBOLS = ("INFY_NS_1d", "TCS_NS_1d")


def _fast_config() -> ResearchValidationConfig:
    return ResearchValidationConfig(
        bars_per_year=252,
        permutation_iterations=15,
        permutation_block_size=10,
        train_window=160,
        test_window=50,
        walk_forward_step=50,
        p_value_threshold=0.05,
        random_seed=99,
    )


# --- CachedDataSource ---

def test_cached_source_fetches_once() -> None:
    source = CSVHistoricalDataSource(directory=_FIXTURE_DIR)
    cached = CachedDataSource(source)

    f1 = cached.fetch_ohlcv("INFY_NS_1d")
    f2 = cached.fetch_ohlcv("INFY_NS_1d")

    assert f1 is f2, "Second call must return the exact same cached object"


def test_cached_source_handles_multiple_symbols() -> None:
    source = CSVHistoricalDataSource(directory=_FIXTURE_DIR)
    cached = CachedDataSource(source)

    for sym in _TEST_SYMBOLS:
        frame = cached.fetch_ohlcv(sym)
        assert not frame.empty

    assert set(cached.cached_symbols) == set(_TEST_SYMBOLS)


def test_cached_source_reraises_fetch_error() -> None:
    source = CSVHistoricalDataSource(directory=_FIXTURE_DIR)
    cached = CachedDataSource(source)

    with pytest.raises(ResearchDataError):
        cached.fetch_ohlcv("DOES_NOT_EXIST")
    # Second call should also raise (error is stored)
    with pytest.raises(ResearchDataError):
        cached.fetch_ohlcv("DOES_NOT_EXIST")


def test_cached_source_prefetch_returns_none_for_success() -> None:
    source = CSVHistoricalDataSource(directory=_FIXTURE_DIR)
    cached = CachedDataSource(source)
    results = cached.prefetch(("INFY_NS_1d", "TCS_NS_1d"))

    assert results["INFY_NS_1d"] is None
    assert results["TCS_NS_1d"] is None


def test_cached_source_prefetch_captures_errors() -> None:
    source = CSVHistoricalDataSource(directory=_FIXTURE_DIR)
    cached = CachedDataSource(source)
    results = cached.prefetch(("INFY_NS_1d", "MISSING_SYM"))

    assert results["INFY_NS_1d"] is None
    assert results["MISSING_SYM"] is not None
    assert isinstance(results["MISSING_SYM"], ResearchDataError)


# --- run_research parallel harness ---

def test_run_research_returns_correct_type() -> None:
    source = CSVHistoricalDataSource(directory=_FIXTURE_DIR)
    strategies = [
        DonchianBreakoutStrategy(lookback=20, name="donchian_test"),
        MomentumStrategy(lookback=20, name="momentum_test"),
    ]
    result = run_research(
        symbols=_TEST_SYMBOLS,
        data_source=source,
        strategies=strategies,
        config=_fast_config(),
        max_workers=2,
    )
    assert isinstance(result, ResearchRunResult)


def test_run_research_comparison_ranked_descending() -> None:
    source = CSVHistoricalDataSource(directory=_FIXTURE_DIR)
    strategies = build_core_strategy_universe()[:3]
    result = run_research(
        symbols=_TEST_SYMBOLS,
        data_source=source,
        strategies=strategies,
        config=_fast_config(),
        max_workers=2,
    )
    scores = [row.composite_score for row in result.comparison.rows]
    assert scores == sorted(scores, reverse=True)


def test_run_research_data_availability_reports_all_symbols() -> None:
    source = CSVHistoricalDataSource(directory=_FIXTURE_DIR)
    strategies = [DonchianBreakoutStrategy(lookback=20, name="donchian_test")]
    result = run_research(
        symbols=_TEST_SYMBOLS,
        data_source=source,
        strategies=strategies,
        config=_fast_config(),
        max_workers=1,
    )
    reported_symbols = {r.symbol for r in result.data_availability}
    assert reported_symbols == {s.upper() for s in _TEST_SYMBOLS}
    assert all(r.available for r in result.data_availability)


def test_run_research_handles_missing_symbol_gracefully() -> None:
    source = CSVHistoricalDataSource(directory=_FIXTURE_DIR)
    strategies = [DonchianBreakoutStrategy(lookback=20, name="donchian_test")]
    result = run_research(
        symbols=("INFY_NS_1d", "NONEXISTENT_SYM"),
        data_source=source,
        strategies=strategies,
        config=_fast_config(),
        max_workers=2,
    )
    avail = {r.symbol: r.available for r in result.data_availability}
    assert avail.get("INFY_NS_1D") is True
    assert avail.get("NONEXISTENT_SYM") is False
    # Comparison still runs on available symbols
    assert len(result.comparison.rows) == 1


def test_run_research_produces_robustness_rows() -> None:
    source = CSVHistoricalDataSource(directory=_FIXTURE_DIR)
    strategies = [
        DonchianBreakoutStrategy(lookback=20, name="donchian_test"),
        MomentumStrategy(lookback=20, name="momentum_test"),
    ]
    result = run_research(
        symbols=_TEST_SYMBOLS,
        data_source=source,
        strategies=strategies,
        config=_fast_config(),
        max_workers=2,
    )
    # 2 strategies × 2 symbols = 4 robustness rows
    assert len(result.symbol_robustness) == 4
    strategy_names = {r.strategy_name for r in result.symbol_robustness}
    assert strategy_names == {"donchian_test", "momentum_test"}


def test_run_research_matches_sequential_harness() -> None:
    """Parallel run must agree with sequential run on the same ranked order."""
    source = CSVHistoricalDataSource(directory=_FIXTURE_DIR)
    strategies = [
        DonchianBreakoutStrategy(lookback=20, name="donchian_test"),
        MomentumStrategy(lookback=20, name="momentum_test"),
    ]
    config = _fast_config()

    sequential = compare_strategies_across_symbols(
        symbols=_TEST_SYMBOLS,
        data_source=source,
        strategies=strategies,
        config=config,
    )
    parallel_result = run_research(
        symbols=_TEST_SYMBOLS,
        data_source=source,
        strategies=strategies,
        config=config,
        max_workers=2,
    )

    seq_names = [row.strategy_name for row in sequential.rows]
    par_names = [row.strategy_name for row in parallel_result.comparison.rows]
    assert seq_names == par_names, (
        f"Sequential order {seq_names} must match parallel order {par_names}"
    )


def test_run_research_is_deterministic_across_two_runs() -> None:
    source = CSVHistoricalDataSource(directory=_FIXTURE_DIR)
    strategies = [
        DonchianBreakoutStrategy(lookback=20, name="donchian_test"),
        MomentumStrategy(lookback=20, name="momentum_test"),
    ]
    config = _fast_config()

    r1 = run_research(
        symbols=_TEST_SYMBOLS,
        data_source=source,
        strategies=strategies,
        config=config,
        max_workers=2,
    )
    r2 = run_research(
        symbols=_TEST_SYMBOLS,
        data_source=source,
        strategies=strategies,
        config=config,
        max_workers=2,
    )

    assert [r.strategy_name for r in r1.comparison.rows] == [
        r.strategy_name for r in r2.comparison.rows
    ]
    for row1, row2 in zip(r1.comparison.rows, r2.comparison.rows):
        assert abs(row1.composite_score - row2.composite_score) < 1e-9


def test_run_research_all_symbols_missing_returns_empty_comparison() -> None:
    source = CSVHistoricalDataSource(directory=_FIXTURE_DIR)
    strategies = [DonchianBreakoutStrategy(lookback=20, name="d")]
    result = run_research(
        symbols=("MISSING_A", "MISSING_B"),
        data_source=source,
        strategies=strategies,
        config=_fast_config(),
        max_workers=1,
    )
    assert result.comparison.rows == ()
    assert all(not r.available for r in result.data_availability)
