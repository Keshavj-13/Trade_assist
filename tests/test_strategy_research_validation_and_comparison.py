"""Tests for four-stage validation and cross-strategy comparison harness."""

from __future__ import annotations

import pandas as pd

from predictor.research.comparison import compare_strategies
from predictor.research.data import CSVHistoricalDataSource
from predictor.research.harness import compare_strategies_across_symbols
from predictor.research.library import build_literature_strategy_universe
from predictor.research.validation import ResearchValidationConfig, validate_strategy


class _FlatStrategy:
    name = "flat"

    def generate_positions(self, frame: pd.DataFrame) -> pd.Series:
        return pd.Series(0.0, index=frame.index, dtype=float)


class _AlwaysLongStrategy:
    name = "always_long"

    def generate_positions(self, frame: pd.DataFrame) -> pd.Series:
        return pd.Series(1.0, index=frame.index, dtype=float)


def _test_config() -> ResearchValidationConfig:
    return ResearchValidationConfig(
        bars_per_year=252,
        permutation_iterations=80,
        permutation_block_size=12,
        train_window=180,
        test_window=60,
        walk_forward_step=60,
        p_value_threshold=0.05,
        random_seed=11,
    )


def test_literature_strategy_universe_contains_required_families() -> None:
    strategies = build_literature_strategy_universe()
    names = {strategy.name for strategy in strategies}

    assert "donchian_breakout" in names
    assert "ma_crossover" in names
    assert "mean_reversion" in names
    assert "momentum" in names
    assert "volatility_breakout" in names
    assert "regime_switching" in names
    assert any("hybrid" in name for name in names)


def test_validate_strategy_runs_all_four_stages(real_ohlcv_frame: pd.DataFrame) -> None:
    strategy = build_literature_strategy_universe()[0]
    from dataclasses import replace
    config = replace(_test_config(), p_value_threshold=0.999)
    report = validate_strategy(real_ohlcv_frame, strategy, config=config)

    assert report.strategy_name == strategy.name
    assert report.in_sample.metrics.sharpe_ratio == report.in_sample.metrics.sharpe_ratio
    assert 0.0 <= report.in_sample_permutation.p_value <= 1.0
    assert len(report.walk_forward_folds) >= 1
    assert 0.0 <= report.walk_forward_permutation.p_value <= 1.0
    assert 0.0 <= report.walk_forward_stability <= 1.0


def test_compare_strategies_returns_ranked_table(real_ohlcv_frame: pd.DataFrame) -> None:
    strategies = build_literature_strategy_universe()[:5]
    result = compare_strategies(real_ohlcv_frame, strategies, config=_test_config())

    assert len(result.rows) == len(strategies)
    scores = [row.composite_score for row in result.rows]
    assert scores == sorted(scores, reverse=True)
    assert all(0.0 <= row.in_sample_p_value <= 1.0 for row in result.rows)
    assert all(0.0 <= row.walk_forward_p_value <= 1.0 for row in result.rows)


def test_flat_strategy_is_rejected_by_validation(real_ohlcv_frame: pd.DataFrame) -> None:
    report = validate_strategy(real_ohlcv_frame, _FlatStrategy(), config=_test_config())
    assert report.is_valid is False


def test_rejected_strategy_preserves_raw_metrics(real_ohlcv_frame: pd.DataFrame) -> None:
    """Rejected reports must retain raw behavior metrics for interpretability."""
    config = ResearchValidationConfig(
        bars_per_year=252,
        permutation_iterations=30,
        permutation_block_size=10,
        train_window=180,
        test_window=60,
        walk_forward_step=60,
        p_value_threshold=1e-12,  # force rejection even if performance looks strong
        random_seed=17,
    )
    report = validate_strategy(real_ohlcv_frame, _AlwaysLongStrategy(), config=config)

    assert report.is_valid is False
    assert report.rejection_reason == "in_sample_permutation_failed"
    assert report.raw_metrics.trade_count > 0
    assert report.validated_metrics.trade_count == 0
    assert report.in_sample_permutation.p_value > 0.0


def test_aggregate_row_uses_raw_metrics_for_rejected_strategies() -> None:
    source = CSVHistoricalDataSource(directory="tests/fixtures/market_data")
    config = ResearchValidationConfig(
        bars_per_year=252,
        permutation_iterations=20,
        permutation_block_size=10,
        train_window=160,
        test_window=50,
        walk_forward_step=50,
        p_value_threshold=1e-12,
        random_seed=19,
    )
    result = compare_strategies_across_symbols(
        symbols=("INFY_NS_1d",),
        data_source=source,
        strategies=(_AlwaysLongStrategy(),),
        config=config,
    )
    assert len(result.rows) == 1
    row = result.rows[0]
    assert row.validation_pass_rate == 0.0
    assert row.avg_trade_count > 0.0
    assert row.avg_sharpe_ratio != 0.0


def test_multi_symbol_harness_aggregates_rankings() -> None:
    source = CSVHistoricalDataSource(directory="tests/fixtures/market_data")
    config = ResearchValidationConfig(
        bars_per_year=252,
        permutation_iterations=20,
        permutation_block_size=10,
        train_window=160,
        test_window=50,
        walk_forward_step=50,
        p_value_threshold=0.05,
        random_seed=13,
    )
    strategies = build_literature_strategy_universe()[:3]
    result = compare_strategies_across_symbols(
        symbols=("INFY_NS_1d", "TCS_NS_1d", "RELIANCE_NS_1d"),
        data_source=source,
        strategies=strategies,
        config=config,
    )

    assert len(result.rows) == len(strategies)
    assert all(row.symbols_tested == 3 for row in result.rows)
    assert all(0.0 <= row.validation_pass_rate <= 1.0 for row in result.rows)
    scores = [row.composite_score for row in result.rows]
    assert scores == sorted(scores, reverse=True)
