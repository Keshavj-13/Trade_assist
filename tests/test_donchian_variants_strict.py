"""Tests for Donchian-only variant research path and strict stability gates."""

from __future__ import annotations

from dataclasses import replace

from predictor.research.donchian import (
    build_donchian_variant_universe,
    build_strict_donchian_validation_config,
    holm_adjust_p_values,
    select_stable_donchian_rows,
)
from predictor.research.harness import compare_strategies_across_symbols
from predictor.research.data import CSVHistoricalDataSource
from predictor.research.types import MultiSymbolStrategyRow


def test_build_donchian_variant_universe_contains_expected_variants() -> None:
    variants = build_donchian_variant_universe()
    names = {strategy.name for strategy in variants}

    assert "donchian_s1_20_10" in names
    assert "donchian_s2_55_20" in names
    assert "donchian_s1_after_loss" in names
    assert "donchian_s1_after_win" in names
    assert "donchian_s1_skip_after_win_failsafe55" in names
    assert "donchian_s1_atr_stop_2n" in names
    assert len(names) == len(variants)


def test_holm_adjust_p_values_matches_reference_example() -> None:
    adjusted = holm_adjust_p_values((0.01, 0.04, 0.03, 0.005))
    expected = (0.03, 0.06, 0.06, 0.02)
    for got, exp in zip(adjusted, expected):
        assert abs(got - exp) < 1e-12


def test_select_stable_donchian_rows_applies_strict_filters() -> None:
    rows = (
        MultiSymbolStrategyRow(
            strategy_name="weak_variant",
            symbols_tested=3,
            symbols_validated=3,
            avg_total_return=0.10,
            avg_sharpe_ratio=0.9,
            avg_max_drawdown=-0.20,
            avg_win_rate=0.50,
            avg_profit_factor=1.2,
            avg_in_sample_p_value=0.03,
            avg_walk_forward_p_value=0.06,
            avg_walk_forward_stability=0.60,
            validation_pass_rate=0.8,
            composite_score=0.7,
        ),
        MultiSymbolStrategyRow(
            strategy_name="stable_variant",
            symbols_tested=3,
            symbols_validated=3,
            avg_total_return=0.08,
            avg_sharpe_ratio=0.6,
            avg_max_drawdown=-0.12,
            avg_win_rate=0.52,
            avg_profit_factor=1.3,
            avg_in_sample_p_value=0.01,
            avg_walk_forward_p_value=0.01,
            avg_walk_forward_stability=0.65,
            validation_pass_rate=0.9,
            composite_score=0.8,
        ),
    )

    stable = select_stable_donchian_rows(
        rows,
        familywise_alpha=0.05,
        min_pass_rate=0.5,
        min_stability=0.55,
        min_avg_sharpe=0.3,
        max_abs_drawdown=0.25,
    )
    assert [row.strategy_name for row in stable] == ["stable_variant"]


def test_compare_donchian_variants_with_strict_config_runs() -> None:
    source = CSVHistoricalDataSource(directory="tests/fixtures/market_data")
    variants = build_donchian_variant_universe()[:4]
    config = replace(
        build_strict_donchian_validation_config(),
        permutation_iterations=20,
        train_window=160,
        test_window=50,
        walk_forward_step=50,
    )

    result = compare_strategies_across_symbols(
        symbols=("INFY_NS_1d", "TCS_NS_1d"),
        data_source=source,
        strategies=variants,
        config=config,
    )

    assert len(result.rows) == len(variants)
