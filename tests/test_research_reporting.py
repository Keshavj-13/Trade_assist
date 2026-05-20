"""Tests for failure rate and robustness reporting."""

from __future__ import annotations

import pytest

from predictor.research.reporting import (
    build_symbol_robustness_rows,
    compute_failure_rates,
    compute_symbol_failure_rates,
    most_common_fail_reason,
    summarise_data_availability,
    worst_symbol_for_strategy,
)
from predictor.research.types import (
    DataAvailabilityReport,
    SymbolRobustnessRow,
)


def _make_row(
    strategy: str,
    symbol: str,
    is_valid: bool,
    fail_reasons: tuple = (),
    sharpe: float = 0.5,
    drawdown: float = -0.20,
    stability: float = 0.60,
) -> SymbolRobustnessRow:
    return SymbolRobustnessRow(
        strategy_name=strategy,
        symbol=symbol,
        is_valid=is_valid,
        fail_reasons=fail_reasons,
        sharpe_ratio=sharpe,
        max_drawdown=drawdown,
        walk_forward_stability=stability,
    )


def _make_avail(symbol: str, available: bool, rows: int = 400) -> DataAvailabilityReport:
    return DataAvailabilityReport(
        symbol=symbol,
        available=available,
        row_count=rows if available else 0,
        date_range="2020-01-01 to 2022-01-01" if available else "",
        failure_reason=None if available else "download_failed",
    )


# --- compute_failure_rates ---

def test_compute_failure_rates_correct_fraction() -> None:
    rows = [
        _make_row("strat_a", "SYM1", is_valid=True),
        _make_row("strat_a", "SYM2", is_valid=False),
        _make_row("strat_a", "SYM3", is_valid=False),
        _make_row("strat_b", "SYM1", is_valid=True),
        _make_row("strat_b", "SYM2", is_valid=True),
    ]
    rates = compute_failure_rates(rows)

    assert abs(rates["strat_a"] - 2 / 3) < 1e-9
    assert rates["strat_b"] == 0.0


def test_compute_failure_rates_all_valid() -> None:
    rows = [_make_row("s", f"SYM{i}", is_valid=True) for i in range(5)]
    rates = compute_failure_rates(rows)
    assert rates["s"] == 0.0


def test_compute_failure_rates_all_invalid() -> None:
    rows = [_make_row("s", f"SYM{i}", is_valid=False) for i in range(3)]
    rates = compute_failure_rates(rows)
    assert rates["s"] == 1.0


# --- compute_symbol_failure_rates ---

def test_compute_symbol_failure_rates_per_symbol() -> None:
    rows = [
        _make_row("strat_a", "SYM1", is_valid=True),
        _make_row("strat_b", "SYM1", is_valid=False),
        _make_row("strat_a", "SYM2", is_valid=False),
        _make_row("strat_b", "SYM2", is_valid=False),
    ]
    rates = compute_symbol_failure_rates(rows)
    assert abs(rates["SYM1"] - 0.5) < 1e-9
    assert rates["SYM2"] == 1.0


# --- worst_symbol_for_strategy ---

def test_worst_symbol_for_strategy_returns_lowest_sharpe() -> None:
    rows = [
        _make_row("strat", "SYM1", is_valid=True, sharpe=1.0),
        _make_row("strat", "SYM2", is_valid=True, sharpe=-0.5),
        _make_row("strat", "SYM3", is_valid=False, sharpe=0.2),
    ]
    worst = worst_symbol_for_strategy(rows, "strat")
    assert worst is not None
    assert worst.symbol == "SYM2"


def test_worst_symbol_for_strategy_returns_none_if_no_match() -> None:
    rows = [_make_row("other", "SYM1", is_valid=True)]
    assert worst_symbol_for_strategy(rows, "missing") is None


# --- most_common_fail_reason ---

def test_most_common_fail_reason_finds_dominant_reason() -> None:
    rows = [
        _make_row("s", "S1", False, ("in_sample_permutation_failed",)),
        _make_row("s", "S2", False, ("in_sample_permutation_failed", "walk_forward_stability_failed")),
        _make_row("s", "S3", False, ("walk_forward_stability_failed",)),
    ]
    reason = most_common_fail_reason(rows)
    # in_sample_permutation_failed appears twice, walk_forward_stability_failed twice too
    # Both appear twice; either is acceptable
    assert reason in {"in_sample_permutation_failed", "walk_forward_stability_failed"}


def test_most_common_fail_reason_returns_none_for_empty() -> None:
    assert most_common_fail_reason([]) is None


def test_most_common_fail_reason_returns_none_for_all_valid() -> None:
    rows = [_make_row("s", f"S{i}", True) for i in range(3)]
    assert most_common_fail_reason(rows) is None


# --- summarise_data_availability ---

def test_summarise_data_availability_correct_counts() -> None:
    availability = [
        _make_avail("SYM1", True),
        _make_avail("SYM2", True),
        _make_avail("SYM3", False),
    ]
    summary = summarise_data_availability(availability)
    assert summary["total"] == 3
    assert summary["available"] == 2
    assert summary["missing"] == 1
    assert abs(summary["availability_rate"] - 2 / 3) < 1e-3


def test_summarise_data_availability_empty() -> None:
    summary = summarise_data_availability([])
    assert summary["total"] == 0
    assert summary["availability_rate"] == 0.0


def test_summarise_data_availability_all_available() -> None:
    availability = [_make_avail(f"SYM{i}", True, rows=500) for i in range(5)]
    summary = summarise_data_availability(availability)
    assert summary["availability_rate"] == 1.0
    assert summary["missing"] == 0
    assert summary["avg_rows_available"] == 500.0
