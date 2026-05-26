"""Failure rate, robustness, and data availability reporting."""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np

from predictor.research.types import (
    DataAvailabilityReport,
    MultiSymbolStrategyRow,
    StrategyValidationReport,
    SymbolRobustnessRow,
)


def build_symbol_robustness_rows(
    reports: Sequence[Tuple[str, str, StrategyValidationReport]],
) -> Tuple[SymbolRobustnessRow, ...]:
    """Build per-symbol robustness rows from (strategy_name, symbol, report) triples."""
    rows: List[SymbolRobustnessRow] = []
    for strategy_name, symbol, report in reports:
        raw = report.resolved_raw_metrics
        rows.append(
            SymbolRobustnessRow(
                strategy_name=strategy_name,
                symbol=symbol,
                is_valid=report.is_valid,
                fail_reasons=report.fail_reasons,
                sharpe_ratio=raw.sharpe_ratio,
                max_drawdown=raw.max_drawdown,
                walk_forward_stability=report.walk_forward_stability,
            )
        )
    return tuple(rows)


def compute_failure_rates(
    robustness_rows: Sequence[SymbolRobustnessRow],
) -> Dict[str, float]:
    """Return per-strategy failure rate (fraction of symbols where is_valid is False)."""
    by_strategy: Dict[str, List[bool]] = {}
    for row in robustness_rows:
        by_strategy.setdefault(row.strategy_name, []).append(row.is_valid)
    return {
        name: float(1.0 - np.mean(valid_flags))
        for name, valid_flags in by_strategy.items()
    }


def compute_symbol_failure_rates(
    robustness_rows: Sequence[SymbolRobustnessRow],
) -> Dict[str, float]:
    """Return per-symbol failure rate across all strategies."""
    by_symbol: Dict[str, List[bool]] = {}
    for row in robustness_rows:
        by_symbol.setdefault(row.symbol, []).append(row.is_valid)
    return {
        sym: float(1.0 - np.mean(valid_flags))
        for sym, valid_flags in by_symbol.items()
    }


def worst_symbol_for_strategy(
    robustness_rows: Sequence[SymbolRobustnessRow],
    strategy_name: str,
) -> SymbolRobustnessRow | None:
    """Return the symbol row with the lowest Sharpe ratio for a given strategy."""
    filtered = [r for r in robustness_rows if r.strategy_name == strategy_name]
    if not filtered:
        return None
    return min(filtered, key=lambda r: r.sharpe_ratio)


def compute_global_fail_reason_rates(
    robustness_rows: Sequence[SymbolRobustnessRow],
) -> Dict[str, float]:
    """Return the proportion of failed validations that include each reason."""
    failed_rows = [r for r in robustness_rows if not r.is_valid]
    if not failed_rows:
        return {}
    
    total_failures = len(failed_rows)
    counts: Dict[str, int] = {}
    for row in failed_rows:
        # Avoid counting the same reason multiple times per row if it happens
        for reason in set(row.fail_reasons):
            counts[reason] = counts.get(reason, 0) + 1
            
    return {
        reason: float(count / total_failures)
        for reason, count in counts.items()
    }

def most_common_fail_reason(
    robustness_rows: Sequence[SymbolRobustnessRow],
) -> str | None:
    """Return the most frequently occurring fail reason across all rows."""
    counts: Dict[str, int] = {}
    for row in robustness_rows:
        for reason in row.fail_reasons:
            counts[reason] = counts.get(reason, 0) + 1
    if not counts:
        return None
    return max(counts, key=lambda k: counts[k])


def summarise_data_availability(
    availability: Sequence[DataAvailabilityReport],
) -> Dict[str, int | float]:
    """Return a summary dict of data availability across symbols."""
    total = len(availability)
    if total == 0:
        return {"total": 0, "available": 0, "missing": 0, "availability_rate": 0.0}
    n_available = sum(1 for r in availability if r.available)
    avg_rows = float(
        np.mean([r.row_count for r in availability if r.available]) if n_available else 0.0
    )
    return {
        "total": total,
        "available": n_available,
        "missing": total - n_available,
        "availability_rate": round(n_available / total, 4),
        "avg_rows_available": round(avg_rows, 1),
    }
