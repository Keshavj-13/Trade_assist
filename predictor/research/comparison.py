"""Cross-strategy comparison harness on a shared validation framework."""

from __future__ import annotations

from typing import List, Sequence

import numpy as np
import pandas as pd

from predictor.research.metrics import finite_profit_factor
from predictor.research.strategies import TradingStrategy, to_strategy_tuple
from predictor.research.types import StrategyComparisonResult, StrategyComparisonRow
from predictor.research.validation import ResearchValidationConfig, validate_strategy


def _composite_score(row: StrategyComparisonRow) -> float:
    """Compute objective score balancing performance and robustness."""

    sharpe_component = np.tanh(row.sharpe_ratio / 2.0)
    return_component = np.tanh(row.total_return * 4.0)
    drawdown_component = 1.0 - min(1.0, abs(row.max_drawdown))
    profit_component = np.tanh(finite_profit_factor(row.profit_factor) / 4.0)
    p_value_component = 1.0 - ((row.in_sample_p_value + row.walk_forward_p_value) / 2.0)
    stability_component = row.walk_forward_stability
    valid_bonus = 0.2 if row.is_valid else 0.0
    return float(
        0.22 * sharpe_component
        + 0.18 * return_component
        + 0.15 * drawdown_component
        + 0.10 * profit_component
        + 0.20 * p_value_component
        + 0.15 * stability_component
        + valid_bonus
    )


def compare_strategies(
    frame: pd.DataFrame,
    strategies: Sequence[TradingStrategy],
    *,
    config: ResearchValidationConfig,
) -> StrategyComparisonResult:
    """Validate and rank strategies, variants, and combinations fairly."""

    strategy_tuple = to_strategy_tuple(strategies)
    rows: List[StrategyComparisonRow] = []
    for strategy in strategy_tuple:
        report = validate_strategy(frame, strategy, config=config)
        raw = report.resolved_raw_metrics
        row = StrategyComparisonRow(
            strategy_name=report.strategy_name,
            is_valid=report.is_valid,
            total_return=raw.total_return,
            sharpe_ratio=raw.sharpe_ratio,
            max_drawdown=raw.max_drawdown,
            win_rate=raw.win_rate,
            profit_factor=raw.profit_factor,
            in_sample_p_value=report.in_sample_permutation.p_value,
            walk_forward_p_value=report.walk_forward_permutation.p_value,
            walk_forward_stability=report.walk_forward_stability,
            composite_score=0.0,
        )
        rows.append(row)

    scored = [row.__class__(**{**row.__dict__, "composite_score": _composite_score(row)}) for row in rows]
    ranked = tuple(sorted(scored, key=lambda item: item.composite_score, reverse=True))
    return StrategyComparisonResult(rows=ranked)
