"""Candidate elimination and dominance pruning for strategy selection."""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd

from predictor.research.backtest import backtest_strategy
from predictor.research.errors import ResearchInputError
from predictor.research.strategies import TradingStrategy
from predictor.research.types import (
    EliminatedStrategy,
    MultiSymbolStrategyRow,
    PruningResult,
)
from predictor.research.validation import ResearchValidationConfig


def quick_reject_strategy(
    frame: pd.DataFrame,
    strategy: TradingStrategy,
    config: ResearchValidationConfig,
) -> Tuple[bool, str]:
    """Check if a strategy should be rejected before full validation.

    Runs only the in-sample backtest (no permutation tests). Returns
    (should_reject, reason_string). Reason is empty string when not rejected.

    Only rejects strategies with obviously pathological behaviour: almost no
    trades, catastrophic drawdown, or extreme negative Sharpe. Does not reject
    on weak performance — full validation handles borderline cases.
    """
    try:
        run = backtest_strategy(
            frame,
            strategy,
            bars_per_year=config.bars_per_year,
            transaction_cost_bps=config.transaction_cost_bps,
        )
    except Exception as exc:
        return True, f"backtest_failed:{exc}"

    active = int((run.positions != 0).sum())
    if active < 10:
        return True, "too_few_trades"
    if run.metrics.max_drawdown < -0.95:
        return True, "catastrophic_drawdown"
    if run.metrics.sharpe_ratio < -3.0:
        return True, "extreme_negative_sharpe"
    return False, ""


def _dominates(winner: MultiSymbolStrategyRow, loser: MultiSymbolStrategyRow) -> bool:
    """Return True if winner strictly dominates loser on all four criteria."""
    return (
        winner.avg_sharpe_ratio > loser.avg_sharpe_ratio
        and winner.avg_total_return > loser.avg_total_return
        and winner.avg_max_drawdown > loser.avg_max_drawdown  # less negative = better
        and winner.validation_pass_rate > loser.validation_pass_rate
    )


def prune_dominated_strategies(
    rows: Sequence[MultiSymbolStrategyRow],
) -> PruningResult:
    """Remove strategies that are strictly dominated by at least one other strategy.

    A strategy is dominated if another strategy beats it on ALL of:
    avg_sharpe_ratio, avg_total_return, avg_max_drawdown (less negative),
    and validation_pass_rate.

    Strict dominance is rare in practice — this removes only clear deadwood.
    """
    if not rows:
        return PruningResult(kept=(), eliminated=())

    kept: List[str] = []
    eliminated: List[EliminatedStrategy] = []

    for candidate in rows:
        dominated_by = next(
            (
                other.strategy_name
                for other in rows
                if other.strategy_name != candidate.strategy_name
                and _dominates(other, candidate)
            ),
            None,
        )
        if dominated_by is not None:
            eliminated.append(
                EliminatedStrategy(
                    strategy_name=candidate.strategy_name,
                    reason=f"dominated_by:{dominated_by}",
                )
            )
        else:
            kept.append(candidate.strategy_name)

    return PruningResult(kept=tuple(kept), eliminated=tuple(eliminated))


def prune_unstable_strategies(
    rows: Sequence[MultiSymbolStrategyRow],
    *,
    min_stability: float = 0.40,
    min_pass_rate: float = 0.30,
    max_abs_drawdown: float = 0.50,
) -> PruningResult:
    """Remove strategies that fail minimum stability thresholds.

    Thresholds are intentionally lenient — this catches clearly broken
    candidates without over-filtering. Strict selection belongs in the
    Donchian/family-specific validation layer.
    """
    if not 0.0 <= min_stability <= 1.0:
        raise ResearchInputError("min_stability must be in [0, 1]")
    if not 0.0 <= min_pass_rate <= 1.0:
        raise ResearchInputError("min_pass_rate must be in [0, 1]")
    if max_abs_drawdown <= 0.0:
        raise ResearchInputError("max_abs_drawdown must be > 0")

    kept: List[str] = []
    eliminated: List[EliminatedStrategy] = []

    for row in rows:
        reasons: List[str] = []
        if row.avg_walk_forward_stability < min_stability:
            reasons.append("low_stability")
        if row.validation_pass_rate < min_pass_rate:
            reasons.append("low_pass_rate")
        if abs(row.avg_max_drawdown) > max_abs_drawdown:
            reasons.append("excessive_drawdown")
        if reasons:
            eliminated.append(
                EliminatedStrategy(
                    strategy_name=row.strategy_name,
                    reason=",".join(reasons),
                )
            )
        else:
            kept.append(row.strategy_name)

    return PruningResult(kept=tuple(kept), eliminated=tuple(eliminated))


def apply_quick_rejects(
    frame: pd.DataFrame,
    strategies: Sequence[TradingStrategy],
    config: ResearchValidationConfig,
) -> Tuple[Tuple[TradingStrategy, ...], PruningResult]:
    """Screen strategies with quick_reject before full validation.

    Returns (passing_strategies, pruning_report). Strategies that pass
    quick rejection are returned unchanged for downstream validation.
    """
    passing: List[TradingStrategy] = []
    eliminated: List[EliminatedStrategy] = []

    for strategy in strategies:
        should_reject, reason = quick_reject_strategy(frame, strategy, config)
        if should_reject:
            eliminated.append(EliminatedStrategy(strategy_name=strategy.name, reason=reason))
        else:
            passing.append(strategy)

    kept_names = tuple(s.name for s in passing)
    return tuple(passing), PruningResult(kept=kept_names, eliminated=tuple(eliminated))
