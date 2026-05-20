"""Typed value objects for strategy backtests and validation outcomes."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple

import pandas as pd


@dataclass(frozen=True)
class PerformanceMetrics:
    """Core performance metrics for fair cross-strategy comparison."""

    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    trade_count: int = 0
    exposure_time: float = 0.0
    avg_holding_period: float = 0.0
    expectancy: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0


@dataclass(frozen=True)
class BacktestRun:
    """One deterministic backtest run for a strategy."""

    strategy_name: str
    returns: pd.Series
    positions: pd.Series
    equity_curve: pd.Series
    metrics: PerformanceMetrics


@dataclass(frozen=True)
class PermutationTestResult:
    """Result of a Monte Carlo permutation test."""

    observed_statistic: float
    null_distribution: Tuple[float, ...]
    p_value: float
    passes: bool


@dataclass(frozen=True)
class WalkForwardSplit:
    """Index boundaries for one walk-forward fold."""

    train_start: int
    train_end: int
    test_start: int
    test_end: int


@dataclass(frozen=True)
class WalkForwardFoldResult:
    """Out-of-sample fold result for one walk-forward split."""

    split_index: int
    split: WalkForwardSplit
    run: BacktestRun
    permutation: PermutationTestResult


@dataclass(frozen=True)
class StrategyValidationReport:
    """Full four-stage validation report for one strategy."""

    strategy_name: str
    in_sample: BacktestRun
    in_sample_permutation: PermutationTestResult
    walk_forward_folds: Tuple[WalkForwardFoldResult, ...]
    walk_forward_aggregate: BacktestRun
    walk_forward_permutation: PermutationTestResult
    walk_forward_stability: float
    walk_forward_fold_pass_rate: float
    is_valid: bool
    fail_reasons: Tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class StrategyComparisonRow:
    """Comparable row used in final ranking output."""

    strategy_name: str
    is_valid: bool
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    in_sample_p_value: float
    walk_forward_p_value: float
    walk_forward_stability: float
    composite_score: float


@dataclass(frozen=True)
class StrategyComparisonResult:
    """Cross-strategy ranking output."""

    rows: Tuple[StrategyComparisonRow, ...]


@dataclass(frozen=True)
class MultiSymbolStrategyRow:
    """Aggregated strategy ranking row across multiple symbols."""

    strategy_name: str
    symbols_tested: int
    symbols_validated: int
    avg_total_return: float
    avg_sharpe_ratio: float
    avg_max_drawdown: float
    avg_win_rate: float
    avg_profit_factor: float
    avg_in_sample_p_value: float
    avg_walk_forward_p_value: float
    avg_walk_forward_stability: float
    avg_trade_count: float = 0.0
    avg_holding_period: float = 0.0
    avg_expectancy: float = 0.0
    validation_pass_rate: float = 0.0
    composite_score: float = 0.0


@dataclass(frozen=True)
class MultiSymbolComparisonResult:
    """Research harness output across symbols and strategies."""

    rows: Tuple[MultiSymbolStrategyRow, ...]


@dataclass(frozen=True)
class StressSliceResult:
    """Performance of one strategy on one worst-case market slice."""

    label: str
    strategy_name: str
    metrics: "PerformanceMetrics"
    bar_count: int
    start_date: str
    end_date: str


@dataclass(frozen=True)
class StressTestResult:
    """All stress slice results for one strategy."""

    strategy_name: str
    slices: Tuple[StressSliceResult, ...]


@dataclass(frozen=True)
class EliminatedStrategy:
    """Record of a strategy removed during candidate pruning."""

    strategy_name: str
    reason: str


@dataclass(frozen=True)
class PruningResult:
    """Outcome of a pruning pass over strategy candidates."""

    kept: Tuple[str, ...]
    eliminated: Tuple[EliminatedStrategy, ...]


@dataclass(frozen=True)
class DataAvailabilityReport:
    """Data fetch outcome for one symbol."""

    symbol: str
    available: bool
    row_count: int
    date_range: str
    failure_reason: "str | None"


@dataclass(frozen=True)
class SymbolRobustnessRow:
    """Per-symbol validation outcome for one strategy."""

    strategy_name: str
    symbol: str
    is_valid: bool
    fail_reasons: Tuple[str, ...]
    sharpe_ratio: float
    max_drawdown: float
    walk_forward_stability: float


@dataclass(frozen=True)
class ResearchRunResult:
    """Full parallel research run result with rankings, robustness, and availability."""

    comparison: MultiSymbolComparisonResult
    symbol_robustness: Tuple[SymbolRobustnessRow, ...]
    data_availability: Tuple[DataAvailabilityReport, ...]
    pruning: PruningResult
