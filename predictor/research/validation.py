"""Four-stage validation harness for strategy research."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Sequence, Tuple

import numpy as np
import pandas as pd

from predictor.research.backtest import backtest_strategy
from predictor.research.data import validate_research_frame
from predictor.research.diagnostics import (
    PipelineStage,
    assert_impossible_states,
    decompose_positions,
    log_pipeline_stage,
)
from predictor.research.errors import ResearchInputError, ResearchValidationError
from predictor.research.metrics import compute_performance_metrics
from predictor.research.permutation import block_permute_ohlcv, run_permutation_test
from predictor.research.strategies import TradingStrategy
from predictor.research.types import (
    BacktestRun,
    PerformanceMetrics,
    PermutationTestResult,
    StrategyValidationReport,
    WalkForwardFoldResult,
    WalkForwardSplit,
)


def _zero_performance_metrics() -> PerformanceMetrics:
    """Return an explicit zeroed metrics object for rejected validation views."""
    return PerformanceMetrics(
        total_return=0.0,
        sharpe_ratio=0.0,
        max_drawdown=0.0,
        win_rate=0.0,
        profit_factor=0.0,
        trade_count=0,
        exposure_time=0.0,
        avg_holding_period=0.0,
        expectancy=0.0,
        avg_win=0.0,
        avg_loss=0.0,
        largest_win=0.0,
        largest_loss=0.0,
    )


@dataclass(frozen=True)
class ResearchValidationConfig:
    """Configuration for fair strategy validation and comparison."""

    bars_per_year: int = 252
    transaction_cost_bps: float = 0.0
    permutation_iterations: int = 300
    permutation_block_size: int = 20
    p_value_threshold: float = 0.05
    train_window: int = 240
    test_window: int = 60
    walk_forward_step: int = 60
    minimum_walk_forward_stability: float = 0.35
    minimum_walk_forward_fold_pass_rate: float = 0.0
    require_positive_walk_forward_return: bool = False
    minimum_walk_forward_sharpe: float | None = None
    random_seed: int = 7


def build_walk_forward_splits(
    *,
    total_rows: int,
    train_size: int,
    test_size: int,
    step_size: int,
) -> Tuple[WalkForwardSplit, ...]:
    """Build deterministic rolling walk-forward splits."""

    for value, label in (
        (total_rows, "total_rows"),
        (train_size, "train_size"),
        (test_size, "test_size"),
        (step_size, "step_size"),
    ):
        if not isinstance(value, int) or value <= 0:
            raise ResearchInputError(f"{label} must be a positive integer")

    if train_size + test_size > total_rows:
        raise ResearchInputError("train_size + test_size must be <= total_rows")

    splits: List[WalkForwardSplit] = []
    cursor = 0
    while cursor + train_size + test_size <= total_rows:
        train_start = cursor
        train_end = train_start + train_size - 1
        test_start = train_end + 1
        test_end = test_start + test_size - 1
        splits.append(
            WalkForwardSplit(
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
            )
        )
        cursor += step_size

    if not splits:
        raise ResearchInputError("no walk-forward splits can be formed with provided windows")
    return tuple(splits)


def _aggregate_runs(strategy_name: str, runs: Sequence[BacktestRun], bars_per_year: int) -> BacktestRun:
    """Aggregate multiple fold runs into one synthetic out-of-sample run."""

    if not runs:
        raise ResearchValidationError("cannot aggregate empty walk-forward runs")
    returns = pd.concat([run.returns for run in runs], axis=0)
    positions = pd.concat([run.positions for run in runs], axis=0)
    equity_curve = (1.0 + returns.fillna(0.0)).cumprod()
    metrics = compute_performance_metrics(
        returns,
        bars_per_year=bars_per_year,
        positions=positions,
    )
    return BacktestRun(
        strategy_name=strategy_name,
        returns=returns,
        positions=positions,
        equity_curve=equity_curve,
        metrics=metrics,
    )


def _walk_forward_stability(folds: Sequence[WalkForwardFoldResult]) -> float:
    """Estimate stability from dispersion of fold-level Sharpe and return signs."""

    if not folds:
        return 0.0
    sharpes = np.asarray([fold.run.metrics.sharpe_ratio for fold in folds], dtype=float)
    mean_abs = float(np.mean(np.abs(sharpes)))
    dispersion = float(np.std(sharpes)) / max(mean_abs, 1e-9)
    consistency = 1.0 / (1.0 + dispersion)

    signs = np.asarray([np.sign(fold.run.metrics.total_return) for fold in folds], dtype=float)
    positive_share = float((signs > 0).sum()) / max(len(signs), 1)
    return float(np.clip(0.65 * consistency + 0.35 * positive_share, 0.0, 1.0))


def _score_frame_sharpe(
    frame: pd.DataFrame,
    strategy: TradingStrategy,
    config: ResearchValidationConfig,
) -> float:
    """Return strategy Sharpe on one frame for permutation null evaluation."""

    run = backtest_strategy(
        frame,
        strategy,
        bars_per_year=config.bars_per_year,
        transaction_cost_bps=config.transaction_cost_bps,
    )
    return run.metrics.sharpe_ratio


def _walk_forward_permutation(
    *,
    validated: pd.DataFrame,
    strategy: TradingStrategy,
    splits: Sequence[WalkForwardSplit],
    observed_sharpe: float,
    config: ResearchValidationConfig,
) -> "PermutationTestResult":
    """Permutation test on stitched walk-forward out-of-sample outcomes."""

    from predictor.research.types import PermutationTestResult

    rng = np.random.default_rng(config.random_seed + 3_003)
    null_values: List[float] = []
    for _ in range(config.permutation_iterations):
        stitched_returns: List[pd.Series] = []
        for split in splits:
            test_frame = validated.iloc[split.test_start : split.test_end + 1]
            perm_seed = int(rng.integers(0, 2**31 - 1))
            permuted = block_permute_ohlcv(
                test_frame,
                block_size=min(config.permutation_block_size, max(len(test_frame) - 1, 1)),
                seed=perm_seed,
            )
            perm_run = backtest_strategy(
                permuted,
                strategy,
                bars_per_year=config.bars_per_year,
                transaction_cost_bps=config.transaction_cost_bps,
            )
            stitched_returns.append(perm_run.returns)
        joined = pd.concat(stitched_returns, axis=0)
        metric = compute_performance_metrics(joined, bars_per_year=config.bars_per_year)
        null_values.append(float(metric.sharpe_ratio))

    extremes = sum(value >= observed_sharpe for value in null_values)
    p_value = float((extremes + 1) / (len(null_values) + 1))
    return PermutationTestResult(
        observed_statistic=float(observed_sharpe),
        null_distribution=tuple(null_values),
        p_value=p_value,
        passes=p_value <= config.p_value_threshold,
    )


def validate_strategy(
    frame: pd.DataFrame,
    strategy: TradingStrategy,
    *,
    config: ResearchValidationConfig,
) -> StrategyValidationReport:
    """Run the mandatory four-stage validation framework for one strategy.

    Pipeline stages executed in order:
    1. LOAD_DATA          — validate and normalise the OHLCV frame
    2. GENERATE_POSITIONS — call strategy.generate_positions()
    3. CONSTRUCT_BACKTEST — build returns / equity curve
    4. COMPUTE_IS_METRIC  — compute in-sample Sharpe ratio
    5. RUN_IS_PERMUTATIONS — block-permutation null test on IS metric
    6. RUN_WALKFORWARD    — walk-forward fold backtests (skipped if IS fails)
    7. RUN_WF_PERMUTATIONS — permutation test on stitched OOS returns
    """
    log_pipeline_stage(PipelineStage.LOAD_DATA)
    minimum_rows = max(3, config.train_window + config.test_window)
    validated = validate_research_frame(
        frame,
        symbol=getattr(strategy, "name", "STRATEGY"),
        min_rows=minimum_rows,
    )

    log_pipeline_stage(PipelineStage.GENERATE_POSITIONS)
    # Decompose positions for impossible-state checking (signals → trades).
    positions_raw = strategy.generate_positions(validated)
    _decomp = decompose_positions(positions_raw)

    log_pipeline_stage(PipelineStage.CONSTRUCT_BACKTEST)
    in_sample = backtest_strategy(
        validated,
        strategy,
        bars_per_year=config.bars_per_year,
        transaction_cost_bps=config.transaction_cost_bps,
    )

    # Verify signal → trade chain integrity.
    equity_end = float(in_sample.equity_curve.iloc[-1]) if not in_sample.equity_curve.empty else 1.0
    assert_impossible_states(
        positions_nonzero=_decomp.positions_nonzero,
        entries_generated=_decomp.entries_generated,
        trade_count=in_sample.metrics.trade_count,
        equity_end=equity_end,
        strategy_name=getattr(strategy, "name", "STRATEGY"),
    )

    log_pipeline_stage(PipelineStage.COMPUTE_IS_METRIC)
    in_sample_permutation = run_permutation_test(
        frame=validated,
        observed_statistic=in_sample.metrics.sharpe_ratio,
        score_on_frame=lambda perm: _score_frame_sharpe(perm, strategy, config),
        iterations=config.permutation_iterations,
        block_size=min(config.permutation_block_size, max(len(validated) - 1, 1)),
        seed=config.random_seed + 1_001,
        p_value_threshold=config.p_value_threshold,
    )
    log_pipeline_stage(PipelineStage.RUN_IS_PERMUTATIONS)

    is_valid = True
    fail_reasons: List[str] = []
    raw_metrics = in_sample.metrics
    validated_metrics = _zero_performance_metrics()
    rejection_reason: str | None = None
    if not in_sample_permutation.passes:
        fail_reasons.append("in_sample_permutation_failed")
        is_valid = False

    # Short-circuit: skip expensive WF stages when IS permutation fails.
    if is_valid:
        log_pipeline_stage(PipelineStage.RUN_WALKFORWARD)
        splits = build_walk_forward_splits(
            total_rows=len(validated),
            train_size=config.train_window,
            test_size=config.test_window,
            step_size=config.walk_forward_step,
        )
        folds: List[WalkForwardFoldResult] = []
        for split_index, split in enumerate(splits):
            test_frame = validated.iloc[split.test_start : split.test_end + 1]
            fold_run = backtest_strategy(
                test_frame,
                strategy,
                bars_per_year=config.bars_per_year,
                transaction_cost_bps=config.transaction_cost_bps,
            )
            fold_permutation = run_permutation_test(
                frame=test_frame,
                observed_statistic=fold_run.metrics.sharpe_ratio,
                score_on_frame=lambda perm: _score_frame_sharpe(perm, strategy, config),
                iterations=max(50, int(config.permutation_iterations / 2)),
                block_size=min(config.permutation_block_size, max(len(test_frame) - 1, 1)),
                seed=config.random_seed + 2_000 + split_index,
                p_value_threshold=config.p_value_threshold,
            )
            folds.append(
                WalkForwardFoldResult(
                    split_index=split_index,
                    split=split,
                    run=fold_run,
                    permutation=fold_permutation,
                )
            )

        walk_forward_aggregate = _aggregate_runs(
            getattr(strategy, "name", "STRATEGY"),
            [fold.run for fold in folds],
            bars_per_year=config.bars_per_year,
        )
        log_pipeline_stage(PipelineStage.RUN_WF_PERMUTATIONS)
        walk_forward_permutation = _walk_forward_permutation(
            validated=validated,
            strategy=strategy,
            splits=splits,
            observed_sharpe=walk_forward_aggregate.metrics.sharpe_ratio,
            config=config,
        )
        stability = _walk_forward_stability(folds)
        fold_pass_rate = float(
            np.mean([1.0 if fold.permutation.passes else 0.0 for fold in folds])
        )

        if not walk_forward_permutation.passes:
            fail_reasons.append("walk_forward_permutation_failed")
        if stability < config.minimum_walk_forward_stability:
            fail_reasons.append("walk_forward_stability_failed")
        if fold_pass_rate < config.minimum_walk_forward_fold_pass_rate:
            fail_reasons.append("walk_forward_fold_pass_rate_failed")
        if (
            config.require_positive_walk_forward_return
            and walk_forward_aggregate.metrics.total_return <= 0.0
        ):
            fail_reasons.append("walk_forward_return_non_positive")
        if (
            config.minimum_walk_forward_sharpe is not None
            and walk_forward_aggregate.metrics.sharpe_ratio < config.minimum_walk_forward_sharpe
        ):
            fail_reasons.append("walk_forward_sharpe_below_threshold")
        is_valid = not fail_reasons
        raw_metrics = walk_forward_aggregate.metrics
        if is_valid:
            validated_metrics = walk_forward_aggregate.metrics
        else:
            validated_metrics = _zero_performance_metrics()
            rejection_reason = fail_reasons[0]
    else:
        # Dummy values for skipped steps
        folds = []
        walk_forward_aggregate = BacktestRun(
            strategy_name=getattr(strategy, "name", "STRATEGY"),
            returns=pd.Series(dtype=float),
            positions=pd.Series(dtype=float),
            equity_curve=pd.Series(dtype=float),
            metrics=_zero_performance_metrics(),
        )
        walk_forward_permutation = PermutationTestResult(
            observed_statistic=0.0, null_distribution=(), p_value=1.0, passes=False
        )
        stability = 0.0
        fold_pass_rate = 0.0
        raw_metrics = in_sample.metrics
        validated_metrics = _zero_performance_metrics()
        rejection_reason = "in_sample_permutation_failed"

    return StrategyValidationReport(
        strategy_name=getattr(strategy, "name", "STRATEGY"),
        in_sample=in_sample,
        in_sample_permutation=in_sample_permutation,
        walk_forward_folds=tuple(folds),
        walk_forward_aggregate=walk_forward_aggregate,
        walk_forward_permutation=walk_forward_permutation,
        walk_forward_stability=stability,
        walk_forward_fold_pass_rate=fold_pass_rate,
        is_valid=is_valid,
        fail_reasons=tuple(fail_reasons),
        raw_metrics=raw_metrics,
        validated_metrics=validated_metrics,
        rejection_reason=rejection_reason,
    )
