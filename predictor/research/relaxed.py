"""Diagnostic relaxed-mode configuration for pipeline investigation.

WARNING
-------
This module is a DIAGNOSTIC INSTRUMENT ONLY.

Its purpose is to determine whether strategies fail because:
  (a) they have no statistical edge, or
  (b) the validation pipeline rejects them before meaningful execution completes.

Results produced under relaxed mode are NOT production-valid.
Every function in this module prints explicit warnings.

Public API
----------
RelaxedDiagnosticConfig  -- Diagnostic override flags wrapping ResearchValidationConfig.
apply_relaxed_config     -- Produce a modified config with diagnostic overrides applied.
validate_strategy_relaxed -- Run standard validation under relaxed config with warnings.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from predictor.research.validation import ResearchValidationConfig


_RELAXED_WARNING = (
    "\n"
    "╔══════════════════════════════════════════════════════════════╗\n"
    "║  [RELAXED MODE ACTIVE]  Results are NOT production-valid.   ║\n"
    "║  Use only to diagnose pipeline ordering or threshold bugs.  ║\n"
    "╚══════════════════════════════════════════════════════════════╝\n"
)


@dataclass(frozen=True)
class RelaxedDiagnosticConfig:
    """Diagnostic override flags on top of a base ResearchValidationConfig.

    All overrides default to None / False (i.e. no change to base config).
    Only set fields that are needed for the specific diagnostic investigation.

    Attributes
    ----------
    base_config:
        The production ResearchValidationConfig to start from.
    disable_is_rejection:
        When True, the in-sample permutation test result is IGNORED and the
        strategy always passes IS validation. This reveals whether downstream
        stages (walk-forward, metrics) produce meaningful output.
    relaxed_alpha:
        Override p_value_threshold with a more permissive value. Useful to
        determine how close strategies come to the standard threshold.
    reduced_permutation_count:
        Override permutation_iterations for faster diagnostic runs.
    skip_holm_correction:
        When True, Holm-adjusted p-values are not applied during
        select_stable_donchian_rows. Does not affect per-strategy validation.
    """

    base_config: ResearchValidationConfig
    disable_is_rejection: bool = False
    relaxed_alpha: Optional[float] = None
    reduced_permutation_count: Optional[int] = None
    skip_holm_correction: bool = False


def apply_relaxed_config(relaxed: RelaxedDiagnosticConfig) -> ResearchValidationConfig:
    """Produce a ResearchValidationConfig with diagnostic overrides applied.

    Prints a warning to stdout whenever any override is active.

    Parameters
    ----------
    relaxed:
        RelaxedDiagnosticConfig specifying which overrides to apply.

    Returns
    -------
    ResearchValidationConfig with overrides applied. Unmodified base config
    fields are preserved verbatim.
    """
    base = relaxed.base_config
    any_override = (
        relaxed.disable_is_rejection
        or relaxed.relaxed_alpha is not None
        or relaxed.reduced_permutation_count is not None
    )

    if any_override:
        print(_RELAXED_WARNING)

    alpha = (
        float(relaxed.relaxed_alpha)
        if relaxed.relaxed_alpha is not None
        else base.p_value_threshold
    )
    perm_count = (
        int(relaxed.reduced_permutation_count)
        if relaxed.reduced_permutation_count is not None
        else base.permutation_iterations
    )

    if relaxed.relaxed_alpha is not None:
        print(
            f"[RELAXED] p_value_threshold overridden: "
            f"{base.p_value_threshold} → {alpha}"
        )
    if relaxed.reduced_permutation_count is not None:
        print(
            f"[RELAXED] permutation_iterations overridden: "
            f"{base.permutation_iterations} → {perm_count}"
        )
    if relaxed.disable_is_rejection:
        print(
            "[RELAXED] disable_is_rejection=True — "
            "IS permutation failure will NOT reject strategies."
        )

    return ResearchValidationConfig(
        bars_per_year=base.bars_per_year,
        transaction_cost_bps=base.transaction_cost_bps,
        permutation_iterations=perm_count,
        permutation_block_size=base.permutation_block_size,
        p_value_threshold=alpha,
        train_window=base.train_window,
        test_window=base.test_window,
        walk_forward_step=base.walk_forward_step,
        minimum_walk_forward_stability=base.minimum_walk_forward_stability,
        minimum_walk_forward_fold_pass_rate=base.minimum_walk_forward_fold_pass_rate,
        require_positive_walk_forward_return=base.require_positive_walk_forward_return,
        minimum_walk_forward_sharpe=base.minimum_walk_forward_sharpe,
        random_seed=base.random_seed,
    )


def validate_strategy_relaxed(
    frame: object,
    strategy: object,
    *,
    relaxed: RelaxedDiagnosticConfig,
) -> object:
    """Run strategy validation under relaxed diagnostic configuration.

    When disable_is_rejection is True, the IS permutation outcome is patched
    to always pass so that walk-forward stages always execute. This reveals
    whether downstream metrics produce meaningful values.

    Parameters
    ----------
    frame:
        OHLCV DataFrame.
    strategy:
        TradingStrategy instance.
    relaxed:
        RelaxedDiagnosticConfig with desired overrides.

    Returns
    -------
    StrategyValidationReport. When disable_is_rejection=True the
    in_sample_permutation field will reflect actual computed values but
    fail_reasons will NOT include 'in_sample_permutation_failed'.

    Notes
    -----
    This function is intentionally not part of the standard validate_strategy
    call path. It always prints warnings and must never be used in production
    runs.
    """
    import pandas as pd

    from predictor.research.backtest import backtest_strategy
    from predictor.research.metrics import compute_performance_metrics
    from predictor.research.permutation import run_permutation_test
    from predictor.research.types import (
        BacktestRun,
        PerformanceMetrics,
        PermutationTestResult,
    )
    from predictor.research.validation import (
        _aggregate_runs,  # type: ignore[attr-defined]
        _zero_performance_metrics,  # type: ignore[attr-defined]
        _walk_forward_stability,  # type: ignore[attr-defined]
        _walk_forward_permutation,  # type: ignore[attr-defined]
        build_walk_forward_splits,
        ResearchValidationConfig,
        StrategyValidationReport,
        WalkForwardFoldResult,
    )
    from predictor.research.data import validate_research_frame

    print(_RELAXED_WARNING)

    cfg = apply_relaxed_config(relaxed)

    minimum_rows = max(3, cfg.train_window + cfg.test_window)
    validated = validate_research_frame(
        frame,  # type: ignore[arg-type]
        symbol=getattr(strategy, "name", "STRATEGY"),
        min_rows=minimum_rows,
    )

    in_sample = backtest_strategy(
        validated,
        strategy,  # type: ignore[arg-type]
        bars_per_year=cfg.bars_per_year,
        transaction_cost_bps=cfg.transaction_cost_bps,
    )

    def _score_frame_sharpe(perm_frame: pd.DataFrame) -> float:
        run = backtest_strategy(
            perm_frame,
            strategy,  # type: ignore[arg-type]
            bars_per_year=cfg.bars_per_year,
            transaction_cost_bps=cfg.transaction_cost_bps,
        )
        return run.metrics.sharpe_ratio

    in_sample_permutation = run_permutation_test(
        frame=validated,
        observed_statistic=in_sample.metrics.sharpe_ratio,
        score_on_frame=_score_frame_sharpe,
        iterations=cfg.permutation_iterations,
        block_size=min(cfg.permutation_block_size, max(len(validated) - 1, 1)),
        seed=cfg.random_seed + 1_001,
        p_value_threshold=cfg.p_value_threshold,
    )

    # When disable_is_rejection is active, treat IS as passed regardless.
    is_passes_for_routing = (
        True if relaxed.disable_is_rejection else in_sample_permutation.passes
    )

    fail_reasons = []
    raw_metrics = in_sample.metrics
    validated_metrics = _zero_performance_metrics()
    rejection_reason: str | None = None

    if is_passes_for_routing:
        splits = build_walk_forward_splits(
            total_rows=len(validated),
            train_size=cfg.train_window,
            test_size=cfg.test_window,
            step_size=cfg.walk_forward_step,
        )
        folds = []
        import numpy as np

        for split_index, split in enumerate(splits):
            test_frame = validated.iloc[split.test_start : split.test_end + 1]
            fold_run = backtest_strategy(
                test_frame,
                strategy,  # type: ignore[arg-type]
                bars_per_year=cfg.bars_per_year,
                transaction_cost_bps=cfg.transaction_cost_bps,
            )
            fold_permutation = run_permutation_test(
                frame=test_frame,
                observed_statistic=fold_run.metrics.sharpe_ratio,
                score_on_frame=_score_frame_sharpe,
                iterations=max(50, cfg.permutation_iterations // 2),
                block_size=min(cfg.permutation_block_size, max(len(test_frame) - 1, 1)),
                seed=cfg.random_seed + 2_000 + split_index,
                p_value_threshold=cfg.p_value_threshold,
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
            bars_per_year=cfg.bars_per_year,
        )
        walk_forward_permutation = _walk_forward_permutation(
            validated=validated,
            strategy=strategy,  # type: ignore[arg-type]
            splits=splits,
            observed_sharpe=walk_forward_aggregate.metrics.sharpe_ratio,
            config=cfg,
        )
        stability = _walk_forward_stability(folds)
        fold_pass_rate = float(
            np.mean([1.0 if fold.permutation.passes else 0.0 for fold in folds])
        )

        if not walk_forward_permutation.passes:
            fail_reasons.append("walk_forward_permutation_failed")
        if stability < cfg.minimum_walk_forward_stability:
            fail_reasons.append("walk_forward_stability_failed")
        if fold_pass_rate < cfg.minimum_walk_forward_fold_pass_rate:
            fail_reasons.append("walk_forward_fold_pass_rate_failed")
        if (
            cfg.require_positive_walk_forward_return
            and walk_forward_aggregate.metrics.total_return <= 0.0
        ):
            fail_reasons.append("walk_forward_return_non_positive")
        if (
            cfg.minimum_walk_forward_sharpe is not None
            and walk_forward_aggregate.metrics.sharpe_ratio < cfg.minimum_walk_forward_sharpe
        ):
            fail_reasons.append("walk_forward_sharpe_below_threshold")
        raw_metrics = walk_forward_aggregate.metrics
        if not fail_reasons:
            validated_metrics = walk_forward_aggregate.metrics
        else:
            validated_metrics = _zero_performance_metrics()
            rejection_reason = fail_reasons[0]

    else:
        # IS permutation actually failed and disable_is_rejection is False.
        fail_reasons.append("in_sample_permutation_failed")
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

    is_valid = len(fail_reasons) == 0

    print(
        f"[RELAXED] validate_strategy_relaxed complete: "
        f"is_valid={is_valid}  fail_reasons={fail_reasons}  "
        f"trade_count={walk_forward_aggregate.metrics.trade_count}"
    )

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
