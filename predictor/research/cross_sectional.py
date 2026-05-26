"""Cross-sectional factor backtesting, permutation testing, and walk-forward validation.

Skeptical scientific engine for verifying predictive edge.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from predictor.research.errors import ResearchInputError, ResearchValidationError
from predictor.research.factors import RankingFactor
from predictor.research.ranking import (
    RankingMetrics,
    compute_daily_regimes,
    compute_next_day_returns,
    evaluate_ranking,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CrossSectionalResearchConfig:
    """Configuration for cross-sectional research backtesting and validation."""

    top_k: int = 5
    transaction_cost_bps: float = 5.0
    slippage_bps: float = 0.0
    permutation_iterations: int = 300
    permutation_block_size: int = 20
    p_value_threshold: float = 0.05
    train_window: int = 240
    test_window: int = 60
    walk_forward_step: int = 60
    minimum_ic: float = 0.01
    random_seed: int = 42


@dataclass(frozen=True)
class CrossSectionalPermutationResult:
    """Outcomes of the joint-block permutation validation."""

    observed_ic: float
    observed_sharpe: float
    ic_p_value: float
    sharpe_p_value: float
    null_ic: Tuple[float, ...]
    null_sharpe: Tuple[float, ...]
    passes_ic: bool
    passes_sharpe: bool


@dataclass(frozen=True)
class CSWalkForwardFoldResult:
    """One walk-forward validation fold."""

    fold_index: int
    train_start_date: str
    train_end_date: str
    test_start_date: str
    test_end_date: str
    out_of_sample_metrics: RankingMetrics


@dataclass(frozen=True)
class CrossSectionalValidationReport:
    """Full scientific validation report for a RankingFactor."""

    factor_name: str
    full_sample_metrics: RankingMetrics
    permutation_result: CrossSectionalPermutationResult
    walk_forward_folds: Tuple[CSWalkForwardFoldResult, ...]
    walk_forward_metrics: RankingMetrics
    is_valid: bool
    fail_reasons: Tuple[str, ...] = field(default_factory=tuple)


def run_cross_sectional_backtest(
    factor: RankingFactor,
    symbol_data: Dict[str, pd.DataFrame],
    config: CrossSectionalResearchConfig,
    *,
    regimes: pd.Series | None = None,
    custom_targets: pd.DataFrame | None = None,
) -> Tuple[pd.Series, RankingMetrics]:
    """Run a vectorized cross-sectional factor backtest.

    Parameters
    ----------
    factor : RankingFactor
        The factor to test.
    symbol_data : Dict[str, pd.DataFrame]
        Dict mapping symbol to OHLCV DataFrame.
    config : CrossSectionalResearchConfig
        Validation config.
    regimes : pd.Series | None
        Pre-computed market regimes.
    custom_targets : pd.DataFrame | None
        Override actual targets (useful for permutation/shuffling).

    Returns
    -------
    Tuple[pd.Series, RankingMetrics]
        Daily portfolio returns Series, and RankingMetrics.
    """
    if not symbol_data:
        raise ResearchInputError("symbol_data must not be empty")

    # 1. Compute factor scores panel
    scores = factor.compute_scores(symbol_data)
    
    # 2. Compute targets (next-day intraday returns)
    if custom_targets is not None:
        targets = custom_targets
    else:
        targets = compute_next_day_returns(symbol_data)
        
    if regimes is None:
        regimes = compute_daily_regimes(symbol_data)

    # 3. Evaluate portfolio performance and IC
    port_rets, metrics = evaluate_ranking(
        scores=scores,
        targets=targets,
        top_k=config.top_k,
        transaction_cost_bps=config.transaction_cost_bps,
        slippage_bps=config.slippage_bps,
        regimes=regimes,
    )
    return port_rets, metrics


def joint_block_permute_targets(
    targets: pd.DataFrame,
    block_size: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Block permute targets across time.

    Keeps cross-sectional asset correlations intact by shuffling the entire
    daily row blocks jointly across time.
    """
    n_rows = len(targets)
    if n_rows <= block_size:
        return targets

    # Construct block boundaries
    indices = np.arange(n_rows)
    blocks = [indices[i : i + block_size] for i in range(0, n_rows, block_size)]
    
    # Shuffle the blocks
    rng.shuffle(blocks)
    shuffled_indices = np.concatenate(blocks)
    
    # Reconstruct the target panel with shuffled dates, keeping index aligned to scores
    shuffled_targets = pd.DataFrame(
        targets.values[shuffled_indices],
        index=targets.index,
        columns=targets.columns
    )
    return shuffled_targets


def run_cross_sectional_permutation(
    factor: RankingFactor,
    symbol_data: Dict[str, pd.DataFrame],
    observed_metrics: RankingMetrics,
    config: CrossSectionalResearchConfig,
) -> CrossSectionalPermutationResult:
    """Run a Monte Carlo joint-block permutation validation test.

    Generates the true empirical null distribution by shuffling target returns.
    """
    scores = factor.compute_scores(symbol_data)
    targets = compute_next_day_returns(symbol_data)
    regimes = compute_daily_regimes(symbol_data)
    return run_cross_sectional_permutation_on_panels(
        scores=scores,
        targets=targets,
        observed_metrics=observed_metrics,
        config=config,
        regimes=regimes,
    )


def run_cross_sectional_permutation_on_panels(
    *,
    scores: pd.DataFrame,
    targets: pd.DataFrame,
    observed_metrics: RankingMetrics,
    config: CrossSectionalResearchConfig,
    regimes: pd.Series | None = None,
) -> CrossSectionalPermutationResult:
    """Run permutation testing directly on score/target panels.

    This supports synthetic calibration probes where no OHLCV source or factor
    object is required.
    """
    if regimes is None:
        regimes = pd.Series(index=scores.index, data="SIDEWAYS")

    rng = np.random.default_rng(config.random_seed)
    
    null_ic = []
    null_sharpe = []

    for _ in range(config.permutation_iterations):
        # 1. Permute targets jointly across time
        permuted_targets = joint_block_permute_targets(targets, config.permutation_block_size, rng)
        
        # 2. Rerun backtest
        _, perm_metrics = evaluate_ranking(
            scores=scores,
            targets=permuted_targets,
            top_k=config.top_k,
            transaction_cost_bps=config.transaction_cost_bps,
            slippage_bps=config.slippage_bps,
            regimes=regimes,
        )
        null_ic.append(perm_metrics.mean_ic)
        null_sharpe.append(perm_metrics.sharpe_ratio)

    null_ic_arr = np.array(null_ic)
    null_sharpe_arr = np.array(null_sharpe)

    # Compute empirical p-values
    # Fraction of permuted trials exceeding or matching observed statistic
    ic_p = float(np.mean(null_ic_arr >= observed_metrics.mean_ic))
    sharpe_p = float(np.mean(null_sharpe_arr >= observed_metrics.sharpe_ratio))

    passes_ic = ic_p <= config.p_value_threshold and observed_metrics.mean_ic >= config.minimum_ic
    passes_sharpe = sharpe_p <= config.p_value_threshold and observed_metrics.sharpe_ratio > 0.0

    return CrossSectionalPermutationResult(
        observed_ic=observed_metrics.mean_ic,
        observed_sharpe=observed_metrics.sharpe_ratio,
        ic_p_value=ic_p,
        sharpe_p_value=sharpe_p,
        null_ic=tuple(null_ic),
        null_sharpe=tuple(null_sharpe),
        passes_ic=passes_ic,
        passes_sharpe=passes_sharpe,
    )


def run_cross_sectional_walk_forward(
    factor: RankingFactor,
    symbol_data: Dict[str, pd.DataFrame],
    config: CrossSectionalResearchConfig,
) -> Tuple[Tuple[CSWalkForwardFoldResult, ...], RankingMetrics]:
    """Perform out-of-sample walk-forward panel validation for the RankingFactor."""
    # Find all common sorted dates
    all_dates = sorted(list(set().union(*(df.index for df in symbol_data.values()))))
    total_dates = len(all_dates)

    train_size = config.train_window
    test_size = config.test_window
    step_size = config.walk_forward_step

    if train_size + test_size > total_dates:
        raise ResearchValidationError(
            f"Insufficient dates for walk-forward validation ({total_dates} < {train_size + test_size})"
        )

    folds = []
    cursor = 0
    fold_idx = 0
    
    scores = factor.compute_scores(symbol_data)
    targets = compute_next_day_returns(symbol_data)
    regimes = compute_daily_regimes(symbol_data)

    out_of_sample_returns = []
    out_of_sample_ic = []
    
    while cursor + train_size + test_size <= total_dates:
        train_start = cursor
        train_end = train_start + train_size - 1
        test_start = train_end + 1
        test_end = test_start + test_size - 1

        train_start_date = all_dates[train_start]
        train_end_date = all_dates[train_end]
        test_start_date = all_dates[test_start]
        test_end_date = all_dates[test_end]

        # Extract Out-Of-Sample test slice
        test_dates = all_dates[test_start : test_end + 1]
        test_scores = scores.loc[test_dates]
        test_targets = targets.loc[test_dates]
        test_regimes = regimes.loc[test_dates]

        # Evaluate OOS metrics for this fold
        fold_rets, fold_metrics = evaluate_ranking(
            scores=test_scores,
            targets=test_targets,
            top_k=config.top_k,
            transaction_cost_bps=config.transaction_cost_bps,
            slippage_bps=config.slippage_bps,
            regimes=test_regimes,
        )

        folds.append(
            CSWalkForwardFoldResult(
                fold_index=fold_idx,
                train_start_date=str(train_start_date.date()) if hasattr(train_start_date, "date") else str(train_start_date),
                train_end_date=str(train_end_date.date()) if hasattr(train_end_date, "date") else str(train_end_date),
                test_start_date=str(test_start_date.date()) if hasattr(test_start_date, "date") else str(test_start_date),
                test_end_date=str(test_end_date.date()) if hasattr(test_end_date, "date") else str(test_end_date),
                out_of_sample_metrics=fold_metrics,
            )
        )
        
        # Accumulate out-of-sample predictions
        out_of_sample_returns.append(fold_rets)
        cursor += step_size
        fold_idx += 1

    if not folds:
        raise ResearchValidationError("Could not form any walk-forward folds.")

    # Aggregate synthetic out-of-sample metrics
    oos_rets_df = pd.concat(out_of_sample_returns).sort_index()
    oos_scores = scores.loc[oos_rets_df.index]
    oos_targets = targets.loc[oos_rets_df.index]
    oos_regimes = regimes.loc[oos_rets_df.index]

    _, aggregated_metrics = evaluate_ranking(
        scores=oos_scores,
        targets=oos_targets,
        top_k=config.top_k,
        transaction_cost_bps=config.transaction_cost_bps,
        slippage_bps=config.slippage_bps,
        regimes=oos_regimes,
    )

    return tuple(folds), aggregated_metrics


def validate_factor(
    factor: RankingFactor,
    symbol_data: Dict[str, pd.DataFrame],
    config: CrossSectionalResearchConfig,
) -> CrossSectionalValidationReport:
    """Run full, skeptical cross-sectional validation for a RankingFactor."""
    # 1. Full-sample backtest
    _, full_metrics = run_cross_sectional_backtest(factor, symbol_data, config)

    # 2. Monte Carlo joint-block permutation
    perm_result = run_cross_sectional_permutation(factor, symbol_data, full_metrics, config)

    # 3. Out-of-sample walk-forward validation
    try:
        folds, wf_metrics = run_cross_sectional_walk_forward(factor, symbol_data, config)
    except ResearchValidationError as exc:
        logger.warning("Walk-forward validation failed for %s: %s", factor.name, exc)
        folds = ()
        _, wf_metrics = evaluate_ranking(
            scores=pd.DataFrame(index=[]),
            targets=pd.DataFrame(index=[]),
            top_k=config.top_k,
            transaction_cost_bps=config.transaction_cost_bps,
            slippage_bps=config.slippage_bps,
        )

    # 4. Synthesize overall verdict
    fail_reasons = []
    if not perm_result.passes_ic:
        fail_reasons.append("ic_permutation_failed")
    if not perm_result.passes_sharpe:
        fail_reasons.append("sharpe_permutation_failed")
    if wf_metrics.mean_ic < config.minimum_ic:
        fail_reasons.append("out_of_sample_ic_too_low")

    is_valid = len(fail_reasons) == 0

    return CrossSectionalValidationReport(
        factor_name=factor.name,
        full_sample_metrics=full_metrics,
        permutation_result=perm_result,
        walk_forward_folds=folds,
        walk_forward_metrics=wf_metrics,
        is_valid=is_valid,
        fail_reasons=tuple(fail_reasons),
    )
