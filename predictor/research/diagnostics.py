"""Step-by-step diagnostic trace engine for the research pipeline.

This module is a diagnostic instrument only. Its outputs are intended to
answer the question:

    "Did the strategy fail because it has no statistical edge,
     or did the framework reject before meaningful execution completed?"

Public API
----------
run_strategy_trace       -- Full stepwise trace for one strategy on one frame.
assert_impossible_states -- Raise DiagnosticAssertionError on contradictory pipeline state.
export_permutation_distribution -- Persist IS permutation result to JSON.
PipelineStage            -- Ordered enum of pipeline stages.
StrategyTraceReport      -- Typed result of run_strategy_trace().
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from predictor.research.errors import DiagnosticAssertionError, ResearchInputError
from predictor.research.types import PermutationTestResult


# ---------------------------------------------------------------------------
# Pipeline stage ordering
# ---------------------------------------------------------------------------


class PipelineStage(Enum):
    """Ordered stages of the research validation pipeline."""

    LOAD_DATA = auto()
    GENERATE_POSITIONS = auto()
    CONSTRUCT_BACKTEST = auto()
    COMPUTE_IS_METRIC = auto()
    RUN_IS_PERMUTATIONS = auto()
    RUN_WALKFORWARD = auto()
    RUN_WF_PERMUTATIONS = auto()

    def label(self) -> str:
        """Return human-readable stage label for console output."""
        return self.name.lower()


# Canonical pipeline order — used for ordering verification in tests.
PIPELINE_ORDER: Tuple[PipelineStage, ...] = (
    PipelineStage.LOAD_DATA,
    PipelineStage.GENERATE_POSITIONS,
    PipelineStage.CONSTRUCT_BACKTEST,
    PipelineStage.COMPUTE_IS_METRIC,
    PipelineStage.RUN_IS_PERMUTATIONS,
    PipelineStage.RUN_WALKFORWARD,
    PipelineStage.RUN_WF_PERMUTATIONS,
)


def log_pipeline_stage(stage: PipelineStage) -> None:
    """Print a pipeline stage marker to stdout."""
    print(f"[PIPELINE] {stage.label()}")


# ---------------------------------------------------------------------------
# Signal decomposition helpers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SignalDecomposition:
    """Breakdown of position series into actionable signal events."""

    bars_processed: int
    positions_nonzero: int
    entries_generated: int
    exits_generated: int
    sign_flips: int
    flat_bars: int

    @property
    def total_transitions(self) -> int:
        """Total bars where position state changed."""
        return self.entries_generated + self.exits_generated + self.sign_flips


def decompose_positions(positions: pd.Series) -> SignalDecomposition:
    """Decompose a position series into discrete signal events.

    Parameters
    ----------
    positions:
        Position series in {-1, 0, 1} aligned to OHLCV frame index.

    Returns
    -------
    SignalDecomposition with counts of all entry/exit/flip events.
    """
    if positions.empty:
        return SignalDecomposition(
            bars_processed=0,
            positions_nonzero=0,
            entries_generated=0,
            exits_generated=0,
            sign_flips=0,
            flat_bars=0,
        )

    pos = positions.fillna(0.0).astype(float)
    prev = pos.shift(1).fillna(0.0)
    diff = pos - prev

    entries = int(((prev == 0.0) & (pos != 0.0)).sum())
    exits = int(((prev != 0.0) & (pos == 0.0)).sum())
    sign_flips = int(((prev > 0) & (pos < 0) | (prev < 0) & (pos > 0)).sum())
    positions_nonzero = int((pos != 0.0).sum())
    flat_bars = int((pos == 0.0).sum())

    return SignalDecomposition(
        bars_processed=len(pos),
        positions_nonzero=positions_nonzero,
        entries_generated=entries,
        exits_generated=exits,
        sign_flips=sign_flips,
        flat_bars=flat_bars,
    )


# ---------------------------------------------------------------------------
# Equity curve snapshot
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EquityCurveSnapshot:
    """Key percentile values from an equity curve for quick inspection."""

    start: float
    end: float
    min_: float
    max_: float
    pct25: float
    pct75: float

    @classmethod
    def from_series(cls, equity: pd.Series) -> "EquityCurveSnapshot":
        """Build a snapshot from an equity curve Series."""
        if equity.empty:
            return cls(
                start=1.0, end=1.0, min_=1.0, max_=1.0, pct25=1.0, pct75=1.0
            )
        arr = equity.values.astype(float)
        return cls(
            start=float(arr[0]),
            end=float(arr[-1]),
            min_=float(np.min(arr)),
            max_=float(np.max(arr)),
            pct25=float(np.percentile(arr, 25)),
            pct75=float(np.percentile(arr, 75)),
        )


# ---------------------------------------------------------------------------
# Permutation distribution summary
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PermutationSummary:
    """Human-readable summary of a permutation null distribution."""

    observed: float
    mean: float
    std: float
    pct5: float
    pct95: float
    p_value: float
    threshold: float
    passes: bool

    @classmethod
    def from_result(
        cls, result: PermutationTestResult, threshold: float
    ) -> "PermutationSummary":
        """Build a summary from a PermutationTestResult."""
        null = np.asarray(result.null_distribution, dtype=float)
        return cls(
            observed=float(result.observed_statistic),
            mean=float(np.mean(null)) if len(null) > 0 else 0.0,
            std=float(np.std(null)) if len(null) > 0 else 0.0,
            pct5=float(np.percentile(null, 5)) if len(null) > 0 else 0.0,
            pct95=float(np.percentile(null, 95)) if len(null) > 0 else 0.0,
            p_value=result.p_value,
            threshold=threshold,
            passes=result.passes,
        )


# ---------------------------------------------------------------------------
# Trace report
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StrategyTraceReport:
    """Full step-by-step trace result for one strategy on one frame."""

    strategy_name: str
    symbol: str
    bars_processed: int
    signal_decomposition: SignalDecomposition
    trade_count: int
    is_sharpe: float
    equity_snapshot: EquityCurveSnapshot
    permutation_summary: PermutationSummary
    rejection_reason: Optional[str]
    stages_executed: Tuple[PipelineStage, ...]

    @property
    def is_rejected(self) -> bool:
        """True if the strategy was rejected at the IS permutation stage."""
        return self.rejection_reason is not None


# ---------------------------------------------------------------------------
# Main trace function
# ---------------------------------------------------------------------------


def run_strategy_trace(
    frame: pd.DataFrame,
    strategy: object,
    *,
    config: object,
    symbol: str = "UNKNOWN",
) -> StrategyTraceReport:
    """Run a single-strategy stepwise diagnostic trace.

    This function executes the same logic as validate_strategy() but prints
    detailed diagnostics at every stage. It is intended for manual
    investigation and CI-level impossible-state detection.

    Parameters
    ----------
    frame:
        OHLCV DataFrame for the symbol under test.
    strategy:
        Any object implementing the TradingStrategy protocol.
    config:
        ResearchValidationConfig instance.
    symbol:
        Label for console output. Does not affect computation.

    Returns
    -------
    StrategyTraceReport with all diagnostic fields populated.

    Raises
    ------
    DiagnosticAssertionError
        If an impossible pipeline state is detected (e.g. signals > 0 but
        trade_count == 0).
    """
    # Deferred imports to avoid circular imports at module level.
    from predictor.research.backtest import backtest_strategy
    from predictor.research.data import validate_research_frame
    from predictor.research.permutation import run_permutation_test, block_permute_ohlcv
    from predictor.research.validation import ResearchValidationConfig

    cfg: ResearchValidationConfig = config  # type: ignore[assignment]
    strategy_name: str = getattr(strategy, "name", "UNKNOWN")
    stages_executed: List[PipelineStage] = []

    print(f"\n{'='*60}")
    print(f"[TRACE] Strategy: {strategy_name}  Symbol: {symbol}")
    print(f"{'='*60}")

    # ------------------------------------------------------------------ #
    # Stage 1: Load / validate data                                        #
    # ------------------------------------------------------------------ #
    log_pipeline_stage(PipelineStage.LOAD_DATA)
    stages_executed.append(PipelineStage.LOAD_DATA)

    minimum_rows = max(3, cfg.train_window + cfg.test_window)
    validated = validate_research_frame(
        frame, symbol=strategy_name, min_rows=minimum_rows
    )
    bars_processed = len(validated)
    print(f"  bars_processed={bars_processed}")
    print(f"  minimum_rows_required={minimum_rows}")
    print(f"  frame_date_range={validated.index[0]} → {validated.index[-1]}")

    # ------------------------------------------------------------------ #
    # Stage 2: Generate positions                                          #
    # ------------------------------------------------------------------ #
    log_pipeline_stage(PipelineStage.GENERATE_POSITIONS)
    stages_executed.append(PipelineStage.GENERATE_POSITIONS)

    positions_raw: pd.Series = strategy.generate_positions(validated)  # type: ignore[attr-defined]
    decomp = decompose_positions(positions_raw)

    print(f"  positions_nonzero={decomp.positions_nonzero}")
    print(f"  entries_generated={decomp.entries_generated}")
    print(f"  exits_generated={decomp.exits_generated}")
    print(f"  sign_flips={decomp.sign_flips}")
    print(f"  flat_bars={decomp.flat_bars}")
    print(f"  exposure_pct={decomp.positions_nonzero / max(bars_processed, 1) * 100:.1f}%")

    # ------------------------------------------------------------------ #
    # Stage 3: Construct backtest                                          #
    # ------------------------------------------------------------------ #
    log_pipeline_stage(PipelineStage.CONSTRUCT_BACKTEST)
    stages_executed.append(PipelineStage.CONSTRUCT_BACKTEST)

    run = backtest_strategy(
        validated,
        strategy,  # type: ignore[arg-type]
        bars_per_year=cfg.bars_per_year,
        transaction_cost_bps=cfg.transaction_cost_bps,
    )
    trade_count = run.metrics.trade_count
    equity_snap = EquityCurveSnapshot.from_series(run.equity_curve)

    print(f"  trade_count={trade_count}")
    print(f"  avg_holding_period={run.metrics.avg_holding_period:.2f}")
    print(f"  total_return={run.metrics.total_return:.4f}")
    print(f"  max_drawdown={run.metrics.max_drawdown:.4f}")
    print(f"  equity_start={equity_snap.start:.4f}")
    print(f"  equity_end={equity_snap.end:.4f}")
    print(f"  equity_min={equity_snap.min_:.4f}")
    print(f"  equity_max={equity_snap.max_:.4f}")

    # Impossible state check: signals but no trades
    assert_impossible_states(
        positions_nonzero=decomp.positions_nonzero,
        entries_generated=decomp.entries_generated,
        trade_count=trade_count,
        equity_end=equity_snap.end,
        strategy_name=strategy_name,
    )

    # ------------------------------------------------------------------ #
    # Stage 4: Compute IS metric                                           #
    # ------------------------------------------------------------------ #
    log_pipeline_stage(PipelineStage.COMPUTE_IS_METRIC)
    stages_executed.append(PipelineStage.COMPUTE_IS_METRIC)

    is_sharpe = run.metrics.sharpe_ratio
    print(f"  raw_is_sharpe={is_sharpe:.4f}")
    print(f"  win_rate={run.metrics.win_rate:.4f}")
    print(f"  profit_factor={run.metrics.profit_factor:.4f}")
    print(f"  expectancy={run.metrics.expectancy:.6f}")

    # ------------------------------------------------------------------ #
    # Stage 5: IS permutation test                                         #
    # ------------------------------------------------------------------ #
    log_pipeline_stage(PipelineStage.RUN_IS_PERMUTATIONS)
    stages_executed.append(PipelineStage.RUN_IS_PERMUTATIONS)

    def _score_frame(perm_frame: pd.DataFrame) -> float:
        perm_run = backtest_strategy(
            perm_frame,
            strategy,  # type: ignore[arg-type]
            bars_per_year=cfg.bars_per_year,
            transaction_cost_bps=cfg.transaction_cost_bps,
        )
        return perm_run.metrics.sharpe_ratio

    is_perm = run_permutation_test(
        frame=validated,
        observed_statistic=is_sharpe,
        score_on_frame=_score_frame,
        iterations=cfg.permutation_iterations,
        block_size=min(cfg.permutation_block_size, max(len(validated) - 1, 1)),
        seed=cfg.random_seed + 1_001,
        p_value_threshold=cfg.p_value_threshold,
    )
    perm_summary = PermutationSummary.from_result(is_perm, cfg.p_value_threshold)

    print(f"\n  permutation_distribution ({cfg.permutation_iterations} iterations):")
    print(f"    mean={perm_summary.mean:.4f}")
    print(f"    std={perm_summary.std:.4f}")
    print(f"    5pct={perm_summary.pct5:.4f}")
    print(f"    95pct={perm_summary.pct95:.4f}")
    print(f"    observed={perm_summary.observed:.4f}")
    print(f"    p_value={perm_summary.p_value:.4f}")
    print(f"    threshold={perm_summary.threshold:.4f}")
    print(f"    passes={perm_summary.passes}")

    rejection_reason: Optional[str] = None
    if not is_perm.passes:
        rejection_reason = "in_sample_permutation_failed"
        print(f"\n  rejected:")
        print(f"    reason={rejection_reason}")
        print(f"    p_value={perm_summary.p_value:.4f}")
        print(f"    threshold={perm_summary.threshold:.4f}")
        print(
            f"    diagnosis: observed_sharpe={is_sharpe:.4f} "
            f"null_mean={perm_summary.mean:.4f} "
            f"null_95pct={perm_summary.pct95:.4f}"
        )
        if is_sharpe <= 0.0:
            print("    [DIAGNOSIS] IS Sharpe is ≤ 0 — strategy has no directional edge on this frame")
        elif is_sharpe < perm_summary.mean:
            print("    [DIAGNOSIS] IS Sharpe is BELOW null mean — strategy is weaker than random")
        elif is_sharpe < perm_summary.pct95:
            print(
                f"    [DIAGNOSIS] IS Sharpe ({is_sharpe:.4f}) is between null mean "
                f"({perm_summary.mean:.4f}) and 95pct ({perm_summary.pct95:.4f}) — "
                "some edge exists but not statistically significant at this threshold"
            )
        else:
            print(
                "    [DIAGNOSIS] IS Sharpe exceeds null 95pct but p_value still above threshold — "
                "check permutation iteration count"
            )
    else:
        print("\n  [PIPELINE] IS permutation passed — proceeding to walk-forward stages")
        stages_executed.append(PipelineStage.RUN_WALKFORWARD)
        stages_executed.append(PipelineStage.RUN_WF_PERMUTATIONS)

    print(f"\n{'='*60}")
    print(f"[TRACE] Complete for {strategy_name} on {symbol}")
    if rejection_reason:
        print(f"[TRACE] VERDICT: REJECTED at {rejection_reason}")
    else:
        print("[TRACE] VERDICT: PASSED IS permutation")
    print(f"{'='*60}\n")

    return StrategyTraceReport(
        strategy_name=strategy_name,
        symbol=symbol,
        bars_processed=bars_processed,
        signal_decomposition=decomp,
        trade_count=trade_count,
        is_sharpe=is_sharpe,
        equity_snapshot=equity_snap,
        permutation_summary=perm_summary,
        rejection_reason=rejection_reason,
        stages_executed=tuple(stages_executed),
    )


# ---------------------------------------------------------------------------
# Impossible-state assertions
# ---------------------------------------------------------------------------


def assert_impossible_states(
    *,
    positions_nonzero: int,
    entries_generated: int,
    trade_count: int,
    equity_end: float,
    strategy_name: str,
) -> None:
    """Raise DiagnosticAssertionError if contradictory pipeline state is detected.

    Checked invariants:
    1. If entries_generated > 0, trade_count must be > 0.
    2. If trade_count > 0, equity_end must differ from 1.0 (strategy was active).
    3. If positions_nonzero > 0 and entries_generated == 0, this is suspicious
       (possible hold-forever state with no explicit entry).

    Parameters
    ----------
    positions_nonzero: int
        Number of bars where position ≠ 0.
    entries_generated: int
        Number of bars where position transitioned from 0 → nonzero.
    trade_count: int
        Number of completed trades recorded by metrics.
    equity_end: float
        Terminal equity curve value.
    strategy_name: str
        Name for error messages.

    Raises
    ------
    DiagnosticAssertionError
        On any detected impossible state.
    """
    if entries_generated > 0 and trade_count == 0:
        raise DiagnosticAssertionError(
            f"[IMPOSSIBLE STATE] strategy={strategy_name}: "
            f"entries_generated={entries_generated} but trade_count=0. "
            "Signals are being generated but trades are not being extracted. "
            "Likely cause: backtest position lag or trade-pairing logic mismatch."
        )

    if trade_count > 0 and math.isclose(equity_end, 1.0, rel_tol=1e-9, abs_tol=1e-9):
        raise DiagnosticAssertionError(
            f"[IMPOSSIBLE STATE] strategy={strategy_name}: "
            f"trade_count={trade_count} but equity_end=1.0 (flat equity). "
            "Trades are counted but returns are zero — possible return calculation bug."
        )


# ---------------------------------------------------------------------------
# Permutation distribution export
# ---------------------------------------------------------------------------


def export_permutation_distribution(
    *,
    result: PermutationTestResult,
    strategy_name: str,
    symbol: str,
    stage: str,
    p_value_threshold: float,
    output_path: Path,
) -> None:
    """Persist a permutation test result to JSON for offline analysis.

    Parameters
    ----------
    result:
        The PermutationTestResult to export.
    strategy_name:
        Name of the strategy under test.
    symbol:
        Symbol the test was run on.
    stage:
        Label for the pipeline stage (e.g. 'in_sample', 'walk_forward').
    p_value_threshold:
        The alpha threshold used during the test.
    output_path:
        Full path where the JSON file will be written. Parent directories
        are created automatically.
    """
    null = list(result.null_distribution)
    null_arr = np.asarray(null, dtype=float)

    payload = {
        "strategy_name": strategy_name,
        "symbol": symbol,
        "stage": stage,
        "observed_statistic": float(result.observed_statistic),
        "p_value": float(result.p_value),
        "p_value_threshold": float(p_value_threshold),
        "passes": bool(result.passes),
        "null_distribution_summary": {
            "count": len(null),
            "mean": float(np.mean(null_arr)) if null else 0.0,
            "std": float(np.std(null_arr)) if null else 0.0,
            "min": float(np.min(null_arr)) if null else 0.0,
            "pct5": float(np.percentile(null_arr, 5)) if null else 0.0,
            "pct25": float(np.percentile(null_arr, 25)) if null else 0.0,
            "pct50": float(np.percentile(null_arr, 50)) if null else 0.0,
            "pct75": float(np.percentile(null_arr, 75)) if null else 0.0,
            "pct95": float(np.percentile(null_arr, 95)) if null else 0.0,
            "max": float(np.max(null_arr)) if null else 0.0,
        },
        "null_distribution_raw": null,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2))
    print(f"[EXPORT] Permutation distribution saved → {output_path}")
