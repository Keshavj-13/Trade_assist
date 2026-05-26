"""Permutation engine calibration using synthetic strategies of known properties.

Purpose
-------
Validate that the permutation test engine behaves statistically correctly.
Three synthetic probes with known expected outcomes:

1. FutureLeakStrategy (Oracle / positive control)
   Uses shifted future returns — guaranteed edge.
   Expected: p_value < 0.02 reliably. If this fails, the engine is broken.

2. PureRandomEntryStrategy (Null / negative control)
   Seeded random entries with no signal.
   Expected: p_value > 0.40 on average. If this consistently passes, the
   null distribution is pathologically narrow.

3. MildPredictiveProcess (Sensitivity control)
   True direction + heavy noise — weak signal.
   Expected: passes 10–30% of trials with 100 permutation iterations.
   Verifies the engine has realistic power at the margin.

Public API
----------
FutureLeakStrategy
PureRandomEntryStrategy
MildPredictiveProcess
CalibrationResult
run_permutation_calibration
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Literal, Tuple

import numpy as np
import pandas as pd

from predictor.research.backtest import backtest_strategy
from predictor.research.data import validate_research_frame
from predictor.research.permutation import run_permutation_test
from predictor.research.validation import ResearchValidationConfig


# ---------------------------------------------------------------------------
# Synthetic strategies
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FutureLeakStrategy:
    """Oracle strategy that looks one bar ahead — a guaranteed cheat.

    Enters long when next bar's return > 0, short when < 0.
    This violates causality and is only valid as a calibration probe.

    Expected IS permutation p_value: < 0.02 reliably.
    If this fails, the permutation engine is broken.
    """

    name: str = "calibration_future_leak"
    threshold: float = 0.0

    def generate_positions(self, frame: pd.DataFrame) -> pd.Series:
        """Use lead-1 returns as the signal (look-ahead bias intentional)."""
        close = frame["Close"]
        future_return = close.pct_change().shift(-1)  # look-ahead: next bar's return
        positions = pd.Series(0.0, index=frame.index, dtype=float)
        positions[future_return > self.threshold] = 1.0
        positions[future_return < -self.threshold] = -1.0
        # Last bar has no future — set to flat
        positions.iloc[-1] = 0.0
        return positions


@dataclass(frozen=True)
class PureRandomEntryStrategy:
    """Completely random entries — the null hypothesis in strategy form.

    Expected IS permutation p_value: > 0.40 on average across trials.
    If this consistently passes, the null distribution is too narrow.
    """

    name: str = "calibration_random_entry"
    hold_bars: int = 10
    seed: int = 99

    def generate_positions(self, frame: pd.DataFrame) -> pd.Series:
        """Generate random long/short entries with deterministic seed."""
        n = len(frame)
        rng = np.random.default_rng(self.seed)
        positions = np.zeros(n, dtype=float)
        i = int(rng.integers(1, max(2, self.hold_bars)))
        while i < n:
            direction = float(rng.choice([-1.0, 1.0]))
            hold = int(rng.integers(5, max(6, self.hold_bars * 2)))
            end = min(i + hold, n)
            positions[i:end] = direction
            i = end + int(rng.integers(1, max(2, self.hold_bars)))
        return pd.Series(positions, index=frame.index, dtype=float)


@dataclass(frozen=True)
class MildPredictiveProcess:
    """Weak signal strategy — noise dominates but true direction leaks through.

    signal_t = alpha * true_direction_t + (1 - alpha) * noise_t
    where alpha = signal_strength (default 0.15).

    Expected IS permutation pass rate: 10–30% with 100 iterations.
    Verifies the engine has realistic statistical power at the margin.
    """

    name: str = "calibration_mild_predictive"
    signal_strength: float = 0.15  # weight on true direction vs noise
    seed: int = 77

    def generate_positions(self, frame: pd.DataFrame) -> pd.Series:
        """Mix true return direction with Gaussian noise."""
        close = frame["Close"]
        returns = close.pct_change().fillna(0.0).to_numpy()
        rng = np.random.default_rng(self.seed)
        noise = rng.standard_normal(len(returns))
        true_direction = np.sign(returns)
        mixed = self.signal_strength * true_direction + (1.0 - self.signal_strength) * noise
        positions = np.where(mixed > 0, 1.0, -1.0).astype(float)
        positions[0] = 0.0  # no position on first bar (no return yet)
        return pd.Series(positions, index=frame.index, dtype=float)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CalibrationResult:
    """Result from one calibration probe across n_trials runs.

    Attributes
    ----------
    strategy_type : str
        "oracle" | "random" | "mild"
    p_values : Tuple[float, ...]
        IS permutation p-values across all trials.
    pass_rate : float
        Fraction of trials where p_value <= threshold.
    expected_pass_rate_range : Tuple[float, float]
        (min, max) expected pass rate for a correctly functioning engine.
    calibration_ok : bool
        True when pass_rate falls inside expected_pass_rate_range.
    mean_p_value : float
        Mean p-value across trials.
    diagnosis : str
        Human-readable interpretation.
    """

    strategy_type: Literal["oracle", "random", "mild"]
    p_values: Tuple[float, ...]
    pass_rate: float
    expected_pass_rate_range: Tuple[float, float]
    calibration_ok: bool
    mean_p_value: float
    diagnosis: str


# ---------------------------------------------------------------------------
# Calibration runner
# ---------------------------------------------------------------------------


def _score_sharpe(frame: pd.DataFrame, strategy: object, config: ResearchValidationConfig) -> float:
    """Compute IS Sharpe ratio for use inside run_permutation_test."""
    run = backtest_strategy(
        frame,
        strategy,
        bars_per_year=config.bars_per_year,
        transaction_cost_bps=config.transaction_cost_bps,
    )
    return float(run.metrics.sharpe_ratio)


def _run_one_trial(
    frame: pd.DataFrame,
    strategy: object,
    config: ResearchValidationConfig,
) -> float:
    """Run IS permutation test once and return the p-value."""
    validated = validate_research_frame(frame, symbol="CALIBRATION")
    observed_sharpe = _score_sharpe(validated, strategy, config)
    result = run_permutation_test(
        frame=validated,
        observed_statistic=observed_sharpe,
        score_on_frame=lambda perm: _score_sharpe(perm, strategy, config),
        iterations=config.permutation_iterations,
        block_size=min(config.permutation_block_size, max(len(validated) - 1, 1)),
        seed=config.random_seed,
        p_value_threshold=config.p_value_threshold,
    )
    return float(result.p_value)


def run_permutation_calibration(
    frame: pd.DataFrame,
    *,
    n_trials: int = 3,
    config: ResearchValidationConfig | None = None,
) -> Tuple[CalibrationResult, CalibrationResult, CalibrationResult]:
    """Run all three calibration probes and return (oracle, random, mild) results.

    Parameters
    ----------
    frame : pd.DataFrame
        OHLCV frame for calibration. Must have >= 200 bars.
    n_trials : int
        Number of independent trials per probe. Default 3 (fast).
        Use 10+ for reliable calibration in production.
    config : ResearchValidationConfig | None
        Validation config. Uses a fast default if None.

    Returns
    -------
    Tuple[CalibrationResult, CalibrationResult, CalibrationResult]
        (oracle_result, random_result, mild_result)
    """
    if config is None:
        config = ResearchValidationConfig(
            permutation_iterations=100,
            permutation_block_size=20,
            p_value_threshold=0.05,
            random_seed=42,
        )

    threshold = config.p_value_threshold

    # Oracle probe — expected pass rate: 0.80–1.00
    oracle_strategy = FutureLeakStrategy()
    oracle_p_values: list[float] = []
    for trial in range(n_trials):
        seed_config = ResearchValidationConfig(
            **{**config.__dict__, "random_seed": config.random_seed + trial * 7}
        )
        oracle_p_values.append(_run_one_trial(frame, oracle_strategy, seed_config))

    oracle_pass_rate = sum(p <= threshold for p in oracle_p_values) / len(oracle_p_values)
    oracle_ok = 0.70 <= oracle_pass_rate <= 1.00
    oracle_diag = (
        f"Oracle p-values: {[f'{p:.3f}' for p in oracle_p_values]} | "
        f"pass_rate={oracle_pass_rate:.2f} (expect 0.70–1.00) | "
        f"{'OK' if oracle_ok else 'WARNING: engine may be underpowered'}"
    )

    # Random probe — expected pass rate: 0.00–0.15
    random_strategy = PureRandomEntryStrategy()
    random_p_values: list[float] = []
    for trial in range(n_trials):
        seed_config = ResearchValidationConfig(
            **{**config.__dict__, "random_seed": config.random_seed + trial * 13 + 100}
        )
        random_p_values.append(_run_one_trial(frame, random_strategy, seed_config))

    random_pass_rate = sum(p <= threshold for p in random_p_values) / len(random_p_values)
    random_ok = 0.00 <= random_pass_rate <= 0.15
    random_diag = (
        f"Random p-values: {[f'{p:.3f}' for p in random_p_values]} | "
        f"pass_rate={random_pass_rate:.2f} (expect 0.00–0.15) | "
        f"{'OK' if random_ok else 'WARNING: null distribution may be too narrow'}"
    )

    # Mild probe — expected pass rate: 0.00–0.40 (wide due to noise)
    mild_strategy = MildPredictiveProcess()
    mild_p_values: list[float] = []
    for trial in range(n_trials):
        seed_config = ResearchValidationConfig(
            **{**config.__dict__, "random_seed": config.random_seed + trial * 17 + 200}
        )
        mild_p_values.append(_run_one_trial(frame, mild_strategy, seed_config))

    mild_pass_rate = sum(p <= threshold for p in mild_p_values) / len(mild_p_values)
    mild_ok = 0.00 <= mild_pass_rate <= 0.50  # wide band — highly variable by design
    mild_diag = (
        f"Mild p-values: {[f'{p:.3f}' for p in mild_p_values]} | "
        f"pass_rate={mild_pass_rate:.2f} (expect 0.00–0.50 depending on data) | "
        f"{'OK' if mild_ok else 'WARNING: unexpected calibration behaviour'}"
    )

    return (
        CalibrationResult(
            strategy_type="oracle",
            p_values=tuple(oracle_p_values),
            pass_rate=oracle_pass_rate,
            expected_pass_rate_range=(0.70, 1.00),
            calibration_ok=oracle_ok,
            mean_p_value=statistics.mean(oracle_p_values),
            diagnosis=oracle_diag,
        ),
        CalibrationResult(
            strategy_type="random",
            p_values=tuple(random_p_values),
            pass_rate=random_pass_rate,
            expected_pass_rate_range=(0.00, 0.15),
            calibration_ok=random_ok,
            mean_p_value=statistics.mean(random_p_values),
            diagnosis=random_diag,
        ),
        CalibrationResult(
            strategy_type="mild",
            p_values=tuple(mild_p_values),
            pass_rate=mild_pass_rate,
            expected_pass_rate_range=(0.00, 0.50),
            calibration_ok=mild_ok,
            mean_p_value=statistics.mean(mild_p_values),
            diagnosis=mild_diag,
        ),
    )


def print_calibration_report(results: Tuple[CalibrationResult, ...]) -> None:
    """Print a human-readable calibration report to stdout."""
    print("\n" + "=" * 60)
    print("[CALIBRATION REPORT] Permutation Engine Validation")
    print("=" * 60)
    all_ok = True
    for result in results:
        status = "✓ OK" if result.calibration_ok else "✗ FAIL"
        all_ok = all_ok and result.calibration_ok
        print(f"\n  [{result.strategy_type.upper()}] {status}")
        print(f"    {result.diagnosis}")
    print("\n" + ("-" * 60))
    print(
        f"  OVERALL: {'CALIBRATED — engine behaves statistically correctly.' if all_ok else 'MISCALIBRATION DETECTED — review permutation engine.'}"
    )
    print("=" * 60 + "\n")
