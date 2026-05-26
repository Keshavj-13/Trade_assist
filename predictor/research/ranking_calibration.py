"""Calibration probes for cross-sectional ranking permutation validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Tuple

import numpy as np
import pandas as pd

from predictor.research.cross_sectional import (
    CrossSectionalPermutationResult,
    CrossSectionalResearchConfig,
    run_cross_sectional_permutation_on_panels,
)
from predictor.research.errors import ResearchInputError
from predictor.research.ranking import evaluate_ranking


@dataclass(frozen=True)
class RankingCalibrationResult:
    """One synthetic probe result for permutation calibration."""

    probe_type: Literal["cheating", "random", "weak"]
    ic_p_values: Tuple[float, ...]
    sharpe_p_values: Tuple[float, ...]
    pass_rate: float
    mean_ic_p_value: float
    expected_pass_rate_range: Tuple[float, float]
    calibration_ok: bool


@dataclass(frozen=True)
class RankingCalibrationSuite:
    """All synthetic probe outcomes."""

    cheating: RankingCalibrationResult
    random: RankingCalibrationResult
    weak: RankingCalibrationResult


def build_synthetic_intraday_targets(
    *,
    num_days: int = 160,
    num_symbols: int = 16,
    seed: int = 42,
) -> pd.DataFrame:
    """Create deterministic synthetic next-day intraday target panel."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=num_days, freq="B")
    symbols = [f"S{i:02d}" for i in range(num_symbols)]

    market = rng.normal(0.0004, 0.007, size=num_days)
    idio = rng.normal(0.0, 0.012, size=(num_days, num_symbols))
    sector = rng.normal(0.0, 0.004, size=(num_days, max(1, num_symbols // 4)))

    data = np.empty((num_days, num_symbols), dtype=float)
    for j in range(num_symbols):
        sector_idx = min(j // 4, sector.shape[1] - 1)
        data[:, j] = market + sector[:, sector_idx] + idio[:, j]

    return pd.DataFrame(data, index=dates, columns=symbols)


def _probe_once(
    *,
    scores: pd.DataFrame,
    targets: pd.DataFrame,
    config: CrossSectionalResearchConfig,
    seed: int,
) -> CrossSectionalPermutationResult:
    _, observed_metrics = evaluate_ranking(
        scores=scores,
        targets=targets,
        top_k=config.top_k,
        transaction_cost_bps=config.transaction_cost_bps,
        slippage_bps=config.slippage_bps,
    )
    trial_config = CrossSectionalResearchConfig(
        **{**config.__dict__, "random_seed": seed}
    )
    return run_cross_sectional_permutation_on_panels(
        scores=scores,
        targets=targets,
        observed_metrics=observed_metrics,
        config=trial_config,
    )


def _build_result(
    probe_type: Literal["cheating", "random", "weak"],
    results: Tuple[CrossSectionalPermutationResult, ...],
    expected_range: Tuple[float, float],
) -> RankingCalibrationResult:
    ic_p_values = tuple(float(r.ic_p_value) for r in results)
    sharpe_p_values = tuple(float(r.sharpe_p_value) for r in results)
    passes = [1.0 if r.passes_ic else 0.0 for r in results]
    pass_rate = float(np.mean(passes)) if passes else 0.0
    low, high = expected_range
    return RankingCalibrationResult(
        probe_type=probe_type,
        ic_p_values=ic_p_values,
        sharpe_p_values=sharpe_p_values,
        pass_rate=pass_rate,
        mean_ic_p_value=float(np.mean(ic_p_values)) if ic_p_values else 1.0,
        expected_pass_rate_range=expected_range,
        calibration_ok=(low <= pass_rate <= high),
    )


def run_ranking_calibration_suite(
    targets: pd.DataFrame,
    *,
    config: CrossSectionalResearchConfig,
    n_trials: int = 5,
    weak_signal_strength: float = 0.12,
) -> RankingCalibrationSuite:
    """Run cheating/random/weak synthetic probes for permutation calibration."""
    if n_trials <= 0:
        raise ResearchInputError("n_trials must be > 0")
    if not 0.0 < weak_signal_strength < 1.0:
        raise ResearchInputError("weak_signal_strength must be in (0, 1)")

    clean_targets = targets.fillna(0.0).astype(float)
    rng = np.random.default_rng(config.random_seed)

    cheating_results = []
    random_results = []
    weak_results = []

    for trial in range(n_trials):
        trial_seed = int(rng.integers(0, 2**31 - 1))
        trial_rng = np.random.default_rng(trial_seed)

        noise = pd.DataFrame(
            trial_rng.normal(0.0, 1.0, size=clean_targets.shape),
            index=clean_targets.index,
            columns=clean_targets.columns,
        )

        cheating_scores = clean_targets + (noise * 1e-6)
        random_scores = pd.DataFrame(
            trial_rng.normal(0.0, 1.0, size=clean_targets.shape),
            index=clean_targets.index,
            columns=clean_targets.columns,
        )
        weak_scores = (clean_targets * weak_signal_strength) + (noise * (1.0 - weak_signal_strength))

        cheating_results.append(
            _probe_once(
                scores=cheating_scores,
                targets=clean_targets,
                config=config,
                seed=trial_seed + 11,
            )
        )
        random_results.append(
            _probe_once(
                scores=random_scores,
                targets=clean_targets,
                config=config,
                seed=trial_seed + 29,
            )
        )
        weak_results.append(
            _probe_once(
                scores=weak_scores,
                targets=clean_targets,
                config=config,
                seed=trial_seed + 47,
            )
        )

    return RankingCalibrationSuite(
        cheating=_build_result("cheating", tuple(cheating_results), expected_range=(0.80, 1.00)),
        random=_build_result("random", tuple(random_results), expected_range=(0.00, 0.20)),
        weak=_build_result("weak", tuple(weak_results), expected_range=(0.00, 0.70)),
    )
