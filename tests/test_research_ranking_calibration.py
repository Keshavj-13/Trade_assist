"""Calibration tests for cross-sectional ranking permutation engine."""

from __future__ import annotations

from predictor.research.cross_sectional import CrossSectionalResearchConfig
from predictor.research.ranking_calibration import (
    build_synthetic_intraday_targets,
    run_ranking_calibration_suite,
)


def test_ranking_calibration_behaviour_ordering():
    targets = build_synthetic_intraday_targets(num_days=140, num_symbols=14, seed=21)
    config = CrossSectionalResearchConfig(
        top_k=3,
        transaction_cost_bps=5.0,
        slippage_bps=2.0,
        permutation_iterations=80,
        permutation_block_size=10,
        p_value_threshold=0.05,
        minimum_ic=0.01,
        random_seed=42,
    )

    suite = run_ranking_calibration_suite(
        targets,
        config=config,
        n_trials=4,
        weak_signal_strength=0.12,
    )

    assert suite.cheating.pass_rate >= 0.75
    assert suite.random.pass_rate <= 0.25
    assert suite.cheating.mean_ic_p_value < suite.random.mean_ic_p_value
    assert suite.cheating.mean_ic_p_value <= suite.weak.mean_ic_p_value
