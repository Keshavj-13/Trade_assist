"""Baseline ranking systems for cross-sectional next-day evaluation."""

from __future__ import annotations

from typing import Tuple

from predictor.research.factors import (
    BuyAndHoldBaselineFactor,
    EqualWeightSelectionFactor,
    RandomRankingFactor,
    RankingFactor,
    SimpleMomentumRankFactor,
    VolatilityRankFactor,
)


def build_ranking_baseline_universe() -> Tuple[RankingFactor, ...]:
    """Return mandatory baseline ranking systems for contextual performance."""
    return (
        RandomRankingFactor(name="random_ranking", seed=42),
        BuyAndHoldBaselineFactor(name="buy_and_hold_baseline"),
        SimpleMomentumRankFactor(name="simple_momentum_rank", lookback=5),
        VolatilityRankFactor(name="volatility_rank", window=20),
        EqualWeightSelectionFactor(name="equal_weight_selection"),
    )
