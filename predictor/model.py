"""Deterministic probabilistic predictor model focused on market structure signals."""

from __future__ import annotations

import math
from typing import Optional

from predictor.config import PredictorConfig
from predictor.types import FeatureVector, MarketRegime, Prediction, PredictorAction


def _clamp(value: float, low: float = -1.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


def _softmax3(a: float, b: float, c: float) -> tuple[float, float, float]:
    max_logit = max(a, b, c)
    ea = math.exp(a - max_logit)
    eb = math.exp(b - max_logit)
    ec = math.exp(c - max_logit)
    total = ea + eb + ec
    return ea / total, eb / total, ec / total


def classify_regime(features: FeatureVector, config: PredictorConfig) -> MarketRegime:
    """Classify a coarse regime from trend and return structure."""

    if (
        features.trend_strength > config.regime_threshold
        and features.long_return > 0
    ):
        return MarketRegime.BULL
    if (
        features.trend_strength < -config.regime_threshold
        and features.long_return < 0
    ):
        return MarketRegime.BEAR
    return MarketRegime.RANGE


def predict_probabilities(
    symbol: str,
    features: FeatureVector,
    config: PredictorConfig,
    *,
    cross_asset_consensus: Optional[float] = None,
    open_position: bool = False,
) -> Prediction:
    """Predict BUY/SELL/HOLD probabilities with explicit uncertainty.

    This model intentionally targets short-horizon probabilistic ranking rather
    than exact price forecasts.
    """

    regime = classify_regime(features, config)
    regime_bias = {
        MarketRegime.BULL: 0.4,
        MarketRegime.BEAR: -0.4,
        MarketRegime.RANGE: 0.0,
    }[regime]

    cross = 0.0 if cross_asset_consensus is None else _clamp(cross_asset_consensus)
    liquidity_signal = _clamp((features.volume_ratio - 1.0) / 2.0)
    mean_reversion_signal = _clamp(-features.vwap_distance_pct / 2.0)
    low_proximity = _clamp((2.0 - features.pct_from_low) / 2.0)
    high_proximity = _clamp((2.0 - features.pct_from_high) / 2.0)
    volatility_penalty = _clamp((features.atr_pct - 2.0) / 4.0, 0.0, 1.0)

    buy_logit = (
        0.45 * liquidity_signal
        + 0.35 * mean_reversion_signal
        + 0.30 * low_proximity
        + 0.25 * regime_bias
        + config.cross_asset_weight * cross
        - 0.25 * volatility_penalty
    )
    sell_logit = (
        0.45 * liquidity_signal
        - 0.35 * mean_reversion_signal
        + 0.30 * high_proximity
        - 0.25 * regime_bias
        - config.cross_asset_weight * cross
        - 0.25 * volatility_penalty
    )
    hold_logit = (
        0.15
        + 0.50 * volatility_penalty
        + 0.35 * (1.0 - abs(liquidity_signal))
    )

    buy_prob, sell_prob, hold_prob = _softmax3(buy_logit, sell_logit, hold_logit)
    confidence = max(buy_prob, sell_prob, hold_prob)
    uncertainty = 1.0 - confidence

    action = PredictorAction.HOLD
    if confidence >= config.min_confidence and uncertainty <= config.max_uncertainty:
        if buy_prob >= sell_prob and buy_prob >= hold_prob:
            action = PredictorAction.BUY
        elif sell_prob >= buy_prob and sell_prob >= hold_prob:
            action = PredictorAction.SELL

    if action == PredictorAction.SELL and not open_position:
        action = PredictorAction.HOLD
    if action == PredictorAction.BUY and open_position:
        # Keep position management outside the predictor; default to HOLD.
        action = PredictorAction.HOLD

    reasons = [
        f"regime={regime.value}",
        f"liquidity={liquidity_signal:.3f}",
        f"order_flow={mean_reversion_signal:.3f}",
        f"cross_asset={cross:.3f}" if cross_asset_consensus is not None else "cross_asset=unavailable",
    ]

    return Prediction(
        symbol=symbol,
        price=features.price,
        action=action,
        regime=regime,
        buy_probability=buy_prob,
        sell_probability=sell_prob,
        hold_probability=hold_prob,
        confidence=confidence,
        uncertainty=uncertainty,
        score=buy_prob - sell_prob,
        reasons=tuple(reasons),
    )
