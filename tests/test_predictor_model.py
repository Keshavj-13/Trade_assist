"""Tests for deterministic probabilistic predictor model."""

from __future__ import annotations

from predictor.config import PredictorConfig
from predictor.features import build_feature_vector
from predictor.model import classify_regime, predict_probabilities
from predictor.types import FeatureVector, MarketRegime, PredictorAction


def test_classify_regime_detects_bull_and_bear(make_ohlcv_frame) -> None:
    cfg = PredictorConfig(regime_threshold=0.001)
    bull_features = build_feature_vector(make_ohlcv_frame(trend=0.15))
    bear_features = build_feature_vector(make_ohlcv_frame(trend=-0.15))

    assert classify_regime(bull_features, cfg) == MarketRegime.BULL
    assert classify_regime(bear_features, cfg) == MarketRegime.BEAR


def test_predict_probabilities_sum_to_one(make_ohlcv_frame) -> None:
    cfg = PredictorConfig()
    features = build_feature_vector(make_ohlcv_frame())
    pred = predict_probabilities("INFY", features, cfg)

    total = pred.buy_probability + pred.sell_probability + pred.hold_probability
    assert abs(total - 1.0) < 1e-9


def test_predict_probabilities_respects_uncertainty_threshold(make_ohlcv_frame) -> None:
    cfg = PredictorConfig(min_confidence=0.95, max_uncertainty=0.05)
    features = build_feature_vector(make_ohlcv_frame())
    pred = predict_probabilities("INFY", features, cfg)

    assert pred.action == PredictorAction.HOLD


def test_cross_asset_consensus_increases_buy_probability(make_ohlcv_frame) -> None:
    cfg = PredictorConfig()
    features = build_feature_vector(make_ohlcv_frame(trend=0.10))

    without_consensus = predict_probabilities("INFY", features, cfg, cross_asset_consensus=None)
    with_consensus = predict_probabilities("INFY", features, cfg, cross_asset_consensus=1.0)

    assert with_consensus.buy_probability > without_consensus.buy_probability


def test_confidence_and_uncertainty_consistency(make_ohlcv_frame) -> None:
    cfg = PredictorConfig()
    features = build_feature_vector(make_ohlcv_frame(trend=0.05))
    pred = predict_probabilities("INFY", features, cfg)

    assert pred.confidence == max(pred.buy_probability, pred.sell_probability, pred.hold_probability)
    assert abs(pred.uncertainty - (1.0 - pred.confidence)) < 1e-12


def test_sell_action_is_blocked_without_open_position() -> None:
    cfg = PredictorConfig(min_confidence=0.10, max_uncertainty=1.0)
    bearish = FeatureVector(
        price=100.0,
        ema20=98.0,
        ema50=102.0,
        rsi=35.0,
        atr_pct=1.0,
        avg_volume=200_000.0,
        volume=250_000.0,
        volume_ratio=1.25,
        vwap=95.0,
        vwap_distance_pct=5.0,
        session_low=94.0,
        session_high=106.0,
        pct_from_low=6.0,
        pct_from_high=1.0,
        trend_strength=-0.02,
        short_return=-0.01,
        long_return=-0.03,
        realized_volatility=0.01,
    )

    no_position = predict_probabilities("INFY", bearish, cfg, open_position=False)
    with_position = predict_probabilities("INFY", bearish, cfg, open_position=True)

    assert no_position.action != PredictorAction.SELL
    assert with_position.action in {PredictorAction.SELL, PredictorAction.HOLD}
