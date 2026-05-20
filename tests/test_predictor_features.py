"""Tests for predictor feature engineering logic."""

from __future__ import annotations

import pytest

from predictor.errors import DataUnavailableError
from predictor.features import build_feature_vector


def test_build_feature_vector_returns_expected_schema(make_ohlcv_frame) -> None:
    frame = make_ohlcv_frame()
    features = build_feature_vector(frame)

    assert features.price > 0
    assert features.ema20 > 0
    assert features.ema50 > 0
    assert 0 <= features.rsi <= 100
    assert features.avg_volume > 0
    assert features.volume > 0
    assert features.session_high >= features.session_low


def test_build_feature_vector_reflects_trend_direction(make_ohlcv_frame) -> None:
    bullish = build_feature_vector(make_ohlcv_frame(trend=0.12))
    bearish = build_feature_vector(make_ohlcv_frame(trend=-0.12))

    assert bullish.trend_strength > bearish.trend_strength
    assert bullish.long_return > bearish.long_return


def test_build_feature_vector_rejects_invalid_last_price(make_ohlcv_frame) -> None:
    frame = make_ohlcv_frame()
    frame.loc[frame.index[-1], "Close"] = 0.0
    with pytest.raises(DataUnavailableError):
        build_feature_vector(frame)
