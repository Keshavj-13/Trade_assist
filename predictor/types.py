"""Typed value objects for predictor data, features, and inference outputs."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, Tuple


class MarketRegime(str, Enum):
    """Coarse short-horizon market regime label."""

    BULL = "bull"
    BEAR = "bear"
    RANGE = "range"


class PredictorAction(str, Enum):
    """Action labels emitted by predictor inference."""

    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


@dataclass(frozen=True)
class FeatureVector:
    """Feature schema consumed by the predictor model.

    Attributes:
        price: Last traded price.
        ema20: Short moving average.
        ema50: Medium moving average.
        rsi: Relative Strength Index in [0, 100].
        atr_pct: Average true range as percentage of price.
        avg_volume: Mean recent volume.
        volume: Latest candle volume.
        volume_ratio: Latest volume divided by recent mean volume.
        vwap: Volume weighted average price.
        vwap_distance_pct: Signed distance from VWAP in percent.
        session_low: Session low price.
        session_high: Session high price.
        pct_from_low: Price distance from session low, in percent.
        pct_from_high: Price distance from session high, in percent.
        trend_strength: Relative EMA spread (ema20 / ema50 - 1).
        short_return: Recent short-horizon return.
        long_return: Recent medium-horizon return.
        realized_volatility: Rolling realized volatility estimate.
    """

    price: float
    ema20: float
    ema50: float
    rsi: float
    atr_pct: float
    avg_volume: float
    volume: float
    volume_ratio: float
    vwap: float
    vwap_distance_pct: float
    session_low: float
    session_high: float
    pct_from_low: float
    pct_from_high: float
    trend_strength: float
    short_return: float
    long_return: float
    realized_volatility: float


@dataclass(frozen=True)
class Prediction:
    """Single-symbol predictor output with explicit uncertainty."""

    symbol: str
    price: float
    action: PredictorAction
    regime: MarketRegime
    buy_probability: float
    sell_probability: float
    hold_probability: float
    confidence: float
    uncertainty: float
    score: float
    reasons: Tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class PipelineRun:
    """Aggregate output for a predictor scan run."""

    timestamp: datetime
    predictions: Tuple[Prediction, ...]
    buy_candidates: Tuple[Prediction, ...]
    sell_candidates: Tuple[Prediction, ...]
    hold_candidates: Tuple[Prediction, ...]
    skipped: Dict[str, str]
