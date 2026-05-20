"""Feature engineering for short-horizon market-structure prediction."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from predictor.errors import DataUnavailableError
from predictor.types import FeatureVector


def _safe_float(value: object) -> float:
    if isinstance(value, pd.Series):
        value = value.iloc[-1]
    return float(value)


def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = -delta.clip(upper=0).rolling(window).mean()
    rsi = pd.Series(50.0, index=close.index, dtype=float)
    valid = (gain > 0) & (loss > 0)
    rsi.loc[valid] = 100.0 - (100.0 / (1.0 + (gain.loc[valid] / loss.loc[valid])))
    rsi.loc[(gain > 0) & (loss == 0)] = 100.0
    rsi.loc[(gain == 0) & (loss > 0)] = 0.0
    return rsi


def _atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    true_range = np.maximum(
        high - low,
        np.maximum((high - close.shift()).abs(), (low - close.shift()).abs()),
    )
    return true_range.rolling(window).mean()


def _vwap(df: pd.DataFrame) -> pd.Series:
    typical_price = (df["High"] + df["Low"] + df["Close"]) / 3.0
    traded_value = typical_price * df["Volume"]
    cumulative_volume = df["Volume"].cumsum().replace(0, 1)
    return traded_value.cumsum() / cumulative_volume


def build_feature_vector(df: pd.DataFrame) -> FeatureVector:
    """Build deterministic predictor features from validated OHLCV data.

    Args:
        df: Validated OHLCV dataframe sorted in ascending time order.

    Returns:
        A `FeatureVector` suitable for predictor inference.

    Raises:
        DataUnavailableError: If derived values are not computable.
    """

    working = df.copy()
    working["ema20"] = working["Close"].ewm(span=20).mean()
    working["ema50"] = working["Close"].ewm(span=50).mean()
    working["atr"] = _atr(working, 14)
    working["rsi"] = _rsi(working["Close"], 14)
    working["vwap"] = _vwap(working)

    last = working.iloc[-1]
    price = _safe_float(last["Close"])
    if price <= 0:
        raise DataUnavailableError("invalid last close price")

    ema20 = _safe_float(last["ema20"])
    ema50 = _safe_float(last["ema50"])
    atr = _safe_float(last["atr"])
    rsi = _safe_float(last["rsi"])
    vwap = _safe_float(last["vwap"])
    volume = _safe_float(last["Volume"])

    avg_volume = _safe_float(working["Volume"].tail(20).mean())
    volume_ratio = volume / avg_volume if avg_volume > 0 else 0.0
    atr_pct = atr / price * 100.0
    vwap_distance_pct = ((price - vwap) / vwap * 100.0) if vwap else 0.0

    if isinstance(working.index, pd.DatetimeIndex):
        current_date = working.index[-1].date()
        session = working[working.index.date == current_date]
        if session.empty:
            session = working
    else:
        session = working

    session_low = _safe_float(session["Low"].min())
    session_high = _safe_float(session["High"].max())
    pct_from_low = ((price - session_low) / session_low * 100.0) if session_low else 0.0
    pct_from_high = ((session_high - price) / session_high * 100.0) if session_high else 0.0

    trend_strength = (ema20 / ema50 - 1.0) if ema50 else 0.0
    short_return = _safe_float(working["Close"].pct_change(5).iloc[-1])
    long_return = _safe_float(working["Close"].pct_change(20).iloc[-1])
    realized_volatility = _safe_float(working["Close"].pct_change().tail(20).std(ddof=0))
    if math.isnan(short_return):
        short_return = 0.0
    if math.isnan(long_return):
        long_return = 0.0
    if math.isnan(realized_volatility):
        realized_volatility = 0.0

    return FeatureVector(
        price=price,
        ema20=ema20,
        ema50=ema50,
        rsi=rsi,
        atr_pct=atr_pct,
        avg_volume=avg_volume,
        volume=volume,
        volume_ratio=volume_ratio,
        vwap=vwap,
        vwap_distance_pct=vwap_distance_pct,
        session_low=session_low,
        session_high=session_high,
        pct_from_low=pct_from_low,
        pct_from_high=pct_from_high,
        trend_strength=trend_strength,
        short_return=short_return,
        long_return=long_return,
        realized_volatility=realized_volatility,
    )
