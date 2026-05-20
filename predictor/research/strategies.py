"""Strategy family implementations used by the research validation harness."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence, Tuple

import numpy as np
import pandas as pd

from predictor.research.errors import ResearchInputError


class TradingStrategy(Protocol):
    """Interface for pluggable strategy signal generators."""

    name: str

    def generate_positions(self, frame: pd.DataFrame) -> pd.Series:
        """Return position series in {-1, 0, 1} aligned with frame index."""


def _validate_windows(*values: int) -> None:
    """Ensure all rolling windows are strictly positive."""

    for value in values:
        if not isinstance(value, int) or value <= 0:
            raise ResearchInputError("window parameters must be positive integers")


def _to_position_series(signal: pd.Series) -> pd.Series:
    """Normalize continuous signal to clipped discrete {-1,0,1} positions."""

    clipped = signal.clip(lower=-1.0, upper=1.0).fillna(0.0)
    rounded = np.sign(clipped).astype(float)
    return pd.Series(rounded, index=signal.index, dtype=float)


def _hold_until_flip(entries: pd.Series) -> pd.Series:
    """Convert sparse entry signals into held positions until opposite flip."""

    state = pd.Series(np.nan, index=entries.index, dtype=float)
    state.loc[entries > 0] = 1.0
    state.loc[entries < 0] = -1.0
    state = state.ffill().fillna(0.0)
    return state.astype(float)


@dataclass(frozen=True)
class DonchianBreakoutStrategy:
    """Trend-following breakout strategy based on Donchian channels."""

    lookback: int = 55
    name: str = "donchian_breakout"

    def generate_positions(self, frame: pd.DataFrame) -> pd.Series:
        _validate_windows(self.lookback)
        close = frame["Close"]
        upper = frame["High"].rolling(self.lookback).max().shift(1)
        lower = frame["Low"].rolling(self.lookback).min().shift(1)
        entries = pd.Series(0.0, index=close.index, dtype=float)
        entries.loc[close > upper] = 1.0
        entries.loc[close < lower] = -1.0
        return _hold_until_flip(entries)


@dataclass(frozen=True)
class MovingAverageCrossoverStrategy:
    """Classical moving-average crossover trend strategy."""

    short_window: int = 20
    long_window: int = 60
    name: str = "ma_crossover"

    def generate_positions(self, frame: pd.DataFrame) -> pd.Series:
        _validate_windows(self.short_window, self.long_window)
        if self.short_window >= self.long_window:
            raise ResearchInputError("short_window must be < long_window")
        close = frame["Close"]
        short_ma = close.rolling(self.short_window).mean()
        long_ma = close.rolling(self.long_window).mean()
        signal = pd.Series(0.0, index=close.index, dtype=float)
        signal.loc[short_ma > long_ma] = 1.0
        signal.loc[short_ma < long_ma] = -1.0
        return _hold_until_flip(signal)


@dataclass(frozen=True)
class MeanReversionStrategy:
    """Z-score mean-reversion strategy around rolling average."""

    window: int = 20
    zscore_threshold: float = 1.0
    exit_zscore: float = 0.25
    name: str = "mean_reversion"

    def generate_positions(self, frame: pd.DataFrame) -> pd.Series:
        _validate_windows(self.window)
        if self.zscore_threshold <= 0:
            raise ResearchInputError("zscore_threshold must be > 0")
        close = frame["Close"]
        mean = close.rolling(self.window).mean()
        std = close.rolling(self.window).std(ddof=0).replace(0.0, np.nan)
        zscore = (close - mean) / std

        state = pd.Series(np.nan, index=close.index, dtype=float)
        state.loc[zscore < -self.zscore_threshold] = 1.0
        state.loc[zscore > self.zscore_threshold] = -1.0
        held = state.ffill().fillna(0.0)
        held.loc[zscore.abs() <= self.exit_zscore] = 0.0
        return held.ffill().fillna(0.0).astype(float)


@dataclass(frozen=True)
class MomentumStrategy:
    """Directional time-series momentum rule."""

    lookback: int = 20
    name: str = "momentum"

    def generate_positions(self, frame: pd.DataFrame) -> pd.Series:
        _validate_windows(self.lookback)
        close = frame["Close"]
        momentum = close.pct_change(self.lookback)
        signal = pd.Series(0.0, index=close.index, dtype=float)
        signal.loc[momentum > 0] = 1.0
        signal.loc[momentum < 0] = -1.0
        return _hold_until_flip(signal)


def _average_true_range(frame: pd.DataFrame, window: int) -> pd.Series:
    """Compute ATR from OHLC inputs."""

    prev_close = frame["Close"].shift(1)
    high_low = frame["High"] - frame["Low"]
    high_close = (frame["High"] - prev_close).abs()
    low_close = (frame["Low"] - prev_close).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(window).mean()


@dataclass(frozen=True)
class VolatilityBreakoutStrategy:
    """Breakout rule requiring move larger than ATR-scaled threshold."""

    lookback: int = 20
    atr_window: int = 14
    atr_multiplier: float = 1.5
    name: str = "volatility_breakout"

    def generate_positions(self, frame: pd.DataFrame) -> pd.Series:
        _validate_windows(self.lookback, self.atr_window)
        if self.atr_multiplier <= 0:
            raise ResearchInputError("atr_multiplier must be > 0")

        close = frame["Close"]
        atr = _average_true_range(frame, self.atr_window)
        rolling_high = frame["High"].rolling(self.lookback).max().shift(1)
        rolling_low = frame["Low"].rolling(self.lookback).min().shift(1)
        upper_trigger = rolling_high + self.atr_multiplier * atr
        lower_trigger = rolling_low - self.atr_multiplier * atr

        entries = pd.Series(0.0, index=close.index, dtype=float)
        entries.loc[close > upper_trigger] = 1.0
        entries.loc[close < lower_trigger] = -1.0
        return _hold_until_flip(entries)


@dataclass(frozen=True)
class RegimeSwitchingStrategy:
    """Simple volatility-regime strategy with trend and mean-reversion modes."""

    volatility_window: int = 20
    trend_window: int = 50
    zscore_threshold: float = 1.0
    name: str = "regime_switching"

    def generate_positions(self, frame: pd.DataFrame) -> pd.Series:
        _validate_windows(self.volatility_window, self.trend_window)

        close = frame["Close"]
        returns = close.pct_change()
        realized_vol = returns.rolling(self.volatility_window).std(ddof=0)
        vol_median = realized_vol.rolling(self.volatility_window * 3).median()
        low_vol_regime = realized_vol <= vol_median

        trend = close / close.rolling(self.trend_window).mean() - 1.0
        trend_signal = pd.Series(0.0, index=close.index, dtype=float)
        trend_signal.loc[trend > 0] = 1.0
        trend_signal.loc[trend < 0] = -1.0

        mean = close.rolling(self.volatility_window).mean()
        std = close.rolling(self.volatility_window).std(ddof=0).replace(0.0, np.nan)
        zscore = (close - mean) / std
        reversion_signal = pd.Series(0.0, index=close.index, dtype=float)
        reversion_signal.loc[zscore < -self.zscore_threshold] = 1.0
        reversion_signal.loc[zscore > self.zscore_threshold] = -1.0

        combined = pd.Series(0.0, index=close.index, dtype=float)
        combined.loc[low_vol_regime] = trend_signal.loc[low_vol_regime]
        combined.loc[~low_vol_regime] = reversion_signal.loc[~low_vol_regime]
        return _hold_until_flip(combined)


@dataclass(frozen=True)
class WeightedStrategyEnsemble:
    """Weighted vote ensemble for strategy combinations and hybrids."""

    name: str
    components: Tuple[TradingStrategy, ...]
    weights: Tuple[float, ...]
    threshold: float = 0.05

    def generate_positions(self, frame: pd.DataFrame) -> pd.Series:
        if not self.components:
            raise ResearchInputError("ensemble requires at least one component strategy")
        if len(self.components) != len(self.weights):
            raise ResearchInputError("components and weights lengths must match")
        total_abs_weight = float(sum(abs(weight) for weight in self.weights))
        if total_abs_weight <= 0:
            raise ResearchInputError("ensemble weights must not all be zero")

        weighted_signal = pd.Series(0.0, index=frame.index, dtype=float)
        for strategy, weight in zip(self.components, self.weights):
            component_positions = _to_position_series(strategy.generate_positions(frame))
            weighted_signal = weighted_signal.add(weight * component_positions, fill_value=0.0)
        normalized = weighted_signal / total_abs_weight
        discrete = pd.Series(0.0, index=frame.index, dtype=float)
        discrete.loc[normalized >= self.threshold] = 1.0
        discrete.loc[normalized <= -self.threshold] = -1.0
        return _hold_until_flip(discrete)


@dataclass(frozen=True)
class TripleMAcrossoverStrategy:
    """Three-way moving-average crossover requiring full stack alignment.

    Long only when fast > medium > slow MA. Short only when fast < medium < slow.
    Flat when the stack is partially aligned (mixed signal).
    Literature: Faber (2007), Hurst/Ooi/Pedersen (AQR TSMOM).
    """

    fast: int = 10
    medium: int = 30
    slow: int = 100
    name: str = "triple_ma_crossover"

    def generate_positions(self, frame: pd.DataFrame) -> pd.Series:
        _validate_windows(self.fast, self.medium, self.slow)
        if not (self.fast < self.medium < self.slow):
            raise ResearchInputError("fast < medium < slow required")
        close = frame["Close"]
        f_ma = close.rolling(self.fast).mean()
        m_ma = close.rolling(self.medium).mean()
        s_ma = close.rolling(self.slow).mean()
        signal = pd.Series(0.0, index=close.index, dtype=float)
        signal.loc[(f_ma > m_ma) & (m_ma > s_ma)] = 1.0
        signal.loc[(f_ma < m_ma) & (m_ma < s_ma)] = -1.0
        return _hold_until_flip(signal)


@dataclass(frozen=True)
class KeltnerBreakoutStrategy:
    """EMA + ATR channel breakout (Keltner Channel).

    Distinct from Donchian: uses EMA centre with ATR-scaled bands.
    Literature: Chester Keltner (1960); widely used in systematic CTA programmes.
    """

    ema_window: int = 20
    atr_window: int = 14
    atr_multiplier: float = 2.0
    name: str = "keltner_breakout"

    def generate_positions(self, frame: pd.DataFrame) -> pd.Series:
        _validate_windows(self.ema_window, self.atr_window)
        if self.atr_multiplier <= 0:
            raise ResearchInputError("atr_multiplier must be > 0")
        close = frame["Close"]
        ema = close.ewm(span=self.ema_window, adjust=False).mean()
        atr = _average_true_range(frame, self.atr_window)
        upper = ema + self.atr_multiplier * atr
        lower = ema - self.atr_multiplier * atr
        entries = pd.Series(0.0, index=close.index, dtype=float)
        entries.loc[close > upper] = 1.0
        entries.loc[close < lower] = -1.0
        return _hold_until_flip(entries)


def _compute_rsi(close: pd.Series, window: int) -> pd.Series:
    """Compute Wilder RSI from Close prices."""
    delta = close.diff()
    gain = delta.clip(lower=0.0).rolling(window).mean()
    loss = (-delta.clip(upper=0.0)).rolling(window).mean()
    rs = gain / loss.replace(0.0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


@dataclass(frozen=True)
class RSIMeanReversionStrategy:
    """RSI-triggered mean-reversion rule.

    Long on oversold (RSI < oversold_threshold), short on overbought.
    Exits when RSI returns to neutral zone.
    Literature: Wilder (1978); extensively backtested in Lo/MacKinlay (1990).
    """

    rsi_window: int = 14
    oversold: float = 30.0
    overbought: float = 70.0
    neutral_band: float = 10.0
    name: str = "rsi_mean_reversion"

    def generate_positions(self, frame: pd.DataFrame) -> pd.Series:
        _validate_windows(self.rsi_window)
        if not (0.0 < self.oversold < self.overbought < 100.0):
            raise ResearchInputError("oversold < overbought required, both in (0,100)")
        if self.neutral_band <= 0.0:
            raise ResearchInputError("neutral_band must be > 0")
        close = frame["Close"]
        rsi = _compute_rsi(close, self.rsi_window)
        neutral_low = self.oversold + self.neutral_band
        neutral_high = self.overbought - self.neutral_band

        state = pd.Series(np.nan, index=close.index, dtype=float)
        state.loc[rsi < self.oversold] = 1.0
        state.loc[rsi > self.overbought] = -1.0
        held = state.ffill().fillna(0.0)
        held.loc[(rsi >= neutral_low) & (rsi <= neutral_high)] = 0.0
        return held.ffill().fillna(0.0).astype(float)


def to_strategy_tuple(strategies: Sequence[TradingStrategy]) -> Tuple[TradingStrategy, ...]:
    """Validate and normalize a sequence of strategy instances."""

    if strategies is None:
        raise ResearchInputError("strategies must not be None")
    normalized = tuple(strategies)
    if not normalized:
        raise ResearchInputError("strategies must contain at least one element")
    names = [getattr(strategy, "name", None) for strategy in normalized]
    if any(not isinstance(name, str) or not name.strip() for name in names):
        raise ResearchInputError("every strategy must expose a non-empty `name` attribute")
    if len(set(names)) != len(names):
        raise ResearchInputError("strategy names must be unique for fair comparison")
    return normalized
