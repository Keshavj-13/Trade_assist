"""Broader market hypothesis families with documented economic rationale.

Scientific discipline: every strategy family must state its theoretical basis,
expected market conditions, and known failure modes BEFORE being tested.
This module enforces that contract via the MarketHypothesis base dataclass.

Hypothesis families included
----------------------------
1. GapFadeStrategy              — Behavioral: retail overreaction on open gaps
2. VolatilityCompressionBreakout — Squeeze→expansion: tight range precedes move
3. VolatilityExpansionStrategy  — Trade direction of volatility expansion bursts
4. RegimeFilteredTrendStrategy  — Trend-following gated by regime classification

Deferred (requires multi-frame protocol extension)
---------------------------------------------------
- RelativeStrengthRankStrategy — Cross-sectional momentum (future iteration)

Public API
----------
GapFadeStrategy
VolatilityCompressionBreakoutStrategy
VolatilityExpansionStrategy
RegimeFilteredTrendStrategy
build_hypothesis_universe
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple

import numpy as np
import pandas as pd

from predictor.research.errors import ResearchInputError
from predictor.research.strategies import TradingStrategy


# ---------------------------------------------------------------------------
# Base dataclass — enforces documentation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MarketHypothesis:
    """Base for all hypothesis families. Documents the theory before backtest.

    Subclasses must provide non-empty strings for theoretical_basis,
    expected_market_condition, and known_failure_modes.
    """

    name: str
    theoretical_basis: str
    expected_market_condition: str
    known_failure_modes: str

    def __post_init__(self) -> None:
        for attr in ("theoretical_basis", "expected_market_condition", "known_failure_modes"):
            val = getattr(self, attr)
            if not isinstance(val, str) or not val.strip():
                raise ResearchInputError(
                    f"{self.__class__.__name__}.{attr} must be a non-empty string. "
                    "State the hypothesis before testing it."
                )


# ---------------------------------------------------------------------------
# 1. Gap Fade
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GapFadeStrategy(MarketHypothesis):
    """Fade opening gaps — trade the close of the gap rather than its direction.

    Theoretical basis:
        Behavioral finance: retail participants overreact to overnight news,
        driving exaggerated opening moves that partially reverse intraday
        (Bhattacharya et al., 2009; Caginalp & DeSantis, 2011).

    Expected market conditions:
        Liquid markets with significant overnight news flow. High gap
        frequency. Daily OHLCV bars capture O/C spread as gap proxy.

    Known failure modes:
        On daily bars, gap fade is a coarse approximation — Open represents
        the first trade of the day. Works best with intraday data. May miss
        the intraday reversal window. Momentum regimes (gapping in trend
        direction) cause persistent losses.

    Signal construction:
        Gap = (Open_t - Close_{t-1}) / Close_{t-1}
        When gap > threshold: go short (expect fade)
        When gap < -threshold: go long (expect fade)
        Hold for hold_bars bars, then exit.
    """

    gap_threshold: float = 0.005
    hold_bars: int = 3
    name: str = "gap_fade"
    theoretical_basis: str = (
        "Behavioral overreaction on open gaps (Bhattacharya et al. 2009). "
        "Retail over-reads overnight news; gap partially reverts."
    )
    expected_market_condition: str = (
        "Liquid markets with frequent material gap-ups/downs. "
        "Low momentum regime (gaps don't run)."
    )
    known_failure_modes: str = (
        "Coarse on daily bars — gap may have already closed by open. "
        "Losses in momentum regimes where gaps continue in direction."
    )

    def generate_positions(self, frame: pd.DataFrame) -> pd.Series:
        """Fade gaps exceeding gap_threshold, hold for hold_bars bars."""
        if self.gap_threshold <= 0:
            raise ResearchInputError("gap_threshold must be > 0")
        if self.hold_bars <= 0:
            raise ResearchInputError("hold_bars must be > 0")

        open_px = frame["Open"]
        prev_close = frame["Close"].shift(1)
        gap = (open_px - prev_close) / prev_close.replace(0, np.nan)
        gap = gap.fillna(0.0)

        n = len(frame)
        positions = np.zeros(n, dtype=float)

        for i in range(1, n):
            g = gap.iloc[i]
            if g > self.gap_threshold:
                # Gap up → fade short
                end = min(i + self.hold_bars, n)
                positions[i:end] = -1.0
            elif g < -self.gap_threshold:
                # Gap down → fade long
                end = min(i + self.hold_bars, n)
                positions[i:end] = 1.0

        return pd.Series(positions, index=frame.index, dtype=float)


# ---------------------------------------------------------------------------
# 2. Volatility Compression Breakout
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VolatilityCompressionBreakoutStrategy(MarketHypothesis):
    """Enter breakouts only after a volatility squeeze (compression phase).

    Theoretical basis:
        Volatility mean-reversion: periods of unusually low volatility
        (Bollinger Band squeeze) precede explosive directional moves
        (Bollinger, 1983; Connors & Alvarez, 2009). The compression phase
        represents market indecision that resolves directionally.

    Expected market conditions:
        Markets alternating between consolidation and expansion phases.
        Works best in assets with well-defined volatility cycles.

    Known failure modes:
        False breakouts in persistently mean-reverting regimes. The squeeze
        resolves sideways rather than directionally in choppy markets.
        High false positive rate requires additional confirmation.

    Signal construction:
        Squeeze when ATR < ATR_low_percentile (compression detected)
        On next bar after squeeze: enter in direction of price vs midpoint
        Exit after exit_bars or when ATR > ATR_high_percentile (expansion done)
    """

    atr_window: int = 14
    atr_lookback: int = 50
    compression_pct: float = 25.0   # ATR below this percentile = squeeze
    expansion_pct: float = 60.0     # ATR above this percentile = expansion done
    exit_bars: int = 15
    name: str = "volatility_compression_breakout"
    theoretical_basis: str = (
        "Volatility mean-reversion (Bollinger 1983): low-vol compression "
        "precedes explosive directional expansion."
    )
    expected_market_condition: str = (
        "Markets with clear volatility cycles — consolidation then expansion."
    )
    known_failure_modes: str = (
        "False breakouts in persistently ranging markets. "
        "Squeeze resolves sideways — high false positive rate."
    )

    def _compute_atr(self, frame: pd.DataFrame) -> pd.Series:
        high = frame["High"]
        low = frame["Low"]
        prev_close = frame["Close"].shift(1)
        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ], axis=1).max(axis=1)
        return tr.rolling(self.atr_window).mean()

    def generate_positions(self, frame: pd.DataFrame) -> pd.Series:
        """Enter in price direction on bars following a volatility squeeze."""
        if not (0 < self.compression_pct < self.expansion_pct < 100):
            raise ResearchInputError(
                "Must have 0 < compression_pct < expansion_pct < 100"
            )

        atr = self._compute_atr(frame)
        atr_low = atr.rolling(self.atr_lookback).quantile(self.compression_pct / 100.0)
        atr_high = atr.rolling(self.atr_lookback).quantile(self.expansion_pct / 100.0)

        close = frame["Close"]
        midpoint = (frame["High"].rolling(self.atr_lookback).max() +
                    frame["Low"].rolling(self.atr_lookback).min()) / 2.0

        in_squeeze = atr < atr_low
        n = len(frame)
        positions = np.zeros(n, dtype=float)

        i = 1
        while i < n:
            # Detected a squeeze on prior bar — enter on current bar
            if in_squeeze.iloc[i - 1] and not in_squeeze.iloc[i]:
                direction = 1.0 if close.iloc[i] >= midpoint.iloc[i] else -1.0
                j = i
                while j < n and j < i + self.exit_bars:
                    if atr.iloc[j] > atr_high.iloc[j]:
                        break
                    positions[j] = direction
                    j += 1
                i = max(j, i + 1)
            else:
                i += 1

        return pd.Series(positions, index=frame.index, dtype=float)


# ---------------------------------------------------------------------------
# 3. Volatility Expansion
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VolatilityExpansionStrategy(MarketHypothesis):
    """Trade in the direction of volatility expansion bursts.

    Theoretical basis:
        Volatility clustering (Mandelbrot 1963; Engle 1982 ARCH):
        large moves beget large moves. When ATR accelerates sharply above
        its recent level, the market is in an impulsive expansion phase —
        trade in the direction of the price move that caused it.

    Expected market conditions:
        Any market with trending volatility. Particularly effective during
        early-stage trend initiations after quiet consolidation periods.

    Known failure modes:
        Late entries — by the time ATR accelerates, the initial move may
        be exhausted. Choppy indecisive expansions cause whipsaws.
    """

    atr_window: int = 14
    expansion_multiplier: float = 1.5   # ATR > multiplier * slow_atr
    slow_atr_window: int = 40
    hold_bars: int = 10
    name: str = "volatility_expansion"
    theoretical_basis: str = (
        "Volatility clustering (Mandelbrot 1963; Engle 1982 ARCH): "
        "large moves accelerate — trade direction of expansion burst."
    )
    expected_market_condition: str = (
        "Trending volatility. Early-stage impulsive moves after consolidation."
    )
    known_failure_modes: str = (
        "Late entry after initial move is exhausted. "
        "Whipsaws in choppy indecisive expansions."
    )

    def generate_positions(self, frame: pd.DataFrame) -> pd.Series:
        """Enter in price direction when ATR crosses above expansion threshold."""
        if self.expansion_multiplier <= 1.0:
            raise ResearchInputError("expansion_multiplier must be > 1.0")

        high = frame["High"]
        low = frame["Low"]
        close = frame["Close"]
        prev_close = close.shift(1)

        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ], axis=1).max(axis=1)
        fast_atr = tr.rolling(self.atr_window).mean()
        slow_atr = fast_atr.rolling(self.slow_atr_window).mean()

        expanding = fast_atr > (slow_atr * self.expansion_multiplier)
        price_direction = np.sign(close.pct_change().fillna(0.0))

        n = len(frame)
        positions = np.zeros(n, dtype=float)
        i = 1
        while i < n:
            if expanding.iloc[i] and not expanding.iloc[i - 1]:
                direction = price_direction.iloc[i]
                if direction != 0.0:
                    end = min(i + self.hold_bars, n)
                    positions[i:end] = direction
                    i = end
                    continue
            i += 1

        return pd.Series(positions, index=frame.index, dtype=float)


# ---------------------------------------------------------------------------
# 4. Regime-Filtered Trend
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RegimeFilteredTrendStrategy(MarketHypothesis):
    """Donchian trend-following gated by market regime classification.

    Only active in BULL and LOW_VOL regimes. Sits flat in CRASH, HIGH_VOL,
    BEAR, RECOVERY, and SIDEWAYS regimes.

    Theoretical basis:
        Regime-conditional trend following: the profitability of trend
        strategies is known to be regime-dependent (Faber 2013; AQR 2017).
        Trend works in trending low-vol regimes; fails in high-vol and
        mean-reverting regimes. Regime filtering reduces drawdown at the cost
        of reduced exposure and potentially lower returns in pure bull regimes.

    Expected market conditions:
        Sustained low-volatility bull trends. The regime filter removes the
        strategy from high-vol and crash environments.

    Known failure modes:
        Regime detection lag at turning points — the filter is backward-looking
        and will not prevent initial drawdown at regime transitions.
        Reduces exposure and therefore reduces absolute returns in bull markets
        vs unfiltered buy-and-hold.
    """

    entry_lookback: int = 20
    exit_lookback: int = 10
    active_regimes: Tuple[str, ...] = ("BULL", "LOW_VOL")
    name: str = "regime_filtered_trend"
    theoretical_basis: str = (
        "Regime-conditional trend following (Faber 2013; AQR 2017). "
        "Trend works in bull/low-vol; fails in crash/high-vol. "
        "Regime gating reduces drawdown at cost of exposure."
    )
    expected_market_condition: str = (
        "Sustained low-volatility bull trends with clear regime structure."
    )
    known_failure_modes: str = (
        "Regime detection lag at turning points. "
        "Reduces exposure vs unfiltered buy-and-hold in pure bull markets."
    )

    def generate_positions(self, frame: pd.DataFrame) -> pd.Series:
        """Donchian breakout long-only, active only in specified regimes."""
        from predictor.research.regime import classify_regimes

        if self.entry_lookback <= self.exit_lookback:
            raise ResearchInputError("entry_lookback must be > exit_lookback")

        close = frame["Close"]
        high_roll = frame["High"].rolling(self.entry_lookback).max()
        low_roll = frame["Low"].rolling(self.exit_lookback).min()

        regimes = classify_regimes(frame)
        in_active_regime = regimes.isin(self.active_regimes)

        n = len(frame)
        positions = np.zeros(n, dtype=float)
        in_trade = False

        for i in range(1, n):
            if not in_active_regime.iloc[i]:
                in_trade = False
                positions[i] = 0.0
                continue

            if in_trade:
                if close.iloc[i] < low_roll.iloc[i]:
                    in_trade = False
                    positions[i] = 0.0
                else:
                    positions[i] = 1.0
            else:
                prev_high = high_roll.iloc[i - 1]
                if not np.isnan(prev_high) and close.iloc[i] > prev_high:
                    in_trade = True
                    positions[i] = 1.0

        return pd.Series(positions, index=frame.index, dtype=float)


# ---------------------------------------------------------------------------
# Universe builder
# ---------------------------------------------------------------------------


def build_hypothesis_universe() -> Tuple[TradingStrategy, ...]:
    """Return all hypothesis-family strategies for research.

    Each strategy has documented theoretical basis, expected conditions,
    and known failure modes. None have been parameter-tuned against data.
    """
    return (
        GapFadeStrategy(gap_threshold=0.005, hold_bars=3),
        GapFadeStrategy(gap_threshold=0.010, hold_bars=5, name="gap_fade_wide"),
        VolatilityCompressionBreakoutStrategy(
            atr_window=14, atr_lookback=50, exit_bars=15
        ),
        VolatilityExpansionStrategy(
            atr_window=14, expansion_multiplier=1.5, hold_bars=10
        ),
        RegimeFilteredTrendStrategy(
            entry_lookback=20, exit_lookback=10, active_regimes=("BULL", "LOW_VOL")
        ),
        RegimeFilteredTrendStrategy(
            entry_lookback=40, exit_lookback=20,
            active_regimes=("BULL",),
            name="regime_filtered_trend_bull_only",
            theoretical_basis=(
                "Regime-conditional trend following (Faber 2013): "
                "bull-regime only variant — more selective than BULL+LOW_VOL."
            ),
            expected_market_condition="Pure bull-regime only.",
            known_failure_modes="Even less exposure than BULL+LOW_VOL variant.",
        ),
    )
