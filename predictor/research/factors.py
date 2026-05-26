"""Baseline and predictive factor definitions for cross-sectional ranking.

Every factor documents its theoretical basis, expected market conditions,
and known failure modes BEFORE being tested to preserve scientific discipline.

Public API
----------
RankingFactor
RandomRankingFactor
MarketBenchmarkFactor
BuyAndHoldBaselineFactor
EqualWeightSelectionFactor
SimpleMomentumRankFactor
VolatilityRankFactor
Momentum20Factor
PreviousDayWinnerContinuation
PreviousDayReturnFactor
PreviousDayLoserRebound
ShortTermMomentumFactor
SectorMomentumFactor
SectorRelativeStrengthFactor
OvernightGapFactor
VolatilityCompressionFactor
RelativeVolumeFactor
ATRExpansionFactor
RollingBetaAdjustedStrengthFactor
build_factor_universe
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from predictor.research.errors import ResearchInputError

logger = logging.getLogger(__name__)

# Static sector map for major NSE stocks to support the sector momentum factor
SECTOR_MAP: Dict[str, str] = {
    # Technology / IT
    "TCS": "IT", "INFY": "IT", "WIPRO": "IT", "HCLTECH": "IT", "TECHM": "IT",
    "LTIM": "IT", "COFORGE": "IT", "KPITTECH": "IT", "ZENSARTECH": "IT",
    # Financial Services / Banking
    "HDFCBANK": "FIN", "ICICIBANK": "FIN", "SBIN": "FIN", "KOTAKBANK": "FIN",
    "AXISBANK": "FIN", "INDUSINDBK": "FIN", "PNB": "FIN", "CANBK": "FIN",
    "BSE": "FIN", "MCX": "FIN", "MUTHOOTFIN": "FIN", "RECLTD": "FIN", "PFC": "FIN",
    # Energy / Oil & Gas / Utilities
    "RELIANCE": "ENERGY", "ONGC": "ENERGY", "NTPC": "ENERGY", "POWERGRID": "ENERGY",
    "BPCL": "ENERGY", "IOC": "ENERGY", "HINDPETRO": "ENERGY", "TATAPOWER": "ENERGY",
    # Consumer Goods / Automobiles
    "ITC": "CONSUMER", "HINDUNILVR": "CONSUMER", "NESTLEIND": "CONSUMER",
    "BRITANNIA": "CONSUMER", "MARUTI": "CONSUMER", "TATACONSUM": "CONSUMER",
    "M&M": "CONSUMER", "HEROMOTOCO": "CONSUMER", "EICHERMOT": "CONSUMER", "TVSMOTOR": "CONSUMER",
    # Materials / Industrial / Infrastructure
    "LT": "IND", "TATASTEEL": "IND", "JSWSTEEL": "IND", "HINDALCO": "IND",
    "ULTRACEMCO": "IND", "BEL": "IND", "BHEL": "IND", "DLF": "IND",
}


class _DocumentedFactor:
    """Mixin enforcing documentation fields on concrete RankingFactor subclasses."""

    theoretical_basis: str = ""
    expected_market_condition: str = ""
    known_failure_modes: str = ""

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        if cls.__name__ in {"_DocumentedFactor", "RankingFactor"}:
            return
        for attr in ("theoretical_basis", "expected_market_condition", "known_failure_modes"):
            val = getattr(cls, attr, "")
            if not isinstance(val, str) or not val.strip():
                raise TypeError(
                    f"{cls.__name__}.{attr} must be a non-empty string. "
                    "State the predictive hypothesis before testing."
                )


@dataclass(frozen=True)
class RankingFactor(_DocumentedFactor, ABC):
    """Abstract base for all predictive ranking factors."""

    abstract = True
    name: str

    @abstractmethod
    def compute_scores(self, symbol_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Compute cross-sectional factor scores.

        Parameters
        ----------
        symbol_data : Dict[str, pd.DataFrame]
            Dict mapping symbol name to its historical OHLCV DataFrame.

        Returns
        -------
        pd.DataFrame
            DataFrame with DatetimeIndex and columns corresponding to symbols,
            representing the calculated scores.
        """
        raise NotImplementedError("Subclasses must implement compute_scores")


# ---------------------------------------------------------------------------
# 1. Random Ranking Factor (Null Baseline)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RandomRankingFactor(RankingFactor):
    """Null baseline factor that assigns random scores to each symbol.

    Used to calibrate the permutation tests and verify that non-predictive
    factors produce p-values consistent with the null distribution.
    """

    seed: int = 42

    theoretical_basis: str = (
        "Pure random walk null hypothesis. Used to verify the calibration "
        "and empirical p-value distribution of the ranking backtest."
    )
    expected_market_condition: str = "None — this is a random null control."
    known_failure_modes: str = "Expected to fail out-of-sample backtests consistently."

    def compute_scores(self, symbol_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        if not symbol_data:
            return pd.DataFrame()

        # Find the union of all indices
        all_dates = sorted(list(set().union(*(df.index for df in symbol_data.values()))))
        rng = np.random.default_rng(self.seed)
        
        scores = {}
        for symbol, df in symbol_data.items():
            # Generate random numbers aligned to symbol's index
            scores[symbol] = pd.Series(rng.standard_normal(len(df)), index=df.index)

        panel = pd.DataFrame(scores, index=all_dates)
        return panel.fillna(0.0)


# ---------------------------------------------------------------------------
# 2. Equal Weight Universe Factor (Market Baseline)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MarketBenchmarkFactor(RankingFactor):
    """Market baseline factor that assigns identical constant scores to all symbols.

    Equivalent to holding the equal-weighted market universe.
    """

    theoretical_basis: str = (
        "Market beta / equal-weighted benchmark. Represents passive intraday "
        "exposure to the entire asset universe."
    )
    expected_market_condition: str = "Overall broad market uptrend with positive drift."
    known_failure_modes: str = "Flat or declining markets; misses cross-sectional outperformance."

    def compute_scores(self, symbol_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        if not symbol_data:
            return pd.DataFrame()

        all_dates = sorted(list(set().union(*(df.index for df in symbol_data.values()))))
        scores = {}
        for symbol, df in symbol_data.items():
            scores[symbol] = pd.Series(1.0, index=df.index)

        return pd.DataFrame(scores, index=all_dates).fillna(0.0)


# ---------------------------------------------------------------------------
# 3. Baseline Ranking Controls
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BuyAndHoldBaselineFactor(RankingFactor):
    """Constant score baseline mirroring an equal-weight universe selection."""

    theoretical_basis: str = (
        "Passive market exposure baseline. Any predictive ranking should beat this null of no selection edge."
    )
    expected_market_condition: str = "Broad market drift dominates cross-sectional effects."
    known_failure_modes: str = "No differentiation across symbols; cannot discover relative winners."

    def compute_scores(self, symbol_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        return MarketBenchmarkFactor(name=self.name).compute_scores(symbol_data)


@dataclass(frozen=True)
class EqualWeightSelectionFactor(RankingFactor):
    """Tie-score control: identical scores so selection is index-order neutral."""

    theoretical_basis: str = "Selection neutrality control. Tests if ranking logic adds value over tied scores."
    expected_market_condition: str = "None; this is a calibration baseline."
    known_failure_modes: str = "No predictive content; expected to fail strict validation."

    def compute_scores(self, symbol_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        if not symbol_data:
            return pd.DataFrame()
        all_dates = sorted(list(set().union(*(df.index for df in symbol_data.values()))))
        scores = {symbol: pd.Series(0.0, index=df.index) for symbol, df in symbol_data.items()}
        return pd.DataFrame(scores, index=all_dates).fillna(0.0)


@dataclass(frozen=True)
class SimpleMomentumRankFactor(RankingFactor):
    """Simple baseline momentum rank using close-to-close return over a short lookback."""

    lookback: int = 5

    theoretical_basis: str = "Recent winners may continue due to short-lived flow persistence."
    expected_market_condition: str = "Trending tape with persistent order-flow imbalance."
    known_failure_modes: str = "Mean-reversion and sharp reversal sessions."

    def compute_scores(self, symbol_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        if self.lookback <= 0:
            raise ResearchInputError("lookback must be > 0")
        if not symbol_data:
            return pd.DataFrame()
        all_dates = sorted(list(set().union(*(df.index for df in symbol_data.values()))))
        scores = {
            symbol: df["Close"].pct_change(self.lookback)
            for symbol, df in symbol_data.items()
        }
        return pd.DataFrame(scores, index=all_dates).fillna(0.0)


@dataclass(frozen=True)
class VolatilityRankFactor(RankingFactor):
    """Volatility baseline rank preferring lower realized volatility."""

    window: int = 20

    theoretical_basis: str = "Low-volatility names can outperform on a risk-adjusted basis intraday."
    expected_market_condition: str = "Orderly risk-off or low-dispersion sessions."
    known_failure_modes: str = "Momentum ignition regimes where high-vol leaders dominate."

    def compute_scores(self, symbol_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        if self.window <= 1:
            raise ResearchInputError("window must be > 1")
        if not symbol_data:
            return pd.DataFrame()
        all_dates = sorted(list(set().union(*(df.index for df in symbol_data.values()))))
        scores = {}
        for symbol, df in symbol_data.items():
            vol = df["Close"].pct_change().rolling(self.window).std(ddof=0)
            scores[symbol] = -vol
        return pd.DataFrame(scores, index=all_dates).fillna(0.0)


# ---------------------------------------------------------------------------
# 4. 20-Day Relative Momentum (Primary first-factor hypothesis)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Momentum20Factor(RankingFactor):
    """20-day relative momentum for cross-sectional next-day ranking.

    momentum_20_t = (Close_t / Close_{t-20}) - 1
    """

    lookback: int = 20

    theoretical_basis: str = (
        "Medium-short horizon relative strength. Recent outperformers may persist due to institutional flow inertia."
    )
    expected_market_condition: str = "Directional tapes with stable leadership across sectors."
    known_failure_modes: str = "Sharp mean-reversion and macro regime flips causing leadership rotation."

    def compute_scores(self, symbol_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        if self.lookback <= 0:
            raise ResearchInputError("lookback must be > 0")
        if not symbol_data:
            return pd.DataFrame()

        all_dates = sorted(list(set().union(*(df.index for df in symbol_data.values()))))
        scores = {}
        for symbol, df in symbol_data.items():
            close = df["Close"].astype(float)
            scores[symbol] = (close / close.shift(self.lookback)) - 1.0
        return pd.DataFrame(scores, index=all_dates).fillna(0.0)


# ---------------------------------------------------------------------------
# 5. Naive Momentum (Previous Day Winner Continuation)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PreviousDayWinnerContinuation(RankingFactor):
    """Naive momentum benchmark that buys yesterday's top performing stocks.

    Theoretical basis:
       Intraday trend continuation. Positive news flow or institutional flow
       spills over to the next trading day (Jegadeesh & Titman 1993).
    """

    theoretical_basis: str = (
        "Short-term momentum continuation: buying previous day's winners."
    )
    expected_market_condition: str = "Strongly trending bull markets with persistent flows."
    known_failure_modes: str = "Mean-reverting, choppy, or highly volatile regimes."

    def compute_scores(self, symbol_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        if not symbol_data:
            return pd.DataFrame()

        all_dates = sorted(list(set().union(*(df.index for df in symbol_data.values()))))
        scores = {}
        for symbol, df in symbol_data.items():
            # Return of day t Close vs t-1 Close
            ret = df["Close"].pct_change()
            scores[symbol] = ret

        # Yesterday's winner => factor score at day t is simply the return of day t (to rank for t+1)
        return pd.DataFrame(scores, index=all_dates).fillna(0.0)


# ---------------------------------------------------------------------------
# 5. Previous Day Return (explicit hypothesis alias)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PreviousDayReturnFactor(RankingFactor):
    """Explicit previous-day return factor for next-day ranking research."""

    theoretical_basis: str = "One-day continuation: prior session strength carries into next open-close window."
    expected_market_condition: str = "Persistent single-name flow and post-news follow-through."
    known_failure_modes: str = "Gap reversals and mean-reverting churn."

    def compute_scores(self, symbol_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        return PreviousDayWinnerContinuation(name=self.name).compute_scores(symbol_data)


# ---------------------------------------------------------------------------
# 6. Naive Mean Reversion (Previous Day Loser Rebound)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PreviousDayLoserRebound(RankingFactor):
    """Naive mean reversion benchmark that buys yesterday's worst performing stocks.

    Theoretical basis:
       Overreaction and short-term liquidity constraints. Oversold stocks rebound
       due to liquidity provision and correction of panic selling (De Bondt & Thaler 1985).
    """

    theoretical_basis: str = (
        "Short-term mean reversion: buying previous day's losers."
    )
    expected_market_condition: str = "Mean-reverting, sideways, or range-bound markets."
    known_failure_modes: str = "Strongly trending environments (falling knives continue to fall)."

    def compute_scores(self, symbol_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        if not symbol_data:
            return pd.DataFrame()

        all_dates = sorted(list(set().union(*(df.index for df in symbol_data.values()))))
        scores = {}
        for symbol, df in symbol_data.items():
            # Negative of yesterday's daily return
            ret = -df["Close"].pct_change()
            scores[symbol] = ret

        return pd.DataFrame(scores, index=all_dates).fillna(0.0)


# ---------------------------------------------------------------------------
# 5. Sector Momentum Factor (Beta Benchmark)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SectorMomentumFactor(RankingFactor):
    """Sector relative strength factor.

    A stock's score is determined by the average 5-day performance of its
    respective sector.
    """

    theoretical_basis: str = (
        "Industry momentum / sector herd flow. Stocks in leading sectors "
        "benefit from institutional industry-level rotations (Moskowitz & Grinblatt 1999)."
    )
    expected_market_condition: str = "Clear institutional rotation across sectors."
    known_failure_modes: str = "Broad market-wide panics or idiosyncratic stock-specific moves."

    def compute_scores(self, symbol_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        if not symbol_data:
            return pd.DataFrame()

        all_dates = sorted(list(set().union(*(df.index for df in symbol_data.values()))))
        
        # 1. Compute individual 5-day returns
        individual_returns = {}
        for symbol, df in symbol_data.items():
            individual_returns[symbol] = df["Close"].pct_change(5)
        
        ind_ret_df = pd.DataFrame(individual_returns, index=all_dates)
        
        # 2. Group columns by sector and compute daily sector means
        sectors = pd.Series([SECTOR_MAP.get(sym, "OTHER") for sym in ind_ret_df.columns], index=ind_ret_df.columns)
        sector_means = ind_ret_df.T.groupby(sectors).mean().T
        
        # 3. Map sector means back to each symbol
        scores = {}
        for symbol in symbol_data.keys():
            sec = SECTOR_MAP.get(symbol, "OTHER")
            if sec in sector_means.columns:
                scores[symbol] = sector_means[sec]
            else:
                scores[symbol] = pd.Series(0.0, index=all_dates)
                
        return pd.DataFrame(scores, index=all_dates).fillna(0.0)


# ---------------------------------------------------------------------------
# 8. Short-Term Momentum
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ShortTermMomentumFactor(RankingFactor):
    """Short lookback momentum factor for next-day cross-sectional ranking."""

    lookback: int = 3

    theoretical_basis: str = (
        "Very-short-horizon persistence from unresolved order imbalance and delayed reaction."
    )
    expected_market_condition: str = "Directional sessions with sustained intraday leadership."
    known_failure_modes: str = "Event-driven whipsaws and overnight sentiment regime shifts."

    def compute_scores(self, symbol_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        if self.lookback <= 0:
            raise ResearchInputError("lookback must be > 0")
        if not symbol_data:
            return pd.DataFrame()
        all_dates = sorted(list(set().union(*(df.index for df in symbol_data.values()))))
        scores = {
            symbol: df["Close"].pct_change(self.lookback)
            for symbol, df in symbol_data.items()
        }
        return pd.DataFrame(scores, index=all_dates).fillna(0.0)


# ---------------------------------------------------------------------------
# 9. Sector Relative Strength
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SectorRelativeStrengthFactor(RankingFactor):
    """Stock strength relative to its sector benchmark."""

    lookback: int = 5

    theoretical_basis: str = (
        "Stock selection edge comes from outperformers within strong sectors, not sector beta alone."
    )
    expected_market_condition: str = "Rotational markets where intra-sector dispersion is meaningful."
    known_failure_modes: str = "One-factor macro shocks that move all sector constituents together."

    def compute_scores(self, symbol_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        if self.lookback <= 0:
            raise ResearchInputError("lookback must be > 0")
        if not symbol_data:
            return pd.DataFrame()

        all_dates = sorted(list(set().union(*(df.index for df in symbol_data.values()))))
        returns = {
            symbol: df["Close"].pct_change(self.lookback)
            for symbol, df in symbol_data.items()
        }
        ret_df = pd.DataFrame(returns, index=all_dates)
        sectors = pd.Series([SECTOR_MAP.get(sym, "OTHER") for sym in ret_df.columns], index=ret_df.columns)
        sector_mean = ret_df.T.groupby(sectors).mean().T

        scores = {}
        for symbol in ret_df.columns:
            sector = SECTOR_MAP.get(symbol, "OTHER")
            benchmark = sector_mean[sector] if sector in sector_mean.columns else 0.0
            scores[symbol] = ret_df[symbol] - benchmark
        return pd.DataFrame(scores, index=all_dates).fillna(0.0)


# ---------------------------------------------------------------------------
# 10. Overnight Gap Factor
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class OvernightGapFactor(RankingFactor):
    """Overnight gap size relative to previous close.

    Can be used to capture continuation or fade setups. By default, high positive
    overnight gap ranks higher.
    """

    theoretical_basis: str = (
        "Overnight information digestion. Represents O/C overnight price pressure."
    )
    expected_market_condition: str = "Earnings season or news-driven regimes."
    known_failure_modes: str = "Whipsaws and immediate fade of large openings."

    def compute_scores(self, symbol_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        if not symbol_data:
            return pd.DataFrame()

        all_dates = sorted(list(set().union(*(df.index for df in symbol_data.values()))))
        scores = {}
        for symbol, df in symbol_data.items():
            prev_close = df["Close"].shift(1)
            gap = (df["Open"] - prev_close) / prev_close
            scores[symbol] = gap

        return pd.DataFrame(scores, index=all_dates).fillna(0.0)


# ---------------------------------------------------------------------------
# 7. Volatility Compression Factor
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class VolatilityCompressionFactor(RankingFactor):
    """Ranks assets higher when their current volatility is exceptionally compressed.

    Computed as the inverse of normalized high-low range (or rolling median range).
    """

    window: int = 20

    theoretical_basis: str = (
        "Volatility cycles (Bollinger 1983). Quiet consolidation precedes "
        "explosive breakout expansions."
    )
    expected_market_condition: str = "Late consolidation phases before major announcements."
    known_failure_modes: str = "Persistent low-volatility dead markets with no eventual expansion."

    def compute_scores(self, symbol_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        if not symbol_data:
            return pd.DataFrame()

        all_dates = sorted(list(set().union(*(df.index for df in symbol_data.values()))))
        scores = {}
        for symbol, df in symbol_data.items():
            hl_range = (df["High"] - df["Low"]) / df["Close"]
            rolling_median = hl_range.rolling(self.window).median()
            # Compression = Inverse of median range (higher = more compressed)
            scores[symbol] = 1.0 / rolling_median.replace(0, np.nan)

        return pd.DataFrame(scores, index=all_dates).fillna(0.0)


# ---------------------------------------------------------------------------
# 8. Relative Volume Factor
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RelativeVolumeFactor(RankingFactor):
    """Ranks assets higher when their daily volume is higher than recent average.

    Computed as current volume divided by its 20-day SMA.
    """

    window: int = 20

    theoretical_basis: str = (
        "Institutional buying interest (Karpoff 1987). Abnormal volume "
        "indicates retail surge or institutional entry, predicting continuation."
    )
    expected_market_condition: str = "High-liquidity trending markets."
    known_failure_modes: str = "Volume exhaustion spikes that lead to immediate reversals."

    def compute_scores(self, symbol_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        if not symbol_data:
            return pd.DataFrame()

        all_dates = sorted(list(set().union(*(df.index for df in symbol_data.values()))))
        scores = {}
        for symbol, df in symbol_data.items():
            vol_sma = df["Volume"].rolling(self.window).mean()
            rel_vol = df["Volume"] / vol_sma.replace(0, np.nan)
            scores[symbol] = rel_vol

        return pd.DataFrame(scores, index=all_dates).fillna(0.0)


# ---------------------------------------------------------------------------
# 9. ATR Expansion Factor
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ATRExpansionFactor(RankingFactor):
    """Ranks assets higher when volatility is expanding.

    Computed as the ratio of 5-day ATR to 20-day ATR.
    """

    fast_window: int = 5
    slow_window: int = 20

    theoretical_basis: str = (
        "Volatility clustering (Mandelbrot 1963). High volatility is "
        "accompanied by active trading and large price movements."
    )
    expected_market_condition: str = "Early trend breakout and expansion phases."
    known_failure_modes: str = "Late-stage volatility spikes representing exhaustion."

    def _compute_atr(self, df: pd.DataFrame, window: int) -> pd.Series:
        high = df["High"]
        low = df["Low"]
        prev_close = df["Close"].shift(1)
        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ], axis=1).max(axis=1)
        return tr.rolling(window).mean()

    def compute_scores(self, symbol_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        if not symbol_data:
            return pd.DataFrame()

        all_dates = sorted(list(set().union(*(df.index for df in symbol_data.values()))))
        scores = {}
        for symbol, df in symbol_data.items():
            fast_atr = self._compute_atr(df, self.fast_window)
            slow_atr = self._compute_atr(df, self.slow_window)
            scores[symbol] = fast_atr / slow_atr.replace(0, np.nan)

        return pd.DataFrame(scores, index=all_dates).fillna(0.0)


# ---------------------------------------------------------------------------
# 13. Rolling Beta-Adjusted Strength
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RollingBetaAdjustedStrengthFactor(RankingFactor):
    """Excess strength after scaling by rolling market beta."""

    beta_window: int = 40
    return_lookback: int = 5

    theoretical_basis: str = (
        "Cross-sectional alpha should be evaluated after controlling for market beta exposure."
    )
    expected_market_condition: str = "Broad market moves where idiosyncratic leadership still matters."
    known_failure_modes: str = "Thin data windows and unstable beta estimates in shock regimes."

    def compute_scores(self, symbol_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        if self.beta_window <= 1:
            raise ResearchInputError("beta_window must be > 1")
        if self.return_lookback <= 0:
            raise ResearchInputError("return_lookback must be > 0")
        if not symbol_data:
            return pd.DataFrame()

        all_dates = sorted(list(set().union(*(df.index for df in symbol_data.values()))))
        close_returns = {
            symbol: df["Close"].pct_change()
            for symbol, df in symbol_data.items()
        }
        ret_df = pd.DataFrame(close_returns, index=all_dates)
        market = ret_df.mean(axis=1)
        market_var = market.rolling(self.beta_window).var(ddof=0).replace(0.0, np.nan)

        scores: Dict[str, pd.Series] = {}
        for symbol in ret_df.columns:
            cov = ret_df[symbol].rolling(self.beta_window).cov(market)
            beta = cov / market_var
            strength = ret_df[symbol].rolling(self.return_lookback).mean()
            beta_adjusted = strength - (beta * market.rolling(self.return_lookback).mean())
            scores[symbol] = beta_adjusted
        return pd.DataFrame(scores, index=all_dates).fillna(0.0)


# ---------------------------------------------------------------------------
# Factor Universe Builder
# ---------------------------------------------------------------------------

def build_factor_universe() -> Tuple[RankingFactor, ...]:
    """Return all baseline and candidate ranking factors for research."""
    return (
        # Ranking baselines
        RandomRankingFactor(name="random_ranking"),
        MarketBenchmarkFactor(name="equal_weight_market"),
        BuyAndHoldBaselineFactor(name="buy_and_hold_baseline"),
        EqualWeightSelectionFactor(name="equal_weight_selection"),
        SimpleMomentumRankFactor(name="simple_momentum_rank"),
        VolatilityRankFactor(name="volatility_rank"),
        Momentum20Factor(name="momentum_20", lookback=20),
        # Legacy baselines maintained for backward compatibility
        PreviousDayWinnerContinuation(name="winner_continuation"),
        PreviousDayLoserRebound(name="loser_rebound"),
        SectorMomentumFactor(name="sector_momentum"),
        # Required predictive factors
        PreviousDayReturnFactor(name="previous_day_return"),
        ShortTermMomentumFactor(name="short_term_momentum"),
        SectorRelativeStrengthFactor(name="sector_relative_strength"),
        OvernightGapFactor(name="overnight_gap"),
        VolatilityCompressionFactor(name="volatility_compression"),
        RelativeVolumeFactor(name="relative_volume"),
        ATRExpansionFactor(name="atr_expansion"),
        RollingBetaAdjustedStrengthFactor(name="rolling_beta_adjusted_strength"),
    )
