"""Trivial benchmark strategies for contextualising strategy research results.

Every strategy here documents its theoretical basis, expected market conditions,
and known failure modes BEFORE being used in any backtest. This enforces the
scientific discipline of stating hypotheses prior to seeing results.

Baseline strategies serve two purposes:
1. Context: a candidate strategy must beat at minimum buy-and-hold and random
   entry to be worth further investigation.
2. Calibration: random entry should fail IS permutation; this confirms the
   null distribution is correctly calibrated.

Public API
----------
BuyAndHoldStrategy
ShortAndHoldStrategy
RandomEntryFixedHoldStrategy
SimpleMABaseline
VolatilityTargetedHoldStrategy
build_baseline_universe
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd

from predictor.research.errors import ResearchInputError
from predictor.research.strategies import TradingStrategy


# ---------------------------------------------------------------------------
# Shared mixin — enforces documentation before use
# ---------------------------------------------------------------------------


class _DocumentedStrategy:
    """Mixin that enforces theoretical_basis is a non-empty string."""

    theoretical_basis: str = ""
    expected_market_condition: str = ""
    known_failure_modes: str = ""

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        # Only enforce on concrete instantiable subclasses
        if not getattr(cls, "__abstractmethods__", None):
            for attr in ("theoretical_basis", "expected_market_condition", "known_failure_modes"):
                val = getattr(cls, attr, "")
                if not isinstance(val, str) or not val.strip():
                    raise TypeError(
                        f"{cls.__name__}.{attr} must be a non-empty string. "
                        "Document the hypothesis before creating the strategy."
                    )


# ---------------------------------------------------------------------------
# Buy-and-hold
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BuyAndHoldStrategy(_DocumentedStrategy):
    """Always hold a long position — the passive equity benchmark.

    Theoretical basis:
        Equity risk premium: long-run equities compensate for bearing market
        risk. This is not a trading edge; it is the baseline passive return.

    Expected market condition:
        Secular bull markets. Fails in prolonged bear markets or flat regimes.

    Known failure modes:
        Maximum drawdown equals peak-to-trough decline of the asset — no
        protection during crashes or mean-reverting regimes.
    """

    name: str = "buy_and_hold"

    theoretical_basis: str = (
        "Equity risk premium — passive long exposure to the asset's secular return."
    )
    expected_market_condition: str = "Secular bull market with positive drift."
    known_failure_modes: str = (
        "Full drawdown exposure. No edge over permuted null in trending-neutral periods."
    )

    def generate_positions(self, frame: pd.DataFrame) -> pd.Series:
        """Return +1 for every bar (always long, never trade)."""
        return pd.Series(1.0, index=frame.index, dtype=float)


# ---------------------------------------------------------------------------
# Short-and-hold
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ShortAndHoldStrategy(_DocumentedStrategy):
    """Always hold a short position — the passive bearish benchmark.

    Theoretical basis:
        Inverse of equity risk premium. Useful for detecting strategies
        that merely exploit bearish regimes without selective timing.

    Expected market condition:
        Sustained bear markets or sharp drawdown periods.

    Known failure modes:
        Loses in any positive-drift market. Not a viable long-term strategy.
    """

    name: str = "short_and_hold"

    theoretical_basis: str = "Inverse equity risk premium — passive short exposure."
    expected_market_condition: str = "Sustained bear market or drawdown period."
    known_failure_modes: str = "Loses monotonically in any positive-drift market."

    def generate_positions(self, frame: pd.DataFrame) -> pd.Series:
        """Return -1 for every bar (always short, never trade)."""
        return pd.Series(-1.0, index=frame.index, dtype=float)


# ---------------------------------------------------------------------------
# Random entry with fixed holding period
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RandomEntryFixedHoldStrategy(_DocumentedStrategy):
    """Random entries with a fixed holding period — the null hypothesis benchmark.

    This strategy should fail IS permutation consistently. If it does not,
    the permutation engine is miscalibrated.

    Theoretical basis:
        Pure random walk null hypothesis. No market edge. Useful as a
        calibration probe — any strategy that fails to outperform this
        on permuted data has no detectable edge.

    Expected market condition:
        None — this strategy has no directional hypothesis.

    Known failure modes:
        Always. By design.
    """

    name: str = "random_entry_fixed_hold"
    hold_bars: int = 20
    long_only: bool = True
    seed: int = 42

    theoretical_basis: str = (
        "Random walk null — no market hypothesis. Calibration probe only."
    )
    expected_market_condition: str = "None — this is a null-hypothesis baseline."
    known_failure_modes: str = (
        "Expected to fail IS permutation consistently. If it passes, "
        "the permutation engine may be miscalibrated."
    )

    def generate_positions(self, frame: pd.DataFrame) -> pd.Series:
        """Generate random entries with fixed hold, using deterministic seed."""
        if self.hold_bars <= 0:
            raise ResearchInputError("hold_bars must be > 0")
        n = len(frame)
        rng = np.random.default_rng(self.seed)
        positions = np.zeros(n, dtype=float)
        i = 0
        while i < n:
            # Random skip before next entry (1 to 2*hold_bars bars)
            skip = int(rng.integers(1, max(2, 2 * self.hold_bars)))
            i += skip
            if i >= n:
                break
            direction = 1.0 if self.long_only else float(rng.choice([-1.0, 1.0]))
            hold = min(self.hold_bars, n - i)
            positions[i : i + hold] = direction
            i += hold
        return pd.Series(positions, index=frame.index, dtype=float)


# ---------------------------------------------------------------------------
# Simple MA baseline
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SimpleMABaseline(_DocumentedStrategy):
    """Minimal 20/60 MA crossover — the weakest trend-following baseline.

    Any serious trend strategy must outperform this. If it cannot, it adds
    no value over the simplest possible rule.

    Theoretical basis:
        Price momentum via moving average crossover (Faber 2007, Lo & MacKinlay
        1988). The weakest version of trend following — no regime gating,
        no volatility normalisation.

    Expected market condition:
        Strong sustained trends with low choppiness.

    Known failure modes:
        High turnover in sideways markets. Significant lag at trend reversals.
    """

    name: str = "simple_ma_baseline"
    short_window: int = 20
    long_window: int = 60

    theoretical_basis: str = (
        "Minimal MA crossover (Faber 2007). Weakest possible trend baseline."
    )
    expected_market_condition: str = "Strong sustained trends, low choppiness."
    known_failure_modes: str = (
        "High turnover in sideways markets. Significant lag at reversals."
    )

    def generate_positions(self, frame: pd.DataFrame) -> pd.Series:
        """Return +1 when short MA > long MA, -1 otherwise."""
        if self.short_window >= self.long_window:
            raise ResearchInputError("short_window must be < long_window")
        close = frame["Close"]
        short_ma = close.rolling(self.short_window).mean()
        long_ma = close.rolling(self.long_window).mean()
        signal = pd.Series(0.0, index=close.index, dtype=float)
        signal.loc[short_ma > long_ma] = 1.0
        signal.loc[short_ma < long_ma] = -1.0
        return signal.fillna(0.0)


# ---------------------------------------------------------------------------
# Volatility-targeted hold
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VolatilityTargetedHoldStrategy(_DocumentedStrategy):
    """Long when realized volatility is below its rolling median — the low-vol anomaly.

    Theoretical basis:
        Low-volatility anomaly (Ang et al. 2006, Baker & Haugen 2012).
        Low-volatility stocks and periods tend to have better risk-adjusted
        returns than high-volatility equivalents, contrary to CAPM predictions.

    Expected market condition:
        Persistent low-volatility regimes. May capture risk-off flight to
        quality if combined with sector rotation.

    Known failure modes:
        Misses strongly trending high-vol bull markets. Underperforms
        buy-and-hold in strongly directional bull phases.
    """

    name: str = "volatility_targeted_hold"
    vol_window: int = 21
    baseline_window: int = 63

    theoretical_basis: str = (
        "Low-volatility anomaly (Ang et al. 2006). "
        "Low-vol periods have superior risk-adjusted returns."
    )
    expected_market_condition: str = "Persistent low-volatility regimes."
    known_failure_modes: str = (
        "Underperforms in strongly trending high-vol bull markets."
    )

    def generate_positions(self, frame: pd.DataFrame) -> pd.Series:
        """Return +1 when recent vol < rolling vol median, else 0."""
        close = frame["Close"]
        returns = close.pct_change()
        realized_vol = returns.rolling(self.vol_window).std(ddof=0)
        vol_median = realized_vol.rolling(self.baseline_window).median()
        in_low_vol = realized_vol < vol_median
        positions = pd.Series(0.0, index=frame.index, dtype=float)
        positions.loc[in_low_vol] = 1.0
        return positions.fillna(0.0)


# ---------------------------------------------------------------------------
# Universe builder
# ---------------------------------------------------------------------------


def build_baseline_universe() -> Tuple[TradingStrategy, ...]:
    """Return all baseline strategies for use as benchmarks.

    These should be included in every research run to provide context.
    Any candidate strategy that fails to outperform buy_and_hold and
    random_entry_fixed_hold on the permutation test has no detectable edge.
    """
    return (
        BuyAndHoldStrategy(),
        ShortAndHoldStrategy(),
        RandomEntryFixedHoldStrategy(hold_bars=20, long_only=True, seed=42),
        SimpleMABaseline(short_window=20, long_window=60),
        VolatilityTargetedHoldStrategy(vol_window=21, baseline_window=63),
    )
