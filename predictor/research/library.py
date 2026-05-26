"""Literature-guided strategy families, variants, and hybrid combinations."""

from __future__ import annotations

from typing import Tuple

from predictor.research.baselines import build_baseline_universe
from predictor.research.hypotheses import build_hypothesis_universe
from predictor.research.strategies import (
    DonchianBreakoutStrategy,
    KeltnerBreakoutStrategy,
    MeanReversionStrategy,
    MomentumStrategy,
    MovingAverageCrossoverStrategy,
    RSIMeanReversionStrategy,
    RegimeSwitchingStrategy,
    TripleMAcrossoverStrategy,
    TradingStrategy,
    VolatilityBreakoutStrategy,
    WeightedStrategyEnsemble,
)


def build_literature_strategy_universe() -> Tuple[TradingStrategy, ...]:
    """Build the full multi-family strategy universe grounded in research literature.

    Families included:
        Trend-following  — Donchian (turtle), MA crossover (Faber), triple MA,
                           Keltner channel breakout
        Momentum         — Time-series momentum at 20 and 60 bar lookbacks
        Mean reversion   — Z-score (20 bar) and RSI-triggered (Wilder)
        Volatility       — ATR breakout
        Regime-adaptive  — Volatility-regime switching
        Hybrids          — Ensemble blends of complementary families

    Variants probe sensitivity to lookback and threshold parameters.
    """
    # --- Trend-following ---
    donchian = DonchianBreakoutStrategy(lookback=55, name="donchian_breakout")
    donchian_fast = DonchianBreakoutStrategy(lookback=20, name="donchian_breakout_fast")

    ma_cross = MovingAverageCrossoverStrategy(short_window=20, long_window=60, name="ma_crossover")
    ma_cross_fast = MovingAverageCrossoverStrategy(
        short_window=10, long_window=40, name="ma_crossover_fast"
    )
    ma_cross_slow = MovingAverageCrossoverStrategy(
        short_window=50, long_window=200, name="ma_crossover_slow"
    )

    triple_ma = TripleMAcrossoverStrategy(fast=10, medium=30, slow=100, name="triple_ma_crossover")

    keltner = KeltnerBreakoutStrategy(
        ema_window=20, atr_window=14, atr_multiplier=2.0, name="keltner_breakout"
    )

    # --- Momentum ---
    momentum = MomentumStrategy(lookback=20, name="momentum")
    momentum_slow = MomentumStrategy(lookback=60, name="momentum_slow")

    # --- Mean reversion ---
    mean_reversion = MeanReversionStrategy(window=20, zscore_threshold=1.0, name="mean_reversion")
    rsi_reversion = RSIMeanReversionStrategy(
        rsi_window=14, oversold=30.0, overbought=70.0, name="rsi_mean_reversion"
    )

    # --- Volatility breakout ---
    vol_breakout = VolatilityBreakoutStrategy(
        lookback=20, atr_window=14, atr_multiplier=1.5, name="volatility_breakout"
    )

    # --- Regime-adaptive ---
    regime = RegimeSwitchingStrategy(
        volatility_window=20, trend_window=50, name="regime_switching"
    )

    # --- Hybrids (literature-motivated ensembles) ---
    # Trend + reversion blend: momentum anchors, mean-reversion and regime modulate
    hybrid_trend_reversion = WeightedStrategyEnsemble(
        name="hybrid_trend_reversion",
        components=(momentum, mean_reversion, regime),
        weights=(0.5, 0.2, 0.3),
        threshold=0.10,
    )
    # Breakout consensus: Donchian, Keltner, and ATR breakout must broadly agree
    hybrid_breakout_consensus = WeightedStrategyEnsemble(
        name="hybrid_breakout_consensus",
        components=(donchian, keltner, vol_breakout),
        weights=(0.4, 0.3, 0.3),
        threshold=0.12,
    )
    # RSI + regime: fade extremes when low-vol, follow trend when high-vol
    hybrid_rsi_regime = WeightedStrategyEnsemble(
        name="hybrid_rsi_regime",
        components=(rsi_reversion, regime),
        weights=(0.5, 0.5),
        threshold=0.10,
    )

    return (
        # Anchor family (Donchian)
        donchian,
        donchian_fast,
        # MA crossover family
        ma_cross,
        ma_cross_fast,
        ma_cross_slow,
        triple_ma,
        # Keltner
        keltner,
        # Momentum
        momentum,
        momentum_slow,
        # Mean reversion
        mean_reversion,
        rsi_reversion,
        # Volatility
        vol_breakout,
        # Regime
        regime,
        # Hybrids
        hybrid_trend_reversion,
        hybrid_breakout_consensus,
        hybrid_rsi_regime,
    )


def build_core_strategy_universe() -> Tuple[TradingStrategy, ...]:
    """Return a smaller set of one representative per family.

    Use for fast exploratory runs where you want coverage without full depth.
    """
    return (
        DonchianBreakoutStrategy(lookback=55, name="donchian_breakout"),
        MovingAverageCrossoverStrategy(short_window=20, long_window=60, name="ma_crossover"),
        TripleMAcrossoverStrategy(fast=10, medium=30, slow=100, name="triple_ma_crossover"),
        KeltnerBreakoutStrategy(ema_window=20, atr_window=14, name="keltner_breakout"),
        MomentumStrategy(lookback=20, name="momentum"),
        MeanReversionStrategy(window=20, zscore_threshold=1.0, name="mean_reversion"),
        RSIMeanReversionStrategy(name="rsi_mean_reversion"),
        VolatilityBreakoutStrategy(lookback=20, atr_window=14, name="volatility_breakout"),
        RegimeSwitchingStrategy(volatility_window=20, trend_window=50, name="regime_switching"),
    )


# Re-export for convenience — library.py is the single import point for strategy universes
__all__ = [
    "build_literature_strategy_universe",
    "build_core_strategy_universe",
    "build_baseline_universe",
    "build_hypothesis_universe",
]
