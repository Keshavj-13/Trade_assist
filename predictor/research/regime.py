"""Market regime detection and slice extraction for regime-aware evaluation."""

from __future__ import annotations

from typing import Dict, List, Literal, Tuple

import numpy as np
import pandas as pd

from predictor.research.errors import ResearchInputError


RegimeLabel = Literal["BULL", "BEAR", "HIGH_VOL", "LOW_VOL", "CRASH", "RECOVERY", "SIDEWAYS"]

ALL_REGIMES: Tuple[str, ...] = (
    "BULL", "BEAR", "HIGH_VOL", "LOW_VOL", "CRASH", "RECOVERY", "SIDEWAYS"
)

_TREND_WINDOW = 63
_VOL_WINDOW = 21
_VOL_BASELINE_WINDOW = 252
_RECOVERY_LOOKBACK = 40
_CRASH_THRESHOLD = -0.20
_RECOVERING_UPPER = -0.10
_BULL_THRESHOLD = 0.05
_BEAR_THRESHOLD = -0.05
_HIGH_VOL_MULTIPLIER = 1.5
_LOW_VOL_MULTIPLIER = 0.65


def classify_regimes(frame: pd.DataFrame) -> pd.Series:
    """Return per-bar regime labels from Close prices.

    Priority order: CRASH > HIGH_VOL > LOW_VOL > BULL > BEAR > RECOVERY > SIDEWAYS.
    Labels are strings from ALL_REGIMES.
    """
    if frame is None or not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ResearchInputError("frame must be a non-empty DataFrame")
    if "Close" not in frame.columns:
        raise ResearchInputError("frame must contain a Close column")

    close = frame["Close"].astype(float)
    returns = close.pct_change()

    trend = close / close.rolling(_TREND_WINDOW, min_periods=_TREND_WINDOW // 2).mean() - 1.0

    vol = returns.rolling(_VOL_WINDOW, min_periods=_VOL_WINDOW // 2).std(ddof=0) * np.sqrt(252)
    vol_baseline = vol.rolling(_VOL_BASELINE_WINDOW, min_periods=_VOL_WINDOW * 2).median()

    peak = close.cummax()
    drawdown = (close / peak) - 1.0

    labels = pd.Series("SIDEWAYS", index=frame.index, dtype=object)

    # RECOVERY: drawdown was crash-level but has partially healed
    in_crash_zone = drawdown < _CRASH_THRESHOLD
    recent_crash = in_crash_zone.rolling(_RECOVERY_LOOKBACK, min_periods=1).max().astype(bool)
    in_recovery = (drawdown > _CRASH_THRESHOLD) & (drawdown < _RECOVERING_UPPER) & recent_crash
    labels.loc[in_recovery] = "RECOVERY"

    # Trend
    labels.loc[trend > _BULL_THRESHOLD] = "BULL"
    labels.loc[trend < _BEAR_THRESHOLD] = "BEAR"

    # Vol — overrides trend labels
    high_vol = vol > (vol_baseline * _HIGH_VOL_MULTIPLIER)
    low_vol = (vol < (vol_baseline * _LOW_VOL_MULTIPLIER)) & vol_baseline.notna()
    labels.loc[high_vol] = "HIGH_VOL"
    labels.loc[low_vol] = "LOW_VOL"

    # CRASH overrides everything
    labels.loc[in_crash_zone] = "CRASH"

    return labels


def extract_regime_slices(
    frame: pd.DataFrame,
    target_regime: str,
    *,
    min_bars: int = 20,
) -> List[pd.DataFrame]:
    """Extract contiguous slices where target_regime dominates.

    Each returned slice is a copy of the sub-frame with the original DatetimeIndex.
    Slices shorter than min_bars are dropped.
    """
    if target_regime not in ALL_REGIMES:
        raise ResearchInputError(
            f"target_regime must be one of {ALL_REGIMES}, got {target_regime!r}"
        )
    if min_bars <= 0:
        raise ResearchInputError("min_bars must be > 0")

    labels = classify_regimes(frame)
    mask = (labels == target_regime).to_numpy()

    slices: List[pd.DataFrame] = []
    start_idx: int | None = None
    n = len(frame)

    for i in range(n):
        if mask[i] and start_idx is None:
            start_idx = i
        elif not mask[i] and start_idx is not None:
            if i - start_idx >= min_bars:
                slices.append(frame.iloc[start_idx:i].copy())
            start_idx = None

    if start_idx is not None and n - start_idx >= min_bars:
        slices.append(frame.iloc[start_idx:].copy())

    return slices


def regime_distribution(frame: pd.DataFrame) -> Dict[str, float]:
    """Return fraction of bars assigned to each regime."""
    labels = classify_regimes(frame)
    total = max(len(labels), 1)
    return {regime: float((labels == regime).sum() / total) for regime in ALL_REGIMES}


def dominant_regimes(frame: pd.DataFrame, *, top_n: int = 3) -> Tuple[str, ...]:
    """Return the top_n most frequent regimes in descending order."""
    if top_n <= 0:
        raise ResearchInputError("top_n must be > 0")
    dist = regime_distribution(frame)
    sorted_regimes = sorted(dist.items(), key=lambda kv: kv[1], reverse=True)
    return tuple(name for name, _ in sorted_regimes[:top_n])
