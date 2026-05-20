"""Worst-case scenario stress testing for strategy robustness."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from predictor.research.backtest import backtest_strategy
from predictor.research.data import validate_research_frame
from predictor.research.errors import ResearchInputError
from predictor.research.strategies import TradingStrategy
from predictor.research.types import StressSliceResult, StressTestResult


_MIN_STRESS_BARS = 30
_DRAWDOWN_WINDOW = 252
_VOL_SPIKE_WINDOW = 21
_DROP_WINDOW_RADIUS = 30


def _find_worst_drawdown_slice(close: pd.Series) -> Optional[pd.RangeIndex]:
    """Return (peak_idx, trough_idx) of the maximum peak-to-trough drawdown."""
    arr = close.to_numpy(dtype=float)
    n = len(arr)
    if n < 2:
        return None
    peak = arr[0]
    peak_i = 0
    best_dd = 0.0
    best_peak_i = 0
    best_trough_i = 0
    for i in range(1, n):
        if arr[i] > peak:
            peak = arr[i]
            peak_i = i
        dd = arr[i] / peak - 1.0
        if dd < best_dd:
            best_dd = dd
            best_peak_i = peak_i
            best_trough_i = i
    if best_trough_i <= best_peak_i:
        return None
    return best_peak_i, best_trough_i


def detect_worst_case_slices(frame: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Detect worst-case historical slices from OHLCV data.

    Returns dict mapping label -> sub-frame. Labels:
        full_history     — complete frame (always present as baseline)
        worst_drawdown   — peak-to-trough of maximum sustained drawdown
        worst_vol_spike  — window centred on peak realized volatility
        largest_day_drop — window around the single worst daily close return
    """
    if frame is None or not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ResearchInputError("frame must be a non-empty DataFrame")
    if "Close" not in frame.columns:
        raise ResearchInputError("frame must contain a Close column")

    close = frame["Close"].astype(float)
    returns = close.pct_change().fillna(0.0)
    n = len(frame)

    slices: Dict[str, pd.DataFrame] = {"full_history": frame}

    # Worst drawdown slice
    dd_result = _find_worst_drawdown_slice(close)
    if dd_result is not None:
        pi, ti = dd_result
        if ti - pi >= _MIN_STRESS_BARS:
            slices["worst_drawdown"] = frame.iloc[pi : ti + 1].copy()

    # Worst vol spike: window centred on peak 21-day realized vol
    vol = returns.rolling(_VOL_SPIKE_WINDOW, min_periods=5).std(ddof=0)
    valid_vol = vol.dropna()
    if len(valid_vol) >= _MIN_STRESS_BARS:
        vol_peak_pos = int(np.nanargmax(vol.values))
        start = max(0, vol_peak_pos - _VOL_SPIKE_WINDOW * 3)
        end = min(n - 1, vol_peak_pos + _VOL_SPIKE_WINDOW)
        if end - start >= _MIN_STRESS_BARS:
            slices["worst_vol_spike"] = frame.iloc[start : end + 1].copy()

    # Largest single-day close drop
    if n > 1:
        worst_day_pos = int(np.nanargmin(returns.values))
        start = max(0, worst_day_pos - _DROP_WINDOW_RADIUS)
        end = min(n - 1, worst_day_pos + _DROP_WINDOW_RADIUS)
        if end - start >= _MIN_STRESS_BARS:
            slices["largest_day_drop"] = frame.iloc[start : end + 1].copy()

    return slices


@dataclass(frozen=True)
class StressTestConfig:
    """Configuration for strategy stress testing."""

    bars_per_year: int = 252
    transaction_cost_bps: float = 5.0
    min_slice_bars: int = 30


def run_stress_tests(
    frame: pd.DataFrame,
    strategy: TradingStrategy,
    *,
    config: StressTestConfig,
) -> StressTestResult:
    """Run strategy on worst-case historical slices and collect per-slice metrics.

    Slices too short for meaningful analysis are silently skipped.
    Slice-level exceptions are caught individually so one bad slice
    does not abort the full test.
    """
    slices = detect_worst_case_slices(frame)
    results: List[StressSliceResult] = []

    for label, slice_frame in slices.items():
        if len(slice_frame) < config.min_slice_bars:
            continue
        try:
            validated = validate_research_frame(
                slice_frame,
                symbol=label,
                min_rows=config.min_slice_bars,
            )
            run = backtest_strategy(
                validated,
                strategy,
                bars_per_year=config.bars_per_year,
                transaction_cost_bps=config.transaction_cost_bps,
            )
            results.append(
                StressSliceResult(
                    label=label,
                    strategy_name=strategy.name,
                    metrics=run.metrics,
                    bar_count=len(validated),
                    start_date=str(validated.index[0].date()),
                    end_date=str(validated.index[-1].date()),
                )
            )
        except Exception:
            # Slice malformed or too sparse — not a code bug
            continue

    return StressTestResult(strategy_name=strategy.name, slices=tuple(results))


def worst_stress_drawdown(result: StressTestResult) -> float:
    """Return the worst max drawdown observed across all stress slices."""
    if not result.slices:
        return 0.0
    return min(s.metrics.max_drawdown for s in result.slices)


def worst_stress_sharpe(result: StressTestResult) -> float:
    """Return the worst Sharpe ratio observed across all stress slices."""
    if not result.slices:
        return 0.0
    return min(s.metrics.sharpe_ratio for s in result.slices)
