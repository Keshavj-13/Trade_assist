"""Performance metrics used by all strategy validation stages."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from predictor.research.errors import ResearchInputError
from predictor.research.types import PerformanceMetrics


def _max_drawdown_from_equity(equity_curve: pd.Series) -> float:
    """Compute maximum drawdown from an equity curve."""

    running_peak = equity_curve.cummax()
    drawdown = equity_curve / running_peak - 1.0
    return float(drawdown.min())


def compute_performance_metrics(
    returns: pd.Series,
    *,
    bars_per_year: int,
    positions: pd.Series | None = None,
) -> PerformanceMetrics:
    """Compute return, risk, and robustness metrics from period returns."""

    if not isinstance(returns, pd.Series):
        raise ResearchInputError("returns must be a pandas Series")
    if bars_per_year <= 0:
        raise ResearchInputError("bars_per_year must be > 0")

    clean = returns.fillna(0.0).astype(float)
    growth = (1.0 + clean).cumprod()
    total_return = float(growth.iloc[-1] - 1.0) if not growth.empty else 0.0

    std = float(clean.std(ddof=0))
    mean = float(clean.mean())
    if std <= 0.0:
        sharpe = 0.0
    else:
        sharpe = float((mean / std) * math.sqrt(bars_per_year))

    max_drawdown = _max_drawdown_from_equity(growth) if not growth.empty else 0.0

    trade_count = 0
    exposure_time = 0.0
    avg_holding_period = 0.0
    expectancy = 0.0
    avg_win = 0.0
    avg_loss = 0.0
    largest_win = 0.0
    largest_loss = 0.0

    if positions is not None and len(positions) > 0:
        clean_pos = positions.fillna(0.0).astype(float)
        exposure_time = float((clean_pos != 0).sum() / len(clean_pos))
        
        # Calculate individual trades
        sign_changes = clean_pos.diff()
        # A trade starts when position becomes non-zero or changes sign, and ends when it changes.
        # But this is a simple approximation. Let's just group by contiguous non-zero values.
        active_periods = clean_pos != 0
        blocks = (active_periods != active_periods.shift()).cumsum()
        
        trade_returns = []
        trade_lengths = []
        for _, group in clean.groupby(blocks):
            # Only consider blocks where position is active
            idx = group.index[0]
            if clean_pos.loc[idx] != 0:
                tr = float((1.0 + group).prod() - 1.0)
                trade_returns.append(tr)
                trade_lengths.append(len(group))
                
        trade_count = len(trade_returns)
        if trade_count > 0:
            avg_holding_period = float(np.mean(trade_lengths))
            wins = [r for r in trade_returns if r > 0]
            losses = [r for r in trade_returns if r < 0]
            
            avg_win = float(np.mean(wins)) if wins else 0.0
            avg_loss = float(np.mean(losses)) if losses else 0.0
            largest_win = float(np.max(wins)) if wins else 0.0
            largest_loss = float(np.min(losses)) if losses else 0.0
            
            win_prob = len(wins) / trade_count if trade_count > 0 else 0.0
            loss_prob = 1.0 - win_prob
            expectancy = float((win_prob * avg_win) + (loss_prob * avg_loss))
            
            if win_prob > 0:
                win_rate = float(win_prob) # Override simple win_rate
            else:
                win_rate = 0.0
        else:
            win_rate = 0.0
    else:
        active = clean[clean != 0.0]
        if active.empty:
            win_rate = 0.0
        else:
            win_rate = float((active > 0.0).sum() / len(active))

    gains = float(clean[clean > 0.0].sum())
    losses = float(clean[clean < 0.0].sum())
    if losses == 0.0:
        profit_factor = float("inf") if gains > 0.0 else 0.0
    else:
        profit_factor = float(gains / abs(losses))

    return PerformanceMetrics(
        total_return=total_return,
        sharpe_ratio=sharpe,
        max_drawdown=max_drawdown,
        win_rate=win_rate,
        profit_factor=profit_factor,
        trade_count=trade_count,
        exposure_time=exposure_time,
        avg_holding_period=avg_holding_period,
        expectancy=expectancy,
        avg_win=avg_win,
        avg_loss=avg_loss,
        largest_win=largest_win,
        largest_loss=largest_loss,
    )


def finite_profit_factor(value: float) -> float:
    """Clamp infinite profit factors for stable cross-strategy scoring."""

    if np.isfinite(value):
        return float(value)
    return 10.0
