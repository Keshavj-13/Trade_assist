"""Deterministic backtest engine shared by all validation stages."""

from __future__ import annotations

import numpy as np
import pandas as pd

from predictor.research.data import validate_research_frame
from predictor.research.errors import ResearchInputError
from predictor.research.metrics import compute_performance_metrics
from predictor.research.strategies import TradingStrategy
from predictor.research.types import BacktestRun


def _validate_positions(positions: pd.Series, index: pd.Index) -> pd.Series:
    """Validate and normalize strategy positions to {-1, 0, 1}."""

    if not isinstance(positions, pd.Series):
        raise ResearchInputError("strategy.generate_positions must return pandas Series")
    aligned = positions.reindex(index).fillna(0.0).astype(float)
    if not np.isfinite(aligned.values).all():
        raise ResearchInputError("positions contain non-finite values")
    clipped = np.sign(aligned.clip(lower=-1.0, upper=1.0))
    return pd.Series(clipped, index=index, dtype=float)


def backtest_strategy(
    frame: pd.DataFrame,
    strategy: TradingStrategy,
    *,
    bars_per_year: int,
    transaction_cost_bps: float = 0.0,
) -> BacktestRun:
    """Backtest one strategy on one OHLCV frame at input data granularity."""

    if transaction_cost_bps < 0:
        raise ResearchInputError("transaction_cost_bps must be >= 0")
    validated = validate_research_frame(frame, symbol=getattr(strategy, "name", "STRAT"), min_rows=2)
    strategy_name = getattr(strategy, "name", None)
    if not isinstance(strategy_name, str) or not strategy_name.strip():
        raise ResearchInputError("strategy must expose a non-empty `name`")

    positions_raw = strategy.generate_positions(validated)
    positions = _validate_positions(positions_raw, validated.index)
    lagged_positions = positions.shift(1).fillna(0.0)

    close_returns = validated["Close"].pct_change().fillna(0.0)
    gross_returns = lagged_positions * close_returns

    if transaction_cost_bps > 0:
        turnover = lagged_positions.diff().abs().fillna(lagged_positions.abs())
        costs = turnover * (transaction_cost_bps / 10_000.0)
    else:
        costs = pd.Series(0.0, index=validated.index, dtype=float)
    net_returns = gross_returns - costs
    equity_curve = (1.0 + net_returns).cumprod()
    metrics = compute_performance_metrics(
        net_returns,
        bars_per_year=bars_per_year,
        positions=lagged_positions,
    )
    return BacktestRun(
        strategy_name=strategy_name,
        returns=net_returns,
        positions=lagged_positions.astype(float),
        equity_curve=equity_curve.astype(float),
        metrics=metrics,
    )
