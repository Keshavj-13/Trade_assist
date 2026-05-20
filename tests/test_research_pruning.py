"""Tests for candidate pruning and early elimination logic."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from predictor.research.errors import ResearchInputError
from predictor.research.pruning import (
    apply_quick_rejects,
    prune_dominated_strategies,
    prune_unstable_strategies,
    quick_reject_strategy,
)
from predictor.research.strategies import DonchianBreakoutStrategy, MomentumStrategy
from predictor.research.types import EliminatedStrategy, MultiSymbolStrategyRow, PruningResult
from predictor.research.validation import ResearchValidationConfig


def _make_frame(rows: int = 200) -> pd.DataFrame:
    idx = pd.date_range("2020-01-02", periods=rows, freq="B")
    rng = np.random.default_rng(7)
    returns = rng.normal(0.0005, 0.01, rows)
    close = 100.0 * np.cumprod(1.0 + returns)
    high = close * 1.003
    low = close * 0.997
    return pd.DataFrame(
        {
            "Open": close * 0.999,
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": 1_000_000.0,
        },
        index=idx,
    )


def _make_row(
    name: str,
    sharpe: float = 0.5,
    ret: float = 0.10,
    drawdown: float = -0.20,
    pass_rate: float = 0.60,
) -> MultiSymbolStrategyRow:
    return MultiSymbolStrategyRow(
        strategy_name=name,
        symbols_tested=4,
        symbols_validated=4,
        avg_total_return=ret,
        avg_sharpe_ratio=sharpe,
        avg_max_drawdown=drawdown,
        avg_win_rate=0.50,
        avg_profit_factor=1.2,
        avg_in_sample_p_value=0.03,
        avg_walk_forward_p_value=0.03,
        avg_walk_forward_stability=0.55,
        validation_pass_rate=pass_rate,
        composite_score=0.5,
    )


class _AlwaysFlatStrategy:
    name = "flat_strategy"

    def generate_positions(self, frame: pd.DataFrame) -> pd.Series:
        return pd.Series(0.0, index=frame.index, dtype=float)


# --- quick_reject_strategy ---

def test_quick_reject_passes_normal_strategy() -> None:
    frame = _make_frame()
    strategy = DonchianBreakoutStrategy(lookback=20, name="donchian_test")
    config = ResearchValidationConfig(bars_per_year=252)
    reject, reason = quick_reject_strategy(frame, strategy, config)
    assert reject is False
    assert reason == ""


def test_quick_reject_rejects_flat_strategy() -> None:
    frame = _make_frame()
    config = ResearchValidationConfig(bars_per_year=252)
    reject, reason = quick_reject_strategy(frame, _AlwaysFlatStrategy(), config)
    assert reject is True
    assert "too_few_trades" in reason


def test_quick_reject_is_deterministic() -> None:
    frame = _make_frame()
    strategy = MomentumStrategy(lookback=20, name="momentum_test")
    config = ResearchValidationConfig(bars_per_year=252)
    r1, reason1 = quick_reject_strategy(frame, strategy, config)
    r2, reason2 = quick_reject_strategy(frame, strategy, config)
    assert r1 == r2
    assert reason1 == reason2


# --- prune_dominated_strategies ---

def test_prune_dominated_removes_strictly_dominated_row() -> None:
    strong = _make_row("strong", sharpe=1.5, ret=0.20, drawdown=-0.10, pass_rate=0.90)
    weak = _make_row("weak", sharpe=0.3, ret=0.05, drawdown=-0.40, pass_rate=0.30)

    result = prune_dominated_strategies([strong, weak])

    assert "strong" in result.kept
    assert "weak" not in result.kept
    assert any(e.strategy_name == "weak" for e in result.eliminated)
    assert any("dominated_by:strong" in e.reason for e in result.eliminated)


def test_prune_dominated_keeps_both_when_neither_dominates() -> None:
    a = _make_row("strategy_a", sharpe=1.0, ret=0.15, drawdown=-0.10, pass_rate=0.80)
    # b has worse sharpe but better drawdown — not dominated
    b = _make_row("strategy_b", sharpe=0.8, ret=0.12, drawdown=-0.05, pass_rate=0.75)

    result = prune_dominated_strategies([a, b])
    assert "strategy_a" in result.kept
    assert "strategy_b" in result.kept
    assert len(result.eliminated) == 0


def test_prune_dominated_empty_input() -> None:
    result = prune_dominated_strategies([])
    assert result.kept == ()
    assert result.eliminated == ()


def test_prune_dominated_single_strategy() -> None:
    row = _make_row("solo")
    result = prune_dominated_strategies([row])
    assert result.kept == ("solo",)
    assert result.eliminated == ()


# --- prune_unstable_strategies ---

def test_prune_unstable_removes_low_stability() -> None:
    stable = _make_row("stable_strat")
    # Override stability via new row
    unstable_row = MultiSymbolStrategyRow(
        strategy_name="unstable_strat",
        symbols_tested=4,
        symbols_validated=4,
        avg_total_return=0.05,
        avg_sharpe_ratio=0.4,
        avg_max_drawdown=-0.25,
        avg_win_rate=0.50,
        avg_profit_factor=1.1,
        avg_in_sample_p_value=0.04,
        avg_walk_forward_p_value=0.04,
        avg_walk_forward_stability=0.20,   # below 0.40 threshold
        validation_pass_rate=0.50,
        composite_score=0.3,
    )

    result = prune_unstable_strategies([stable, unstable_row], min_stability=0.40)
    assert "stable_strat" in result.kept
    assert "unstable_strat" not in result.kept
    eliminated_names = {e.strategy_name for e in result.eliminated}
    assert "unstable_strat" in eliminated_names


def test_prune_unstable_raises_on_invalid_params() -> None:
    with pytest.raises(ResearchInputError):
        prune_unstable_strategies([], min_stability=1.5)
    with pytest.raises(ResearchInputError):
        prune_unstable_strategies([], min_pass_rate=-0.1)
    with pytest.raises(ResearchInputError):
        prune_unstable_strategies([], max_abs_drawdown=0.0)


# --- apply_quick_rejects ---

def test_apply_quick_rejects_separates_flat_from_normal() -> None:
    frame = _make_frame()
    strategies = [
        DonchianBreakoutStrategy(lookback=20, name="donchian_test"),
        _AlwaysFlatStrategy(),
    ]
    config = ResearchValidationConfig(bars_per_year=252)
    passing, pruning_result = apply_quick_rejects(frame, strategies, config)

    passing_names = {s.name for s in passing}
    assert "donchian_test" in passing_names
    assert "flat_strategy" not in passing_names
    eliminated_names = {e.strategy_name for e in pruning_result.eliminated}
    assert "flat_strategy" in eliminated_names
