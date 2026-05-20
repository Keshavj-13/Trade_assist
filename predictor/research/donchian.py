"""Donchian-specific variant library and strict stability selection helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal, Sequence, Tuple

import numpy as np
import pandas as pd

from predictor.research.errors import ResearchInputError
from predictor.research.types import MultiSymbolStrategyRow
from predictor.research.validation import ResearchValidationConfig


DependencyRule = Literal["none", "after_loss", "after_win", "skip_after_win"]


def _validate_positive_int(value: int, label: str) -> None:
    """Validate a positive integer parameter."""

    if not isinstance(value, int) or value <= 0:
        raise ResearchInputError(f"{label} must be a positive integer")


def _validate_dependency_rule(rule: str) -> None:
    """Validate trade-dependence rule name."""

    if rule not in {"none", "after_loss", "after_win", "skip_after_win"}:
        raise ResearchInputError(
            "dependency_rule must be one of: none, after_loss, after_win, skip_after_win"
        )


def _average_true_range(frame: pd.DataFrame, window: int) -> pd.Series:
    """Compute Average True Range (ATR) from OHLC data."""

    prev_close = frame["Close"].shift(1)
    high_low = frame["High"] - frame["Low"]
    high_close = (frame["High"] - prev_close).abs()
    low_close = (frame["Low"] - prev_close).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return true_range.rolling(window).mean()


def _entry_breakout_signal(
    *,
    close: float,
    upper: float,
    lower: float,
) -> float:
    """Return Donchian breakout direction for one bar."""

    if np.isnan(upper) or np.isnan(lower):
        return 0.0
    if close > upper:
        return 1.0
    if close < lower:
        return -1.0
    return 0.0


def _dependency_allows_entry(last_outcome: int | None, rule: DependencyRule) -> bool:
    """Return whether dependency rule permits an entry."""

    if rule == "none":
        return True
    if last_outcome is None:
        return True
    if rule == "after_loss":
        return last_outcome <= 0
    if rule == "after_win":
        return last_outcome > 0
    # skip_after_win
    return last_outcome <= 0


def _initial_stop_price(direction: float, close: float, atr: float, multiplier: float) -> float:
    """Build initial ATR stop around entry price."""

    if direction > 0:
        return close - multiplier * atr
    return close + multiplier * atr


def _update_trailing_stop(
    *,
    direction: float,
    current_stop: float,
    close: float,
    atr: float,
    multiplier: float,
) -> float:
    """Update ATR trailing stop while preserving one-way tightening."""

    candidate = _initial_stop_price(direction, close, atr, multiplier)
    if direction > 0:
        return max(current_stop, candidate)
    return min(current_stop, candidate)


def _stop_hit(direction: float, close: float, stop: float | None) -> bool:
    """Check whether price has crossed stop for current position."""

    if stop is None:
        return False
    if direction > 0:
        return close <= stop
    return close >= stop


def _exit_signal(
    *,
    direction: float,
    close: float,
    exit_upper: float,
    exit_lower: float,
) -> bool:
    """Return whether the Donchian exit channel triggers."""

    if direction > 0:
        return not np.isnan(exit_lower) and close < exit_lower
    return not np.isnan(exit_upper) and close > exit_upper


@dataclass(frozen=True)
class DonchianVariantStrategy:
    """Configurable Donchian trend-following variant with dependency-aware entries."""

    name: str
    entry_lookback: int
    exit_lookback: int
    dependency_rule: DependencyRule = "none"
    failsafe_entry_lookback: int | None = None
    atr_window: int = 20
    atr_stop_multiplier: float | None = None
    trailing_stop: bool = True
    atr_filter_min_quantile: float | None = None

    def generate_positions(self, frame: pd.DataFrame) -> pd.Series:
        """Generate long/short/flat position series from OHLC data."""

        _validate_positive_int(self.entry_lookback, "entry_lookback")
        _validate_positive_int(self.exit_lookback, "exit_lookback")
        _validate_positive_int(self.atr_window, "atr_window")
        _validate_dependency_rule(self.dependency_rule)
        if self.failsafe_entry_lookback is not None:
            _validate_positive_int(self.failsafe_entry_lookback, "failsafe_entry_lookback")
            if self.failsafe_entry_lookback < self.entry_lookback:
                raise ResearchInputError(
                    "failsafe_entry_lookback must be >= entry_lookback"
                )
        if self.atr_stop_multiplier is not None and self.atr_stop_multiplier <= 0:
            raise ResearchInputError("atr_stop_multiplier must be > 0 when provided")
        if self.atr_filter_min_quantile is not None:
            if not 0.0 <= self.atr_filter_min_quantile <= 1.0:
                raise ResearchInputError("atr_filter_min_quantile must be in [0,1]")

        close = frame["Close"]
        entry_upper = frame["High"].rolling(self.entry_lookback).max().shift(1)
        entry_lower = frame["Low"].rolling(self.entry_lookback).min().shift(1)
        exit_upper = frame["High"].rolling(self.exit_lookback).max().shift(1)
        exit_lower = frame["Low"].rolling(self.exit_lookback).min().shift(1)
        atr = _average_true_range(frame, self.atr_window)

        if self.failsafe_entry_lookback is not None:
            failsafe_upper = frame["High"].rolling(self.failsafe_entry_lookback).max().shift(1)
            failsafe_lower = frame["Low"].rolling(self.failsafe_entry_lookback).min().shift(1)
        else:
            failsafe_upper = pd.Series(np.nan, index=frame.index)
            failsafe_lower = pd.Series(np.nan, index=frame.index)

        if self.atr_filter_min_quantile is not None:
            threshold = float(atr.quantile(self.atr_filter_min_quantile))
        else:
            threshold = float("-inf")

        positions = pd.Series(0.0, index=frame.index, dtype=float)
        current_position = 0.0
        entry_price = 0.0
        stop_price: float | None = None
        last_outcome: int | None = None

        for index in frame.index:
            current_close = float(close.loc[index])
            current_atr = float(atr.loc[index]) if np.isfinite(atr.loc[index]) else np.nan

            if current_position == 0.0:
                entry_signal = _entry_breakout_signal(
                    close=current_close,
                    upper=float(entry_upper.loc[index]) if np.isfinite(entry_upper.loc[index]) else np.nan,
                    lower=float(entry_lower.loc[index]) if np.isfinite(entry_lower.loc[index]) else np.nan,
                )
                if not _dependency_allows_entry(last_outcome, self.dependency_rule):
                    entry_signal = 0.0
                if (
                    entry_signal == 0.0
                    and self.dependency_rule == "skip_after_win"
                    and self.failsafe_entry_lookback is not None
                    and last_outcome is not None
                    and last_outcome > 0
                ):
                    entry_signal = _entry_breakout_signal(
                        close=current_close,
                        upper=float(failsafe_upper.loc[index])
                        if np.isfinite(failsafe_upper.loc[index])
                        else np.nan,
                        lower=float(failsafe_lower.loc[index])
                        if np.isfinite(failsafe_lower.loc[index])
                        else np.nan,
                    )

                if entry_signal != 0.0 and current_atr >= threshold and np.isfinite(current_atr):
                    current_position = entry_signal
                    entry_price = current_close
                    if self.atr_stop_multiplier is not None:
                        stop_price = _initial_stop_price(
                            direction=current_position,
                            close=current_close,
                            atr=current_atr,
                            multiplier=self.atr_stop_multiplier,
                        )
                positions.loc[index] = current_position
                continue

            channel_exit = _exit_signal(
                direction=current_position,
                close=current_close,
                exit_upper=float(exit_upper.loc[index]) if np.isfinite(exit_upper.loc[index]) else np.nan,
                exit_lower=float(exit_lower.loc[index]) if np.isfinite(exit_lower.loc[index]) else np.nan,
            )
            stop_exit = _stop_hit(current_position, current_close, stop_price)
            should_exit = channel_exit or stop_exit

            if should_exit:
                pnl = current_position * (current_close - entry_price)
                last_outcome = 1 if pnl > 0 else -1
                current_position = 0.0
                entry_price = 0.0
                stop_price = None
                positions.loc[index] = 0.0
                continue

            if (
                self.trailing_stop
                and self.atr_stop_multiplier is not None
                and np.isfinite(current_atr)
                and stop_price is not None
            ):
                stop_price = _update_trailing_stop(
                    direction=current_position,
                    current_stop=stop_price,
                    close=current_close,
                    atr=current_atr,
                    multiplier=self.atr_stop_multiplier,
                )
            positions.loc[index] = current_position
        return positions


def build_donchian_variant_universe() -> Tuple[DonchianVariantStrategy, ...]:
    """Build a Donchian-only variant universe grounded in turtle-style research."""

    return (
        DonchianVariantStrategy(
            name="donchian_s1_20_10",
            entry_lookback=20,
            exit_lookback=10,
            dependency_rule="none",
        ),
        DonchianVariantStrategy(
            name="donchian_s2_55_20",
            entry_lookback=55,
            exit_lookback=20,
            dependency_rule="none",
        ),
        DonchianVariantStrategy(
            name="donchian_s1_after_loss",
            entry_lookback=20,
            exit_lookback=10,
            dependency_rule="after_loss",
        ),
        DonchianVariantStrategy(
            name="donchian_s1_after_win",
            entry_lookback=20,
            exit_lookback=10,
            dependency_rule="after_win",
        ),
        DonchianVariantStrategy(
            name="donchian_s1_skip_after_win_failsafe55",
            entry_lookback=20,
            exit_lookback=10,
            dependency_rule="skip_after_win",
            failsafe_entry_lookback=55,
        ),
        DonchianVariantStrategy(
            name="donchian_s1_atr_stop_2n",
            entry_lookback=20,
            exit_lookback=10,
            dependency_rule="none",
            atr_stop_multiplier=2.0,
            trailing_stop=True,
        ),
        DonchianVariantStrategy(
            name="donchian_s2_atr_stop_2n",
            entry_lookback=55,
            exit_lookback=20,
            dependency_rule="none",
            atr_stop_multiplier=2.0,
            trailing_stop=True,
        ),
        DonchianVariantStrategy(
            name="donchian_s1_after_loss_atr_filter",
            entry_lookback=20,
            exit_lookback=10,
            dependency_rule="after_loss",
            atr_filter_min_quantile=0.40,
        ),
    )


def build_strict_donchian_validation_config() -> ResearchValidationConfig:
    """Return stricter defaults for Donchian variant selection."""

    return ResearchValidationConfig(
        bars_per_year=252,
        transaction_cost_bps=5.0,
        permutation_iterations=1000,
        permutation_block_size=20,
        p_value_threshold=0.02,
        train_window=260,
        test_window=80,
        walk_forward_step=40,
        minimum_walk_forward_stability=0.55,
        minimum_walk_forward_fold_pass_rate=0.70,
        require_positive_walk_forward_return=True,
        random_seed=19,
    )


def holm_adjust_p_values(p_values: Sequence[float]) -> Tuple[float, ...]:
    """Return Holm step-down adjusted p-values in original order."""

    if p_values is None:
        raise ResearchInputError("p_values must not be None")
    if not p_values:
        return ()
    raw = np.asarray(p_values, dtype=float)
    if np.any(~np.isfinite(raw)) or np.any(raw < 0.0) or np.any(raw > 1.0):
        raise ResearchInputError("p_values must be finite numbers in [0,1]")

    order = np.argsort(raw)
    sorted_p = raw[order]
    m = len(sorted_p)
    adjusted_sorted = np.zeros(m, dtype=float)
    running_max = 0.0
    for i, p_value in enumerate(sorted_p, start=1):
        corrected = min(1.0, (m - i + 1) * p_value)
        running_max = max(running_max, corrected)
        adjusted_sorted[i - 1] = running_max

    adjusted = np.zeros(m, dtype=float)
    for sorted_index, original_index in enumerate(order):
        adjusted[original_index] = adjusted_sorted[sorted_index]
    return tuple(float(value) for value in adjusted)


def _row_passes_thresholds(
    row: MultiSymbolStrategyRow,
    *,
    corrected_in_sample_p: float,
    corrected_walk_forward_p: float,
    familywise_alpha: float,
    min_pass_rate: float,
    min_stability: float,
    min_avg_sharpe: float,
    max_abs_drawdown: float,
    require_positive_avg_return: bool,
) -> bool:
    """Evaluate strict stability criteria for one aggregated strategy row."""

    if corrected_in_sample_p > familywise_alpha:
        return False
    if corrected_walk_forward_p > familywise_alpha:
        return False
    if row.validation_pass_rate < min_pass_rate:
        return False
    if row.avg_walk_forward_stability < min_stability:
        return False
    if row.avg_sharpe_ratio < min_avg_sharpe:
        return False
    if abs(row.avg_max_drawdown) > max_abs_drawdown:
        return False
    if require_positive_avg_return and row.avg_total_return <= 0.0:
        return False
    return True


def select_stable_donchian_rows(
    rows: Iterable[MultiSymbolStrategyRow],
    *,
    familywise_alpha: float = 0.05,
    min_pass_rate: float = 0.60,
    min_stability: float = 0.55,
    min_avg_sharpe: float = 0.25,
    max_abs_drawdown: float = 0.25,
    require_positive_avg_return: bool = True,
) -> Tuple[MultiSymbolStrategyRow, ...]:
    """Filter ranked Donchian rows into strict stability-approved candidates."""

    if not 0.0 < familywise_alpha < 1.0:
        raise ResearchInputError("familywise_alpha must be in (0,1)")
    if not 0.0 <= min_pass_rate <= 1.0:
        raise ResearchInputError("min_pass_rate must be in [0,1]")
    if not 0.0 <= min_stability <= 1.0:
        raise ResearchInputError("min_stability must be in [0,1]")
    if max_abs_drawdown <= 0.0:
        raise ResearchInputError("max_abs_drawdown must be > 0")

    row_tuple = tuple(rows)
    if not row_tuple:
        return ()
    is_p = tuple(row.avg_in_sample_p_value for row in row_tuple)
    wf_p = tuple(row.avg_walk_forward_p_value for row in row_tuple)
    is_holm = holm_adjust_p_values(is_p)
    wf_holm = holm_adjust_p_values(wf_p)

    stable = []
    for row, is_adj, wf_adj in zip(row_tuple, is_holm, wf_holm):
        if _row_passes_thresholds(
            row,
            corrected_in_sample_p=is_adj,
            corrected_walk_forward_p=wf_adj,
            familywise_alpha=familywise_alpha,
            min_pass_rate=min_pass_rate,
            min_stability=min_stability,
            min_avg_sharpe=min_avg_sharpe,
            max_abs_drawdown=max_abs_drawdown,
            require_positive_avg_return=require_positive_avg_return,
        ):
            stable.append(row)
    return tuple(stable)
