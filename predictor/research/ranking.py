"""Portfolio construction and evaluation for cross-sectional next-day ranking."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from predictor.research.errors import ResearchInputError
from predictor.research.regime import classify_regimes


def _coarse_regime_label(label: object) -> str:
    raw = str(label).upper()
    if raw == "CRASH":
        return "CRISIS"
    if raw in {"BULL", "BEAR"}:
        return "TRENDING"
    if raw in {"HIGH_VOL", "LOW_VOL", "RECOVERY", "SIDEWAYS"}:
        return raw
    return "SIDEWAYS"


@dataclass(frozen=True)
class RankingMetrics:
    """Evaluation metrics aligned with cross-sectional ranking objectives."""

    mean_ic: float
    ic_std: float
    ic_t_stat: float
    mean_selected_return: float
    annualised_return: float
    sharpe_ratio: float
    max_drawdown: float
    precision_at_k: float
    top_1_hit_rate: float
    mean_turnover: float
    turnover_adjusted_return: float
    regime_ic: Dict[str, float]
    top_k_mean_return: float = 0.0
    rank_correlation: float = 0.0
    information_coefficient: float = 0.0
    average_selected_return: float = 0.0
    intraday_drawdown: float = 0.0
    regime_stability: float = 0.0


@dataclass(frozen=True)
class IntradayExecutionAssumptions:
    """Execution assumptions for open-entry/close-exit next-day portfolios."""

    transaction_cost_bps: float = 5.0
    entry_slippage_bps: float = 0.0
    exit_slippage_bps: float = 0.0


def _empty_metrics() -> RankingMetrics:
    return RankingMetrics(
        mean_ic=0.0,
        ic_std=0.0,
        ic_t_stat=0.0,
        mean_selected_return=0.0,
        annualised_return=0.0,
        sharpe_ratio=0.0,
        max_drawdown=0.0,
        precision_at_k=0.0,
        top_1_hit_rate=0.0,
        mean_turnover=0.0,
        turnover_adjusted_return=0.0,
        regime_ic={},
        top_k_mean_return=0.0,
        rank_correlation=0.0,
        information_coefficient=0.0,
        average_selected_return=0.0,
        intraday_drawdown=0.0,
        regime_stability=0.0,
    )


def compute_next_day_returns(symbol_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Build next-day intraday target aligned to decision date.

    target_t = (Close_{t+1} - Open_{t+1}) / Open_{t+1}
    """
    if not symbol_data:
        raise ResearchInputError("symbol_data must not be empty")

    returns_dict: Dict[str, pd.Series] = {}
    for symbol, df in symbol_data.items():
        if "Open" not in df.columns or "Close" not in df.columns:
            raise ResearchInputError(f"symbol {symbol!r} is missing Open/Close columns")
        intraday_ret = (df["Close"].astype(float) - df["Open"].astype(float)) / df["Open"].astype(float)
        returns_dict[symbol] = intraday_ret.shift(-1)

    all_dates = sorted(list(set().union(*(df.index for df in symbol_data.values()))))
    return pd.DataFrame(returns_dict, index=all_dates)


def compute_daily_regimes(symbol_data: Dict[str, pd.DataFrame]) -> pd.Series:
    """Classify regime from equal-weighted close-to-close market proxy."""
    if not symbol_data:
        raise ResearchInputError("symbol_data must not be empty")

    all_dates = sorted(list(set().union(*(df.index for df in symbol_data.values()))))
    daily_returns: Dict[str, pd.Series] = {}
    for symbol, df in symbol_data.items():
        if "Close" not in df.columns:
            raise ResearchInputError(f"symbol {symbol!r} is missing Close column")
        daily_returns[symbol] = df["Close"].astype(float).pct_change()

    ret_df = pd.DataFrame(daily_returns, index=all_dates)
    market_proxy = ret_df.mean(axis=1).fillna(0.0)
    synthetic_close = 100.0 * (1.0 + market_proxy).cumprod()

    proxy_frame = pd.DataFrame(
        {
            "Open": synthetic_close,
            "High": synthetic_close * 1.001,
            "Low": synthetic_close * 0.999,
            "Close": synthetic_close,
            "Volume": 1_000_000.0,
        },
        index=market_proxy.index,
    )
    labels = classify_regimes(proxy_frame)
    return labels.map(_coarse_regime_label)


def evaluate_ranking(
    scores: pd.DataFrame,
    targets: pd.DataFrame,
    *,
    top_k: int = 5,
    transaction_cost_bps: float = 5.0,
    slippage_bps: float = 0.0,
    regimes: pd.Series | None = None,
) -> Tuple[pd.Series, RankingMetrics]:
    """Evaluate next-day ranking quality under open-entry/close-exit assumptions."""
    if top_k <= 0:
        raise ResearchInputError("top_k must be > 0")
    if transaction_cost_bps < 0 or slippage_bps < 0:
        raise ResearchInputError("transaction_cost_bps and slippage_bps must be >= 0")

    common_dates = scores.index.intersection(targets.index)
    if len(common_dates) == 0:
        return pd.Series(dtype=float), _empty_metrics()

    aligned_scores = scores.loc[common_dates]
    aligned_targets = targets.loc[common_dates]

    daily_portfolio_returns: list[float] = []
    daily_ic: list[float] = []
    daily_precision: list[float] = []
    daily_top_1_positive: list[float] = []
    daily_turnover: list[float] = []
    daily_has_trade: list[float] = []
    prev_portfolio: set[str] = set()

    for idx, date in enumerate(common_dates):
        daily_scores = aligned_scores.loc[date].dropna()
        daily_targets = aligned_targets.loc[date].dropna()
        valid_assets = daily_scores.index.intersection(daily_targets.index)

        if len(valid_assets) == 0:
            daily_portfolio_returns.append(0.0)
            daily_ic.append(0.0)
            daily_precision.append(0.0)
            daily_top_1_positive.append(0.0)
            daily_turnover.append(0.0)
            daily_has_trade.append(0.0)
            continue

        s = daily_scores.loc[valid_assets]
        t = daily_targets.loc[valid_assets]

        if len(valid_assets) >= 3 and float(s.std()) > 0.0 and float(t.std()) > 0.0:
            ic = float(s.corr(t, method="spearman"))
        else:
            ic = 0.0
        daily_ic.append(ic)

        ranked = s.sort_values(ascending=False)
        actual_k = min(top_k, len(ranked))
        if actual_k == 0:
            daily_portfolio_returns.append(0.0)
            daily_precision.append(0.0)
            daily_top_1_positive.append(0.0)
            daily_turnover.append(0.0)
            daily_has_trade.append(0.0)
            continue

        top_assets = ranked.index[:actual_k]
        portfolio = set(str(a) for a in top_assets)
        port_ret = float(t.loc[top_assets].mean())
        daily_portfolio_returns.append(port_ret)
        daily_precision.append(float((t.loc[top_assets] > 0).mean()))
        daily_top_1_positive.append(1.0 if float(t.loc[top_assets[0]]) > 0.0 else 0.0)
        daily_has_trade.append(1.0)

        if idx > 0 and prev_portfolio:
            added = portfolio - prev_portfolio
            daily_turnover.append(len(added) / max(len(portfolio), 1))
        else:
            daily_turnover.append(0.0)
        prev_portfolio = portfolio

    port_rets_series = pd.Series(daily_portfolio_returns, index=common_dates, dtype=float)
    ic_series = pd.Series(daily_ic, index=common_dates, dtype=float)
    precision_series = pd.Series(daily_precision, index=common_dates, dtype=float)
    top_1_series = pd.Series(daily_top_1_positive, index=common_dates, dtype=float)
    turnover_series = pd.Series(daily_turnover, index=common_dates, dtype=float)
    traded_series = pd.Series(daily_has_trade, index=common_dates, dtype=float)

    mean_ic = float(ic_series.mean()) if len(ic_series) else 0.0
    ic_std = float(ic_series.std(ddof=1)) if len(ic_series) > 1 else 0.0
    ic_t_stat = float(mean_ic / (ic_std / np.sqrt(len(ic_series)))) if ic_std > 0 else 0.0

    mean_selected = float(port_rets_series.mean()) if len(port_rets_series) else 0.0
    annualised_ret = mean_selected * 252.0
    daily_std = float(port_rets_series.std(ddof=1)) if len(port_rets_series) > 1 else 0.0
    sharpe = float((mean_selected / daily_std) * np.sqrt(252.0)) if daily_std > 0 else 0.0

    cum_ret = (port_rets_series + 1.0).cumprod()
    if len(cum_ret):
        running_max = cum_ret.cummax()
        drawdown = (cum_ret / running_max) - 1.0
        max_dd = float(drawdown.min())
    else:
        max_dd = 0.0

    mean_turnover = float(turnover_series.mean()) if len(turnover_series) else 0.0

    # Open-entry/close-exit always incurs a round-trip cost on traded days.
    round_trip_cost = (2.0 * transaction_cost_bps + slippage_bps) * 0.0001
    daily_execution_cost = traded_series * round_trip_cost
    # Turnover penalty remains as a stability-sensitive adjustment signal.
    daily_turnover_cost = turnover_series * (transaction_cost_bps * 0.0001)
    adjusted_returns = port_rets_series - daily_execution_cost - daily_turnover_cost
    turnover_adj_annualised = float(adjusted_returns.mean() * 252.0) if len(adjusted_returns) else 0.0

    regime_ic_map: Dict[str, float] = {}
    regime_stability = 0.0
    if regimes is not None and len(ic_series):
        aligned_regimes = regimes.reindex(ic_series.index)
        for regime_name, group in ic_series.groupby(aligned_regimes):
            regime_ic_map[str(regime_name)] = float(group.mean())
        if len(regime_ic_map) > 1:
            vals = np.asarray(list(regime_ic_map.values()), dtype=float)
            regime_stability = float(1.0 / (1.0 + vals.std(ddof=0)))

    metrics = RankingMetrics(
        mean_ic=mean_ic,
        ic_std=ic_std,
        ic_t_stat=ic_t_stat,
        mean_selected_return=mean_selected,
        annualised_return=annualised_ret,
        sharpe_ratio=sharpe,
        max_drawdown=max_dd,
        precision_at_k=float(precision_series.mean()) if len(precision_series) else 0.0,
        top_1_hit_rate=float(top_1_series.mean()) if len(top_1_series) else 0.0,
        mean_turnover=mean_turnover,
        turnover_adjusted_return=turnover_adj_annualised,
        regime_ic=regime_ic_map,
        top_k_mean_return=mean_selected,
        rank_correlation=mean_ic,
        information_coefficient=mean_ic,
        average_selected_return=mean_selected,
        intraday_drawdown=max_dd,
        regime_stability=regime_stability,
    )
    return port_rets_series, metrics
