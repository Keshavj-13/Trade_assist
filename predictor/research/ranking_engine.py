"""Cross-sectional ranking engine for next-day open-entry/close-exit research.

Pipeline
--------
market_data -> feature_generation -> daily_feature_snapshot -> ranking_model
-> cross_sectional_scores -> top_k_selection -> open_entry -> close_exit -> evaluation
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Tuple

import numpy as np
import pandas as pd

from predictor.research.errors import ResearchInputError
from predictor.research.factors import RankingFactor
from predictor.research.normalization import zscore_cross_section
from predictor.research.ranking import (
    IntradayExecutionAssumptions,
    RankingMetrics,
    evaluate_ranking,
)


@dataclass(frozen=True)
class DailyFeatureSnapshot:
    """Feature matrix for one decision date.

    `features` has index=symbol and columns=factor names.
    """

    date: pd.Timestamp
    features: pd.DataFrame


@dataclass(frozen=True)
class DailyRanking:
    """Ranking output for one decision date."""

    date: pd.Timestamp
    scores: pd.Series
    ranks: pd.Series
    selected_symbols: Tuple[str, ...]


@dataclass(frozen=True)
class RankingEngineResult:
    """End-to-end output of the ranking engine."""

    feature_snapshots: Tuple[DailyFeatureSnapshot, ...]
    daily_rankings: Tuple[DailyRanking, ...]
    score_panel: pd.DataFrame
    portfolio_returns: pd.Series
    metrics: RankingMetrics


def build_daily_feature_snapshots(
    symbol_data: Dict[str, pd.DataFrame],
    factors: Mapping[str, RankingFactor],
) -> Tuple[DailyFeatureSnapshot, ...]:
    """Compute daily per-symbol feature snapshots from factor definitions."""
    if not symbol_data:
        raise ResearchInputError("symbol_data must not be empty")
    if not factors:
        raise ResearchInputError("factors must not be empty")

    factor_panels: Dict[str, pd.DataFrame] = {}
    for factor_name, factor in factors.items():
        panel = factor.compute_scores(symbol_data)
        if not isinstance(panel, pd.DataFrame):
            raise ResearchInputError(f"factor {factor_name!r} returned non-DataFrame scores")
        factor_panels[factor_name] = panel

    all_dates = sorted(list(set().union(*(panel.index for panel in factor_panels.values()))))
    snapshots: list[DailyFeatureSnapshot] = []

    for date in all_dates:
        per_factor: Dict[str, pd.Series] = {}
        for factor_name, panel in factor_panels.items():
            if date not in panel.index:
                continue
            per_factor[factor_name] = panel.loc[date]

        if not per_factor:
            continue

        feature_df = pd.DataFrame(per_factor).replace([np.inf, -np.inf], np.nan)
        feature_df = feature_df.dropna(how="all")
        if feature_df.empty:
            continue
        snapshots.append(DailyFeatureSnapshot(date=pd.Timestamp(date), features=feature_df))

    return tuple(snapshots)


def rank_snapshot(
    snapshot: DailyFeatureSnapshot,
    *,
    top_k: int,
    factor_weights: Mapping[str, float] | None = None,
) -> DailyRanking:
    """Convert one daily feature snapshot into ranked scores and top-k selection."""
    if top_k <= 0:
        raise ResearchInputError("top_k must be > 0")

    features = snapshot.features.copy()
    if features.empty:
        raise ResearchInputError("snapshot.features must not be empty")

    standardized = pd.DataFrame(
        {col: zscore_cross_section(features[col].fillna(0.0)) for col in features.columns},
        index=features.index,
    )

    if factor_weights is None:
        weights = {col: 1.0 for col in standardized.columns}
    else:
        weights = {col: float(factor_weights.get(col, 0.0)) for col in standardized.columns}

    score_series = pd.Series(0.0, index=standardized.index, dtype=float)
    for col, weight in weights.items():
        score_series = score_series + (standardized[col] * weight)

    ranked = score_series.sort_values(ascending=False)
    actual_k = min(top_k, len(ranked))
    selected = tuple(str(sym) for sym in ranked.index[:actual_k])
    rank_series = ranked.rank(method="first", ascending=False)

    return DailyRanking(
        date=snapshot.date,
        scores=ranked,
        ranks=rank_series,
        selected_symbols=selected,
    )


def generate_daily_rankings(
    snapshots: Tuple[DailyFeatureSnapshot, ...],
    *,
    top_k: int,
    factor_weights: Mapping[str, float] | None = None,
) -> Tuple[DailyRanking, ...]:
    """Rank all snapshots and produce deterministic daily top-k selections."""
    return tuple(
        rank_snapshot(snapshot, top_k=top_k, factor_weights=factor_weights)
        for snapshot in snapshots
    )


def rankings_to_score_panel(rankings: Tuple[DailyRanking, ...]) -> pd.DataFrame:
    """Convert daily ranking objects to a score panel for evaluation."""
    if not rankings:
        return pd.DataFrame()
    rows = {ranking.date: ranking.scores for ranking in rankings}
    return pd.DataFrame.from_dict(rows, orient="index").sort_index()


def evaluate_daily_rankings(
    rankings: Tuple[DailyRanking, ...],
    targets: pd.DataFrame,
    *,
    execution: IntradayExecutionAssumptions,
    top_k: int,
    regimes: pd.Series | None = None,
) -> Tuple[pd.Series, RankingMetrics]:
    """Evaluate open-entry/close-exit outcomes for precomputed daily rankings."""
    scores = rankings_to_score_panel(rankings)
    return evaluate_ranking(
        scores=scores,
        targets=targets,
        top_k=top_k,
        transaction_cost_bps=execution.transaction_cost_bps,
        slippage_bps=execution.entry_slippage_bps + execution.exit_slippage_bps,
        regimes=regimes,
    )


def run_ranking_engine(
    symbol_data: Dict[str, pd.DataFrame],
    factors: Mapping[str, RankingFactor],
    targets: pd.DataFrame,
    *,
    top_k: int,
    execution: IntradayExecutionAssumptions,
    factor_weights: Mapping[str, float] | None = None,
    regimes: pd.Series | None = None,
) -> RankingEngineResult:
    """Execute feature snapshots -> ranking -> top-k -> next-day evaluation."""
    snapshots = build_daily_feature_snapshots(symbol_data, factors)
    daily_rankings = generate_daily_rankings(
        snapshots,
        top_k=top_k,
        factor_weights=factor_weights,
    )
    score_panel = rankings_to_score_panel(daily_rankings)
    portfolio_returns, metrics = evaluate_daily_rankings(
        daily_rankings,
        targets,
        execution=execution,
        top_k=top_k,
        regimes=regimes,
    )
    return RankingEngineResult(
        feature_snapshots=snapshots,
        daily_rankings=daily_rankings,
        score_panel=score_panel,
        portfolio_returns=portfolio_returns,
        metrics=metrics,
    )
