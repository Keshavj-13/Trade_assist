"""Daily factor-health diagnostics for cross-sectional degeneracy detection."""

from __future__ import annotations

import math
from typing import Iterable, List

import numpy as np
import pandas as pd

from predictor.research.errors import ResearchInputError
from predictor.research.normalization import zscore_cross_section


def _shannon_entropy(probabilities: Iterable[float]) -> float:
    probs = [float(p) for p in probabilities if p > 0.0]
    if not probs:
        return 0.0
    return float(-sum(p * math.log(p) for p in probs))


def compute_factor_dispersion_metrics(
    scores: pd.DataFrame,
    *,
    top_k: int,
) -> pd.DataFrame:
    """Compute daily cross-sectional dispersion and degeneracy metrics.

    Columns returned:
    - date
    - cross_sectional_std
    - cross_sectional_mean
    - unique_rank_count
    - nan_ratio
    - factor_entropy
    - top_k_changes
    - rank_turnover
    - z_score_dispersion
    - factor_min
    - factor_max
    - degeneracy_warning
    """
    if top_k <= 0:
        raise ResearchInputError("top_k must be > 0")

    if scores is None or not isinstance(scores, pd.DataFrame):
        raise ResearchInputError("scores must be a DataFrame")

    if scores.empty:
        return pd.DataFrame(
            columns=[
                "date",
                "cross_sectional_std",
                "cross_sectional_mean",
                "unique_rank_count",
                "nan_ratio",
                "factor_entropy",
                "top_k_changes",
                "rank_turnover",
                "z_score_dispersion",
                "factor_min",
                "factor_max",
                "degeneracy_warning",
            ]
        )

    rows: list[dict[str, object]] = []
    prev_top: set[str] = set()

    for date in scores.index:
        raw = scores.loc[date]
        total_assets = len(raw)
        valid = raw.dropna().astype(float)
        nan_ratio = float(1.0 - (len(valid) / max(total_assets, 1)))

        if valid.empty:
            row = {
                "date": pd.Timestamp(date),
                "cross_sectional_std": 0.0,
                "cross_sectional_mean": 0.0,
                "unique_rank_count": 0,
                "nan_ratio": 1.0,
                "factor_entropy": 0.0,
                "top_k_changes": 0,
                "rank_turnover": 0.0,
                "z_score_dispersion": 0.0,
                "factor_min": 0.0,
                "factor_max": 0.0,
                "degeneracy_warning": True,
            }
            rows.append(row)
            prev_top = set()
            continue

        ranked = valid.sort_values(ascending=False)
        actual_k = min(top_k, len(ranked))
        current_top = set(str(symbol) for symbol in ranked.index[:actual_k])
        top_k_changes = len(current_top - prev_top) if prev_top else 0
        rank_turnover = float(top_k_changes / max(actual_k, 1))

        rank_series = valid.rank(method="dense", ascending=False)
        unique_rank_count = int(rank_series.nunique())
        rank_probs = rank_series.value_counts(normalize=True)
        factor_entropy = _shannon_entropy(rank_probs.values)

        z_scores = zscore_cross_section(valid)
        z_score_dispersion = float(z_scores.std(ddof=0)) if len(z_scores) else 0.0

        std = float(valid.std(ddof=0)) if len(valid) else 0.0
        min_v = float(valid.min()) if len(valid) else 0.0
        max_v = float(valid.max()) if len(valid) else 0.0
        warning = bool(unique_rank_count <= 1 or std <= 0.0 or np.isclose(min_v, max_v))

        rows.append(
            {
                "date": pd.Timestamp(date),
                "cross_sectional_std": std,
                "cross_sectional_mean": float(valid.mean()) if len(valid) else 0.0,
                "unique_rank_count": unique_rank_count,
                "nan_ratio": nan_ratio,
                "factor_entropy": factor_entropy,
                "top_k_changes": int(top_k_changes),
                "rank_turnover": rank_turnover,
                "z_score_dispersion": z_score_dispersion,
                "factor_min": min_v,
                "factor_max": max_v,
                "degeneracy_warning": warning,
            }
        )
        prev_top = current_top

    return pd.DataFrame(rows)


def extract_degeneracy_warnings(metrics: pd.DataFrame, *, factor_name: str) -> List[str]:
    """Return formatted warning lines for degenerate daily cross-sections."""
    if metrics.empty:
        return []

    required = {"date", "unique_rank_count", "cross_sectional_std", "degeneracy_warning"}
    missing = required - set(metrics.columns)
    if missing:
        raise ResearchInputError(f"metrics missing required columns: {sorted(missing)}")

    warnings: list[str] = []
    flagged = metrics[metrics["degeneracy_warning"] == True]  # noqa: E712
    for _, row in flagged.iterrows():
        warnings.append(
            "[DEGENERACY WARNING] "
            f"factor={factor_name} "
            f"date={pd.Timestamp(row['date']).date()} "
            f"unique_rank_count={int(row['unique_rank_count'])} "
            f"cross_sectional_std={float(row['cross_sectional_std']):.8f}"
        )
    return warnings
