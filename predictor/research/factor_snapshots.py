"""Persistence helpers for daily factor snapshot tables."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from predictor.research.errors import ResearchInputError
from predictor.research.normalization import zscore_cross_section


def build_factor_snapshot_table(
    *,
    factor_name: str,
    scores: pd.DataFrame,
    targets: pd.DataFrame,
    regimes: pd.Series,
    top_k: int,
) -> pd.DataFrame:
    """Build a long-form daily symbol snapshot table for one factor."""
    if not factor_name or not isinstance(factor_name, str):
        raise ResearchInputError("factor_name must be a non-empty string")
    if top_k <= 0:
        raise ResearchInputError("top_k must be > 0")

    common_dates = scores.index.intersection(targets.index)
    if common_dates.empty:
        return pd.DataFrame(
            columns=[
                "date",
                "symbol",
                "factor_name",
                "raw_factor",
                "z_score",
                "rank",
                "selected_top_k",
                "future_return",
                "daily_ic",
                "regime_label",
            ]
        )

    rows: list[dict[str, object]] = []
    for date in common_dates:
        s = scores.loc[date].dropna().astype(float)
        t = targets.loc[date] if date in targets.index else pd.Series(dtype=float)
        if s.empty:
            continue

        z = zscore_cross_section(s)
        ranks = s.rank(method="first", ascending=False)
        top = set(str(sym) for sym in s.sort_values(ascending=False).index[: min(top_k, len(s))])

        aligned_targets = t.reindex(s.index).astype(float)
        if len(s) >= 3 and float(s.std(ddof=0)) > 0 and float(aligned_targets.std(ddof=0)) > 0:
            daily_ic = float(s.corr(aligned_targets, method="spearman"))
        else:
            daily_ic = float("nan")

        regime_label = str(regimes.get(date, "SIDEWAYS"))
        for symbol in s.index:
            rows.append(
                {
                    "date": pd.Timestamp(date),
                    "symbol": str(symbol),
                    "factor_name": factor_name,
                    "raw_factor": float(s.loc[symbol]),
                    "z_score": float(z.loc[symbol]),
                    "rank": float(ranks.loc[symbol]),
                    "selected_top_k": str(symbol) in top,
                    "future_return": float(aligned_targets.loc[symbol]) if pd.notna(aligned_targets.loc[symbol]) else np.nan,
                    "daily_ic": daily_ic,
                    "regime_label": regime_label,
                }
            )

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["date", "rank", "symbol"]).reset_index(drop=True)


def export_factor_snapshot_bundle(
    snapshot_table: pd.DataFrame,
    *,
    output_dir: Path,
    factor_name: str,
    include_json_summary: bool = True,
) -> Dict[str, Path | None]:
    """Persist snapshot bundle to parquet/csv/(optional)json summary."""
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / f"{factor_name}_daily_snapshots.csv"
    snapshot_table.to_csv(csv_path, index=False)

    parquet_path: Path | None = output_dir / f"{factor_name}_daily_snapshots.parquet"
    try:
        import pyarrow  # noqa: F401
        snapshot_table.to_parquet(parquet_path, index=False)
    except ImportError:
        parquet_path = None

    json_path: Path | None = None
    if include_json_summary:
        json_path = output_dir / f"{factor_name}_daily_snapshots_summary.json"
        summary = {
            "factor_name": factor_name,
            "rows": int(len(snapshot_table)),
            "symbols": int(snapshot_table["symbol"].nunique()) if not snapshot_table.empty else 0,
            "date_start": str(snapshot_table["date"].min()) if not snapshot_table.empty else None,
            "date_end": str(snapshot_table["date"].max()) if not snapshot_table.empty else None,
            "selected_top_k_rows": int(snapshot_table["selected_top_k"].sum()) if not snapshot_table.empty else 0,
            "daily_ic_nan_ratio": float(snapshot_table["daily_ic"].isna().mean()) if not snapshot_table.empty else 0.0,
        }
        json_path.write_text(json.dumps(summary, indent=2))

    return {
        "csv": csv_path,
        "parquet": parquet_path,
        "json_summary": json_path,
    }
