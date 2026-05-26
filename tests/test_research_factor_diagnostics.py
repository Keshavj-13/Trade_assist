"""Tests for factor degeneracy diagnostics and snapshot persistence."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from predictor.research.factor_diagnostics import (
    compute_factor_dispersion_metrics,
    extract_degeneracy_warnings,
)
from predictor.research.factor_snapshots import (
    build_factor_snapshot_table,
    export_factor_snapshot_bundle,
)


def test_compute_factor_dispersion_metrics_detects_degeneracy():
    dates = pd.to_datetime(["2026-01-01", "2026-01-02"])
    scores = pd.DataFrame(
        [[1.0, 1.0, 1.0], [1.0, 2.0, 3.0]],
        index=dates,
        columns=["A", "B", "C"],
    )

    metrics = compute_factor_dispersion_metrics(scores, top_k=2)
    assert set(metrics.columns) >= {
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
    }

    day1 = metrics.iloc[0]
    day2 = metrics.iloc[1]

    assert day1["cross_sectional_std"] == 0.0
    assert day1["unique_rank_count"] == 1
    assert day1["factor_entropy"] == 0.0
    assert day1["top_k_changes"] == 0
    assert day1["rank_turnover"] == 0.0
    assert day1["z_score_dispersion"] == 0.0

    assert day2["cross_sectional_std"] > 0.0
    assert day2["unique_rank_count"] >= 2
    assert day2["factor_entropy"] > 0.0
    assert day2["top_k_changes"] >= 1
    assert day2["rank_turnover"] > 0.0

    warnings = extract_degeneracy_warnings(metrics, factor_name="overnight_gap")
    assert any("[DEGENERACY WARNING]" in w for w in warnings)
    assert any("factor=overnight_gap" in w for w in warnings)


def test_build_and_export_factor_snapshot_bundle(tmp_path: Path):
    dates = pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03"])
    symbols = ["A", "B", "C"]

    scores = pd.DataFrame(
        [[0.0, 1.0, 2.0], [2.0, 1.0, 0.0], [1.0, 1.5, 1.2]],
        index=dates,
        columns=symbols,
    )
    targets = pd.DataFrame(
        [[0.01, 0.02, 0.03], [0.01, -0.01, 0.00], [0.02, 0.01, 0.04]],
        index=dates,
        columns=symbols,
    )
    regimes = pd.Series(["TRENDING", "SIDEWAYS", "HIGH_VOL"], index=dates)

    table = build_factor_snapshot_table(
        factor_name="momentum_20",
        scores=scores,
        targets=targets,
        regimes=regimes,
        top_k=1,
    )

    assert set(table.columns) == {
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
    }
    assert len(table) == len(dates) * len(symbols)
    assert table["selected_top_k"].sum() == len(dates)

    paths = export_factor_snapshot_bundle(
        table,
        output_dir=tmp_path,
        factor_name="momentum_20",
        include_json_summary=True,
    )

    assert paths["csv"].exists()
    assert paths["parquet"] is None or paths["parquet"].exists()
    assert paths["json_summary"] is None or paths["json_summary"].exists()
