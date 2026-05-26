"""Tests for scripts/analyze_factor_correlations.py."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import scripts.analyze_factor_correlations as script


def _snapshot_rows(factor_name: str, values: pd.DataFrame, daily_ic: pd.Series) -> pd.DataFrame:
    rows = []
    for date in values.index:
        for symbol in values.columns:
            rows.append(
                {
                    "date": date,
                    "symbol": symbol,
                    "factor_name": factor_name,
                    "raw_factor": float(values.loc[date, symbol]),
                    "z_score": 0.0,
                    "rank": 1.0,
                    "selected_top_k": False,
                    "future_return": 0.0,
                    "daily_ic": float(daily_ic.loc[date]),
                    "regime_label": "SIDEWAYS",
                }
            )
    return pd.DataFrame(rows)


def test_analyze_factor_correlations_exports_outputs(monkeypatch, tmp_path: Path) -> None:
    dates = pd.date_range("2026-01-01", periods=20, freq="B")
    symbols = ["A", "B", "C", "D", "E"]

    base = np.linspace(-1.0, 1.0, len(symbols))
    momentum = pd.DataFrame(
        [base + i * 0.05 for i in range(len(dates))],
        index=dates,
        columns=symbols,
    )
    gap = momentum * 0.9 + 0.1

    momentum_ic = pd.Series(np.linspace(0.01, 0.05, len(dates)), index=dates)
    gap_ic = pd.Series(np.linspace(0.012, 0.048, len(dates)), index=dates)

    input_dir = tmp_path / "snapshots"
    input_dir.mkdir(parents=True, exist_ok=True)

    _snapshot_rows("momentum_20", momentum, momentum_ic).to_csv(
        input_dir / "momentum_20_daily_snapshots.csv",
        index=False,
    )
    _snapshot_rows("overnight_gap", gap, gap_ic).to_csv(
        input_dir / "overnight_gap_daily_snapshots.csv",
        index=False,
    )

    output_dir = tmp_path / "correlations"
    argv = [
        "analyze_factor_correlations.py",
        "--input-dir",
        str(input_dir),
        "--output-dir",
        str(output_dir),
        "--rolling-window",
        "5",
        "--redundancy-threshold",
        "0.8",
    ]
    monkeypatch.setattr("sys.argv", argv)

    script.main()

    pearson_csv = output_dir / "static_pearson_correlation.csv"
    spearman_csv = output_dir / "static_spearman_correlation.csv"
    ic_csv = output_dir / "ic_correlation_matrix.csv"
    rolling_csv = output_dir / "rolling_pearson_correlation.csv"
    report_json = output_dir / "factor_redundancy_report.json"

    assert pearson_csv.exists()
    assert spearman_csv.exists()
    assert ic_csv.exists()
    assert rolling_csv.exists()
    assert report_json.exists()

    pearson = pd.read_csv(pearson_csv, index_col=0)
    assert float(pearson.loc["momentum_20", "overnight_gap"]) > 0.95

    report = json.loads(report_json.read_text())
    assert "static" in report
    assert "rolling" in report
    assert isinstance(report["static"]["pearson"], list)
