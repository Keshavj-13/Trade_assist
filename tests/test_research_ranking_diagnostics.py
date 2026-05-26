"""Tests for ranking diagnostics plotting and exports."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from predictor.research.cross_sectional import CrossSectionalPermutationResult
from predictor.research.ranking_diagnostics import (
    export_ranking_diagnostics_json,
    export_ranking_diagnostics_parquet,
    plot_cumulative_top_k_return,
    plot_factor_distributions,
    plot_prediction_vs_realized,
    plot_ranking_distribution,
    plot_ranking_permutation_histogram,
    plot_ranking_regime_overlay,
    plot_rolling_ic,
)


def _scores_targets():
    dates = pd.date_range("2026-01-01", periods=60, freq="B")
    symbols = ["A", "B", "C", "D"]
    rng = np.random.default_rng(42)
    scores = pd.DataFrame(rng.normal(size=(60, 4)), index=dates, columns=symbols)
    targets = pd.DataFrame(rng.normal(0.001, 0.01, size=(60, 4)), index=dates, columns=symbols)
    return scores, targets


def test_ranking_plots_and_exports(tmp_path: Path):
    scores, targets = _scores_targets()
    ic = scores.corrwith(targets, axis=1, method="spearman")
    returns = targets.mean(axis=1)
    regimes = pd.Series(["SIDEWAYS", "BULL", "HIGH_VOL"] * 20, index=targets.index)

    perm = CrossSectionalPermutationResult(
        observed_ic=0.05,
        observed_sharpe=0.7,
        ic_p_value=0.03,
        sharpe_p_value=0.08,
        null_ic=tuple(np.random.default_rng(1).normal(0.0, 0.02, size=120)),
        null_sharpe=tuple(np.random.default_rng(2).normal(0.0, 0.5, size=120)),
        passes_ic=True,
        passes_sharpe=False,
    )

    for path, fn in (
        (tmp_path / "ranking_dist.png", lambda p: plot_ranking_distribution(scores, output_path=p)),
        (tmp_path / "factor_dist.png", lambda p: plot_factor_distributions(scores, output_path=p)),
        (tmp_path / "scatter.png", lambda p: plot_prediction_vs_realized(scores, targets, output_path=p)),
        (tmp_path / "perm_hist.png", lambda p: plot_ranking_permutation_histogram(perm, output_path=p)),
        (tmp_path / "rolling_ic.png", lambda p: plot_rolling_ic(ic, output_path=p)),
        (tmp_path / "cum.png", lambda p: plot_cumulative_top_k_return(returns, output_path=p)),
        (tmp_path / "regime_overlay.png", lambda p: plot_ranking_regime_overlay(returns, regimes, output_path=p)),
    ):
        out = fn(path)
        if out is not None:
            assert path.exists()

    json_path = export_ranking_diagnostics_json({"mean_ic": float(ic.mean())}, output_path=tmp_path / "diag.json")
    assert json_path.exists()

    parquet_path = export_ranking_diagnostics_parquet(
        pd.DataFrame({"ic": ic.values}),
        output_path=tmp_path / "diag.parquet",
    )
    if parquet_path is not None:
        assert parquet_path.exists()
