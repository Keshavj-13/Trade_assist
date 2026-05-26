"""Unit tests for the cross-sectional factor backtesting, permutation, and walk-forward harnesses."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from predictor.research.cross_sectional import (
    CrossSectionalResearchConfig,
    CrossSectionalValidationReport,
    joint_block_permute_targets,
    run_cross_sectional_backtest,
    run_cross_sectional_permutation,
    run_cross_sectional_walk_forward,
    validate_factor,
)
from predictor.research.factors import MarketBenchmarkFactor, RandomRankingFactor


def test_joint_block_permute_targets():
    """Verify that joint_block_permute_targets shuffles targets while maintaining asset correlation."""
    dates = pd.date_range("2026-01-01", periods=100)
    symbols = ["A", "B"]
    
    # Highly correlated targets
    x = np.linspace(0, 10, 100)
    targets = pd.DataFrame(
        {
            "A": np.sin(x),
            "B": np.sin(x) + 0.1,
        },
        index=dates,
    )
    
    rng = np.random.default_rng(42)
    shuffled = joint_block_permute_targets(targets, block_size=10, rng=rng)
    
    assert isinstance(shuffled, pd.DataFrame)
    assert set(shuffled.columns) == {"A", "B"}
    assert len(shuffled) == 100
    
    # Check that correlation is perfectly preserved (shuffled jointly)
    orig_corr = targets["A"].corr(targets["B"])
    shuffled_corr = shuffled["A"].corr(shuffled["B"])
    assert abs(orig_corr - shuffled_corr) < 1e-7
    
    # Shuffled timeline should differ from original
    assert not (shuffled["A"] == targets["A"]).all()


def test_run_cross_sectional_backtest(make_ohlcv_frame):
    """Verify that run_cross_sectional_backtest completes successfully and returns portfolio returns."""
    symbol_data = {
        "TCS": make_ohlcv_frame(rows=50),
        "INFY": make_ohlcv_frame(rows=50),
    }
    
    factor = MarketBenchmarkFactor(name="market")
    config = CrossSectionalResearchConfig(top_k=1, transaction_cost_bps=5.0)
    
    port_rets, metrics = run_cross_sectional_backtest(factor, symbol_data, config)
    
    assert isinstance(port_rets, pd.Series)
    assert len(port_rets) == 50
    assert metrics.sharpe_ratio is not None


def test_run_cross_sectional_permutation(make_ohlcv_frame):
    """Verify that run_cross_sectional_permutation correctly computes null distributions and p-values."""
    symbol_data = {
        "TCS": make_ohlcv_frame(rows=60),
        "INFY": make_ohlcv_frame(rows=60),
    }
    
    factor = RandomRankingFactor(name="rand")
    config = CrossSectionalResearchConfig(top_k=1, permutation_iterations=20, random_seed=42)
    
    _, observed_metrics = run_cross_sectional_backtest(factor, symbol_data, config)
    perm_result = run_cross_sectional_permutation(factor, symbol_data, observed_metrics, config)
    
    assert len(perm_result.null_ic) == 20
    assert len(perm_result.null_sharpe) == 20
    assert 0.0 <= perm_result.ic_p_value <= 1.0
    assert 0.0 <= perm_result.sharpe_p_value <= 1.0


def test_run_cross_sectional_walk_forward(make_ohlcv_frame):
    """Verify that run_cross_sectional_walk_forward slides walk-forward folds and returns OOS metrics."""
    symbol_data = {
        "TCS": make_ohlcv_frame(rows=360),
        "INFY": make_ohlcv_frame(rows=360),
    }
    
    factor = MarketBenchmarkFactor(name="market")
    config = CrossSectionalResearchConfig(
        top_k=1,
        train_window=200,
        test_window=50,
        walk_forward_step=50,
    )
    
    folds, wf_metrics = run_cross_sectional_walk_forward(factor, symbol_data, config)
    
    # 200 + 50 = 250 start. dates=360 => folds = 3
    # Fold 0: test 200-249
    # Fold 1: test 250-299
    # Fold 2: test 300-349
    assert len(folds) == 3
    assert folds[0].fold_index == 0
    assert folds[0].out_of_sample_metrics.mean_ic is not None
    assert wf_metrics.sharpe_ratio is not None


def test_validate_factor(make_ohlcv_frame):
    """Verify that validate_factor executes full harness and yields a comprehensive CS Validation Report."""
    symbol_data = {
        "TCS": make_ohlcv_frame(rows=320),
        "INFY": make_ohlcv_frame(rows=320),
    }
    
    factor = MarketBenchmarkFactor(name="market")
    config = CrossSectionalResearchConfig(
        top_k=1,
        permutation_iterations=10,
        train_window=200,
        test_window=50,
        walk_forward_step=50,
    )
    
    report = validate_factor(factor, symbol_data, config)
    
    assert isinstance(report, CrossSectionalValidationReport)
    assert report.factor_name == "market"
    assert len(report.walk_forward_folds) == 2
    assert report.full_sample_metrics.sharpe_ratio is not None
    assert report.permutation_result.ic_p_value is not None
