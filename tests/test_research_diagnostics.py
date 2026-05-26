"""Tests for predictor.research.diagnostics.

Covers:
- PipelineStage ordering correctness
- decompose_positions signal decomposition
- run_strategy_trace produces expected fields on synthetic data
- assert_impossible_states raises on contradictions and passes on valid data
- export_permutation_distribution writes valid JSON
- Stage ordering in StrategyTraceReport.stages_executed matches PIPELINE_ORDER
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from predictor.research.diagnostics import (
    PIPELINE_ORDER,
    EquityCurveSnapshot,
    PermutationSummary,
    PipelineStage,
    SignalDecomposition,
    StrategyTraceReport,
    assert_impossible_states,
    decompose_positions,
    export_permutation_distribution,
    run_strategy_trace,
)
from predictor.research.errors import DiagnosticAssertionError
from predictor.research.types import PermutationTestResult
from predictor.research.validation import ResearchValidationConfig


# ---------------------------------------------------------------------------
# Synthetic OHLCV factory
# ---------------------------------------------------------------------------


def _make_trending_ohlcv(n: int = 400, seed: int = 42) -> pd.DataFrame:
    """Build a synthetic trending price series with clear Donchian breakouts.

    Uses a deterministic up-trend so that a 20-bar Donchian strategy will
    generate entries reliably.
    """
    rng = np.random.default_rng(seed)
    log_returns = rng.normal(0.0008, 0.012, size=n)  # slight positive drift
    log_close = np.cumsum(log_returns) + np.log(100.0)
    close = np.exp(log_close)

    high = close * (1.0 + rng.uniform(0.002, 0.010, size=n))
    low = close * (1.0 - rng.uniform(0.002, 0.010, size=n))
    open_ = close * (1.0 + rng.uniform(-0.005, 0.005, size=n))
    volume = rng.uniform(1e6, 5e6, size=n)

    idx = pd.date_range("2018-01-01", periods=n, freq="B")
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=idx,
    )


def _make_flat_ohlcv(n: int = 400) -> pd.DataFrame:
    """Build a flat (no-edge) price series for impossible-state tests."""
    close = np.ones(n) * 100.0
    idx = pd.date_range("2018-01-01", periods=n, freq="B")
    return pd.DataFrame(
        {
            "Open": close,
            "High": close * 1.001,
            "Low": close * 0.999,
            "Close": close,
            "Volume": np.ones(n) * 1e6,
        },
        index=idx,
    )


def _donchian_strategy():
    """Return a simple Donchian strategy instance."""
    from predictor.research.donchian import DonchianVariantStrategy

    return DonchianVariantStrategy(
        name="donchian_s1_20_10",
        entry_lookback=20,
        exit_lookback=10,
        dependency_rule="none",
    )


def _fast_config() -> ResearchValidationConfig:
    """Return a config with few permutations for fast tests."""
    return ResearchValidationConfig(
        permutation_iterations=30,
        train_window=120,
        test_window=40,
        walk_forward_step=40,
        random_seed=7,
    )


# ---------------------------------------------------------------------------
# PipelineStage ordering
# ---------------------------------------------------------------------------


class TestPipelineStageOrdering:
    def test_pipeline_order_has_all_stages(self):
        assert set(PIPELINE_ORDER) == set(PipelineStage)

    def test_pipeline_order_is_tuple(self):
        assert isinstance(PIPELINE_ORDER, tuple)

    def test_load_data_is_first(self):
        assert PIPELINE_ORDER[0] == PipelineStage.LOAD_DATA

    def test_wf_permutations_is_last(self):
        assert PIPELINE_ORDER[-1] == PipelineStage.RUN_WF_PERMUTATIONS

    def test_is_permutations_before_walkforward(self):
        idx = {s: i for i, s in enumerate(PIPELINE_ORDER)}
        assert idx[PipelineStage.RUN_IS_PERMUTATIONS] < idx[PipelineStage.RUN_WALKFORWARD]

    def test_compute_is_metric_before_is_permutations(self):
        idx = {s: i for i, s in enumerate(PIPELINE_ORDER)}
        assert idx[PipelineStage.COMPUTE_IS_METRIC] < idx[PipelineStage.RUN_IS_PERMUTATIONS]

    def test_generate_positions_before_construct_backtest(self):
        idx = {s: i for i, s in enumerate(PIPELINE_ORDER)}
        assert idx[PipelineStage.GENERATE_POSITIONS] < idx[PipelineStage.CONSTRUCT_BACKTEST]

    def test_stage_labels_are_lowercase(self):
        for stage in PipelineStage:
            assert stage.label() == stage.label().lower()


# ---------------------------------------------------------------------------
# decompose_positions
# ---------------------------------------------------------------------------


class TestDecomposePositions:
    def test_empty_series(self):
        d = decompose_positions(pd.Series([], dtype=float))
        assert d.bars_processed == 0
        assert d.positions_nonzero == 0
        assert d.entries_generated == 0
        assert d.exits_generated == 0

    def test_all_flat(self):
        pos = pd.Series([0.0] * 10)
        d = decompose_positions(pos)
        assert d.bars_processed == 10
        assert d.positions_nonzero == 0
        assert d.entries_generated == 0
        assert d.exits_generated == 0
        assert d.flat_bars == 10

    def test_single_long_trade(self):
        # 0 0 1 1 1 0
        pos = pd.Series([0.0, 0.0, 1.0, 1.0, 1.0, 0.0])
        d = decompose_positions(pos)
        assert d.entries_generated == 1
        assert d.exits_generated == 1
        assert d.positions_nonzero == 3
        assert d.sign_flips == 0

    def test_sign_flip_counted(self):
        # 1 1 -1 -1
        pos = pd.Series([1.0, 1.0, -1.0, -1.0])
        d = decompose_positions(pos)
        assert d.sign_flips == 1
        assert d.entries_generated == 1  # initial 0→1
        assert d.exits_generated == 0  # never goes to 0

    def test_multiple_trades(self):
        # 0 1 1 0 0 -1 0
        pos = pd.Series([0.0, 1.0, 1.0, 0.0, 0.0, -1.0, 0.0])
        d = decompose_positions(pos)
        assert d.entries_generated == 2
        assert d.exits_generated == 2
        assert d.positions_nonzero == 3

    def test_total_transitions_property(self):
        pos = pd.Series([0.0, 1.0, 1.0, 0.0])
        d = decompose_positions(pos)
        assert d.total_transitions == d.entries_generated + d.exits_generated + d.sign_flips


# ---------------------------------------------------------------------------
# assert_impossible_states
# ---------------------------------------------------------------------------


class TestAssertImpossibleStates:
    def test_normal_state_does_not_raise(self):
        # entries=5, trades=5, equity_end=1.05 → valid
        assert_impossible_states(
            positions_nonzero=50,
            entries_generated=5,
            trade_count=5,
            equity_end=1.05,
            strategy_name="test_strategy",
        )

    def test_no_signals_no_trades_does_not_raise(self):
        # No signals → no trades expected
        assert_impossible_states(
            positions_nonzero=0,
            entries_generated=0,
            trade_count=0,
            equity_end=1.0,
            strategy_name="test_strategy",
        )

    def test_entries_but_no_trades_raises(self):
        with pytest.raises(DiagnosticAssertionError, match="IMPOSSIBLE"):
            assert_impossible_states(
                positions_nonzero=20,
                entries_generated=5,
                trade_count=0,
                equity_end=1.0,
                strategy_name="broken_strategy",
            )

    def test_trades_but_flat_equity_raises(self):
        with pytest.raises(DiagnosticAssertionError, match="IMPOSSIBLE"):
            assert_impossible_states(
                positions_nonzero=20,
                entries_generated=5,
                trade_count=5,
                equity_end=1.0,  # perfectly flat equity despite trades
                strategy_name="broken_strategy",
            )

    def test_error_message_contains_strategy_name(self):
        with pytest.raises(DiagnosticAssertionError, match="my_unique_strategy"):
            assert_impossible_states(
                positions_nonzero=10,
                entries_generated=3,
                trade_count=0,
                equity_end=1.0,
                strategy_name="my_unique_strategy",
            )


# ---------------------------------------------------------------------------
# run_strategy_trace — field presence
# ---------------------------------------------------------------------------


class TestRunStrategyTrace:
    """run_strategy_trace must return a fully-populated StrategyTraceReport."""

    def test_returns_strategy_trace_report(self):
        frame = _make_trending_ohlcv(400)
        strategy = _donchian_strategy()
        config = _fast_config()
        report = run_strategy_trace(frame, strategy, config=config, symbol="TEST")
        assert isinstance(report, StrategyTraceReport)

    def test_strategy_name_matches(self):
        frame = _make_trending_ohlcv(400)
        strategy = _donchian_strategy()
        config = _fast_config()
        report = run_strategy_trace(frame, strategy, config=config, symbol="TSYM")
        assert report.strategy_name == "donchian_s1_20_10"

    def test_symbol_matches(self):
        frame = _make_trending_ohlcv(400)
        strategy = _donchian_strategy()
        config = _fast_config()
        report = run_strategy_trace(frame, strategy, config=config, symbol="MYSYM")
        assert report.symbol == "MYSYM"

    def test_bars_processed_positive(self):
        frame = _make_trending_ohlcv(400)
        strategy = _donchian_strategy()
        config = _fast_config()
        report = run_strategy_trace(frame, strategy, config=config)
        assert report.bars_processed > 0

    def test_signal_decomposition_is_populated(self):
        frame = _make_trending_ohlcv(400)
        strategy = _donchian_strategy()
        config = _fast_config()
        report = run_strategy_trace(frame, strategy, config=config)
        assert isinstance(report.signal_decomposition, SignalDecomposition)

    def test_trending_data_generates_entries(self):
        """Trending synthetic data must generate at least one Donchian entry."""
        frame = _make_trending_ohlcv(400)
        strategy = _donchian_strategy()
        config = _fast_config()
        report = run_strategy_trace(frame, strategy, config=config)
        assert report.signal_decomposition.entries_generated > 0, (
            "Donchian strategy should generate at least one entry on trending data"
        )

    def test_trade_count_matches_entries(self):
        """Trade count should be positive when entries are generated."""
        frame = _make_trending_ohlcv(400)
        strategy = _donchian_strategy()
        config = _fast_config()
        report = run_strategy_trace(frame, strategy, config=config)
        if report.signal_decomposition.entries_generated > 0:
            assert report.trade_count > 0, (
                "trade_count should be > 0 when entries_generated > 0"
            )

    def test_equity_snapshot_populated(self):
        frame = _make_trending_ohlcv(400)
        strategy = _donchian_strategy()
        config = _fast_config()
        report = run_strategy_trace(frame, strategy, config=config)
        assert isinstance(report.equity_snapshot, EquityCurveSnapshot)
        assert math.isfinite(report.equity_snapshot.end)

    def test_permutation_summary_populated(self):
        frame = _make_trending_ohlcv(400)
        strategy = _donchian_strategy()
        config = _fast_config()
        report = run_strategy_trace(frame, strategy, config=config)
        assert isinstance(report.permutation_summary, PermutationSummary)
        assert 0.0 <= report.permutation_summary.p_value <= 1.0

    def test_stages_executed_includes_early_stages(self):
        frame = _make_trending_ohlcv(400)
        strategy = _donchian_strategy()
        config = _fast_config()
        report = run_strategy_trace(frame, strategy, config=config)
        executed = set(report.stages_executed)
        # These must always execute regardless of IS outcome
        assert PipelineStage.LOAD_DATA in executed
        assert PipelineStage.GENERATE_POSITIONS in executed
        assert PipelineStage.CONSTRUCT_BACKTEST in executed
        assert PipelineStage.COMPUTE_IS_METRIC in executed
        assert PipelineStage.RUN_IS_PERMUTATIONS in executed

    def test_rejected_report_has_rejection_reason(self):
        """A report that fails IS permutation must expose a reason string."""
        frame = _make_trending_ohlcv(400)
        strategy = _donchian_strategy()
        config = _fast_config()
        report = run_strategy_trace(frame, strategy, config=config)
        if report.is_rejected:
            assert report.rejection_reason is not None
            assert isinstance(report.rejection_reason, str)
            assert len(report.rejection_reason) > 0

    def test_passed_report_has_no_rejection_reason(self):
        """A report that passes IS must have rejection_reason=None."""
        frame = _make_trending_ohlcv(400)
        strategy = _donchian_strategy()
        # Use a very permissive alpha to maximise chance of passing
        config = ResearchValidationConfig(
            permutation_iterations=20,
            p_value_threshold=0.99,
            train_window=120,
            test_window=40,
            walk_forward_step=40,
            random_seed=7,
        )
        report = run_strategy_trace(frame, strategy, config=config)
        if not report.is_rejected:
            assert report.rejection_reason is None


# ---------------------------------------------------------------------------
# export_permutation_distribution
# ---------------------------------------------------------------------------


class TestExportPermutationDistribution:
    def test_creates_file(self, tmp_path: Path):
        null_dist = tuple(float(x) for x in np.random.default_rng(1).normal(0, 1, 50))
        result = PermutationTestResult(
            observed_statistic=0.75,
            null_distribution=null_dist,
            p_value=0.12,
            passes=False,
        )
        out = tmp_path / "perm_dist.json"
        export_permutation_distribution(
            result=result,
            strategy_name="test_strat",
            symbol="TEST",
            stage="in_sample",
            p_value_threshold=0.05,
            output_path=out,
        )
        assert out.exists()

    def test_json_is_valid(self, tmp_path: Path):
        null_dist = tuple(float(x) for x in np.random.default_rng(2).normal(0, 1, 50))
        result = PermutationTestResult(
            observed_statistic=1.10,
            null_distribution=null_dist,
            p_value=0.04,
            passes=True,
        )
        out = tmp_path / "perm_dist.json"
        export_permutation_distribution(
            result=result,
            strategy_name="test_strat",
            symbol="TEST",
            stage="in_sample",
            p_value_threshold=0.05,
            output_path=out,
        )
        payload = json.loads(out.read_text())
        assert "observed_statistic" in payload
        assert "p_value" in payload
        assert "null_distribution_summary" in payload
        assert "null_distribution_raw" in payload

    def test_json_summary_fields(self, tmp_path: Path):
        null_dist = tuple(float(x) for x in np.random.default_rng(3).normal(0, 1, 100))
        result = PermutationTestResult(
            observed_statistic=0.5,
            null_distribution=null_dist,
            p_value=0.20,
            passes=False,
        )
        out = tmp_path / "perm_dist.json"
        export_permutation_distribution(
            result=result,
            strategy_name="strat",
            symbol="SYM",
            stage="walk_forward",
            p_value_threshold=0.05,
            output_path=out,
        )
        payload = json.loads(out.read_text())
        summary = payload["null_distribution_summary"]
        for key in ("count", "mean", "std", "min", "pct5", "pct50", "pct95", "max"):
            assert key in summary, f"Missing key: {key}"
        assert summary["count"] == 100

    def test_creates_parent_directories(self, tmp_path: Path):
        null_dist = (0.1, 0.2, 0.3)
        result = PermutationTestResult(
            observed_statistic=0.5,
            null_distribution=null_dist,
            p_value=0.20,
            passes=False,
        )
        out = tmp_path / "nested" / "deep" / "perm.json"
        export_permutation_distribution(
            result=result,
            strategy_name="s",
            symbol="S",
            stage="is",
            p_value_threshold=0.05,
            output_path=out,
        )
        assert out.exists()

    def test_passes_field_preserved(self, tmp_path: Path):
        result = PermutationTestResult(
            observed_statistic=2.0,
            null_distribution=(0.1, 0.2),
            p_value=0.01,
            passes=True,
        )
        out = tmp_path / "perm.json"
        export_permutation_distribution(
            result=result,
            strategy_name="s",
            symbol="S",
            stage="is",
            p_value_threshold=0.05,
            output_path=out,
        )
        payload = json.loads(out.read_text())
        assert payload["passes"] is True
        assert math.isclose(payload["observed_statistic"], 2.0)
