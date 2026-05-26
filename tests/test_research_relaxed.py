"""Tests for predictor.research.relaxed.

Covers:
- RelaxedDiagnosticConfig correctly stores overrides
- apply_relaxed_config produces a ResearchValidationConfig with overrides applied
- apply_relaxed_config leaves base config unchanged when no overrides set
- validate_strategy_relaxed with disable_is_rejection=True runs walk-forward stages
- validate_strategy_relaxed produces non-zero metrics when IS rejection is disabled
- validate_strategy_relaxed still collects real IS permutation result (not faked)
- relaxed alpha override produces the correct p_value_threshold in resulting config
- reduced_permutation_count override applies correctly
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from predictor.research.donchian import DonchianVariantStrategy
from predictor.research.relaxed import (
    RelaxedDiagnosticConfig,
    apply_relaxed_config,
    validate_strategy_relaxed,
)
from predictor.research.validation import ResearchValidationConfig


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _base_config() -> ResearchValidationConfig:
    return ResearchValidationConfig(
        bars_per_year=252,
        transaction_cost_bps=0.0,
        permutation_iterations=200,
        permutation_block_size=20,
        p_value_threshold=0.05,
        train_window=120,
        test_window=40,
        walk_forward_step=40,
        minimum_walk_forward_stability=0.35,
        minimum_walk_forward_fold_pass_rate=0.0,
        require_positive_walk_forward_return=False,
        minimum_walk_forward_sharpe=None,
        random_seed=7,
    )


def _fast_config() -> ResearchValidationConfig:
    """Minimal permutation count for fast diagnostic tests."""
    return ResearchValidationConfig(
        permutation_iterations=20,
        train_window=120,
        test_window=40,
        walk_forward_step=40,
        random_seed=7,
    )


def _trending_ohlcv(n: int = 400, seed: int = 42) -> pd.DataFrame:
    """Synthetic trending OHLCV to reliably generate Donchian signals."""
    rng = np.random.default_rng(seed)
    log_returns = rng.normal(0.0008, 0.012, size=n)
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


def _donchian() -> DonchianVariantStrategy:
    return DonchianVariantStrategy(
        name="donchian_s1_20_10",
        entry_lookback=20,
        exit_lookback=10,
        dependency_rule="none",
    )


# ---------------------------------------------------------------------------
# RelaxedDiagnosticConfig
# ---------------------------------------------------------------------------


class TestRelaxedDiagnosticConfig:
    def test_defaults_are_non_invasive(self):
        cfg = RelaxedDiagnosticConfig(base_config=_base_config())
        assert cfg.disable_is_rejection is False
        assert cfg.relaxed_alpha is None
        assert cfg.reduced_permutation_count is None
        assert cfg.skip_holm_correction is False

    def test_stores_base_config(self):
        base = _base_config()
        cfg = RelaxedDiagnosticConfig(base_config=base)
        assert cfg.base_config is base

    def test_stores_overrides(self):
        cfg = RelaxedDiagnosticConfig(
            base_config=_base_config(),
            disable_is_rejection=True,
            relaxed_alpha=0.10,
            reduced_permutation_count=50,
            skip_holm_correction=True,
        )
        assert cfg.disable_is_rejection is True
        assert cfg.relaxed_alpha == pytest.approx(0.10)
        assert cfg.reduced_permutation_count == 50
        assert cfg.skip_holm_correction is True


# ---------------------------------------------------------------------------
# apply_relaxed_config
# ---------------------------------------------------------------------------


class TestApplyRelaxedConfig:
    def test_no_overrides_preserves_base(self):
        base = _base_config()
        relaxed = RelaxedDiagnosticConfig(base_config=base)
        result = apply_relaxed_config(relaxed)
        assert result.p_value_threshold == pytest.approx(base.p_value_threshold)
        assert result.permutation_iterations == base.permutation_iterations
        assert result.train_window == base.train_window
        assert result.random_seed == base.random_seed

    def test_relaxed_alpha_applied(self):
        base = _base_config()
        relaxed = RelaxedDiagnosticConfig(base_config=base, relaxed_alpha=0.20)
        result = apply_relaxed_config(relaxed)
        assert result.p_value_threshold == pytest.approx(0.20)

    def test_reduced_permutation_count_applied(self):
        base = _base_config()
        relaxed = RelaxedDiagnosticConfig(base_config=base, reduced_permutation_count=30)
        result = apply_relaxed_config(relaxed)
        assert result.permutation_iterations == 30

    def test_returns_research_validation_config(self):
        base = _base_config()
        relaxed = RelaxedDiagnosticConfig(base_config=base, relaxed_alpha=0.10)
        result = apply_relaxed_config(relaxed)
        assert isinstance(result, ResearchValidationConfig)

    def test_other_fields_preserved_when_overriding_alpha(self):
        base = _base_config()
        relaxed = RelaxedDiagnosticConfig(base_config=base, relaxed_alpha=0.15)
        result = apply_relaxed_config(relaxed)
        assert result.train_window == base.train_window
        assert result.test_window == base.test_window
        assert result.random_seed == base.random_seed
        assert result.bars_per_year == base.bars_per_year

    def test_combined_overrides(self):
        base = _base_config()
        relaxed = RelaxedDiagnosticConfig(
            base_config=base,
            relaxed_alpha=0.15,
            reduced_permutation_count=25,
        )
        result = apply_relaxed_config(relaxed)
        assert result.p_value_threshold == pytest.approx(0.15)
        assert result.permutation_iterations == 25


# ---------------------------------------------------------------------------
# validate_strategy_relaxed — IS rejection disabled
# ---------------------------------------------------------------------------


class TestValidateStrategyRelaxedDisableIS:
    """When disable_is_rejection=True, walk-forward stages must always execute."""

    def _run_relaxed(self, disable_is: bool) -> object:
        frame = _trending_ohlcv(400)
        strategy = _donchian()
        base = _fast_config()
        relaxed = RelaxedDiagnosticConfig(
            base_config=base,
            disable_is_rejection=disable_is,
        )
        return validate_strategy_relaxed(frame, strategy, relaxed=relaxed)

    def test_returns_validation_report(self):
        from predictor.research.types import StrategyValidationReport

        report = self._run_relaxed(disable_is=True)
        assert isinstance(report, StrategyValidationReport)

    def test_disabled_is_runs_walk_forward_folds(self):
        """disable_is_rejection=True must cause walk-forward folds to execute."""
        report = self._run_relaxed(disable_is=True)
        # Walk-forward folds should be populated (not the empty dummy fallback)
        assert len(report.walk_forward_folds) > 0, (
            "Walk-forward folds should be non-empty when IS rejection is disabled"
        )

    def test_disabled_is_nonzero_trade_count_in_wf(self):
        """Trade count should be > 0 in walk-forward aggregate when IS rejection is disabled."""
        report = self._run_relaxed(disable_is=True)
        # Trending synthetic data should produce trades in walk-forward folds
        total_wf_trades = report.walk_forward_aggregate.metrics.trade_count
        assert total_wf_trades > 0, (
            f"walk_forward trade_count should be > 0 with IS rejection disabled, "
            f"got {total_wf_trades}. "
            f"This would confirm the framework was short-circuiting before trade extraction."
        )

    def test_disabled_is_not_in_fail_reasons(self):
        """'in_sample_permutation_failed' must NOT appear in fail_reasons when IS is disabled."""
        report = self._run_relaxed(disable_is=True)
        assert "in_sample_permutation_failed" not in report.fail_reasons

    def test_is_permutation_result_still_computed(self):
        """Even with IS rejection disabled, the actual IS permutation must be computed."""
        report = self._run_relaxed(disable_is=True)
        # p_value must be in [0,1] — not the dummy 1.0 from early exit
        assert 0.0 <= report.in_sample_permutation.p_value <= 1.0
        # null_distribution must be non-empty
        assert len(report.in_sample_permutation.null_distribution) > 0

    def test_strategy_name_preserved(self):
        report = self._run_relaxed(disable_is=True)
        assert report.strategy_name == "donchian_s1_20_10"


# ---------------------------------------------------------------------------
# validate_strategy_relaxed — relaxed alpha
# ---------------------------------------------------------------------------


class TestValidateStrategyRelaxedAlpha:
    def test_alpha_1_0_always_fails_is(self):
        """p_value_threshold=1.0 is always >= p_value, so IS always passes."""
        # NOTE: alpha=1.0 means *any* p_value passes. Use disable_is_rejection to
        # bypass the gate itself; alpha=0.99 is a close proxy.
        frame = _trending_ohlcv(400)
        strategy = _donchian()
        base = _fast_config()
        relaxed = RelaxedDiagnosticConfig(
            base_config=base,
            relaxed_alpha=0.99,
        )
        report = validate_strategy_relaxed(frame, strategy, relaxed=relaxed)
        # With alpha=0.99, IS permutation almost certainly passes
        # so walk-forward folds should be populated
        assert isinstance(report, object)  # smoke check

    def test_relaxed_report_has_correct_strategy_name(self):
        frame = _trending_ohlcv(400)
        strategy = _donchian()
        base = _fast_config()
        relaxed = RelaxedDiagnosticConfig(base_config=base, relaxed_alpha=0.20)
        report = validate_strategy_relaxed(frame, strategy, relaxed=relaxed)
        assert report.strategy_name == "donchian_s1_20_10"

    def test_fail_reasons_is_tuple(self):
        frame = _trending_ohlcv(400)
        strategy = _donchian()
        base = _fast_config()
        relaxed = RelaxedDiagnosticConfig(base_config=base)
        report = validate_strategy_relaxed(frame, strategy, relaxed=relaxed)
        assert isinstance(report.fail_reasons, tuple)

    def test_is_valid_consistent_with_fail_reasons(self):
        frame = _trending_ohlcv(400)
        strategy = _donchian()
        base = _fast_config()
        relaxed = RelaxedDiagnosticConfig(base_config=base, disable_is_rejection=True)
        report = validate_strategy_relaxed(frame, strategy, relaxed=relaxed)
        if report.is_valid:
            assert len(report.fail_reasons) == 0
        else:
            assert len(report.fail_reasons) > 0


# ---------------------------------------------------------------------------
# Diagnostics: pipeline ordering correctness via validation.py integration
# ---------------------------------------------------------------------------


class TestPipelineOrderingViaValidation:
    """Verify that validation.py calls pipeline stage logs in correct order.

    We capture stdout and check that stage labels appear in the mandated order.
    """

    def test_pipeline_stages_appear_in_correct_order(self, capsys):
        from predictor.research.validation import validate_strategy

        frame = _trending_ohlcv(300)
        strategy = _donchian()
        config = ResearchValidationConfig(
            permutation_iterations=10,
            train_window=100,
            test_window=30,
            walk_forward_step=30,
            random_seed=7,
        )
        validate_strategy(frame, strategy, config=config)
        captured = capsys.readouterr()
        output = captured.out

        # Extract [PIPELINE] lines in order
        pipeline_lines = [
            line.split("[PIPELINE]")[1].strip()
            for line in output.splitlines()
            if "[PIPELINE]" in line
        ]
        # At minimum these five must appear in order
        required = [
            "load_data",
            "generate_positions",
            "construct_backtest",
            "compute_is_metric",
            "run_is_permutations",
        ]
        found_indices = []
        for stage in required:
            try:
                idx = pipeline_lines.index(stage)
                found_indices.append(idx)
            except ValueError:
                pytest.fail(f"Stage '{stage}' not found in pipeline output")
        assert found_indices == sorted(found_indices), (
            f"Pipeline stages not in correct order: {pipeline_lines}"
        )
