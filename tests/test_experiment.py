"""Tests for predictor.research.experiment.

Covers:
- ExperimentRecord dataclass structure
- build_experiment_record
- save_experiment
- load_experiment
- JSON round-trip compatibility
"""

from __future__ import annotations

from pathlib import Path
import pytest

from predictor.research.experiment import (
    ExperimentRecord,
    build_experiment_record,
    save_experiment,
    load_experiment,
    ExperimentPersistenceError,
)
from predictor.research.validation import ResearchValidationConfig


def test_build_experiment_record():
    config = ResearchValidationConfig(permutation_iterations=50, random_seed=123)
    record = build_experiment_record(
        runtime_seconds=12.34,
        random_seed=123,
        validation_config=config,
        symbol_universe=("RELIANCE", "TCS"),
        strategy_names=("donchian_s1", "gap_fade"),
        generated_plots=("/tmp/plot1.png", "/tmp/plot2.png"),
        rejection_counts={"is_sharpe_negative": 5, "p_value_high": 2},
        total_runs=10,
        passed_runs=3,
        notes="Test run notes",
    )
    
    assert record.git_hash is not None
    assert record.runtime_seconds == 12.34
    assert record.random_seed == 123
    assert record.validation_config["permutation_iterations"] == 50
    assert record.symbol_universe == ("RELIANCE", "TCS")
    assert record.strategy_names == ("donchian_s1", "gap_fade")
    assert record.generated_plots == ("/tmp/plot1.png", "/tmp/plot2.png")
    assert record.rejection_summary["pass_rate"] == 0.3
    assert record.rejection_summary["total_runs"] == 10
    assert record.rejection_summary["passed_runs"] == 3
    assert record.notes == "Test run notes"


def test_experiment_roundtrip(tmp_path):
    config = ResearchValidationConfig(permutation_iterations=20, random_seed=42)
    record = build_experiment_record(
        runtime_seconds=5.67,
        random_seed=42,
        validation_config=config,
        symbol_universe=("SPY", "GLD"),
        strategy_names=("simple_ma",),
        notes="Testing round-trip persistence",
    )
    
    # Save it
    saved_path = save_experiment(record, tmp_path)
    assert saved_path.exists()
    assert saved_path.name.startswith("experiment_")
    assert saved_path.suffix == ".json"
    
    # Load it back
    loaded = load_experiment(saved_path)
    assert loaded.experiment_id == record.experiment_id
    assert loaded.git_hash == record.git_hash
    assert loaded.runtime_seconds == record.runtime_seconds
    assert loaded.random_seed == record.random_seed
    assert loaded.symbol_universe == record.symbol_universe
    assert loaded.strategy_names == record.strategy_names
    assert loaded.notes == record.notes
    assert loaded.validation_config == record.validation_config


def test_invalid_experiment_loads(tmp_path):
    bad_file = tmp_path / "bad.json"
    bad_file.write_text("invalid json content")
    with pytest.raises(ExperimentPersistenceError):
        load_experiment(bad_file)
