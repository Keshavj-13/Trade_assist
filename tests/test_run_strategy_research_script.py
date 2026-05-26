"""Regression tests for scripts/run_strategy_research.py CLI behavior."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import scripts.run_strategy_research as runner


@dataclass(frozen=True)
class _DummyConfig:
    random_seed: int = 7
    permutation_iterations: int = 1
    train_window: int = 5
    test_window: int = 2
    walk_forward_step: int = 2


@dataclass(frozen=True)
class _DummyStrategy:
    name: str


@dataclass(frozen=True)
class _DummyComparison:
    rows: tuple
    reports: tuple


def test_export_metadata_does_not_raise_path_unbound(monkeypatch, tmp_path: Path) -> None:
    """`--export-metadata` must work even when `--plot` is omitted."""
    saved = {"called": False, "path": None}

    monkeypatch.setattr(runner, "_build_source", lambda args: object())
    monkeypatch.setattr(runner, "_resolve_config", lambda args: _DummyConfig())
    monkeypatch.setattr(runner, "_build_strategies", lambda args: (_DummyStrategy(name="dummy"),))
    monkeypatch.setattr(
        runner,
        "compare_strategies_across_symbols",
        lambda **kwargs: _DummyComparison(rows=(), reports=()),
    )

    import predictor.research.experiment as exp

    monkeypatch.setattr(exp, "build_experiment_record", lambda **kwargs: {"ok": True})

    def _save_experiment(record, output_dir):
        out = Path(output_dir) / "experiment.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text('{"ok": true}')
        saved["called"] = True
        saved["path"] = out
        return out

    monkeypatch.setattr(exp, "save_experiment", _save_experiment)

    export_dir = tmp_path / "meta_out"
    argv = [
        "run_strategy_research.py",
        "--symbols",
        "INFY",
        "--export-metadata",
        f"{export_dir}/",
    ]
    monkeypatch.setattr("sys.argv", argv)

    runner.main()

    assert saved["called"] is True
    assert saved["path"] is not None
    assert Path(saved["path"]).exists()
