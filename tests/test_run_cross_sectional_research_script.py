"""CLI behavior tests for scripts/run_cross_sectional_research.py."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

import scripts.run_cross_sectional_research as runner


@dataclass(frozen=True)
class _DummyFactor:
    name: str

    def compute_scores(self, symbol_data):
        return pd.DataFrame()


def _mini_frame(rows: int = 160) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=rows, freq="B")
    close = pd.Series(range(rows), index=idx, dtype=float) + 100.0
    return pd.DataFrame(
        {
            "Open": close,
            "High": close + 1.0,
            "Low": close - 1.0,
            "Close": close,
            "Volume": 1_000_000.0,
        },
        index=idx,
    )


def test_cli_factor_filter_and_export_diagnostics(monkeypatch, tmp_path: Path) -> None:
    called = {"factors": [], "diag": None, "meta": None}

    factors = [
        _DummyFactor(name="momentum_20"),
        _DummyFactor(name="relative_volume"),
    ]
    monkeypatch.setattr(runner, "build_factor_universe", lambda: tuple(factors))

    symbol_data = {
        "RELIANCE": _mini_frame(),
        "HDFCBANK": _mini_frame(),
        "INFY": _mini_frame(),
        "TATASTEEL": _mini_frame(),
        "ITC": _mini_frame(),
        "BAJFINANCE": _mini_frame(),
    }
    monkeypatch.setattr(runner, "prefetch_ohlcv_data", lambda symbols, source, workers: symbol_data)

    monkeypatch.setattr(
        runner,
        "validate_factor",
        lambda factor, symbol_data, config: called["factors"].append(factor.name) or object(),
    )
    monkeypatch.setattr(runner, "print_cross_sectional_report", lambda report: None)

    def _capture_diag(reports, loaded_symbol_data, config, plot_dir):
        called["diag"] = plot_dir

    monkeypatch.setattr(runner, "generate_factor_plots", _capture_diag)

    def _capture_meta(reports, symbol_universe, config, runtime, output_path):
        called["meta"] = output_path

    monkeypatch.setattr(runner, "save_reproducible_metadata", _capture_meta)

    argv = [
        "run_cross_sectional_research.py",
        "--source",
        "csv",
        "--factors",
        "momentum_20",
        "--symbols",
        "RELIANCE,HDFCBANK,INFY,TATASTEEL,ITC,BAJFINANCE",
        "--top-k",
        "2",
        "--lookback",
        "5y",
        "--permutation-iterations",
        "10",
        "--export-metadata",
        str(tmp_path / "exp.json"),
        "--export-diagnostics",
        str(tmp_path / "diag"),
    ]
    monkeypatch.setattr("sys.argv", argv)

    runner.main()

    assert called["factors"] == ["momentum_20"]
    assert called["diag"] == tmp_path / "diag"
    assert called["meta"] == tmp_path / "exp.json"


def test_symbols_file_txt_supported(monkeypatch, tmp_path: Path) -> None:
    called = {"symbols": None}

    factors = [_DummyFactor(name="momentum_20")]
    monkeypatch.setattr(runner, "build_factor_universe", lambda: tuple(factors))

    symbol_data = {"RELIANCE": _mini_frame(), "INFY": _mini_frame()}

    def _prefetch(symbols, source, workers):
        called["symbols"] = symbols
        return symbol_data

    monkeypatch.setattr(runner, "prefetch_ohlcv_data", _prefetch)
    monkeypatch.setattr(runner, "validate_factor", lambda factor, symbol_data, config: object())
    monkeypatch.setattr(runner, "print_cross_sectional_report", lambda report: None)
    monkeypatch.setattr(runner, "save_reproducible_metadata", lambda *args, **kwargs: None)

    symbols_file = tmp_path / "nse30.txt"
    symbols_file.write_text("RELIANCE\nINFY\n")

    argv = [
        "run_cross_sectional_research.py",
        "--source",
        "csv",
        "--symbols-file",
        str(symbols_file),
        "--factors",
        "momentum_20",
    ]
    monkeypatch.setattr("sys.argv", argv)

    runner.main()
    assert called["symbols"] == ["RELIANCE", "INFY"]
