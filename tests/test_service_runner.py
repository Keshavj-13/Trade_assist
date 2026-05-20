"""Tests for predictor runner side-effect boundaries."""

from __future__ import annotations

import service.runner as runner


def test_telegram_enabled_default_false(monkeypatch) -> None:
    monkeypatch.delenv("FIN_ASSIST_ENABLE_TELEGRAM", raising=False)
    assert runner._telegram_enabled() is False


def test_run_once_invokes_scan_and_persist_without_telegram(monkeypatch) -> None:
    calls = {"scan": 0, "persist": 0}

    def fake_scan(scope: str):
        calls["scan"] += 1
        assert scope == "whole"
        return {"buy_candidates": [], "sell_candidates": []}

    def fake_persist(scan_result, username: str):
        calls["persist"] += 1
        assert username == "system"

    monkeypatch.setattr(runner, "perform_scan", fake_scan)
    monkeypatch.setattr(runner, "persist_scan_results", fake_persist)
    monkeypatch.delenv("FIN_ASSIST_ENABLE_TELEGRAM", raising=False)

    result = runner.run_once()

    assert calls == {"scan": 1, "persist": 1}
    assert result == {"buy_candidates": [], "sell_candidates": []}


def test_start_service_delegates_to_run_once(monkeypatch) -> None:
    calls = {"run_once": 0}

    def fake_run_once():
        calls["run_once"] += 1
        return None

    monkeypatch.setattr(runner, "run_once", fake_run_once)
    runner.start_service()
    assert calls["run_once"] == 1


def test_scan_stats_text_reports_counts() -> None:
    summary = runner._scan_stats_text(
        {
            "symbols_scanned": 10,
            "buy_candidates": [{"symbol": "A"}],
            "sell_candidates": [{"symbol": "B"}, {"symbol": "C"}],
            "hold_candidates": ["D"],
            "skipped_symbols": {"E": "err"},
        },
        12.34,
    )
    assert "scanned=10" in summary
    assert "buy=1" in summary
    assert "sell=2" in summary
    assert "hold=1" in summary
    assert "skipped=1" in summary
