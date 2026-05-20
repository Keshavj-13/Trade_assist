"""Tests for predictor-centric CLI mode boundaries."""

from __future__ import annotations

import pytest

import market_assistant


def test_non_predictor_enabled_defaults_to_false(monkeypatch) -> None:
    monkeypatch.delenv("FIN_ASSIST_ENABLE_NON_PREDICTOR", raising=False)
    assert market_assistant._non_predictor_enabled() is False


def test_maybe_init_db_skips_when_persistence_disabled(monkeypatch) -> None:
    calls = {"init_db": 0}
    monkeypatch.delenv("FIN_ASSIST_ENABLE_PERSISTENCE", raising=False)
    monkeypatch.setattr(market_assistant, "init_db", lambda: calls.__setitem__("init_db", 1))

    market_assistant._maybe_init_db()
    assert calls["init_db"] == 0


def test_maybe_init_db_calls_when_persistence_enabled(monkeypatch) -> None:
    calls = {"init_db": 0}
    monkeypatch.setenv("FIN_ASSIST_ENABLE_PERSISTENCE", "1")
    monkeypatch.setattr(market_assistant, "init_db", lambda: calls.__setitem__("init_db", 1))

    market_assistant._maybe_init_db()
    assert calls["init_db"] == 1


def test_main_runs_once_by_default(monkeypatch) -> None:
    called = {"once": 0}

    monkeypatch.setattr(market_assistant, "_run_once", lambda: called.__setitem__("once", 1))
    monkeypatch.setattr(market_assistant, "_run_daemon", lambda: None)
    monkeypatch.setattr(market_assistant, "_run_scheduler", lambda: None)
    monkeypatch.setattr(market_assistant, "_run_telegram", lambda: None)
    monkeypatch.setattr("sys.argv", ["market_assistant"])
    monkeypatch.delenv("FIN_ASSIST_ENABLE_NON_PREDICTOR", raising=False)

    market_assistant.main()

    assert called["once"] == 1


def test_main_rejects_legacy_mode_when_disabled(monkeypatch) -> None:
    monkeypatch.setattr("sys.argv", ["market_assistant", "daemon"])
    monkeypatch.delenv("FIN_ASSIST_ENABLE_NON_PREDICTOR", raising=False)

    with pytest.raises(SystemExit):
        market_assistant.main()


def test_main_allows_legacy_mode_when_enabled(monkeypatch) -> None:
    called = {"daemon": 0}

    monkeypatch.setattr(market_assistant, "_run_once", lambda: None)
    monkeypatch.setattr(market_assistant, "_run_scheduler", lambda: None)
    monkeypatch.setattr(market_assistant, "_run_telegram", lambda: None)
    monkeypatch.setattr(
        market_assistant,
        "_run_daemon",
        lambda: called.__setitem__("daemon", 1),
    )
    monkeypatch.setenv("FIN_ASSIST_ENABLE_NON_PREDICTOR", "1")
    monkeypatch.setattr("sys.argv", ["market_assistant", "daemon"])

    market_assistant.main()

    assert called["daemon"] == 1
