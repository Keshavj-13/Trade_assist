"""Tests for lazy database service wrapper behavior."""

from __future__ import annotations

import pytest

import service.database as db_service


class _BackendOk:
    @staticmethod
    def initialize_db() -> None:
        return None

    @staticmethod
    def get_open_positions(user_id=None):
        return [{"symbol": "INFY"}]


class _BackendFail:
    @staticmethod
    def initialize_db() -> None:
        raise RuntimeError("db unavailable")

    @staticmethod
    def get_open_positions(user_id=None):
        raise RuntimeError("db unavailable")


def test_init_db_swallow_backend_errors(monkeypatch) -> None:
    monkeypatch.delenv("FIN_ASSIST_ENABLE_PERSISTENCE", raising=False)
    monkeypatch.setattr(db_service, "_backend", lambda: _BackendFail)
    db_service.init_db()


def test_init_db_raises_when_persistence_required(monkeypatch) -> None:
    monkeypatch.setenv("FIN_ASSIST_ENABLE_PERSISTENCE", "1")
    monkeypatch.setattr(db_service, "_backend", lambda: _BackendFail)
    with pytest.raises(RuntimeError):
        db_service.init_db()


def test_get_open_positions_returns_rows(monkeypatch) -> None:
    monkeypatch.delenv("FIN_ASSIST_ENABLE_PERSISTENCE", raising=False)
    monkeypatch.setattr(db_service, "_backend", lambda: _BackendOk)
    rows = db_service.get_open_positions()
    assert rows == [{"symbol": "INFY"}]


def test_get_open_positions_fallback_empty_on_failure(monkeypatch) -> None:
    monkeypatch.delenv("FIN_ASSIST_ENABLE_PERSISTENCE", raising=False)
    monkeypatch.setattr(db_service, "_backend", lambda: _BackendFail)
    rows = db_service.get_open_positions()
    assert rows == []


def test_get_open_positions_raises_when_persistence_required(monkeypatch) -> None:
    monkeypatch.setenv("FIN_ASSIST_ENABLE_PERSISTENCE", "1")
    monkeypatch.setattr(db_service, "_backend", lambda: _BackendFail)
    with pytest.raises(RuntimeError):
        db_service.get_open_positions()
