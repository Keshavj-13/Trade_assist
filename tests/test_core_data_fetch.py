"""Tests for legacy core data fetch compatibility wrapper."""

from __future__ import annotations

import pandas as pd

import core.data_fetch as data_fetch


def test_fetch_data_returns_dataframe(monkeypatch, make_ohlcv_frame) -> None:
    frame = make_ohlcv_frame()
    monkeypatch.setattr(data_fetch, "_DEFAULT_FETCHER", lambda symbol: frame)

    result = data_fetch.fetch_data("INFY")
    assert isinstance(result, pd.DataFrame)
    assert not result.empty


def test_fetch_data_returns_empty_on_fetch_failure(monkeypatch) -> None:
    def _raise(symbol: str):
        raise RuntimeError("boom")

    monkeypatch.setattr(data_fetch, "_DEFAULT_FETCHER", _raise)
    result = data_fetch.fetch_data("INFY")
    assert isinstance(result, pd.DataFrame)
    assert result.empty
