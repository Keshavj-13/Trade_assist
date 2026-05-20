"""Tests for predictor data-source boundaries and error contracts."""

from __future__ import annotations

import sys

import pandas as pd
import pytest

import predictor.data as data_module
from predictor.data import YFinanceDataSource, _flatten_columns, make_default_data_fetcher
from predictor.errors import DataUnavailableError, InputValidationError


class _FakeYFinance:
    def __init__(self, frame: pd.DataFrame, should_fail: bool = False):
        self._frame = frame
        self._should_fail = should_fail

    def download(self, *args, **kwargs):
        if self._should_fail:
            raise RuntimeError("download failed")
        return self._frame


def test_flatten_columns_handles_multiindex(make_ohlcv_frame) -> None:
    frame = make_ohlcv_frame().copy()
    multi = pd.MultiIndex.from_tuples([(col, "x") for col in frame.columns])
    frame.columns = multi

    flattened = _flatten_columns(frame)
    assert list(flattened.columns) == ["Open", "High", "Low", "Close", "Volume"]


def test_fetch_ohlcv_rejects_empty_symbol() -> None:
    source = YFinanceDataSource(interval="5m", lookback="5d", cache_dir=None)

    with pytest.raises(InputValidationError):
        source.fetch_ohlcv("   ")


def test_fetch_ohlcv_wraps_download_failures(make_ohlcv_frame, monkeypatch) -> None:
    source = YFinanceDataSource(interval="5m", lookback="5d", cache_dir=None)
    fake = _FakeYFinance(make_ohlcv_frame(), should_fail=True)
    monkeypatch.setitem(sys.modules, "yfinance", fake)

    with pytest.raises(DataUnavailableError):
        source.fetch_ohlcv("INFY")


def test_fetch_ohlcv_returns_dataframe(make_ohlcv_frame, monkeypatch) -> None:
    frame = make_ohlcv_frame()
    source = YFinanceDataSource(interval="5m", lookback="5d", cache_dir=None)
    fake = _FakeYFinance(frame)
    monkeypatch.setitem(sys.modules, "yfinance", fake)

    result = source.fetch_ohlcv("INFY")
    assert isinstance(result, pd.DataFrame)
    assert not result.empty


def test_make_default_data_fetcher_returns_callable() -> None:
    fetcher = make_default_data_fetcher()
    assert callable(fetcher)


def test_cache_installation_skipped_when_disabled(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("FIN_ASSIST_ENABLE_DATA_CACHE", "0")
    monkeypatch.setattr(data_module, "_CACHE_INSTALLED", False)
    data_module._safe_install_requests_cache(tmp_path / "cache")
    assert data_module._CACHE_INSTALLED is False
