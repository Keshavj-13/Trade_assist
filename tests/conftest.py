"""Shared fixtures for predictor tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def make_ohlcv_frame():
    """Return a deterministic factory for synthetic OHLCV dataframes."""

    def _factory(
        *,
        rows: int = 120,
        base_price: float = 100.0,
        trend: float = 0.05,
        volume: float = 120_000.0,
    ) -> pd.DataFrame:
        idx = pd.date_range("2026-01-01 09:15", periods=rows, freq="5min")
        close = base_price + np.arange(rows) * trend
        high = close + 0.8
        low = close - 0.8
        open_price = close - 0.2
        seasonal = 1.0 + 0.05 * np.sin(np.linspace(0, 6.28, rows))
        vol = volume * seasonal

        return pd.DataFrame(
            {
                "Open": open_price,
                "High": high,
                "Low": low,
                "Close": close,
                "Volume": vol,
            },
            index=idx,
        )

    return _factory


@pytest.fixture
def load_real_ohlcv_frame():
    """Return a loader for deterministic real-market OHLCV fixture data."""

    fixtures_dir = Path(__file__).parent / "fixtures" / "market_data"

    def _loader(filename: str = "INFY_NS_1d.csv", *, tail_rows: int | None = 420) -> pd.DataFrame:
        path = fixtures_dir / filename
        frame = pd.read_csv(path, parse_dates=["Date"]).set_index("Date")
        required = ["Open", "High", "Low", "Close", "Volume"]
        frame = frame[required].astype(float)
        frame = frame.sort_index()
        if tail_rows is not None:
            frame = frame.tail(tail_rows)
        return frame

    return _loader


@pytest.fixture
def real_ohlcv_frame(load_real_ohlcv_frame):
    """Return one real-market OHLCV fixture frame for convenience."""

    return load_real_ohlcv_frame("INFY_NS_1d.csv")
