"""Legacy compatibility wrapper for market data fetching.

Predictor code should use `predictor.data` directly. This module remains for
legacy command paths and keeps a stable `fetch_data` API.
"""

from __future__ import annotations

import pandas as pd

from infra.logging import log
from predictor.data import make_default_data_fetcher


_DEFAULT_FETCHER = make_default_data_fetcher()


def fetch_data(symbol: str) -> pd.DataFrame:
    """Fetch OHLCV data using the default configured data source.

    Returns an empty dataframe on failures for compatibility with legacy call
    sites that treat empty data as a skip condition.
    """

    log.debug(f"Fetching data for {symbol}")
    try:
        frame = _DEFAULT_FETCHER(symbol)
    except Exception as exc:
        log.error(f"Failed to fetch data for {symbol}: {exc}", exc_info=True)
        return pd.DataFrame()

    log.debug(f"Fetched {len(frame)} rows for {symbol}")
    return frame
