"""Market data sources used by the predictor pipeline.

This module isolates external data-fetch side effects from the predictor core.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import os
from pathlib import Path
from typing import Optional

import pandas as pd

from predictor.errors import DataUnavailableError, InputValidationError


_CACHE_INSTALLED = False
_LOG = logging.getLogger("fin_assist.predictor.data")


def _cache_enabled() -> bool:
    """Return whether HTTP response caching for market data is enabled."""

    return os.environ.get("FIN_ASSIST_ENABLE_DATA_CACHE", "1") == "1"


def _safe_install_requests_cache(cache_dir: Path) -> None:
    """Install requests-cache once when available.
    """

    global _CACHE_INSTALLED
    if _CACHE_INSTALLED or not _cache_enabled():
        return

    try:
        import requests_cache
    except ModuleNotFoundError:
        _CACHE_INSTALLED = True
        return

    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / "yf_cache"
    try:
        requests_cache.install_cache(
            cache_name=str(cache_path),
            backend="sqlite",
            expire_after=300,
            allowable_methods=("GET", "POST"),
        )
    except Exception as exc:
        _LOG.warning("Failed to install requests-cache: %s", exc)
        _CACHE_INSTALLED = True
        return
    _CACHE_INSTALLED = True


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize yfinance column shape to a flat dataframe."""

    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = [
            str(column[0]).strip() if isinstance(column, tuple) else str(column)
            for column in df.columns
        ]
    return df


@dataclass(frozen=True)
class YFinanceDataSource:
    """OHLCV data source backed by yfinance."""

    interval: str
    lookback: str
    exchange_suffix: str = ".NS"
    cache_dir: Optional[Path] = None

    def fetch_ohlcv(self, symbol: str) -> pd.DataFrame:
        """Fetch OHLCV data for a single symbol.

        Args:
            symbol: Upper-case symbol code without exchange suffix.

        Returns:
            Dataframe containing OHLCV rows.

        Raises:
            InputValidationError: If symbol input is invalid.
            DataUnavailableError: If data download fails.
        """

        if not isinstance(symbol, str) or not symbol.strip():
            raise InputValidationError("symbol must be a non-empty string")

        if self.cache_dir is not None:
            _safe_install_requests_cache(self.cache_dir)

        try:
            import yfinance as yf
        except ModuleNotFoundError as exc:
            raise DataUnavailableError("yfinance dependency is not installed") from exc

        ticker = f"{symbol.strip().upper()}{self.exchange_suffix}"
        try:
            frame = yf.download(
                ticker,
                period=self.lookback,
                interval=self.interval,
                progress=False,
            )
        except Exception as exc:
            raise DataUnavailableError(f"{symbol}: data fetch failed: {exc}") from exc

        frame = _flatten_columns(frame)
        return frame


def make_default_data_fetcher():
    """Build the default predictor data fetch callable from legacy settings."""

    from config import settings as legacy

    data_dir = Path(getattr(legacy, "DATA_DIR"))
    source = YFinanceDataSource(
        interval=str(getattr(legacy, "INTERVAL")),
        lookback=str(getattr(legacy, "LOOKBACK")),
        cache_dir=data_dir / "yfinance_cache",
    )
    return source.fetch_ohlcv
