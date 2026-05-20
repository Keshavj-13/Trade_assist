"""Historical market data loaders for strategy research."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Protocol

import pandas as pd

from predictor.research.errors import ResearchDataError, ResearchInputError


REQUIRED_OHLCV_COLUMNS = ("Open", "High", "Low", "Close", "Volume")


class HistoricalDataSource(Protocol):
    """Contract for symbol -> OHLCV dataframe sources."""

    def fetch_ohlcv(self, symbol: str) -> pd.DataFrame:
        """Fetch historical OHLCV data for one symbol."""


def _coerce_datetime_index(frame: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Return frame with DatetimeIndex inferred from common column names."""

    if isinstance(frame.index, pd.DatetimeIndex):
        return frame
    for name in ("Date", "Datetime", "timestamp", "Timestamp"):
        if name in frame.columns:
            parsed = frame.copy()
            parsed[name] = pd.to_datetime(parsed[name], utc=False, errors="coerce")
            parsed = parsed.dropna(subset=[name]).set_index(name)
            return parsed
    raise ResearchInputError(f"{symbol}: frame index must be DatetimeIndex or include a date column")


def validate_research_frame(
    frame: pd.DataFrame,
    symbol: str,
    *,
    min_rows: int = 120,
) -> pd.DataFrame:
    """Validate and normalize OHLCV frame for strategy research."""

    if frame is None:
        raise ResearchDataError(f"{symbol}: missing dataframe")
    if not isinstance(frame, pd.DataFrame):
        raise ResearchInputError(f"{symbol}: dataframe expected")
    if frame.empty:
        raise ResearchDataError(f"{symbol}: empty dataframe")

    with_index = _coerce_datetime_index(frame, symbol=symbol)
    missing = [column for column in REQUIRED_OHLCV_COLUMNS if column not in with_index.columns]
    if missing:
        raise ResearchInputError(f"{symbol}: missing required columns {missing}")

    trimmed = with_index.loc[:, REQUIRED_OHLCV_COLUMNS].copy()
    trimmed = trimmed.dropna(subset=list(REQUIRED_OHLCV_COLUMNS))
    for column in REQUIRED_OHLCV_COLUMNS:
        trimmed[column] = pd.to_numeric(trimmed[column], errors="coerce")
    trimmed = trimmed.dropna(subset=list(REQUIRED_OHLCV_COLUMNS))

    if not isinstance(trimmed.index, pd.DatetimeIndex):
        raise ResearchInputError(f"{symbol}: index must be DatetimeIndex")
    if not trimmed.index.is_monotonic_increasing:
        raise ResearchInputError(f"{symbol}: index must be sorted and monotonic increasing")
    if len(trimmed) < min_rows:
        raise ResearchDataError(f"{symbol}: insufficient rows ({len(trimmed)} < {min_rows})")
    if (trimmed["High"] < trimmed["Low"]).any():
        raise ResearchInputError(f"{symbol}: High must be >= Low")
    if (trimmed["Close"] <= 0).any():
        raise ResearchInputError(f"{symbol}: Close values must be > 0")
    if (trimmed["Open"] <= 0).any():
        raise ResearchInputError(f"{symbol}: Open values must be > 0")
    return trimmed


@dataclass(frozen=True)
class CSVHistoricalDataSource:
    """Load OHLCV data from local CSV files by symbol name."""

    directory: str | Path
    min_rows: int = 120

    def fetch_ohlcv(self, symbol: str) -> pd.DataFrame:
        """Load OHLCV frame from `<directory>/<symbol>.csv`."""

        if not isinstance(symbol, str) or not symbol.strip():
            raise ResearchInputError("symbol must be a non-empty string")
        normalized = symbol.strip()
        base = Path(self.directory)
        candidates = [normalized, normalized.upper(), normalized.lower()]
        files = []
        for candidate in candidates:
            if candidate.endswith(".csv"):
                files.append(base / candidate)
            else:
                files.append(base / f"{candidate}.csv")

        target = next((path for path in files if path.exists()), None)
        if target is None and base.exists():
            wanted = normalized.lower().removesuffix(".csv")
            for candidate in base.glob("*.csv"):
                if candidate.stem.lower() == wanted:
                    target = candidate
                    break
        if target is None:
            raise ResearchDataError(f"{symbol}: csv not found in {base}")

        frame = pd.read_csv(target)
        return validate_research_frame(frame, symbol=normalized, min_rows=self.min_rows)


@dataclass(frozen=True)
class YFinanceHistoricalDataSource:
    """Fetch OHLCV history directly from yfinance for live research runs.

    Use lookback="5y" or lookback="10y" for broader regime coverage.
    """

    interval: str = "1d"
    lookback: str = "5y"
    exchange_suffix: str = ".NS"
    min_rows: int = 120

    def fetch_ohlcv(self, symbol: str) -> pd.DataFrame:
        """Download a historical OHLCV frame from Yahoo Finance."""

        if not isinstance(symbol, str) or not symbol.strip():
            raise ResearchInputError("symbol must be a non-empty string")
        ticker = symbol.strip().upper()
        if "." not in ticker and self.exchange_suffix:
            ticker = f"{ticker}{self.exchange_suffix}"
        try:
            import yfinance as yf
        except ModuleNotFoundError as exc:
            raise ResearchDataError("yfinance dependency is not installed") from exc

        try:
            frame = yf.download(
                ticker,
                period=self.lookback,
                interval=self.interval,
                auto_adjust=False,
                progress=False,
            )
        except Exception as exc:
            raise ResearchDataError(f"{symbol}: yfinance download failed: {exc}") from exc
        if frame.empty:
            raise ResearchDataError(f"{symbol}: no data returned by yfinance")
        if isinstance(frame.columns, pd.MultiIndex):
            flat = frame.copy()
            flat.columns = [str(column[0]).strip() for column in flat.columns]
            frame = flat
        return validate_research_frame(frame, symbol=symbol.upper(), min_rows=self.min_rows)


class CachedDataSource:
    """Thread-safe in-memory cache wrapping any HistoricalDataSource.

    Each symbol is fetched exactly once. Subsequent calls return the
    cached frame. Fetch failures are re-raised on every call so the
    caller always sees the real error — the cache never hides failure.
    """

    def __init__(self, source: HistoricalDataSource) -> None:
        self._source = source
        self._cache: Dict[str, pd.DataFrame] = {}
        self._errors: Dict[str, Exception] = {}
        self._lock = threading.Lock()

    def fetch_ohlcv(self, symbol: str) -> pd.DataFrame:
        """Fetch symbol from cache or delegate to the underlying source."""
        with self._lock:
            if symbol in self._errors:
                raise self._errors[symbol]
            if symbol in self._cache:
                return self._cache[symbol]

        # Fetch outside the lock so concurrent threads can fetch different symbols
        try:
            frame = self._source.fetch_ohlcv(symbol)
        except Exception as exc:
            with self._lock:
                self._errors[symbol] = exc
            raise

        with self._lock:
            self._cache[symbol] = frame
        return frame

    def prefetch(self, symbols: tuple[str, ...]) -> Dict[str, Exception | None]:
        """Attempt to fetch all symbols and return {symbol: error_or_None}.

        Errors are stored in the cache and re-raised on future fetch_ohlcv calls.
        This method never raises — all failures are captured in the returned dict.
        """
        results: Dict[str, Exception | None] = {}
        for symbol in symbols:
            try:
                self.fetch_ohlcv(symbol)
                results[symbol] = None
            except Exception as exc:
                results[symbol] = exc
        return results

    @property
    def cached_symbols(self) -> tuple[str, ...]:
        """Return symbols successfully loaded into cache."""
        with self._lock:
            return tuple(self._cache.keys())
