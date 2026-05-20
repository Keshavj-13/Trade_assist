"""Input validators for predictor symbols and OHLCV market data."""

from __future__ import annotations

from typing import Iterable, Tuple

import pandas as pd

from predictor.errors import DataUnavailableError, InputValidationError

REQUIRED_OHLCV_COLUMNS = ("Open", "High", "Low", "Close", "Volume")


def normalize_symbol(symbol: str) -> str:
    """Normalize and validate a market symbol identifier.

    Args:
        symbol: Raw caller-supplied symbol.

    Returns:
        Upper-cased stripped symbol.

    Raises:
        InputValidationError: If symbol is not a non-empty string.
    """

    if not isinstance(symbol, str):
        raise InputValidationError("symbol must be a string")
    normalized = symbol.strip().upper()
    if not normalized:
        raise InputValidationError("symbol must not be empty")
    return normalized


def validate_symbol_list(symbols: Iterable[str]) -> Tuple[str, ...]:
    """Validate a sequence of symbols and return a stable de-duplicated tuple."""

    if symbols is None:
        raise InputValidationError("symbols must not be None")

    seen = set()
    result = []
    for raw in symbols:
        symbol = normalize_symbol(raw)
        if symbol in seen:
            continue
        seen.add(symbol)
        result.append(symbol)

    if not result:
        raise InputValidationError("symbols must contain at least one valid symbol")
    return tuple(result)


def validate_ohlcv_frame(df: pd.DataFrame, symbol: str, min_rows: int = 60) -> pd.DataFrame:
    """Validate and normalize OHLCV data used by the predictor.

    Args:
        df: Candidate market data table.
        symbol: Symbol associated with the dataframe.
        min_rows: Minimum row count needed for stable rolling features.

    Returns:
        Cleaned dataframe sorted by index with required columns.

    Raises:
        DataUnavailableError: If data is missing or insufficient.
        InputValidationError: If required columns are missing.
    """

    if df is None:
        raise DataUnavailableError(f"{symbol}: missing dataframe")
    if not isinstance(df, pd.DataFrame):
        raise InputValidationError(f"{symbol}: dataframe expected")
    if df.empty:
        raise DataUnavailableError(f"{symbol}: empty dataframe")

    missing = [column for column in REQUIRED_OHLCV_COLUMNS if column not in df.columns]
    if missing:
        raise InputValidationError(f"{symbol}: missing required columns {missing}")

    clean = df.copy().sort_index()
    clean = clean.dropna(subset=list(REQUIRED_OHLCV_COLUMNS))
    if len(clean) < min_rows:
        raise DataUnavailableError(
            f"{symbol}: insufficient rows ({len(clean)} < {min_rows})"
        )
    return clean
