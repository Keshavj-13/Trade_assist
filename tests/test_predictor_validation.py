"""Tests for predictor input validation contracts."""

from __future__ import annotations

import pandas as pd
import pytest

from predictor.errors import DataUnavailableError, InputValidationError
from predictor.validation import (
    normalize_symbol,
    validate_ohlcv_frame,
    validate_symbol_list,
)


def test_normalize_symbol_uppercases_and_strips() -> None:
    assert normalize_symbol("  infy ") == "INFY"


def test_normalize_symbol_rejects_empty() -> None:
    with pytest.raises(InputValidationError):
        normalize_symbol("   ")


def test_validate_symbol_list_deduplicates_preserving_order() -> None:
    assert validate_symbol_list(["infy", "INFY", "tcs"]) == ("INFY", "TCS")


def test_validate_symbol_list_rejects_none() -> None:
    with pytest.raises(InputValidationError):
        validate_symbol_list(None)


def test_validate_ohlcv_frame_rejects_missing_columns(make_ohlcv_frame) -> None:
    frame = make_ohlcv_frame().drop(columns=["Volume"])
    with pytest.raises(InputValidationError):
        validate_ohlcv_frame(frame, "INFY")


def test_validate_ohlcv_frame_rejects_insufficient_rows(make_ohlcv_frame) -> None:
    frame = make_ohlcv_frame(rows=20)
    with pytest.raises(DataUnavailableError):
        validate_ohlcv_frame(frame, "INFY", min_rows=60)


def test_validate_ohlcv_frame_rejects_non_dataframe() -> None:
    with pytest.raises(InputValidationError):
        validate_ohlcv_frame([1, 2, 3], "INFY")  # type: ignore[arg-type]


def test_validate_ohlcv_frame_returns_sorted_frame(make_ohlcv_frame) -> None:
    frame = make_ohlcv_frame().iloc[::-1]
    validated = validate_ohlcv_frame(frame, "INFY")
    assert isinstance(validated, pd.DataFrame)
    assert validated.index.is_monotonic_increasing
