"""Tests for predictor service adapter behavior in `service.research`."""

from __future__ import annotations

from datetime import datetime

import pandas as pd
import pytest

import service.research as research
from predictor.types import MarketRegime, PipelineRun, Prediction, PredictorAction


def test_perform_scan_rejects_invalid_scope() -> None:
    with pytest.raises(ValueError):
        research.perform_scan(scope="invalid", symbols=["INFY"])


def test_perform_scan_rejects_non_positive_top_n() -> None:
    with pytest.raises(ValueError):
        research.perform_scan(scope="whole", symbols=["INFY"], top_n=0)


def test_perform_scan_rejects_non_integer_top_n() -> None:
    with pytest.raises(ValueError):
        research.perform_scan(scope="whole", symbols=["INFY"], top_n=1.5)  # type: ignore[arg-type]


def test_perform_scan_rejects_negative_wallet() -> None:
    with pytest.raises(ValueError):
        research.perform_scan(scope="whole", symbols=["INFY"], wallet=-1)


def test_perform_scan_rejects_non_callable_fetcher() -> None:
    with pytest.raises(ValueError):
        research.perform_scan(scope="whole", symbols=["INFY"], data_fetcher=123)  # type: ignore[arg-type]


def test_perform_scan_returns_legacy_compatible_payload(monkeypatch, make_ohlcv_frame) -> None:
    frame = make_ohlcv_frame(trend=0.12)

    monkeypatch.setattr(research, "get_open_positions", lambda: [])
    monkeypatch.setattr(research, "current_market_time", lambda: datetime(2026, 1, 1, 10, 0))
    monkeypatch.setattr(research, "is_market_closed", lambda _: False)
    monkeypatch.setattr(research, "market_time_str", lambda _: "2026-01-01 10:00:00 IST")

    def fetcher(symbol: str) -> pd.DataFrame:
        return frame

    result = research.perform_scan(scope="whole", symbols=["INFY"], data_fetcher=fetcher)

    assert result["scope"] == "whole"
    assert result["symbols_scanned"] == 1
    assert "buy_candidates" in result
    assert "sell_candidates" in result
    assert "hold_candidates" in result
    assert "skipped_symbols" in result
    for entry in result["buy_candidates"] + result["sell_candidates"]:
        probs = entry["probabilities"]
        assert abs((probs["buy"] + probs["sell"] + probs["hold"]) - 1.0) < 1e-9


def test_perform_scan_uses_lazy_default_fetcher(monkeypatch, make_ohlcv_frame) -> None:
    frame = make_ohlcv_frame(trend=0.12)

    monkeypatch.setattr(research, "get_open_positions", lambda: [])
    monkeypatch.setattr(research, "current_market_time", lambda: datetime(2026, 1, 1, 10, 0))
    monkeypatch.setattr(research, "is_market_closed", lambda _: False)
    monkeypatch.setattr(research, "market_time_str", lambda _: "2026-01-01 10:00:00 IST")

    calls = {"factory": 0}

    def _factory():
        calls["factory"] += 1
        return lambda symbol: frame

    monkeypatch.setattr(research, "_get_default_data_fetcher", _factory)

    result = research.perform_scan(scope="whole", symbols=["INFY"])

    assert calls["factory"] == 1
    assert result["symbols_scanned"] == 1


def test_perform_scan_returns_empty_payload_when_symbol_list_is_empty(monkeypatch) -> None:
    monkeypatch.setattr(research, "get_open_positions", lambda: [])
    monkeypatch.setattr(research, "current_market_time", lambda: datetime(2026, 1, 1, 10, 0))
    monkeypatch.setattr(research, "is_market_closed", lambda _: False)
    monkeypatch.setattr(research, "market_time_str", lambda _: "2026-01-01 10:00:00 IST")

    result = research.perform_scan(scope="whole", symbols=[], top_n=1)
    assert result["symbols_scanned"] == 0
    assert result["buy_candidates"] == []
    assert result["sell_candidates"] == []
    assert result["hold_candidates"] == []


def test_predictor_only_scan_does_not_read_positions_from_db(monkeypatch) -> None:
    monkeypatch.setenv("FIN_ASSIST_ENABLE_PERSISTENCE", "0")
    monkeypatch.setattr(
        research,
        "get_open_positions",
        lambda: (_ for _ in ()).throw(AssertionError("DB position read should not happen")),
    )
    monkeypatch.setattr(research, "current_market_time", lambda: datetime(2026, 1, 1, 10, 0))
    monkeypatch.setattr(research, "is_market_closed", lambda _: False)
    monkeypatch.setattr(research, "market_time_str", lambda _: "2026-01-01 10:00:00 IST")

    result = research.perform_scan(scope="whole", symbols=[], top_n=1)
    assert result["symbols_scanned"] == 0


def test_empty_payload_market_close_adds_exit_candidates(monkeypatch) -> None:
    monkeypatch.setenv("FIN_ASSIST_ENABLE_PERSISTENCE", "1")
    monkeypatch.setattr(research, "get_open_positions", lambda: [{"symbol": "INFY"}, {"symbol": "TCS"}])
    monkeypatch.setattr(research, "current_market_time", lambda: datetime(2026, 1, 1, 16, 0))
    monkeypatch.setattr(research, "is_market_closed", lambda _: True)
    monkeypatch.setattr(research, "market_time_str", lambda _: "2026-01-01 16:00:00 IST")

    result = research.perform_scan(scope="portfolio", symbols=[], top_n=1)
    symbols = {entry["symbol"] for entry in result["sell_candidates"]}
    assert symbols == {"INFY", "TCS"}


def test_perform_scan_forces_market_close_exit(monkeypatch, make_ohlcv_frame) -> None:
    monkeypatch.setenv("FIN_ASSIST_ENABLE_PERSISTENCE", "1")
    frame = make_ohlcv_frame(trend=0.10)

    monkeypatch.setattr(research, "get_open_positions", lambda: [{"symbol": "INFY"}])
    monkeypatch.setattr(research, "current_market_time", lambda: datetime(2026, 1, 1, 16, 0))
    monkeypatch.setattr(research, "is_market_closed", lambda _: True)
    monkeypatch.setattr(research, "market_time_str", lambda _: "2026-01-01 16:00:00 IST")

    def fetcher(symbol: str) -> pd.DataFrame:
        return frame

    result = research.perform_scan(scope="portfolio", symbols=["INFY"], data_fetcher=fetcher)
    assert any(
        candidate["symbol"] == "INFY" and candidate["confidence"].startswith("Market closed")
        for candidate in result["sell_candidates"]
    )


def test_perform_scan_filters_buys_when_buying_disabled(monkeypatch, make_ohlcv_frame) -> None:
    frame = make_ohlcv_frame(trend=0.12, volume=150_000)
    monkeypatch.setattr(research, "get_open_positions", lambda: [])
    monkeypatch.setattr(research, "current_market_time", lambda: datetime(2026, 1, 1, 10, 0))
    monkeypatch.setattr(research, "is_market_closed", lambda _: False)
    monkeypatch.setattr(research, "market_time_str", lambda _: "2026-01-01 10:00:00 IST")

    prediction = Prediction(
        symbol="INFY",
        price=101.0,
        action=PredictorAction.BUY,
        regime=MarketRegime.BULL,
        buy_probability=0.7,
        sell_probability=0.1,
        hold_probability=0.2,
        confidence=0.7,
        uncertainty=0.3,
        score=0.6,
        reasons=("regime=bull",),
    )

    class _StubPipeline:
        def __init__(self, data_fetcher, config):
            self.data_fetcher = data_fetcher
            self.config = config

        def run(self, symbols, open_positions, top_n, cross_asset_consensus):
            return PipelineRun(
                timestamp=datetime(2026, 1, 1, 10, 0),
                predictions=(prediction,),
                buy_candidates=(prediction,),
                sell_candidates=(),
                hold_candidates=(),
                skipped={},
            )

    monkeypatch.setattr(research, "PredictorPipeline", _StubPipeline)

    def fetcher(symbol: str) -> pd.DataFrame:
        return frame

    result = research.perform_scan(
        scope="whole",
        symbols=["INFY"],
        wallet=50.0,
        data_fetcher=fetcher,
    )
    assert result["buy_candidates"] == []
    assert result["filtered_buy_count"] == 1


def test_perform_scan_propagates_unexpected_pipeline_errors(monkeypatch) -> None:
    monkeypatch.setattr(research, "get_open_positions", lambda: [])
    monkeypatch.setattr(research, "current_market_time", lambda: datetime(2026, 1, 1, 10, 0))
    monkeypatch.setattr(research, "is_market_closed", lambda _: False)
    monkeypatch.setattr(research, "market_time_str", lambda _: "2026-01-01 10:00:00 IST")

    class _FailingPipeline:
        def __init__(self, data_fetcher, config):
            self.data_fetcher = data_fetcher
            self.config = config

        def run(self, symbols, open_positions, top_n, cross_asset_consensus):
            raise RuntimeError("unexpected pipeline bug")

    monkeypatch.setattr(research, "PredictorPipeline", _FailingPipeline)

    with pytest.raises(RuntimeError):
        research.perform_scan(scope="whole", symbols=["INFY"], data_fetcher=lambda symbol: pd.DataFrame())


def test_persist_scan_results_is_inert_when_disabled(monkeypatch) -> None:
    monkeypatch.delenv("FIN_ASSIST_ENABLE_PERSISTENCE", raising=False)
    research.persist_scan_results({"buy_candidates": [], "sell_candidates": []}, username="system")


def test_format_summary_text_includes_skipped_count() -> None:
    summary = research.format_summary_text(
        {
            "scope": "whole",
            "timestamp": "2026-01-01 10:00",
            "symbols_scanned": 5,
            "buy_candidates": [],
            "sell_candidates": [],
            "hold_candidates": [],
            "skipped_symbols": {"ABC": "error"},
        }
    )
    assert "Skipped symbols: 1" in summary


def test_budgeted_top_n_scales_with_wallet() -> None:
    assert research._budgeted_top_n(10, None) == 10
    assert research._budgeted_top_n(10, 0) == 5
    assert research._budgeted_top_n(10, 10_000) == 10
