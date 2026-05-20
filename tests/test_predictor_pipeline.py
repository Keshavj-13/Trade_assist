"""Integration-style tests for predictor pipeline orchestration."""

from __future__ import annotations

import pandas as pd
import pytest

from predictor.config import PredictorConfig
from predictor.errors import InputValidationError
from predictor.pipeline import PredictorPipeline


def test_pipeline_skips_invalid_data_and_caps_top_n(make_ohlcv_frame) -> None:
    good_a = make_ohlcv_frame(trend=0.12, volume=130_000)
    good_b = make_ohlcv_frame(trend=0.11, volume=125_000)
    low_volume = make_ohlcv_frame(trend=0.10, volume=500)
    short_frame = make_ohlcv_frame(rows=20)

    data = {
        "A": good_a,
        "B": good_b,
        "LOWVOL": low_volume,
        "SHORT": short_frame,
    }

    def fetcher(symbol: str) -> pd.DataFrame:
        return data[symbol]

    pipeline = PredictorPipeline(
        fetcher,
        config=PredictorConfig(
            top_n=1,
            min_confidence=0.20,
            max_uncertainty=1.0,
            min_avg_volume=1_000,
        ),
    )

    run = pipeline.run(["A", "B", "LOWVOL", "SHORT"], top_n=1)

    assert len(run.buy_candidates) <= 1
    assert "LOWVOL" in run.skipped
    assert "SHORT" in run.skipped


def test_pipeline_results_are_deterministic(make_ohlcv_frame) -> None:
    frame = make_ohlcv_frame(trend=0.08)

    def fetcher(symbol: str) -> pd.DataFrame:
        return frame

    pipeline = PredictorPipeline(
        fetcher,
        config=PredictorConfig(min_confidence=0.20, max_uncertainty=1.0),
    )

    run_a = pipeline.run(["INFY", "TCS"])
    run_b = pipeline.run(["INFY", "TCS"])

    snapshot_a = [
        (pred.symbol, pred.action.value, round(pred.buy_probability, 8), round(pred.sell_probability, 8))
        for pred in run_a.predictions
    ]
    snapshot_b = [
        (pred.symbol, pred.action.value, round(pred.buy_probability, 8), round(pred.sell_probability, 8))
        for pred in run_b.predictions
    ]
    assert snapshot_a == snapshot_b


def test_pipeline_open_position_prevents_buy_signal(make_ohlcv_frame) -> None:
    frame = make_ohlcv_frame(trend=0.15)

    def fetcher(symbol: str) -> pd.DataFrame:
        return frame

    pipeline = PredictorPipeline(
        fetcher,
        config=PredictorConfig(min_confidence=0.20, max_uncertainty=1.0),
    )

    run = pipeline.run(["INFY"], open_positions=["INFY"])
    actions = {pred.symbol: pred.action.value for pred in run.predictions}
    assert actions["INFY"] == "HOLD"


def test_pipeline_requires_callable_fetcher() -> None:
    with pytest.raises(InputValidationError):
        PredictorPipeline(None)


def test_pipeline_rejects_invalid_top_n(make_ohlcv_frame) -> None:
    frame = make_ohlcv_frame()

    def fetcher(symbol: str) -> pd.DataFrame:
        return frame

    pipeline = PredictorPipeline(fetcher, config=PredictorConfig())

    with pytest.raises(InputValidationError):
        pipeline.run(["INFY"], top_n=0)


def test_pipeline_propagates_unexpected_fetcher_errors() -> None:
    def fetcher(symbol: str) -> pd.DataFrame:
        raise RuntimeError("unexpected failure")

    pipeline = PredictorPipeline(fetcher, config=PredictorConfig())
    with pytest.raises(RuntimeError):
        pipeline.run(["INFY"])
