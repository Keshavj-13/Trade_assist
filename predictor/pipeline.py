"""Predictor pipeline orchestration from market data to ranked predictions."""

from __future__ import annotations

from datetime import datetime, timezone
import logging
import os
from typing import Dict, Iterable, Mapping, Optional, Protocol

import pandas as pd

from predictor.config import PredictorConfig, seed_predictor_runtime
from predictor.errors import DataUnavailableError, InputValidationError, PredictorError
from predictor.features import build_feature_vector
from predictor.model import predict_probabilities
from predictor.types import PipelineRun, Prediction, PredictorAction
from predictor.validation import normalize_symbol, validate_ohlcv_frame, validate_symbol_list

_LOG = logging.getLogger("fin_assist.predictor.pipeline")


class DataFetcher(Protocol):
    """Callable contract for symbol -> OHLCV dataframe fetchers."""

    def __call__(self, symbol: str) -> pd.DataFrame:
        """Fetch OHLCV dataframe for a normalized symbol."""


class PredictorPipeline:
    """Deterministic, testable predictor pipeline with explicit contracts."""

    def __init__(self, data_fetcher: DataFetcher, config: Optional[PredictorConfig] = None):
        if not callable(data_fetcher):
            raise InputValidationError("data_fetcher must be callable")
        self._data_fetcher = data_fetcher
        self.config = config or PredictorConfig.from_legacy_settings()
        seed_predictor_runtime(self.config.random_seed)

    def predict_symbol(
        self,
        symbol: str,
        *,
        open_position: bool = False,
        cross_asset_consensus: Optional[float] = None,
    ) -> Prediction:
        """Run predictor inference for a single symbol.

        Raises:
            PredictorError: For invalid inputs or unavailable market data.
        """

        normalized_symbol = normalize_symbol(symbol)
        raw_df = self._data_fetcher(normalized_symbol)
        df = validate_ohlcv_frame(raw_df, normalized_symbol)
        features = build_feature_vector(df)

        if features.price < self.config.min_price:
            raise DataUnavailableError(
                f"{normalized_symbol}: price {features.price:.3f} < min_price"
            )
        if features.avg_volume < self.config.min_avg_volume:
            raise DataUnavailableError(
                f"{normalized_symbol}: avg_volume {features.avg_volume:.3f} < min_avg_volume"
            )
        if not (self.config.min_atr_pct <= features.atr_pct <= self.config.max_atr_pct):
            raise DataUnavailableError(
                f"{normalized_symbol}: atr_pct {features.atr_pct:.3f} outside allowed range"
            )

        return predict_probabilities(
            normalized_symbol,
            features,
            self.config,
            cross_asset_consensus=cross_asset_consensus,
            open_position=open_position,
        )

    def run(
        self,
        symbols: Iterable[str],
        *,
        open_positions: Optional[Iterable[str]] = None,
        top_n: Optional[int] = None,
        cross_asset_consensus: Optional[Mapping[str, float]] = None,
    ) -> PipelineRun:
        """Run scan inference for a symbol universe.

        Args:
            symbols: Universe to scan.
            open_positions: Current open position symbols.
            top_n: Buy-candidate cap.
            cross_asset_consensus: Optional symbol->signal map in [-1, 1].

        Returns:
            `PipelineRun` with predictions and skipped-symbol reasons.
        """

        normalized_symbols = validate_symbol_list(symbols)
        total_symbols = len(normalized_symbols)
        open_set = {
            normalize_symbol(symbol)
            for symbol in (open_positions or ())
        }

        predictions = []
        skipped: Dict[str, str] = {}
        cross_asset_consensus = cross_asset_consensus or {}

        progress_every = max(1, int(os.environ.get("FIN_ASSIST_PROGRESS_EVERY", "25")))

        for index, symbol in enumerate(normalized_symbols, start=1):
            try:
                prediction = self.predict_symbol(
                    symbol,
                    open_position=symbol in open_set,
                    cross_asset_consensus=cross_asset_consensus.get(symbol),
                )
                predictions.append(prediction)
            except PredictorError as exc:
                skipped[symbol] = str(exc)
            if index == 1 or index % progress_every == 0 or index == total_symbols:
                _LOG.info(
                    "Predictor progress: processed=%d/%d predicted=%d skipped=%d",
                    index,
                    total_symbols,
                    len(predictions),
                    len(skipped),
                )

        buy_candidates = [
            entry for entry in predictions if entry.action == PredictorAction.BUY
        ]
        sell_candidates = [
            entry for entry in predictions if entry.action == PredictorAction.SELL
        ]
        hold_candidates = [
            entry for entry in predictions if entry.action == PredictorAction.HOLD
        ]

        if top_n is None:
            buy_cap = self.config.top_n
        else:
            if not isinstance(top_n, int):
                raise InputValidationError("top_n must be an integer")
            if top_n <= 0:
                raise InputValidationError("top_n must be > 0")
            buy_cap = top_n
        sorted_buys = tuple(
            sorted(
                buy_candidates,
                key=lambda entry: entry.buy_probability,
                reverse=True,
            )[:buy_cap]
        )
        sorted_sells = tuple(
            sorted(
                sell_candidates,
                key=lambda entry: entry.sell_probability,
                reverse=True,
            )
        )
        sorted_holds = tuple(
            sorted(
                hold_candidates,
                key=lambda entry: entry.symbol,
            )
        )

        return PipelineRun(
            timestamp=datetime.now(timezone.utc),
            predictions=tuple(predictions),
            buy_candidates=sorted_buys,
            sell_candidates=sorted_sells,
            hold_candidates=sorted_holds,
            skipped=skipped,
        )
