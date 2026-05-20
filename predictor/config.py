"""Configuration objects and deterministic seeding for predictor pipeline."""

from __future__ import annotations

from dataclasses import dataclass
import os
import random


@dataclass(frozen=True)
class PredictorConfig:
    """Runtime configuration for the predictor-only inference pipeline."""

    min_price: float = 5.0
    min_avg_volume: float = 50_000.0
    min_atr_pct: float = 0.2
    max_atr_pct: float = 8.0
    top_n: int = 5
    min_confidence: float = 0.40
    max_uncertainty: float = 0.65
    regime_threshold: float = 0.002
    cross_asset_weight: float = 0.25
    random_seed: int = 7

    @classmethod
    def from_legacy_settings(cls) -> "PredictorConfig":
        """Build predictor configuration from the legacy `config.settings` module.

        This method keeps compatibility with existing environment-driven constants
        while centralizing predictor behavior in a typed object.
        """

        from config import settings as legacy

        return cls(
            min_price=float(getattr(legacy, "MIN_PRICE", cls.min_price)),
            min_avg_volume=float(getattr(legacy, "MIN_AVG_VOLUME", cls.min_avg_volume)),
            min_atr_pct=float(getattr(legacy, "MIN_ATR_PCT", cls.min_atr_pct)),
            max_atr_pct=float(getattr(legacy, "MAX_ATR_PCT", cls.max_atr_pct)),
            top_n=int(getattr(legacy, "TOP_N", cls.top_n)),
            random_seed=int(os.environ.get("FIN_ASSIST_RANDOM_SEED", cls.random_seed)),
        )


def seed_predictor_runtime(seed: int) -> None:
    """Seed optional randomness sources used by predictor dependencies.

    The core predictor path is deterministic by design, but seeding protects
    against accidental non-determinism from optional downstream libraries.
    """

    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ModuleNotFoundError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
    except ModuleNotFoundError:
        pass
