"""Tests for predictor configuration helpers."""

from __future__ import annotations

import random

from predictor.config import PredictorConfig, seed_predictor_runtime


def test_from_legacy_settings_returns_typed_config() -> None:
    config = PredictorConfig.from_legacy_settings()
    assert isinstance(config, PredictorConfig)
    assert config.top_n > 0


def test_seed_predictor_runtime_makes_random_reproducible() -> None:
    seed_predictor_runtime(99)
    first = random.random()
    seed_predictor_runtime(99)
    second = random.random()
    assert first == second
