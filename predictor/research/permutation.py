"""Permutation and resampling utilities for anti-data-mining validation."""

from __future__ import annotations

from typing import Callable, List

import numpy as np
import pandas as pd

from predictor.research.data import validate_research_frame
from predictor.research.errors import ResearchInputError
from predictor.research.types import PermutationTestResult


def _circular_block_indices(length: int, block_size: int, rng: np.random.Generator) -> np.ndarray:
    """Sample circular contiguous blocks until `length` indices are produced."""

    if length <= 0:
        return np.array([], dtype=int)
    if block_size <= 0:
        raise ResearchInputError("block_size must be > 0")
    result: List[int] = []
    while len(result) < length:
        start = int(rng.integers(0, length))
        block = [(start + offset) % length for offset in range(block_size)]
        result.extend(block)
    return np.asarray(result[:length], dtype=int)


def block_permute_ohlcv(
    frame: pd.DataFrame,
    *,
    block_size: int,
    seed: int,
) -> pd.DataFrame:
    """Create a pseudo-market path by block-permuting returns and OHLCV shape."""

    validated = validate_research_frame(frame, symbol="PERMUTE", min_rows=3)
    if block_size <= 0:
        raise ResearchInputError("block_size must be > 0")

    close = validated["Close"].astype(float).to_numpy()
    open_ratio = (validated["Open"] / validated["Close"]).to_numpy()
    high_ratio = (validated["High"] / validated["Close"]).to_numpy()
    low_ratio = (validated["Low"] / validated["Close"]).to_numpy()
    volume = validated["Volume"].astype(float).to_numpy()

    rng = np.random.default_rng(seed)
    n = len(validated)
    # Keep the anchor point fixed and permute subsequent bars in blocks.
    tail_indices = _circular_block_indices(n - 1, min(block_size, n - 1), rng)

    log_close = np.log(close)
    log_returns = np.diff(log_close, prepend=log_close[0])
    permuted_log_returns = log_returns.copy()
    permuted_log_returns[1:] = log_returns[1:][tail_indices]
    permuted_log_close = log_close[0] + np.cumsum(permuted_log_returns)
    permuted_close = np.exp(permuted_log_close)

    perm_open_ratio = open_ratio.copy()
    perm_high_ratio = high_ratio.copy()
    perm_low_ratio = low_ratio.copy()
    perm_volume = volume.copy()
    perm_open_ratio[1:] = open_ratio[1:][tail_indices]
    perm_high_ratio[1:] = high_ratio[1:][tail_indices]
    perm_low_ratio[1:] = low_ratio[1:][tail_indices]
    perm_volume[1:] = volume[1:][tail_indices]

    perm_open = permuted_close * perm_open_ratio
    perm_high = permuted_close * perm_high_ratio
    perm_low = permuted_close * perm_low_ratio

    # Enforce OHLC consistency after ratio permutation.
    perm_high = np.maximum.reduce([perm_high, perm_open, permuted_close, perm_low])
    perm_low = np.minimum.reduce([perm_low, perm_open, permuted_close, perm_high])

    permuted = pd.DataFrame(
        {
            "Open": perm_open,
            "High": perm_high,
            "Low": perm_low,
            "Close": permuted_close,
            "Volume": np.maximum(perm_volume, 0.0),
        },
        index=validated.index,
    )
    return validate_research_frame(permuted, symbol="PERMUTED", min_rows=3)


def run_permutation_test(
    *,
    frame: pd.DataFrame,
    observed_statistic: float,
    score_on_frame: Callable[[pd.DataFrame], float],
    iterations: int,
    block_size: int,
    seed: int,
    p_value_threshold: float,
) -> PermutationTestResult:
    """Evaluate whether observed performance exceeds a block-permuted null."""

    if iterations <= 0:
        raise ResearchInputError("iterations must be > 0")
    if not 0.0 < p_value_threshold < 1.0:
        raise ResearchInputError("p_value_threshold must be between 0 and 1")

    rng = np.random.default_rng(seed)
    null_values: List[float] = []
    for _ in range(iterations):
        perm_seed = int(rng.integers(0, 2**31 - 1))
        permuted = block_permute_ohlcv(frame, block_size=block_size, seed=perm_seed)
        statistic = float(score_on_frame(permuted))
        if not np.isfinite(statistic):
            statistic = 0.0
        null_values.append(statistic)

    extremes = sum(value >= observed_statistic for value in null_values)
    p_value = float((extremes + 1) / (len(null_values) + 1))
    return PermutationTestResult(
        observed_statistic=float(observed_statistic),
        null_distribution=tuple(float(value) for value in null_values),
        p_value=p_value,
        passes=p_value <= p_value_threshold,
    )
