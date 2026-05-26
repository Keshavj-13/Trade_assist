"""Cross-sectional normalization utilities.

Responsibility: transform raw factor values into comparable cross-sectional
scores without performing ranking or evaluation.
"""

from __future__ import annotations

import pandas as pd


def zscore_cross_section(series: pd.Series) -> pd.Series:
    """Return z-scored cross-sectional values for one date.

    If dispersion is zero, returns zeros for all finite entries.
    """
    values = series.astype(float)
    std = float(values.std(ddof=0))
    if std <= 0.0:
        return pd.Series(0.0, index=values.index, dtype=float)
    mean = float(values.mean())
    return (values - mean) / std
