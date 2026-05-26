"""Error hierarchy for predictor strategy research and validation workflows."""

from __future__ import annotations


class ResearchError(Exception):
    """Base class for strategy research errors."""


class ResearchInputError(ResearchError):
    """Raised when caller inputs violate research framework contracts."""


class ResearchDataError(ResearchError):
    """Raised when historical market data is unavailable or malformed."""


class ResearchValidationError(ResearchError):
    """Raised when validation stages cannot run to completion."""


class DiagnosticAssertionError(ResearchError):
    """Raised when an impossible pipeline state is detected during diagnostics.

    Examples of impossible states:
    - entries_generated > 0 but trade_count == 0
    - trade_count > 0 but equity_curve is flat at 1.0
    """
