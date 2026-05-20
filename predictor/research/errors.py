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
