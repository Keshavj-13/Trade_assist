"""Predictor-specific exception hierarchy.

These exceptions provide explicit failure semantics for input validation,
data availability, and inference lifecycle failures.
"""


class PredictorError(Exception):
    """Base class for all predictor pipeline exceptions."""


class InputValidationError(PredictorError):
    """Raised when caller input violates predictor interface contracts."""


class DataUnavailableError(PredictorError):
    """Raised when market data is unavailable or insufficient for inference."""
