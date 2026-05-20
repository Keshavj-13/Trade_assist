"""Database service wrapper with lazy backend loading.

The predictor pipeline can run without persistence; this wrapper defers backend
imports to avoid import-time hard dependency failures.
"""

from __future__ import annotations

from typing import Any, List, Optional

from infra.logging import log


def _persistence_required() -> bool:
    """Return whether persistence failures must be surfaced to callers."""

    import os

    return os.environ.get("FIN_ASSIST_ENABLE_PERSISTENCE", "0") == "1"


def _backend():
    """Load and return the persistence backend module lazily."""

    from infra import database

    return database


def init_db() -> None:
    """Initialize persistence schema if backend is available."""

    try:
        _backend().initialize_db()
    except Exception as exc:
        if _persistence_required():
            raise RuntimeError(f"Database initialization failed: {exc}") from exc
        log.warning(f"Database initialization skipped: {exc}")


def get_open_positions(user_id: Optional[int] = None) -> List[dict[str, Any]]:
    """Fetch currently open positions with safe fallback to empty list."""

    try:
        return _backend().get_open_positions(user_id)
    except Exception as exc:
        if _persistence_required():
            raise RuntimeError(f"Failed to fetch open positions: {exc}") from exc
        log.warning(f"Failed to fetch open positions: {exc}")
        return []
