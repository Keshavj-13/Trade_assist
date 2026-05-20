"""Predictor entrypoint for one-shot scan execution."""

from __future__ import annotations

import os

from infra.logging import setup_logging
from infra.startup_orchestrator import StartupOrchestrator
from service.database import init_db
from service.runner import run_once


def _persistence_enabled() -> bool:
    """Return whether persistence side effects are enabled."""

    return os.environ.get("FIN_ASSIST_ENABLE_PERSISTENCE", "0") == "1"


def main():
    """Run predictor pipeline and return scan payload."""
    setup_logging()
    
    # Stage 1-3: Validate config, start infrastructure, set environment
    StartupOrchestrator.run()
    
    # Stage 4: Initialize DB if persistence is enabled
    if _persistence_enabled():
        init_db()
    
    # Stage 5: Run predictor
    return run_once()


if __name__ == "__main__":
    main()
