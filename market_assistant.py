"""CLI launcher for predictor-centric fin_assist runtime.

By default this CLI runs predictor inference once and keeps other operational
modes inactive unless explicitly enabled.
"""

from __future__ import annotations

import argparse
import os

from infra.logging import setup_logging
from service.database import init_db
from service.runner import run_once


def _non_predictor_enabled() -> bool:
    """Return whether legacy non-predictor runtime modes are enabled."""

    return os.environ.get("FIN_ASSIST_ENABLE_NON_PREDICTOR", "0") == "1"


def _persistence_enabled() -> bool:
    """Return whether persistence side effects are enabled."""

    return os.environ.get("FIN_ASSIST_ENABLE_PERSISTENCE", "0") == "1"


def _maybe_init_db() -> None:
    """Initialize DB only when persistence is explicitly enabled."""

    if _persistence_enabled():
        init_db()


def _run_once() -> None:
    setup_logging()
    _maybe_init_db()
    run_once()


def _run_daemon() -> None:
    setup_logging()
    _maybe_init_db()
    from service.daemon import run_forever

    run_forever()


def _run_scheduler() -> None:
    setup_logging()
    _maybe_init_db()
    from service.scheduler import market_scheduler_loop

    market_scheduler_loop()


def _run_telegram() -> None:
    setup_logging()
    _maybe_init_db()
    from service.telegram_bot import telegram_listener_loop

    telegram_listener_loop()


def main() -> None:
    """Parse CLI args and run requested mode."""

    allowed_modes = ["once"]
    if _non_predictor_enabled():
        allowed_modes.extend(["daemon", "scheduler", "telegram"])

    parser = argparse.ArgumentParser(prog="market_assistant")
    parser.add_argument(
        "mode",
        nargs="?",
        choices=allowed_modes,
        default="once",
        help=(
            "Mode to run. Default is 'once'. "
            "Legacy modes require FIN_ASSIST_ENABLE_NON_PREDICTOR=1."
        ),
    )
    args = parser.parse_args()

    if args.mode == "once":
        _run_once()
    elif args.mode == "daemon":
        _run_daemon()
    elif args.mode == "scheduler":
        _run_scheduler()
    elif args.mode == "telegram":
        _run_telegram()


if __name__ == "__main__":
    main()
