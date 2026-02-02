"""Compatibility launcher for fin_assist.

Provides a small CLI to run a single analysis or start one of the
long-running service components. Defaults to one-shot analysis.
"""

import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from infra.logging import setup_logging
from service.database import init_db
from service.runner import run_once

def _run_once():
    setup_logging()
    init_db()
    run_once()


def _run_daemon():
    setup_logging()
    init_db()
    from service.daemon import run_forever
    run_forever()


def _run_scheduler():
    setup_logging()
    init_db()
    from service.scheduler import market_scheduler_loop
    market_scheduler_loop()


def _run_telegram():
    setup_logging()
    init_db()
    from service.telegram_bot import telegram_listener_loop
    telegram_listener_loop()


def main():
    p = argparse.ArgumentParser(prog="market_assistant")
    p.add_argument("mode", nargs="?", choices=["once", "daemon", "scheduler", "telegram"], default="once",
                   help="Mode to run: 'once' runs analysis once; 'daemon' runs full daemon; 'scheduler' runs scheduler; 'telegram' runs telegram listener")
    args = p.parse_args()

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
