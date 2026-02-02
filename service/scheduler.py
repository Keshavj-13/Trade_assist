"""
Scheduler module for market tasks.
"""

import time
from infra.logging import log
from service.runner import run_once


def market_scheduler_loop():
    interval = 300
    log.info("Market scheduler started.")
    while True:
        start = time.time()
        try:
            run_once()
        except Exception as e:
            log.error(f"Scheduler run_once error: {e}", exc_info=True)
        elapsed = time.time() - start
        sleep_time = max(1, interval - elapsed)
        time.sleep(sleep_time)
