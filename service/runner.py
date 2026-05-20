"""Runner module for predictor scan execution.

Non-predictor side effects (Telegram notifications) are disabled unless
explicitly enabled via environment flags.
"""

from __future__ import annotations

import os
import time
from typing import Any, Mapping

from infra.logging import log
from service.research import perform_scan, persist_scan_results


def _telegram_enabled() -> bool:
    """Return whether Telegram notifications are enabled."""

    return os.environ.get("FIN_ASSIST_ENABLE_TELEGRAM", "0") == "1"


def _scan_stats_text(scan_result: Mapping[str, Any], duration_seconds: float) -> str:
    """Build a concise runtime summary for scan observability."""

    buys = len(scan_result.get("buy_candidates", []))
    sells = len(scan_result.get("sell_candidates", []))
    holds = len(scan_result.get("hold_candidates", []))
    skipped = len(scan_result.get("skipped_symbols", {}))
    scanned = int(scan_result.get("symbols_scanned", 0) or 0)
    return (
        "Scan summary: "
        f"scanned={scanned}, buy={buys}, sell={sells}, hold={holds}, skipped={skipped}, "
        f"duration={duration_seconds:.1f}s"
    )


def start_service() -> None:
    """Compatibility wrapper for one-shot predictor execution."""

    run_once()


def run_once():
    """Run a single predictor scan and optionally emit side effects."""

    log.info("Running predictor scan (runner.run_once)")
    started = time.perf_counter()

    scan_result = perform_scan(scope="whole")
    persist_scan_results(scan_result, username="system")

    if _telegram_enabled():
        from infra.telegram import send_message

        timestamp = scan_result.get("timestamp", "unknown")
        for sell in scan_result.get("sell_candidates", []):
            msg = (
                f"{sell['symbol']} - SELL\n"
                f"Confidence: {sell.get('confidence')}\n"
                f"Price: {sell.get('price')}\n"
                f"Time: {timestamp}"
            )
            send_message(msg)

        for buy in scan_result.get("buy_candidates", []):
            msg = (
                f"{buy['symbol']} - BUY\n"
                f"Confidence: {buy.get('confidence')}\n"
                f"Price: {buy.get('price')}\n"
                f"Time: {timestamp}"
            )
            send_message(msg)

    if not (scan_result.get("sell_candidates") or scan_result.get("buy_candidates")):
        log.info("No BUY/SELL signals were issued this scan.")

    elapsed = time.perf_counter() - started
    log.info(_scan_stats_text(scan_result, elapsed))
    log.info("Predictor scan complete.")
    return scan_result
