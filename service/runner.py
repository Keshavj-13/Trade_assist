"""
Runner module for starting the service.
"""

from infra.telegram import send_message
from infra.logging import log
from service.research import perform_scan, persist_scan_results


def start_service():
    pass


def run_once():
    log.info("Running market scan (runner.run_once)")
    scan_result = perform_scan(scope="whole")
    persist_scan_results(scan_result)
    timestamp = scan_result.get("timestamp")
    if not timestamp:
        timestamp = "unknown"

    for sell in scan_result.get("sell_candidates", []):
        msg = (
            f"{sell['symbol']} — SELL\n"
            f"Confidence: {sell['confidence']}\n"
            f"Price: {sell['price']}\n"
            f"Time: {timestamp}"
        )
        send_message(msg)
        log.info(
            f"{sell['symbol']}: SELL decision sent. "
            f"Confidence: {sell['confidence']}, price={sell['price']}"
        )

    for buy in scan_result.get("buy_candidates", []):
        msg = (
            f"{buy['symbol']} — BUY\n"
            f"Confidence: {buy['confidence']}\n"
            f"Price: {buy['price']}\n"
            f"Time: {timestamp}"
        )
        send_message(msg)
        log.info(
            f"{buy['symbol']}: BUY decision sent. "
            f"Confidence: {buy['confidence']}, price={buy['price']}"
        )

    if scan_result.get("filtered_buy_count"):
        log.info(f"Filtered {scan_result['filtered_buy_count']} extra buy candidates after TOP_N cap.")

    if not (scan_result.get("sell_candidates") or scan_result.get("buy_candidates")):
        log.debug("No BUY/SELL signals were issued this scan.")

    log.info("Market scan complete.")
