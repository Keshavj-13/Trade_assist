"""
Helper CLI for the Node frontend to trigger research scans.
"""

import argparse
import json

from infra.user_store import get_wallet
from service.research import format_summary_text, perform_scan, persist_scan_results


def main():
    parser = argparse.ArgumentParser(description="Run a research scan")
    parser.add_argument("--scope", choices=("whole", "portfolio"), default="whole")
    parser.add_argument("--wallet", type=float, default=0.0)
    parser.add_argument("--username", default="system")
    args = parser.parse_args()

    scan = perform_scan(scope=args.scope, wallet=args.wallet)
    persist_scan_results(scan, username=args.username)
    balance = get_wallet(args.username)
    scan["summary"] = format_summary_text(scan)
    scan["wallet"] = balance
    print(json.dumps(scan, default=str))


if __name__ == "__main__":
    main()
