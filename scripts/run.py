#!/usr/bin/env python3
"""Predictor startup entrypoint.

Usage:
    python scripts/run.py              # predictor-only mode (default)
    FIN_ASSIST_ENABLE_PERSISTENCE=1 ORACLE_DB_USER=... ORACLE_DB_PASSWORD=... ORACLE_DB_DSN=... python scripts/run.py
"""

import sys
from pathlib import Path
from typing import Any, Mapping

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from main import main as predictor_main


def _print_summary(scan_result: Mapping[str, Any]) -> None:
    """Print a concise CLI summary for one-shot runs."""

    scanned = int(scan_result.get("symbols_scanned", 0) or 0)
    buys = scan_result.get("buy_candidates", [])
    sells = scan_result.get("sell_candidates", [])
    holds = scan_result.get("hold_candidates", [])
    skipped = scan_result.get("skipped_symbols", {})

    print(
        "\n" +
        "Scan finished | "
        f"scanned={scanned} "
        f"buy={len(buys)} "
        f"sell={len(sells)} "
        f"hold={len(holds)} "
        f"skipped={len(skipped)}"
    )

    if buys:
        preview = ", ".join(f"{entry['symbol']}@{entry.get('price')}" for entry in buys[:5])
        print(f"Top BUY: {preview}")
    if sells:
        preview = ", ".join(f"{entry['symbol']}@{entry.get('price')}" for entry in sells[:5])
        print(f"Top SELL: {preview}")


if __name__ == "__main__":
    try:
        result = predictor_main()
        if isinstance(result, dict):
            _print_summary(result)
    except KeyboardInterrupt:
        print("\nShutdown requested")
        sys.exit(130)
    except Exception as e:
        print(f"Fatal error: {e}", file=sys.stderr)
        sys.exit(1)
