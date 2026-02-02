"""
Master entrypoint for fin_assist.
Initializes logging, database, and runs one market scan.
"""

from infra.logging import setup_logging
from service.database import init_db
from service.runner import run_once

if __name__ == "__main__":
    setup_logging()
    init_db()
    run_once()
