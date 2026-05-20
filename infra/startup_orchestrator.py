"""Startup orchestrator: deterministic state machine for predictor runtime.

Stages:
1. Environment validation (StartupConfig.validate)
2. Infrastructure validation (InfrastructureValidator)
3. Infrastructure startup (if required)
4. Predictor execution (main.py)

No recovery logic. Fail fast on invalid state.
"""

from __future__ import annotations

import os
import sys

from infra.logging import log
from infra.startup_config import RuntimeMode, StartupConfig, StartupError
from infra.startup_validator import InfrastructureValidator


class StartupOrchestrator:
    """Deterministic startup orchestration."""

    @staticmethod
    def run() -> None:
        """Execute full startup sequence.
        
        Raises:
            StartupError: On configuration or infrastructure failure.
        """
        # Stage 1: Validate environment
        log.info("=== Startup Stage 1: Environment Validation ===")
        try:
            config = StartupConfig.validate()
            log.info(f"Configuration valid: {config}")
        except StartupError as e:
            log.error(f"Configuration validation failed: {e}")
            sys.exit(1)

        # Stage 2: Infrastructure validation and startup
        log.info("=== Startup Stage 2: Infrastructure Validation ===")
        if config.mode == RuntimeMode.PREDICTOR_WITH_PERSISTENCE:
            try:
                status = InfrastructureValidator.validate_and_start_oracle(config.oracle_config)
                if not status.oracle_listener_available:
                    raise StartupError(
                        f"Oracle listener not available after startup attempt (DSN: {config.oracle_config.dsn})"
                    )
                log.info("Oracle infrastructure ready")
            except StartupError as e:
                log.error(f"Infrastructure validation failed: {e}")
                sys.exit(1)
        else:
            log.info("Running in predictor-only mode (no persistence)")

        # Stage 3: Set environment for predictor execution
        log.info("=== Startup Stage 3: Environment Setup ===")
        os.environ["FIN_ASSIST_ENABLE_PERSISTENCE"] = "1" if config.enable_persistence else "0"
        if config.oracle_config:
            os.environ["ORACLE_DB_USER"] = config.oracle_config.user
            os.environ["ORACLE_DB_PASSWORD"] = config.oracle_config.password
            os.environ["ORACLE_DB_DSN"] = config.oracle_config.dsn
        log.info("Environment configured")

        # Stage 4: Return to caller (main.py will continue)
        log.info("=== Startup complete; entering predictor runtime ===")
