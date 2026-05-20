"""Startup configuration validation.

Strict validation-first state machine for determining predictor runtime mode
and optional persistence infrastructure requirements.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class RuntimeMode(Enum):
    """Predictor execution mode."""
    PREDICTOR_ONLY = "predictor_only"  # No persistence
    PREDICTOR_WITH_PERSISTENCE = "predictor_with_persistence"  # Oracle enabled


class StartupError(Exception):
    """Configuration or infrastructure validation failure."""


@dataclass(frozen=True)
class OracleConfig:
    """Oracle credentials and DSN."""
    user: str
    password: str
    dsn: str

    @staticmethod
    def from_env() -> Optional[OracleConfig]:
        """Load Oracle config from environment variables.
        
        Returns None if any credential is missing.
        Raises StartupError if credentials are partial or DSN is invalid.
        """
        user = os.getenv("ORACLE_DB_USER")
        password = os.getenv("ORACLE_DB_PASSWORD")
        dsn = os.getenv("ORACLE_DB_DSN")

        # All missing = not configured (ok for predictor-only)
        if not user and not password and not dsn:
            return None

        # Partial config = error
        if not user or not password or not dsn:
            missing = [k for k, v in [("ORACLE_DB_USER", user), ("ORACLE_DB_PASSWORD", password), ("ORACLE_DB_DSN", dsn)] if not v]
            raise StartupError(f"Partial Oracle credentials: missing {missing}. Set all or none.")

        # Validate DSN format (basic check: must contain colon and alphanumeric)
        if ":" not in dsn or not any(c.isalnum() for c in dsn):
            raise StartupError(f"Invalid Oracle DSN format: '{dsn}' (expected 'host:port/service' or similar)")

        return OracleConfig(user=user, password=password, dsn=dsn)


@dataclass(frozen=True)
class StartupConfig:
    """Validated startup configuration."""
    mode: RuntimeMode
    oracle_config: Optional[OracleConfig]
    enable_persistence: bool

    @staticmethod
    def validate() -> StartupConfig:
        """Validate environment and determine runtime mode.
        
        Returns:
            StartupConfig with validated mode and optional Oracle config.
            
        Raises:
            StartupError: If configuration is invalid or contradictory.
        """
        # Check explicit persistence flag
        persistence_flag = os.getenv("FIN_ASSIST_ENABLE_PERSISTENCE", "0")
        if persistence_flag not in ("0", "1"):
            raise StartupError(
                f"Invalid FIN_ASSIST_ENABLE_PERSISTENCE='{persistence_flag}'; expected '0' or '1'"
            )
        enable_persistence = persistence_flag == "1"

        # Parse Oracle credentials only when persistence is explicitly enabled.
        # Predictor-only mode should not fail due to placeholder or partial
        # Oracle environment variables.
        if enable_persistence:
            oracle_config = OracleConfig.from_env()
            if oracle_config is None:
                raise StartupError(
                    "Persistence enabled (FIN_ASSIST_ENABLE_PERSISTENCE=1) but "
                    "Oracle credentials not set (ORACLE_DB_USER, ORACLE_DB_PASSWORD, ORACLE_DB_DSN required)"
                )
        else:
            oracle_config = None

        # Determine mode
        if enable_persistence and oracle_config:
            mode = RuntimeMode.PREDICTOR_WITH_PERSISTENCE
        else:
            mode = RuntimeMode.PREDICTOR_ONLY

        return StartupConfig(
            mode=mode,
            oracle_config=oracle_config,
            enable_persistence=enable_persistence,
        )

    def __str__(self) -> str:
        """Human-readable config summary."""
        if self.mode == RuntimeMode.PREDICTOR_ONLY:
            return "Mode: predictor-only (no persistence)"
        else:
            return f"Mode: predictor + persistence (Oracle DSN: {self.oracle_config.dsn})"
