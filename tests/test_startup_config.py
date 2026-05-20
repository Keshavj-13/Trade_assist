"""Tests for startup configuration and orchestration.

Verifies:
1. Predictor-only mode works without Oracle.
2. Invalid Oracle config fails clearly.
3. Partial Oracle credentials fail.
4. Docker absence does not break predictor-only mode.
5. Persistence requires valid credentials.
6. Startup behavior follows explicit configuration.
"""

from __future__ import annotations

import os
import pytest

from infra.startup_config import OracleConfig, RuntimeMode, StartupConfig, StartupError


class TestStartupConfigValidation:
    """Test configuration validation logic."""

    def test_predictor_only_mode_no_oracle_vars(self, monkeypatch) -> None:
        """Verify predictor-only mode when no Oracle credentials are set."""
        monkeypatch.delenv("FIN_ASSIST_ENABLE_PERSISTENCE", raising=False)
        monkeypatch.delenv("ORACLE_DB_USER", raising=False)
        monkeypatch.delenv("ORACLE_DB_PASSWORD", raising=False)
        monkeypatch.delenv("ORACLE_DB_DSN", raising=False)

        config = StartupConfig.validate()

        assert config.mode == RuntimeMode.PREDICTOR_ONLY
        assert config.enable_persistence is False
        assert config.oracle_config is None

    def test_predictor_only_explicitly_disabled(self, monkeypatch) -> None:
        """Verify predictor-only when persistence is explicitly disabled."""
        monkeypatch.setenv("FIN_ASSIST_ENABLE_PERSISTENCE", "0")
        monkeypatch.delenv("ORACLE_DB_USER", raising=False)
        monkeypatch.delenv("ORACLE_DB_PASSWORD", raising=False)
        monkeypatch.delenv("ORACLE_DB_DSN", raising=False)

        config = StartupConfig.validate()

        assert config.mode == RuntimeMode.PREDICTOR_ONLY
        assert config.enable_persistence is False
        assert config.oracle_config is None

    def test_persistence_requires_all_credentials(self, monkeypatch) -> None:
        """Verify persistence mode requires user, password, and DSN."""
        monkeypatch.setenv("FIN_ASSIST_ENABLE_PERSISTENCE", "1")
        monkeypatch.setenv("ORACLE_DB_USER", "test_user")
        monkeypatch.setenv("ORACLE_DB_PASSWORD", "test_pass")
        monkeypatch.setenv("ORACLE_DB_DSN", "localhost:1521/orcl")

        config = StartupConfig.validate()

        assert config.mode == RuntimeMode.PREDICTOR_WITH_PERSISTENCE
        assert config.enable_persistence is True
        assert config.oracle_config is not None
        assert config.oracle_config.user == "test_user"

    def test_persistence_without_credentials_fails(self, monkeypatch) -> None:
        """Verify persistence mode without credentials raises error."""
        monkeypatch.setenv("FIN_ASSIST_ENABLE_PERSISTENCE", "1")
        monkeypatch.delenv("ORACLE_DB_USER", raising=False)
        monkeypatch.delenv("ORACLE_DB_PASSWORD", raising=False)
        monkeypatch.delenv("ORACLE_DB_DSN", raising=False)

        with pytest.raises(StartupError, match="Persistence enabled.*credentials not set"):
            StartupConfig.validate()

    def test_partial_oracle_credentials_fails(self, monkeypatch) -> None:
        """Verify partial Oracle credentials raise error."""
        monkeypatch.delenv("FIN_ASSIST_ENABLE_PERSISTENCE", raising=False)
        monkeypatch.setenv("ORACLE_DB_USER", "test_user")
        monkeypatch.setenv("ORACLE_DB_PASSWORD", "test_pass")
        monkeypatch.delenv("ORACLE_DB_DSN", raising=False)

        with pytest.raises(StartupError, match="Partial Oracle credentials"):
            OracleConfig.from_env()

    def test_invalid_persistence_flag(self, monkeypatch) -> None:
        """Verify invalid persistence flag raises error."""
        monkeypatch.setenv("FIN_ASSIST_ENABLE_PERSISTENCE", "yes")
        monkeypatch.delenv("ORACLE_DB_USER", raising=False)
        monkeypatch.delenv("ORACLE_DB_PASSWORD", raising=False)
        monkeypatch.delenv("ORACLE_DB_DSN", raising=False)

        with pytest.raises(StartupError, match="Invalid FIN_ASSIST_ENABLE_PERSISTENCE"):
            StartupConfig.validate()

    def test_invalid_dsn_format_fails(self, monkeypatch) -> None:
        """Verify invalid DSN format raises error."""
        monkeypatch.setenv("ORACLE_DB_USER", "test_user")
        monkeypatch.setenv("ORACLE_DB_PASSWORD", "test_pass")
        monkeypatch.setenv("ORACLE_DB_DSN", "invalid_dsn_format")

        with pytest.raises(StartupError, match="Invalid Oracle DSN format"):
            OracleConfig.from_env()

    def test_valid_dsn_formats_accepted(self, monkeypatch) -> None:
        """Verify valid DSN formats are accepted."""
        test_dsns = [
            "localhost:1521/orcl",
            "db.example.com:1521/PROD",
            "192.168.1.100:1521/test",
            "host:1234/service",
        ]
        for dsn in test_dsns:
            monkeypatch.setenv("ORACLE_DB_USER", "test_user")
            monkeypatch.setenv("ORACLE_DB_PASSWORD", "test_pass")
            monkeypatch.setenv("ORACLE_DB_DSN", dsn)

            config = OracleConfig.from_env()
            assert config is not None
            assert config.dsn == dsn

    def test_oracle_credentials_provided_but_persistence_disabled(self, monkeypatch) -> None:
        """Verify Oracle vars can be set while persistence is disabled (explicit flag wins)."""
        monkeypatch.setenv("FIN_ASSIST_ENABLE_PERSISTENCE", "0")
        monkeypatch.setenv("ORACLE_DB_USER", "test_user")
        monkeypatch.setenv("ORACLE_DB_PASSWORD", "test_pass")
        monkeypatch.setenv("ORACLE_DB_DSN", "localhost:1521/orcl")

        # Should not raise; explicit flag takes precedence
        config = StartupConfig.validate()

        assert config.mode == RuntimeMode.PREDICTOR_ONLY
        assert config.enable_persistence is False

    def test_invalid_oracle_env_ignored_when_persistence_disabled(self, monkeypatch) -> None:
        """Predictor-only mode must ignore invalid Oracle placeholders."""
        monkeypatch.setenv("FIN_ASSIST_ENABLE_PERSISTENCE", "0")
        monkeypatch.setenv("ORACLE_DB_USER", "your_user")
        monkeypatch.setenv("ORACLE_DB_PASSWORD", "your_password")
        monkeypatch.setenv("ORACLE_DB_DSN", "your_dsn")

        config = StartupConfig.validate()
        assert config.mode == RuntimeMode.PREDICTOR_ONLY
        assert config.enable_persistence is False
        assert config.oracle_config is None

    def test_config_string_representation(self, monkeypatch) -> None:
        """Verify config string representation for logging."""
        monkeypatch.setenv("FIN_ASSIST_ENABLE_PERSISTENCE", "0")
        monkeypatch.delenv("ORACLE_DB_USER", raising=False)
        monkeypatch.delenv("ORACLE_DB_PASSWORD", raising=False)
        monkeypatch.delenv("ORACLE_DB_DSN", raising=False)

        config = StartupConfig.validate()

        assert "predictor-only" in str(config)

    def test_config_string_with_persistence(self, monkeypatch) -> None:
        """Verify config string with persistence."""
        monkeypatch.setenv("FIN_ASSIST_ENABLE_PERSISTENCE", "1")
        monkeypatch.setenv("ORACLE_DB_USER", "test_user")
        monkeypatch.setenv("ORACLE_DB_PASSWORD", "test_pass")
        monkeypatch.setenv("ORACLE_DB_DSN", "localhost:1521/orcl")

        config = StartupConfig.validate()

        assert "persistence" in str(config)
        assert "localhost:1521/orcl" in str(config)
