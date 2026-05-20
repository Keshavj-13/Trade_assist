"""Infrastructure availability discovery.

Checks for Oracle listener, Docker, and systemd WITHOUT assuming they exist.
Does NOT attempt auto-recovery or hiding failures.
"""

from __future__ import annotations

import shutil
import socket
import subprocess
from dataclasses import dataclass
from typing import Optional

from infra.logging import log
from infra.startup_config import OracleConfig, StartupError


@dataclass(frozen=True)
class InfrastructureStatus:
    """Current infrastructure availability state."""
    oracle_listener_available: bool
    oracle_started_successfully: bool


class InfrastructureValidator:
    """Deterministic infrastructure validation without hidden recovery."""

    @staticmethod
    def _can_reach_oracle_listener(oracle_config: OracleConfig, timeout: int = 2) -> bool:
        """Check if Oracle listener is reachable.
        
        Parses DSN to extract host and port, attempts TCP connect.
        Returns False if host/port cannot be extracted or connection fails.
        """
        try:
            # Parse DSN format: "host:port/service" or similar
            parts = oracle_config.dsn.split(":")
            if len(parts) < 2:
                log.debug(f"Cannot parse host from DSN '{oracle_config.dsn}'")
                return False
            host = parts[0]
            port_part = parts[1].split("/")[0]
            try:
                port = int(port_part)
            except ValueError:
                log.debug(f"Cannot parse port from DSN part '{port_part}'")
                return False

            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(timeout)
            result = sock.connect_ex((host, port))
            sock.close()
            return result == 0
        except Exception as e:
            log.debug(f"Exception checking Oracle listener: {e}")
            return False

    @staticmethod
    def _systemd_exists() -> bool:
        """Check if systemd is available."""
        return shutil.which("systemctl") is not None

    @staticmethod
    def _docker_exists() -> bool:
        """Check if Docker CLI is available."""
        return shutil.which("docker") is not None

    @staticmethod
    def _docker_oracle_container_exists() -> bool:
        """Check if Docker has an Oracle container (not necessarily running)."""
        if not InfrastructureValidator._docker_exists():
            return False
        try:
            result = subprocess.run(
                ["docker", "ps", "-a", "--format", "{{.Image}}"],
                capture_output=True,
                text=True,
                timeout=3,
            )
            return "oracle" in result.stdout.lower()
        except Exception:
            return False

    @staticmethod
    def validate_and_start_oracle(
        oracle_config: OracleConfig,
    ) -> InfrastructureStatus:
        """Validate and attempt to start Oracle infrastructure.
        
        Strategy:
        1. Check if listener is already reachable → return success.
        2. Try systemd (if available).
        3. Try Docker container (if available and exists).
        4. If all fail, raise StartupError with diagnosis.
        
        Args:
            oracle_config: Oracle credentials and DSN.
            
        Returns:
            InfrastructureStatus with final state.
            
        Raises:
            StartupError: If Oracle is required but cannot be started.
        """
        log.info(f"Validating Oracle infrastructure (DSN: {oracle_config.dsn})...")

        # Check if already reachable
        if InfrastructureValidator._can_reach_oracle_listener(oracle_config):
            log.info("Oracle listener is reachable")
            return InfrastructureStatus(
                oracle_listener_available=True,
                oracle_started_successfully=False,  # We didn't start it
            )

        log.warning("Oracle listener not reachable; attempting to start...")

        # Try systemd
        if InfrastructureValidator._systemd_exists():
            log.debug("Attempting to start Oracle via systemd...")
            for service_name in ["oracle", "oracledb"]:
                try:
                    result = subprocess.run(
                        ["systemctl", "start", service_name],
                        capture_output=True,
                        text=True,
                        timeout=5,
                    )
                    if result.returncode == 0:
                        log.info(f"Started Oracle via systemd service '{service_name}'")
                        import time
                        time.sleep(2)  # Give Oracle time to accept connections
                        if InfrastructureValidator._can_reach_oracle_listener(oracle_config):
                            return InfrastructureStatus(
                                oracle_listener_available=True,
                                oracle_started_successfully=True,
                            )
                except subprocess.TimeoutExpired:
                    log.debug(f"systemctl start {service_name} timed out")
                except Exception as e:
                    log.debug(f"systemctl start failed: {e}")

        # Try Docker
        if InfrastructureValidator._docker_exists() and InfrastructureValidator._docker_oracle_container_exists():
            log.debug("Attempting to start Oracle Docker container...")
            try:
                # Get container ID
                result = subprocess.run(
                    ["docker", "ps", "-a", "--format", "{{.ID}} {{.Image}}", "--filter", "ancestor=oracle*"],
                    capture_output=True,
                    text=True,
                    timeout=3,
                )
                if result.stdout.strip():
                    container_id = result.stdout.strip().split()[0]
                    log.debug(f"Found Oracle container: {container_id}")
                    result = subprocess.run(
                        ["docker", "start", container_id],
                        capture_output=True,
                        text=True,
                        timeout=10,
                    )
                    if result.returncode == 0:
                        log.info(f"Started Oracle Docker container {container_id}")
                        import time
                        time.sleep(3)  # Docker startup takes time
                        if InfrastructureValidator._can_reach_oracle_listener(oracle_config):
                            return InfrastructureStatus(
                                oracle_listener_available=True,
                                oracle_started_successfully=True,
                            )
            except Exception as e:
                log.debug(f"Docker start failed: {e}")

        # If we get here, Oracle could not be started
        diagnosis = []
        if not InfrastructureValidator._systemd_exists():
            diagnosis.append("systemd not available")
        if not InfrastructureValidator._docker_exists():
            diagnosis.append("Docker not available")
        if InfrastructureValidator._docker_exists() and not InfrastructureValidator._docker_oracle_container_exists():
            diagnosis.append("no Docker Oracle container found")

        raise StartupError(
            f"Failed to start Oracle infrastructure (DSN: {oracle_config.dsn}). "
            f"Listener not reachable and no recovery available. "
            f"Diagnostics: {'; '.join(diagnosis) if diagnosis else 'unknown'}. "
            f"Either: (1) start Oracle manually, (2) disable persistence (unset FIN_ASSIST_ENABLE_PERSISTENCE), "
            f"or (3) run in predictor-only mode."
        )
