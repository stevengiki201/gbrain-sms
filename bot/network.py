"""Network availability checks for WhatsApp polling."""

from __future__ import annotations

import logging
import socket

logger = logging.getLogger(__name__)

_PING_HOSTS = ("1.1.1.1", "8.8.8.8")


def host_has_network(timeout: float = 2.0) -> bool:
    """True when this PC can reach the internet."""
    for host in _PING_HOSTS:
        try:
            with socket.create_connection((host, 53), timeout=timeout):
                return True
        except OSError:
            continue
    return False


def device_has_network(adb_run, *, timeout_sec: int = 2) -> bool:
    """
    True when the USB-connected phone reports network access.
    adb_run: callable like AdbConnector._run_adb
    """
    try:
        output = adb_run(
            [
                "shell",
                "ping",
                "-c",
                "1",
                "-W",
                str(timeout_sec),
                "1.1.1.1",
            ]
        )
        lower = (output or "").lower()
        if "1 received" in lower or "1 packets received" in lower:
            return True
        if "bytes from" in lower:
            return True
    except Exception as exc:
        logger.debug("Device network ping failed: %s", exc)

    try:
        output = adb_run(["shell", "dumpsys", "connectivity"])
        if "NetworkAgentInfo" in output and "CONNECTED" in output:
            if "NOT_CONNECTED" not in output.split("CONNECTED")[0][-80:]:
                return True
    except Exception as exc:
        logger.debug("Device connectivity dump failed: %s", exc)

    return False
