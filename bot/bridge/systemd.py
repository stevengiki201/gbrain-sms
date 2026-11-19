"""Linux systemd *user* service integration.

Installation never enables autostart unless the caller explicitly asks for it
(``enable=True``), matching the project rule that startup is opt-in.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

from .errors import Unavailable

UNIT_NAME = "audiolink.service"


def unit_dir() -> Path:
    base = os.environ.get("XDG_CONFIG_HOME") or str(Path.home() / ".config")
    return Path(base) / "systemd" / "user"


def unit_path() -> Path:
    return unit_dir() / UNIT_NAME


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _quote(path: str) -> str:
    """Quote a path for systemd when it contains whitespace."""
    return f'"{path}"' if any(ch.isspace() for ch in path) else path


def render_unit(
    *,
    python_executable: str | None = None,
    root: Path | None = None,
    extra_args: str = "",
) -> str:
    python_executable = python_executable or sys.executable
    root = root or project_root()
    return f"""[Unit]
Description=GBrain Bridge - pairing, signaling, SMS and audio
Documentation=file://{root}/README.md
After=network-online.target sound.target
Wants=network-online.target

[Service]
Type=simple
WorkingDirectory={_quote(str(root))}
ExecStart={_quote(python_executable)} -m bot.main run {extra_args}""".rstrip() + """
Environment=PYTHONUNBUFFERED=1
Restart=on-failure
RestartSec=3
TimeoutStopSec=10
KillSignal=SIGINT

[Install]
WantedBy=default.target
"""


def _systemctl(*args: str) -> subprocess.CompletedProcess:
    if not shutil.which("systemctl"):
        raise Unavailable("systemctl is not available on this system")
    return subprocess.run(
        ["systemctl", "--user", *args],
        capture_output=True,
        text=True,
        check=False,
    )


def install(*, enable: bool = False, restart: bool = False) -> dict:
    """Write the user unit, reload systemd and (optionally) enable it."""
    path = unit_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_unit(), encoding="utf-8")

    reload_result = _systemctl("daemon-reload")
    if reload_result.returncode != 0:
        raise Unavailable(
            "systemd could not reload user units",
            details={"stderr": reload_result.stderr.strip()[-500:]},
        )

    enabled = False
    if enable:
        result = _systemctl("enable", UNIT_NAME)
        enabled = result.returncode == 0
        if not enabled:
            raise Unavailable(
                "could not enable the unit",
                details={"stderr": result.stderr.strip()[-500:]},
            )

    if restart:
        _systemctl("restart", UNIT_NAME)

    return {"unit": str(path), "enabled": enabled, "installed": True}


def uninstall(*, disable: bool = True, remove_unit: bool = True) -> dict:
    path = unit_path()
    if disable:
        _systemctl("disable", "--now", UNIT_NAME)
    if remove_unit and path.is_file():
        path.unlink()
    _systemctl("daemon-reload")
    return {"unit": str(path), "installed": False}


def status() -> dict:
    result = _systemctl("is-active", UNIT_NAME)
    return {
        "unit": str(unit_path()),
        "installed": unit_path().is_file(),
        "active": result.stdout.strip() == "active",
    }
