"""GBrain bridge configuration (pairing, signaling, SMS, media).

Values come from (lowest to highest priority):

1. dataclass defaults below,
2. an optional YAML file (``bridge.yaml`` by default),
3. environment variables prefixed with ``AUDIOLINK_``.

Example::

    BRIDGE_PORT=9000 BRIDGE_LOG_LEVEL=DEBUG python -m bot.main run
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "bridge.yaml"
DEFAULT_DATA_DIR = Path("~/.local/share/gbrain/audiolink").expanduser()

SERVICE_TYPE = "_gbrain-alink._tcp.local."  # DNS-SD labels are limited to 15 bytes
PROTOCOL_VERSION = 1


def _default_device_name() -> str:
    import socket

    host = socket.gethostname() or "gbrain-desktop"
    return host.split(".")[0]


def _csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


@dataclass
class AudioLinkConfig:
    # Network
    host: str = "0.0.0.0"
    port: int = 8787
    device_name: str = field(default_factory=_default_device_name)
    service_type: str = SERVICE_TYPE

    # Discovery
    mdns_enabled: bool = True

    # Storage
    data_dir: str = str(DEFAULT_DATA_DIR)

    # Pairing
    pairing_code_ttl_sec: int = 300
    pairing_max_attempts: int = 5
    pairing_request_rate_limit: int = 10
    pairing_request_rate_window_sec: int = 60
    token_ttl_sec: int = 0  # 0 = tokens do not expire; revoked devices still fail

    # Sessions
    session_idle_timeout_sec: int = 300

    # Audio
    capture_device: str = ""
    playback_device: str = ""

    # WebRTC media (phone <-> PC audio on one peer connection)
    stun_servers: list[str] = field(default_factory=list)
    turn_server: str = ""
    turn_username: str = ""
    turn_password: str = ""

    # Service
    log_level: str = "INFO"
    log_file: str = ""

    # gbrain-sms AI server integration
    gbrain_url: str = ""
    gbrain_token_file: str = ""
    auto_reply_enabled: bool = True

    @property
    def data_path(self) -> Path:
        return Path(self.data_dir).expanduser()

    @classmethod
    def load(
        cls, path: str | Path | None = None, environ: dict[str, str] | None = None
    ) -> AudioLinkConfig:
        environ = dict(os.environ if environ is None else environ)
        values: dict[str, Any] = {}

        config_path = Path(path).expanduser() if path else DEFAULT_CONFIG_PATH
        if config_path.is_file():
            values.update(_read_yaml(config_path))

        known = {f.name: f for f in fields(cls)}
        for name, meta in known.items():
            key = f"AUDIOLINK_{name.upper()}"
            if key in environ:
                values[name] = _coerce(environ[key], meta.default)

        filtered = {k: v for k, v in values.items() if k in known}
        return cls(**filtered)


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - dependency is declared
        raise RuntimeError("pyyaml is required to read audiolink.yaml") from exc

    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return data


def _coerce(raw: str, default: Any) -> Any:
    if isinstance(default, bool):
        return raw.strip().lower() in {"1", "true", "yes", "on"}
    if isinstance(default, int):
        return int(raw)
    if isinstance(default, list):
        return _csv(raw)
    return raw
