from __future__ import annotations

import hashlib
import json
import os
import platform
from pathlib import Path

CONFIG_DIR = Path(
    os.environ.get("GBRAIN_CLI_CONFIG_DIR", str(Path.home() / ".config" / "gbrain"))
)
CONFIG_PATH = CONFIG_DIR / "cli.json"
LOCAL_DATA_DIR = Path.home() / ".local" / "share" / "gbrain" / "audiolink"
LOCAL_TOKEN_PATH = LOCAL_DATA_DIR / "admin.token"
LOCAL_IDENTITY_PATH = LOCAL_DATA_DIR / "identity.json"
LOCAL_CA_PATH = Path(
    os.environ.get("GBRAIN_AUDIOLINK_CA_FILE", str(LOCAL_DATA_DIR / "tls" / "cert.pem"))
)


def load_config() -> dict:
    if not CONFIG_PATH.exists():
        return {}
    try:
        return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def save_config(config: dict) -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(json.dumps(config, indent=2), encoding="utf-8")
    CONFIG_PATH.chmod(0o600)


def clear_config() -> None:
    if CONFIG_PATH.exists():
        CONFIG_PATH.unlink()


def local_admin_token() -> str:
    if not LOCAL_TOKEN_PATH.exists():
        raise RuntimeError(
            f"AudioLink admin token not found at {LOCAL_TOKEN_PATH}. Start gbrain-sms first."
        )
    token = LOCAL_TOKEN_PATH.read_text(encoding="utf-8").strip()
    if not token:
        raise RuntimeError(f"AudioLink admin token at {LOCAL_TOKEN_PATH} is empty.")
    return token


def local_ca_file() -> str:
    if not LOCAL_CA_PATH.is_file():
        raise RuntimeError(
            f"AudioLink TLS certificate not found at {LOCAL_CA_PATH}. Start gbrain-sms first."
        )
    return str(LOCAL_CA_PATH)


def device_fingerprint() -> str:
    if LOCAL_IDENTITY_PATH.exists():
        try:
            identity = json.loads(LOCAL_IDENTITY_PATH.read_text(encoding="utf-8"))
            value = identity.get("device_id") or identity.get("fingerprint")
            if value:
                return str(value)[:120]
        except (OSError, json.JSONDecodeError):
            pass
    seed = f"{platform.node()}:{platform.platform()}:{Path.home()}"
    return hashlib.sha256(seed.encode("utf-8")).hexdigest()[:24]


def default_device_name() -> str:
    return platform.node() or "GBrain PC"
