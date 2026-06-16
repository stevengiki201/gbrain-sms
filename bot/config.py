"""Configuration loader with YAML file and environment overrides."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"
EXAMPLE_CONFIG_PATH = PROJECT_ROOT / "config.example.yaml"


@dataclass
class Config:
    adb_path: str = "adb"
    poll_interval_sec: float = 3.0
    sms_uri_authority: str = "adbsms"

    ollama_url: str = "http://127.0.0.1:11434"
    ollama_model: str = "llama3.2:3b"

    reply_mode: str = "immediate"
    auto_reply_enabled: bool = True
    reply_cooldown_sec: float = 5.0
    reply_backend: str = "rules"  # auto | rules | ollama
    reply_delay_min_sec: float = 2.0
    reply_delay_max_sec: float = 8.0
    bot_name: str = ""
    reply_persona: str = "calm"
    agent_pipeline_enabled: bool = True

    dnd_start: str = ""
    dnd_end: str = ""

    whatsapp_enabled: bool = False
    whatsapp_require_network: bool = True
    training_contact: str = ""

    sms_send_strategies: list[str] = field(
        default_factory=lambda: ["respond", "compose", "intent", "isms"]
    )

    # Kiswahili corpus settings
    crawl_swahili: bool = False  # Auto-crawl on startup if True
    crawl_limit: int = 200       # Max entries per crawl session
    corpus_db_path: str = ""     # Empty = same as db_path

    db_path: str = "data/conversations.db"
    log_path: str = "logs/bot.log"
    log_level: str = "INFO"

    config_path: Path = field(default_factory=lambda: DEFAULT_CONFIG_PATH)

    def resolve_path(self, relative: str) -> Path:
        path = Path(relative)
        if path.is_absolute():
            return path
        return PROJECT_ROOT / path


def _coerce_value(key: str, value: Any) -> Any:
    if key == "auto_reply_enabled":
        return bool(value)
    if key == "whatsapp_enabled":
        return bool(value)
    if key == "whatsapp_require_network":
        return bool(value)
    if key == "agent_pipeline_enabled":
        return bool(value)
    if key == "sms_send_strategies":
        if isinstance(value, str):
            return [s.strip() for s in value.split(",") if s.strip()]
        if isinstance(value, list):
            return [str(s) for s in value]
        return value
    if key in ("crawl_swahili",):
        return bool(value)
    if key in ("crawl_limit",):
        return int(value)
    if key == "reply_backend" and isinstance(value, str):
        return value.lower()
    if key == "reply_persona" and isinstance(value, str):
        valid_personas = {"calm", "energetic", "wise", "funny"}
        v = value.lower()
        return v if v in valid_personas else "calm"
    if key in ("poll_interval_sec", "reply_cooldown_sec", "reply_delay_min_sec", "reply_delay_max_sec"):
        return float(value)
    return value


def _env_override(key: str) -> Any | None:
    env_key = f"BOT_{key.upper()}"
    raw = os.environ.get(env_key)
    if raw is None:
        return None
    if key in ("auto_reply_enabled", "whatsapp_enabled", "agent_pipeline_enabled", "whatsapp_require_network"):
        return raw.lower() in ("1", "true", "yes", "on")
    if key == "sms_send_strategies":
        return [s.strip() for s in raw.split(",") if s.strip()]
    if key in ("crawl_swahili",):
        return raw.lower() in ("1", "true", "yes", "on")
    if key in ("crawl_limit",):
        return int(raw)
    if key in ("poll_interval_sec", "reply_cooldown_sec", "reply_delay_min_sec", "reply_delay_max_sec"):
        return float(raw)
    return raw


def load_config(path: Path | None = None) -> Config:
    cfg_path = path or DEFAULT_CONFIG_PATH
    data: dict[str, Any] = {}

    if cfg_path.exists():
        with cfg_path.open(encoding="utf-8") as fh:
            loaded = yaml.safe_load(fh) or {}
            if isinstance(loaded, dict):
                data.update(loaded)
    elif EXAMPLE_CONFIG_PATH.exists() and cfg_path == DEFAULT_CONFIG_PATH:
        with EXAMPLE_CONFIG_PATH.open(encoding="utf-8") as fh:
            loaded = yaml.safe_load(fh) or {}
            if isinstance(loaded, dict):
                data.update(loaded)

    fields = {f.name for f in Config.__dataclass_fields__.values() if f.name != "config_path"}
    kwargs: dict[str, Any] = {"config_path": cfg_path}

    for key in fields:
        if key in data:
            kwargs[key] = _coerce_value(key, data[key])
        env_val = _env_override(key)
        if env_val is not None:
            kwargs[key] = _coerce_value(key, env_val)

    return Config(**kwargs)
