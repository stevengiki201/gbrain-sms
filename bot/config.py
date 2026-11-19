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
    ollama_fallback_model: str = ""

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

    # Incoming-call auto-answer (phase 1: answer + optional hang-up, no speech)
    call_answer_enabled: bool = False
    call_answer_start: str = ""
    call_answer_end: str = ""
    call_answer_at: str = ""
    call_answer_contacts: list[str] = field(default_factory=list)
    call_answer_strategies: list[str] = field(
        default_factory=lambda: ["keyevent_call", "headsethook", "ui_tap"]
    )
    call_hangup_after_sec: float = 0.0

    # Phase 2: spoken greeting when a call is answered
    call_tts_enabled: bool = False
    call_greeting: str = ""
    call_tts_strategies: list[str] = field(
        default_factory=lambda: ["termux", "helper_intent", "pc_tts"]
    )
    call_tts_helper_action: str = "com.gbrain.sms.SPEAK"
    call_tts_command: str = ""
    call_tts_volume: int = 0
    call_speakerphone: bool = True

    # Phase 3: outgoing calls
    call_dial_strategies: list[str] = field(
        default_factory=lambda: ["intent_call", "dialer_keyevent"]
    )
    call_dial_timeout_sec: float = 20.0
    call_outgoing_message: str = ""
    call_outgoing_delay_sec: float = 6.0
    call_outgoing_repeat: int = 2

    # Phase 4: real-time voice engine (acoustic bridge: phone on speakerphone)
    voice_enabled: bool = False
    voice_transport: str = "local"      # local (PC mic/speakers) | audiolink
    audiolink_url: str = "https://127.0.0.1:8787"
    audiolink_admin_token_file: str = "~/.local/share/gbrain/audiolink/admin.token"
    audiolink_ca_file: str = "~/.local/share/gbrain/audiolink/tls/cert.pem"
    audiolink_device_id: str = ""       # empty = first paired phone
    audiolink_call_sink: str = "gbrain-call"
    audiolink_tts_sink: str = "gbrain-tts"
    voice_sample_rate: int = 16000
    voice_input_device: str = "auto"
    voice_output_device: str = "auto"
    voice_language: str = "auto"        # auto | en | sw
    voice_stt_model: str = "small"
    voice_model_dir: str = "data/models"
    voice_tts_provider: str = "espeak"  # espeak | piper
    voice_tts_voice: str = ""           # espeak voice name, e.g. sw or en
    voice_tts_command: str = ""         # optional "<cmd> {out} {text}"
    voice_piper_binary: str = "piper"
    voice_piper_model: str = ""
    voice_vad_enabled: bool = True
    voice_vad_threshold: float = 0.012
    voice_vad_silence_ms: int = 450
    voice_vad_min_speech_ms: int = 250
    voice_interrupt_enabled: bool = True
    voice_echo_cancellation: bool = True
    voice_greeting: str = ""
    voice_system_prompt: str = ""

    sms_send_strategies: list[str] = field(
        default_factory=lambda: ["respond", "compose", "intent", "isms"]
    )

    # Kiswahili corpus settings
    crawl_swahili: bool = False  # Auto-crawl on startup if True
    crawl_limit: int = 200       # Max entries per crawl session
    corpus_db_path: str = ""     # Empty = same as db_path

    # GBrain Android companion API + AudioLink bridge (single server)
    gbrain_api_enabled: bool = True
    gbrain_api_port: int = 8787
    bridge_enabled: bool = True
    bridge_mdns_enabled: bool = True
    bridge_device_name: str = ""
    bridge_auto_reply_enabled: bool = True
    bridge_pairing_code_ttl_sec: int = 300
    bridge_data_dir: str = ""  # empty = directory of audiolink_admin_token_file
    sms_transport: str = "bridge"  # bridge (Wi-Fi) | adb (legacy USB)

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
    if key in (
        "crawl_swahili",
        "bridge_enabled",
        "bridge_mdns_enabled",
        "bridge_auto_reply_enabled",
    ):
        return bool(value)
    if key in ("crawl_limit", "gbrain_api_port", "bridge_pairing_code_ttl_sec"):
        return int(value)
    if key == "reply_backend" and isinstance(value, str):
        return value.lower()
    if key == "sms_transport" and isinstance(value, str):
        transport = value.strip().lower()
        return transport if transport in {"bridge", "adb"} else "bridge"
    if key == "reply_persona" and isinstance(value, str):
        valid_personas = {"calm", "energetic", "wise", "funny"}
        v = value.lower()
        return v if v in valid_personas else "calm"
    if key in ("poll_interval_sec", "reply_cooldown_sec", "reply_delay_min_sec", "reply_delay_max_sec"):
        return float(value)
    if key in ("call_answer_enabled", "call_tts_enabled", "call_speakerphone"):
        return bool(value)
    if key in (
        "call_answer_contacts",
        "call_answer_strategies",
        "call_tts_strategies",
        "call_dial_strategies",
    ):
        if isinstance(value, str):
            return [s.strip() for s in value.split(",") if s.strip()]
        if isinstance(value, list):
            return [str(s) for s in value]
        return value
    if key in (
        "call_hangup_after_sec",
        "call_dial_timeout_sec",
        "call_outgoing_delay_sec",
    ):
        return float(value)
    if key in ("call_tts_volume", "call_outgoing_repeat"):
        return int(value)
    if key in (
        "voice_enabled",
        "voice_vad_enabled",
        "voice_interrupt_enabled",
        "voice_echo_cancellation",
    ):
        return bool(value)
    if key in ("voice_sample_rate", "voice_vad_silence_ms", "voice_vad_min_speech_ms"):
        return int(value)
    if key == "voice_vad_threshold":
        return float(value)
    if key == "voice_transport" and isinstance(value, str):
        transport = value.strip().lower()
        return transport if transport in {"local", "audiolink"} else "local"
    return value


def _env_override(key: str) -> Any | None:
    env_key = f"BOT_{key.upper()}"
    raw = os.environ.get(env_key)
    if raw is None:
        return None
    if key in (
        "auto_reply_enabled",
        "whatsapp_enabled",
        "agent_pipeline_enabled",
        "whatsapp_require_network",
        "bridge_enabled",
        "bridge_mdns_enabled",
        "bridge_auto_reply_enabled",
    ):
        return raw.lower() in ("1", "true", "yes", "on")
    if key == "sms_send_strategies":
        return [s.strip() for s in raw.split(",") if s.strip()]
    if key in ("crawl_swahili",):
        return raw.lower() in ("1", "true", "yes", "on")
    if key in ("crawl_limit", "gbrain_api_port", "bridge_pairing_code_ttl_sec"):
        return int(raw)
    if key in ("poll_interval_sec", "reply_cooldown_sec", "reply_delay_min_sec", "reply_delay_max_sec"):
        return float(raw)
    if key in ("call_answer_enabled", "call_tts_enabled", "call_speakerphone"):
        return raw.lower() in ("1", "true", "yes", "on")
    if key in (
        "call_answer_contacts",
        "call_answer_strategies",
        "call_tts_strategies",
        "call_dial_strategies",
    ):
        return [s.strip() for s in raw.split(",") if s.strip()]
    if key in (
        "call_hangup_after_sec",
        "call_dial_timeout_sec",
        "call_outgoing_delay_sec",
    ):
        return float(raw)
    if key in ("call_tts_volume", "call_outgoing_repeat"):
        return int(raw)
    if key in (
        "voice_enabled",
        "voice_vad_enabled",
        "voice_interrupt_enabled",
        "voice_echo_cancellation",
    ):
        return raw.lower() in ("1", "true", "yes", "on")
    if key in ("voice_sample_rate", "voice_vad_silence_ms", "voice_vad_min_speech_ms"):
        return int(raw)
    if key == "voice_vad_threshold":
        return float(raw)
    if key == "voice_transport":
        transport = raw.strip().lower()
        return transport if transport in {"local", "audiolink"} else "local"
    if key == "sms_transport":
        transport = raw.strip().lower()
        return transport if transport in {"bridge", "adb"} else "bridge"
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
