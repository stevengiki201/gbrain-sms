"""Call audio transport over GBrain AudioLink instead of the room.

A non-rooted phone still does not expose cellular call audio to apps, so the
phone stays on speakerphone and its own microphone/speaker carry the call. What
changes is the PC side: AudioLink's WebRTC session replaces the acoustic hop, so
the PC can sit anywhere on the LAN and the caller audio arrives as PCM instead
of room echo.

Two PipeWire null sinks keep the two directions apart:

    phone -> PC   AudioLink playout -> sink ``gbrain-call``
                  we record ``gbrain-call.monitor`` for speech recognition
    PC -> phone   we play TTS into sink ``gbrain-tts``
                  AudioLink captures ``gbrain-tts.monitor`` and sends it

This class keeps :class:`~bot.voice.audio_bridge.AcousticBridge`'s interface so
``VoiceEngine`` needs no special cases for reading or speaking.
"""

from __future__ import annotations

import logging
import shutil
import ssl
import subprocess
import urllib.error
import urllib.request
import json
from pathlib import Path
from typing import Union

from bot.voice.audio_bridge import AcousticBridge

logger = logging.getLogger(__name__)

DEFAULT_PORT = 8787
CALL_SINK = "gbrain-call"
TTS_SINK = "gbrain-tts"


class AudioLinkError(RuntimeError):
    """AudioLink refused the request, or could not be reached."""


def _expand(path: str) -> str:
    return str(Path(path).expanduser()) if path else ""


class AudioLinkClient:
    """Minimal client for AudioLink's loopback management API."""

    def __init__(
        self,
        *,
        base_url: str,
        admin_token: str,
        ca_file: str,
        device_id: str = "",
        timeout: float = 15.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.admin_token = admin_token
        self.ca_file = ca_file
        self.device_id = device_id
        self.timeout = timeout
        self._ssl = ssl.create_default_context(cafile=ca_file) if ca_file else None

    @classmethod
    def from_config(cls, config) -> "AudioLinkClient":
        token_file = _expand(getattr(config, "audiolink_admin_token_file", ""))
        ca_file = _expand(getattr(config, "audiolink_ca_file", ""))
        try:
            token = Path(token_file).read_text(encoding="utf-8").strip()
        except OSError as exc:
            raise AudioLinkError(
                f"cannot read the AudioLink admin token at {token_file}: {exc}"
            ) from exc
        if not token:
            raise AudioLinkError(f"AudioLink admin token at {token_file} is empty")
        return cls(
            base_url=getattr(config, "audiolink_url", f"https://127.0.0.1:{DEFAULT_PORT}"),
            admin_token=token,
            ca_file=ca_file,
            device_id=getattr(config, "audiolink_device_id", "") or "",
        )

    # ── HTTP ─────────────────────────────────────────────────────────
    def _request(self, path: str, payload: dict | None = None) -> dict:
        url = f"{self.base_url}{path}"
        data = None
        headers = {"X-AudioLink-Admin": self.admin_token}
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")
            headers["content-type"] = "application/json"
        request = urllib.request.Request(url, data=data, headers=headers)
        try:
            with urllib.request.urlopen(
                request, context=self._ssl, timeout=self.timeout
            ) as response:
                return json.loads(response.read() or b"{}")
        except urllib.error.HTTPError as exc:
            detail = ""
            try:
                body = json.loads(exc.read() or b"{}")
                detail = (body.get("error") or {}).get("code", "")
            except (ValueError, OSError):
                pass
            raise AudioLinkError(
                f"AudioLink {path} failed ({exc.code}{': ' + detail if detail else ''})"
            ) from exc
        except (urllib.error.URLError, OSError, ValueError) as exc:
            raise AudioLinkError(f"AudioLink is not reachable at {self.base_url}: {exc}") from exc

    def execute(self, operation: str, params: dict | None = None) -> dict:
        payload = self._request(
            "/v1/operations", {"operation": operation, "params": params or {}}
        )
        return payload.get("result", payload)

    def status(self) -> dict:
        return self._request("/v1/status")

    def resolve_device_id(self) -> str:
        """Explicit id, or the first paired device the desktop knows."""
        if self.device_id:
            return self.device_id
        devices = (self.execute("audiolink.get_paired_devices") or {}).get("devices", [])
        if not devices:
            raise AudioLinkError(
                "no phone is paired with AudioLink; pair it on the desktop first"
            )
        self.device_id = str(devices[0]["device_id"])
        return self.device_id


def ensure_null_sink(name: str, *, rate: int = 48000, channels: int = 1) -> bool:
    """Create a PipeWire null sink (idempotent). Returns True when created."""
    if not shutil.which("pactl"):
        raise RuntimeError(
            "pactl is required for the AudioLink transport (install pipewire-pulse)"
        )
    try:
        listing = subprocess.run(
            ["pactl", "list", "short", "sinks"],
            capture_output=True,
            text=True,
            timeout=8,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError(f"cannot list audio sinks: {exc}") from exc
    for line in listing.stdout.splitlines():
        parts = line.split("\t")
        if len(parts) > 1 and parts[1] == name:
            return False
    result = subprocess.run(
        [
            "pactl",
            "load-module",
            "module-null-sink",
            f"sink_name={name}",
            f"rate={rate}",
            f"channels={channels}",
            f"sink_properties=device.description={name}",
        ],
        capture_output=True,
        text=True,
        timeout=8,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"could not create the {name} sink: {result.stderr.strip()[:200]}"
        )
    logger.info("Created audio sink %s", name)
    return True


class AudioLinkBridge(AcousticBridge):
    """AcousticBridge-compatible audio path carried by AudioLink."""

    def __init__(
        self,
        *,
        config,
        client: AudioLinkClient | None = None,
        sample_rate: int = 16000,
        manage_sinks: bool = True,
    ) -> None:
        self.config = config
        self.call_sink = getattr(config, "audiolink_call_sink", CALL_SINK) or CALL_SINK
        self.tts_sink = getattr(config, "audiolink_tts_sink", TTS_SINK) or TTS_SINK
        self.client = client or AudioLinkClient.from_config(config)
        self.manage_sinks = manage_sinks
        self._started = False
        self._previous_routing: dict | None = None
        super().__init__(
            sample_rate=sample_rate,
            # phone -> PC arrives on the call sink; its monitor is our input.
            input_device=f"{self.call_sink}.monitor",
            # every spoken reply is played into the sink AudioLink captures.
            output_device=self.tts_sink,
        )

    # ── Lifecycle ────────────────────────────────────────────────────
    def start(self) -> dict:
        """Create the sinks, aim AudioLink at them, and open a media session."""
        if self._started:
            return {"already_running": True}
        if self.manage_sinks:
            ensure_null_sink(self.call_sink)
            ensure_null_sink(self.tts_sink)
        device_id = self.client.resolve_device_id()
        self.client.execute("audiolink.connect", {"device_id": device_id})
        # Remember where AudioLink pointed so a normal session can be restored.
        try:
            self._previous_routing = self.client.status().get("routing")
        except AudioLinkError:
            self._previous_routing = None
        routing = self.client.execute(
            "audiolink.set_audio_routing",
            {
                "playback_device": self.call_sink,
                "capture_device": f"{self.tts_sink}.monitor",
            },
        )
        session = self.client.execute(
            "audiolink.start_audio_session", {"device_id": device_id}
        )
        # Desktop capture is what carries our voice into the call.
        self.client.execute("audiolink.set_microphone_enabled", {"enabled": True})
        # Desktop playout carries the caller into this machine.
        self.client.execute("audiolink.set_speaker_enabled", {"enabled": True})
        self._started = True
        logger.info(
            "AudioLink transport ready for %s (call sink %s, tts sink %s)",
            device_id,
            self.call_sink,
            self.tts_sink,
        )
        return {"device_id": device_id, "routing": routing, "media": session}

    def stop(self) -> None:
        if not self._started:
            return
        self._started = False
        try:
            self.client.execute("audiolink.end_audio_session")
        except AudioLinkError as exc:
            logger.warning("Could not end the AudioLink media session: %s", exc)
        previous = self._previous_routing or {}
        self._previous_routing = None
        if previous:
            try:
                self.client.execute(
                    "audiolink.set_audio_routing",
                    {
                        "playback_device": previous.get("playback_device", ""),
                        "capture_device": previous.get("capture_device", ""),
                    },
                )
            except AudioLinkError as exc:
                logger.warning("Could not restore AudioLink routing: %s", exc)
        logger.info("AudioLink transport closed")

    # ── Devices ──────────────────────────────────────────────────────
    def capture_argv(self) -> list[str] | None:
        """Read the call sink's monitor; pw-record falls back to the mic there."""
        target = self.input_device
        if shutil.which("parec"):
            return [
                "parec",
                f"--device={target}",
                f"--rate={self.sample_rate}",
                "--channels=1",
                "--format=s16le",
                "--latency-msec=40",
            ]
        if shutil.which("pw-record"):
            return [
                "pw-record",
                "--rate",
                str(self.sample_rate),
                "--channels",
                "1",
                "--format",
                "s16",
                "--target",
                target,
                "-",
            ]
        return None

    def playback_argv(self, path: Union[Path, str]) -> list[str] | None:
        """Play the reply into the sink AudioLink captures."""
        target = self.output_device
        if shutil.which("pw-play"):
            return ["pw-play", "--target", target, str(path)]
        if shutil.which("paplay"):
            return ["paplay", f"--device={target}", str(path)]
        return None

    # ── Diagnostics ──────────────────────────────────────────────────
    def check(self) -> dict[str, object]:
        info = super().check()
        info.update(
            {
                "transport": "audiolink",
                "base_url": self.client.base_url,
                "call_sink": self.call_sink,
                "tts_sink": self.tts_sink,
                "running": self._started,
            }
        )
        return info

    def describe(self) -> str:
        return (
            f"audiolink {self.client.base_url}: caller audio <- {self.call_sink}.monitor; "
            f"reply -> {self.tts_sink} (captured by AudioLink)"
        )

    def status(self) -> dict:
        """AudioLink's view, for status commands and the dashboard."""
        try:
            payload = self.client.status()
        except AudioLinkError as exc:
            return {"ok": False, "error": str(exc)}
        media = (payload.get("media") or {}).get("sessions") or []
        session = media[0] if media else {}
        return {
            "ok": True,
            "online": bool(payload.get("online_devices")),
            "peer": session.get("peer_state"),
            "ice": session.get("ice_state"),
            "frames_received": session.get("frames_received", 0),
            "frames_sent": session.get("frames_sent", 0),
            "level_dbfs": session.get("level_dbfs"),
            "level_sent_dbfs": session.get("level_sent_dbfs"),
        }
