"""Linux audio device discovery and control.

Uses ``pactl`` (PulseAudio or PipeWire's PulseAudio compatibility layer) so the
dashboard can list real microphones/speakers and adjust volume. Every function
reports honestly when no sound server is reachable instead of inventing
devices.
"""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
from dataclasses import asdict, dataclass

from .errors import MediaUnavailable, ValidationFailed

logger = logging.getLogger("gbrain.bridge.audio")


@dataclass
class AudioDevice:
    id: str
    name: str
    description: str
    kind: str  # source | sink
    is_default: bool = False
    is_monitor: bool = False
    volume_percent: int | None = None
    muted: bool = False

    def public(self) -> dict:
        return asdict(self)


class AudioDeviceManager:
    def __init__(self) -> None:
        self._available: bool | None = None
        self._last_error = ""
        self._lock = asyncio.Lock()

    @property
    def tools_available(self) -> bool:
        return shutil.which("pactl") is not None

    async def _run(self, *args: str) -> str:
        proc = await asyncio.create_subprocess_exec(
            "pactl",
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        out, err = await proc.communicate()
        if proc.returncode != 0:
            raise MediaUnavailable(
                "pactl command failed",
                details={"args": list(args), "stderr": err.decode()[-500:]},
                code="AUDIO_SERVER_UNAVAILABLE",
            )
        return out.decode()

    async def _list(self, kind: str) -> list[AudioDevice]:
        output = await self._run("-f", "json", "list", f"{kind}s")
        try:
            items = json.loads(output)
        except json.JSONDecodeError as exc:
            raise MediaUnavailable(
                "could not parse pactl output", code="AUDIO_PARSE_ERROR"
            ) from exc

        devices: list[AudioDevice] = []
        for item in items:
            name = item.get("name", "")
            if not name:
                continue
            volume: int | None = None
            channel_map = (item.get("volume") or {}).get("channel_map") or []
            if channel_map:
                raw = channel_map[0].get("volume")
                if isinstance(raw, int):
                    volume = round(raw / 65536 * 100)
            devices.append(
                AudioDevice(
                    id=name,
                    name=name,
                    description=item.get("description") or name,
                    kind=kind,
                    is_default=False,
                    is_monitor=kind == "source" and name.endswith(".monitor"),
                    volume_percent=volume,
                    muted=bool(item.get("mute", False)),
                )
            )
        return devices

    async def list_devices(self) -> dict:
        async with self._lock:
            if not self.tools_available:
                self._available = False
                self._last_error = "pactl is not installed"
                return {
                    "available": False,
                    "error": self._last_error,
                    "sources": [],
                    "sinks": [],
                    "default_source": "",
                    "default_sink": "",
                }
            try:
                sources, sinks = await asyncio.gather(
                    self._list("source"), self._list("sink")
                )
                default_source = (
                    await self._run("get-default-source")
                ).strip()
                default_sink = (await self._run("get-default-sink")).strip()
            except MediaUnavailable as exc:
                self._available = False
                self._last_error = exc.message
                return {
                    "available": False,
                    "error": exc.message,
                    "sources": [],
                    "sinks": [],
                    "default_source": "",
                    "default_sink": "",
                }
            self._available = True
            self._last_error = ""
            for source in sources:
                source.is_default = source.name == default_source
            for sink in sinks:
                sink.is_default = sink.name == default_sink
            return {
                "available": True,
                "error": "",
                "sources": [s.public() for s in sources],
                "sinks": [s.public() for s in sinks],
                "default_source": default_source,
                "default_sink": default_sink,
            }

    async def set_sink_volume(self, level: float, sink: str = "") -> dict:
        if not 0 <= level <= 100:
            raise ValidationFailed("volume must be between 0 and 100")
        target = sink or "@DEFAULT_SINK@"
        await self._run("set-sink-volume", target, f"{int(level)}%")
        return {"sink": target, "volume_percent": int(level)}

    async def set_source_mute(self, muted: bool, source: str = "") -> dict:
        target = source or "@DEFAULT_SOURCE@"
        await self._run("set-source-mute", target, "1" if muted else "0")
        return {"source": target, "muted": muted}

    async def set_sink_mute(self, muted: bool, sink: str = "") -> dict:
        target = sink or "@DEFAULT_SINK@"
        await self._run("set-sink-mute", target, "1" if muted else "0")
        return {"sink": target, "muted": muted}

    def diagnostics(self) -> dict:
        return {
            "tools_available": self.tools_available,
            "server_available": self._available,
            "last_error": self._last_error,
        }
