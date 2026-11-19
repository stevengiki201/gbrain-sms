"""AudioLink desktop service: lifecycle and the operations GBrain calls.

The service owns every subsystem (identity, registry, pairing, mDNS discovery,
signaling hub, audio devices) and can run head-less behind systemd. It never
requires the dashboard to be open.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import secrets
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import __version__
from .audio_devices import AudioDeviceManager
from .config import AudioLinkConfig
from .discovery import DiscoveryBrowser, DiscoveredDevice, ServiceAdvertiser
from .errors import AudioLinkError, Conflict, MediaUnavailable, NotFound, Unavailable
from .identity import Identity, load_or_create_identity
from .media import MediaManager
from .pairing import PairingManager, PairingRequest
from .registry import DeviceRegistry
from .signaling import HELLO_ACK, PONG, PROTOCOL, SMS_SEND, SignalingHub
from .tasks import ScheduledTask, TaskStore, normalize_scheduled_at

logger = logging.getLogger("gbrain.bridge.service")

ADMIN_TOKEN_FILE = "admin.token"
SERVICE_NAME = "audiolink"


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class ControlSession:
    session_id: str
    device_id: str
    device_name: str
    started_at: float = field(default_factory=time.time)
    microphone_enabled: bool = True
    speaker_enabled: bool = True
    volume: int = 100
    state: str = "connected"  # connected | ended

    def public(self) -> dict:
        duration = max(0.0, time.time() - self.started_at)
        return {
            "session_id": self.session_id,
            "device_id": self.device_id,
            "device_name": self.device_name,
            "state": self.state,
            "started_at": self.started_at,
            "duration_sec": round(duration, 1),
            "microphone_enabled": self.microphone_enabled,
            "speaker_enabled": self.speaker_enabled,
            "volume": self.volume,
            "media_transport": "not_negotiated",
        }


class AudioLinkService:
    def __init__(self, config: AudioLinkConfig | None = None) -> None:
        self.config = config or AudioLinkConfig.load()
        self.identity: Identity | None = None
        self.registry: DeviceRegistry | None = None
        self.pairing: PairingManager | None = None
        self.advertiser: ServiceAdvertiser | None = None
        self.browser: DiscoveryBrowser | None = None
        self.hub = SignalingHub()
        self.audio = AudioDeviceManager()
        self.media = MediaManager(self.config)
        self.admin_token = ""
        self._sessions: dict[str, ControlSession] = {}
        self._started_at: float = 0.0
        self._running = False
        self._lock = asyncio.Lock()
        self._events: deque[dict] = deque(maxlen=200)
        self._started_event = asyncio.Event()
        # SMS / task scheduler (initialised in start())
        self.task_store: TaskStore | None = None
        self._scheduler_task: asyncio.Task | None = None
        self.gbrain = None  # GbrainClient fallback, set in start()
        self.bot = None     # BotApp, injected by the combined gbrain server
        # Highest sms_id handled per device, used to dedupe re-sent messages.
        self._sms_seen: dict[str, int] = {}

    # --------------------------------------------------------------- lifecycle
    async def start(self) -> None:
        if self._running:
            return
        self.identity = load_or_create_identity(self.config.data_path, self.config.device_name)
        self.admin_token = _load_or_create_admin_token(self.config.data_path)
        self.registry = DeviceRegistry(
            self.config.data_path, token_ttl_sec=self.config.token_ttl_sec
        )
        self.pairing = PairingManager(
            self.registry,
            ttl_sec=self.config.pairing_code_ttl_sec,
            max_attempts=self.config.pairing_max_attempts,
            rate_limit=self.config.pairing_request_rate_limit,
            rate_window_sec=self.config.pairing_request_rate_window_sec,
        )
        self.pairing.add_listener(self._on_pairing_request)

        if self.config.mdns_enabled:
            self.advertiser = ServiceAdvertiser(
                self.identity,
                port=self.config.port,
                service_type=self.config.service_type,
            )
            try:
                await self.advertiser.start()
            except Exception as exc:
                logger.warning("mDNS advertising unavailable: %s", exc)
                self.advertiser = None

            self.browser = DiscoveryBrowser(
                service_type=self.config.service_type,
                exclude_device_id=self.identity.device_id,
                on_update=self._on_discovery_update,
            )
            try:
                await self.browser.start()
            except Exception as exc:
                logger.warning("mDNS browsing unavailable: %s", exc)
                self.browser = None

        self._started_at = time.time()
        self._running = True
        self._started_event.set()
        self._record("service.started", {"device_id": self.identity.device_id, "port": self.config.port})

        # SMS task scheduler
        self.task_store = TaskStore(self.config.data_path)
        self._scheduler_task = asyncio.create_task(self._scheduler_loop())

        # gbrain-sms AI client
        if self.config.gbrain_url:
            try:
                from .gbrain_client import GbrainClient
                self.gbrain = GbrainClient.from_config(self.config)
                logger.info("GbrainClient ready → %s", self.config.gbrain_url)
            except Exception as exc:
                logger.warning("GbrainClient could not be initialised: %s", exc)

        logger.info(
            "AudioLink service started as %s (%s) on port %s",
            self.identity.name,
            self.identity.device_id,
            self.config.port,
        )

    async def stop(self) -> None:
        if not self._running:
            return
        self._running = False
        # Stop scheduler
        if self._scheduler_task is not None:
            self._scheduler_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._scheduler_task
            self._scheduler_task = None
        try:
            await self.media.shutdown()
        except Exception as exc:  # pragma: no cover - best effort
            logger.debug("media shutdown failed: %s", exc)
        await self.hub.shutdown()
        if self.browser is not None:
            try:
                await self.browser.stop()
            except Exception as exc:  # pragma: no cover
                logger.debug("browser stop failed: %s", exc)
            self.browser = None
        if self.advertiser is not None:
            try:
                await self.advertiser.stop()
            except Exception as exc:  # pragma: no cover
                logger.debug("advertiser stop failed: %s", exc)
            self.advertiser = None
        self._sessions.clear()
        self._record("service.stopped", {})
        logger.info("AudioLink service stopped")

    @property
    def running(self) -> bool:
        return self._running

    async def update_port(self, port: int) -> None:
        """Called once uvicorn knows the real port (supports port 0)."""
        self.config.port = port
        if self.advertiser is not None and port != self.advertiser.port:
            try:
                await self.advertiser.update_port(port)
            except Exception as exc:  # pragma: no cover
                logger.warning("could not update mDNS port: %s", exc)

    def _require_identity(self) -> Identity:
        if not self._running or self.identity is None:
            raise Unavailable("AudioLink service is not running")
        if self.registry is None or self.pairing is None:
            raise Unavailable("AudioLink service is not initialized")
        return self.identity

    # ---------------------------------------------------------------- pairing
    def _on_pairing_request(self, request: PairingRequest) -> None:
        logger.warning(
            "pairing request from %s (%s): show code %s to the user",
            request.device_name,
            request.device_id,
            request.code,
        )
        self._record(
            "pairing.requested",
            {
                "device_id": request.device_id,
                "device_name": request.device_name,
                "code": request.code,
                "expires_at": request.expires_at.isoformat(),
            },
        )

    async def create_pairing_request(
        self,
        device_id: str,
        device_name: str = "",
        platform: str = "android",
        app_version: str = "",
    ) -> dict:
        self._require_identity()
        assert self.pairing is not None
        request = await self.pairing.request_pairing(
            device_id=device_id,
            device_name=device_name,
            platform=platform,
            app_version=app_version,
        )
        return request.public(include_code=True)

    async def pending_pairings(self, include_code: bool = False) -> list[dict]:
        self._require_identity()
        assert self.pairing is not None
        return await self.pairing.pending(include_code=include_code)

    async def confirm_pairing(self, request_id: str, code: str) -> dict:
        self._require_identity()
        assert self.pairing is not None
        assert self.identity is not None
        result = await self.pairing.confirm(request_id, code)
        self._record(
            "pairing.approved",
            {"device_id": result.device.device_id, "device_name": result.device.name},
        )
        return {
            "device": result.device.public(),
            "token": result.token,
            "desktop": {
                "device_id": self.identity.device_id,
                "name": self.identity.name,
                "fingerprint": self.identity.fingerprint,
                "protocol": PROTOCOL,
            },
        }

    async def unpair(self, device_id: str) -> dict:
        self._require_identity()
        assert self.registry is not None
        await self.registry.remove(device_id)
        await self.media.end(device_id)
        self._end_session_unlocked(device_id)
        self._record("device.unpaired", {"device_id": device_id})
        return {"device_id": device_id, "paired": False}

    async def paired_devices(self) -> list[dict]:
        self._require_identity()
        assert self.registry is not None
        devices = await self.registry.list()
        online = {p["device_id"] for p in await self.hub.online_devices()}
        return [
            {**device.public(), "online": device.device_id in online}
            for device in devices
        ]

    # -------------------------------------------------------------- discovery
    async def _on_discovery_update(self, devices: list[DiscoveredDevice]) -> None:
        self._record("discovery.updated", {"count": len(devices)})

    async def discover_devices(self) -> list[dict]:
        self._require_identity()
        if self.browser is None:
            raise Unavailable(
                "mDNS discovery is disabled on this service",
                code="DISCOVERY_DISABLED",
            )
        return [device.public() for device in self.browser.devices]

    async def discovery_status(self) -> dict:
        self._require_identity()
        return {
            "enabled": self.browser is not None,
            "advertising": self.advertiser is not None,
            "count": len(self.browser.devices) if self.browser else 0,
        }

    # --------------------------------------------------------------- sessions
    async def connect(self, device_id: str) -> dict:
        identity = self._require_identity()
        assert self.registry is not None
        device = await self.registry.get(device_id)
        if not await self.hub.is_online(device_id):
            raise Conflict(
                "device is not connected to the signaling channel",
                code="DEVICE_OFFLINE",
                details={"device_id": device_id},
            )
        async with self._lock:
            existing = self._sessions.get(device_id)
            if existing and existing.state == "connected":
                return existing.public()
            session = ControlSession(
                session_id=secrets.token_hex(8),
                device_id=device_id,
                device_name=device.name,
            )
            self._sessions[device_id] = session
        await self.hub.send(
            device_id,
            {
                "type": "control.session",
                "action": "start",
                "session_id": session.session_id,
                "desktop": {
                    "device_id": identity.device_id,
                    "name": identity.name,
                    "fingerprint": identity.fingerprint,
                },
            },
        )
        self._record("session.connected", {"device_id": device_id})
        return session.public()

    async def disconnect(self, device_id: str) -> dict:
        self._require_identity()
        await self.media.end(device_id)
        session = self._end_session_unlocked(device_id)
        if session is None:
            raise NotFound("no active session for this device", code="NO_SESSION")
        await self.hub.send(
            device_id,
            {"type": "control.session", "action": "end", "session_id": session.session_id},
        )
        self._record("session.disconnected", {"device_id": device_id})
        return {"device_id": device_id, "session_id": session.session_id, "state": "ended"}

    def _end_session_unlocked(self, device_id: str) -> ControlSession | None:
        session = self._sessions.pop(device_id, None)
        if session is not None:
            session.state = "ended"
        return session

    async def start_audio_session(self, device_id: str | None = None) -> dict:
        """Start the WebRTC media session for a paired, connected device.

        The phone sends the offer and the desktop answers with its capture
        track, so phone -> PC and PC -> phone audio share one peer connection.
        Desktop capture stays off until ``set_microphone_enabled(true)``;
        until the offer arrives the session reports ``waiting_for_offer``
        rather than pretending audio flows.
        """
        self._require_identity()
        assert self.registry is not None
        if device_id is None:
            if not self._sessions:
                raise Conflict(
                    "no connected device; connect to a device first",
                    code="NO_SESSION",
                )
            device_id = next(iter(self._sessions))
        device = await self.registry.get(device_id)
        if not await self.hub.is_online(device_id):
            raise Conflict(
                "device is not connected to the signaling channel",
                code="DEVICE_OFFLINE",
                details={"device_id": device_id},
            )
        if device_id not in self._sessions:
            await self.connect(device_id)
        control = self._sessions[device_id]
        result = await self.media.start(device_id, device.name)
        await self.hub.send(
            device_id,
            {
                "type": "media.start",
                "transport": "webrtc",
                "session_id": control.session_id,
                "codec": "opus",
                "sample_rate": 48_000,
                "channels": 1,
                "direction": "both",
                "capture_enabled": self.media.capture_enabled,
            },
        )
        self._record(
            "media.start_requested",
            {"device_id": device_id, "state": result["state"]},
        )
        return {**result, "session_id": control.session_id, "codec": "opus"}

    async def end_audio_session(self) -> dict:
        self._require_identity()
        for device_id in list(self._sessions):
            await self.hub.send(device_id, {"type": "media.stop"})
        result = await self.media.end_all()
        if result["ended"]:
            self._record("media.ended", {"count": len(result["ended"])})
        return {**result, "media_transport": "webrtc"}

    async def set_microphone_enabled(self, enabled: bool) -> dict:
        """Authorize desktop microphone capture sent to the phone (M3)."""
        self._require_identity()
        assert self.registry is not None
        capture = await self.media.set_capture_enabled(enabled)
        for session in self._sessions.values():
            session.microphone_enabled = enabled
        try:
            if self.audio.tools_available:
                await self.audio.set_source_mute(not enabled, self.config.capture_device)
        except MediaUnavailable:
            logger.debug("could not mute desktop source; no sound server")
        self._record("control.microphone", {"enabled": enabled})
        return {"microphone_enabled": enabled, **capture}

    async def set_speaker_enabled(self, enabled: bool) -> dict:
        self._require_identity()
        for session in self._sessions.values():
            session.speaker_enabled = enabled
        self.media.set_playout_enabled(enabled)
        try:
            if self.audio.tools_available:
                await self.audio.set_sink_mute(not enabled, self.config.playback_device)
        except MediaUnavailable:
            logger.debug("could not mute desktop sink; no sound server")
        self._record("control.speaker", {"enabled": enabled})
        return {"speaker_enabled": enabled}

    async def set_volume(self, level: int) -> dict:
        self._require_identity()
        if not 0 <= level <= 100:
            from .errors import ValidationFailed

            raise ValidationFailed("volume must be between 0 and 100")
        for session in self._sessions.values():
            session.volume = level
        self.media.set_volume(level)
        try:
            result = await self.audio.set_sink_volume(
                level, self.config.playback_device
            )
        except MediaUnavailable:
            result = {"volume_percent": level, "applied": False}
        self._record("control.volume", {"level": level})
        return {**result, "volume": level}

    async def set_audio_routing(
        self,
        playback_device: str | None = None,
        capture_device: str | None = None,
    ) -> dict:
        """Point phone -> PC playout and PC -> phone capture at different devices.

        Used by call integrations that route audio through virtual sinks instead
        of the default speakers and microphone. Live media sessions are closed
        and re-opened so the phone renegotiates against the new routing.
        """
        self._require_identity()
        if playback_device is not None:
            self.config.playback_device = str(playback_device)
        if capture_device is not None:
            self.config.capture_device = str(capture_device)
        restarted: list[str] = []
        failed: list[dict] = []
        for status in self.media.all_sessions():
            await self.media.end(status["device_id"])
        for device_id in list(self._sessions):
            # The phone keeps its peer connection otherwise and never offers
            # again; media.stop makes it tear down so media.start renegotiates.
            try:
                await self.hub.send(device_id, {"type": "media.stop"})
            except Exception as exc:  # pragma: no cover - peer vanishing mid-call
                logger.debug("could not ask %s to stop media: %s", device_id, exc)
        for device_id in list(self._sessions):
            try:
                await self.start_audio_session(device_id)
                restarted.append(device_id)
            except AudioLinkError as exc:
                # The routing change still applies to the next session.
                failed.append({"device_id": device_id, "code": exc.code})
                logger.warning("could not restart media for %s: %s", device_id, exc)
        self._record(
            "media.routing",
            {
                "playback_device": self.config.playback_device,
                "capture_device": self.config.capture_device,
            },
        )
        return {
            "playback_device": self.config.playback_device,
            "capture_device": self.config.capture_device,
            "restarted": restarted,
            "not_restarted": failed,
        }

    async def get_audio_devices(self) -> dict:
        self._require_identity()
        return await self.audio.list_devices()

    async def send_signal(self, device_id: str, payload: dict) -> dict:
        """Push a raw JSON message into a connected phone's WebSocket session."""
        self._require_identity()
        # Try the specified device first, then any connected device.
        target = device_id if device_id else None
        if target and not await self.hub.is_online(target):
            # fall back to any online device
            online = await self.hub.online_devices()
            if not online:
                raise Unavailable("no connected device to send signal to")
            target = online[0]["device_id"]
        elif not target:
            online = await self.hub.online_devices()
            if not online:
                raise Unavailable("no connected device to send signal to")
            target = online[0]["device_id"]
        sent = await self.hub.send(target, payload)
        self._record("signal.sent", {"device_id": target, "type": payload.get("type", "")})
        return {"sent": sent, "device_id": target}

    # -------------------------------------------------------------- scheduler
    async def _scheduler_loop(self) -> None:
        """Check for due tasks every 5 seconds and fire them as sms.send signals."""
        while self._running:
            await asyncio.sleep(5)
            if self.task_store is None:
                continue
            try:
                due = await self.task_store.list_due()
                for task in due:
                    await self._fire_task(task)
            except Exception:
                logger.exception("scheduler loop error")

    def generate_reply(self, contact: str, body: str, source: str = "sms") -> str:
        """Generate an in-process reply through the bot (used by the signaling hub)."""
        if self.bot is None:
            return ""
        return self.bot.generate_bridge_reply(contact, body, source=source)

    def note_sms_seen(self, device_id: str, sms_id: int) -> bool:
        """Return False when this sms_id was already handled for the device."""
        try:
            sms_id = int(sms_id)
        except (TypeError, ValueError):
            return True
        if sms_id <= 0:
            return True
        previous = self._sms_seen.get(device_id, 0)
        if sms_id <= previous:
            return False
        self._sms_seen[device_id] = sms_id
        return True

    async def create_sms_task(
        self,
        *,
        to_number: str,
        message: str,
        scheduled_at: str,
        task_type: str = "sms",
        label: str = "",
        source_device_id: str = "",
    ) -> ScheduledTask:
        """Validate and persist a task for the scheduler."""
        if self.task_store is None:
            raise Unavailable("task store is not initialised")
        number = (to_number or "").strip()
        body = (message or "").strip()
        if not number:
            from .errors import ValidationFailed

            raise ValidationFailed("to_number is required")
        if not body:
            from .errors import ValidationFailed

            raise ValidationFailed("message is required")
        task_type = "call" if (task_type or "").strip().lower() == "call" else "sms"
        try:
            when = normalize_scheduled_at(scheduled_at)
        except (ValueError, TypeError) as exc:
            from .errors import ValidationFailed

            raise ValidationFailed(f"invalid scheduled_at: {scheduled_at!r}") from exc
        task = ScheduledTask(
            task_id=secrets.token_hex(4),
            task_type=task_type,
            to_number=number,
            message=body,
            scheduled_at=when,
            label=(label or "").strip(),
            source_device_id=source_device_id or "",
            created_at=datetime.now().isoformat(),
        )
        await self.task_store.add(task)
        self._record(
            "task.created",
            {
                "task_id": task.task_id,
                "label": task.label,
                "to": task.to_number,
                "status": task.status,
                "scheduled_at": task.scheduled_at,
            },
        )
        return task

    async def _fire_task(self, task: ScheduledTask) -> None:
        """Signal the phone to send the SMS for a due task."""
        assert self.task_store is not None
        await self.dispatch_sms_task(task, mark_done_on_dispatch=True)

    async def dispatch_sms_task(
        self,
        task: ScheduledTask,
        *,
        mark_done_on_dispatch: bool = False,
        record_attempt: bool = True,
    ) -> dict[str, Any]:
        """Send an SMS task to the phone. Returns dispatch status."""
        assert self.task_store is not None
        payload = {
            "type": SMS_SEND,
            "to_number": task.to_number,
            "body": task.message,
            "task_id": task.task_id,
        }
        try:
            result = await self.send_signal(task.source_device_id, payload)
            if result.get("sent"):
                self._record(
                    "task.dispatched",
                    {"task_id": task.task_id, "to": task.to_number, "device_id": result.get("device_id")},
                )
                logger.info("Task %s dispatched → %s", task.task_id, task.to_number)
                if mark_done_on_dispatch:
                    # Fallback completion when phone delivery receipt is not expected.
                    await self.task_store.mark_done(task.task_id)
                    self._record("task.fired", {"task_id": task.task_id, "to": task.to_number})
                return {"sent": True, "device_id": result.get("device_id"), "task": asdict(task)}
            if record_attempt:
                still_pending = await self.task_store.bump_attempt(
                    task.task_id, "no connected device"
                )
                logger.warning(
                    "Task %s could not be delivered (attempt %d)%s",
                    task.task_id,
                    task.attempts,
                    "" if still_pending else " — giving up",
                )
            return {"sent": False, "error": "no connected device", "task": asdict(task)}
        except Exception as exc:
            if record_attempt:
                still_pending = await self.task_store.bump_attempt(task.task_id, str(exc))
                logger.exception(
                    "Task %s execution failed (attempt %d)%s: %s",
                    task.task_id,
                    task.attempts,
                    "" if still_pending else " — giving up",
                    exc,
                )
            return {"sent": False, "error": str(exc), "task": asdict(task)}

    async def wait_for_task_terminal(
        self, task_id: str, timeout: float = 45.0, interval: float = 0.5
    ) -> ScheduledTask | None:
        """Wait until a task leaves pending state or timeout."""
        if self.task_store is None:
            return None
        deadline = time.time() + max(0.0, timeout)
        while True:
            task = await self.task_store.get(task_id)
            if task is None:
                return None
            if task.status != "pending":
                return task
            if time.time() >= deadline:
                return task
            await asyncio.sleep(interval)

    # -------------------------------------------------------------- observability
    def _record(self, event: str, fields: dict[str, Any]) -> None:
        self._events.append({"ts": utcnow_iso(), "event": event, **fields})

    async def events(self, limit: int = 50) -> list[dict]:
        return list(self._events)[-limit:]

    async def status(self) -> dict:
        identity = self._require_identity()
        online = await self.hub.online_devices()
        devices = await self.registry.list() if self.registry else []
        media = self.media.diagnostics()
        sessions = []
        for control in self._sessions.values():
            public = control.public()
            media_status = self.media.session_status(control.device_id)
            public["media_transport"] = "webrtc" if media_status else "not_negotiated"
            if media_status is not None:
                public["media"] = media_status
            sessions.append(public)
        return {
            "service": SERVICE_NAME,
            "version": __version__,
            "running": True,
            "uptime_sec": round(time.time() - self._started_at, 1),
            "device_id": identity.device_id,
            "device_name": identity.name,
            "platform": identity.platform,
            "fingerprint": identity.fingerprint,
            "port": self.config.port,
            "mdns": {
                "advertising": self.advertiser is not None,
                "browsing": self.browser is not None,
            },
            "paired_device_count": len(devices),
            "online_devices": online,
            "sessions": sessions,
            "audio": self.audio.diagnostics(),
            "routing": {
                "capture_device": self.config.capture_device,
                "playback_device": self.config.playback_device,
            },
            "media": media,
            "media_transport": "webrtc" if media["available"] else "unavailable",
        }

    async def diagnostics(self) -> dict:
        identity = self._require_identity()
        audio = await self.audio.list_devices()
        return {
            "service": await self.status(),
            "identity": {
                "device_id": identity.device_id,
                "name": identity.name,
                "fingerprint": identity.fingerprint,
                "data_dir": str(identity.data_dir),
                "tls_cert": str(identity.cert_path),
            },
            "signaling_peers": await self.hub.online_devices(),
            "pairing_pending": await self.pending_pairings(include_code=False),
            "audio": audio,
            "events": await self.events(limit=25),
            "milestones": {
                "M1_pairing_and_signaling": "implemented",
                "M2_phone_to_pc_audio": (
                    "implemented" if self.media.available else "media engine unavailable"
                ),
                "M3_pc_to_phone_audio": (
                    "implemented"
                    if self.media.available and self.media.capture_available
                    else "capture tool unavailable"
                ),
                "M4_two_way": "pending",
                "M5_sms_integration": (
                    "implemented" if self.gbrain is not None else "gbrain_url not configured"
                ),
            },
        }


def _load_or_create_admin_token(data_dir: Path) -> str:
    """Admin token gating the local management API."""
    data_dir = Path(data_dir).expanduser()
    data_dir.mkdir(parents=True, exist_ok=True)
    path = data_dir / ADMIN_TOKEN_FILE
    if path.is_file():
        token = path.read_text(encoding="utf-8").strip()
        if token:
            return token
    token = secrets.token_urlsafe(32)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(token + "\n")
    os.chmod(path, 0o600)
    return token
