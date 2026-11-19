"""GBrain-facing operation table (``audiolink.*``).

GBrain can call these either in-process through :class:`AudioLinkOperations`
or over HTTP via ``POST /v1/operations``. Every method is async and failures
are :class:`~audiolink.errors.AudioLinkError` instances carrying a stable code.
"""

from __future__ import annotations

import secrets
from dataclasses import asdict
from datetime import datetime
from typing import Any, Awaitable, Callable

from .errors import ValidationFailed
from .service import AudioLinkService
from .signaling import SMS_SEND
from .tasks import ScheduledTask


class AudioLinkOperations:
    def __init__(self, service: AudioLinkService) -> None:
        self.service = service
        self._table: dict[str, tuple[Callable[..., Awaitable[Any]], tuple[str, ...]]] = {
            "audiolink.get_status": (self._get_status, ()),
            "audiolink.start_service": (self._start_service, ()),
            "audiolink.stop_service": (self._stop_service, ()),
            "audiolink.discover_devices": (self._discover_devices, ()),
            "audiolink.get_paired_devices": (self._paired_devices, ()),
            "audiolink.pair_device": (self._pair_device, ("device_id",)),
            "audiolink.connect": (self._connect, ("device_id",)),
            "audiolink.disconnect": (self._disconnect, ("device_id",)),
            "audiolink.start_audio_session": (self._start_audio, ("device_id",)),
            "audiolink.end_audio_session": (self._end_audio, ()),
            "audiolink.set_microphone_enabled": (self._set_mic, ("enabled",)),
            "audiolink.set_speaker_enabled": (self._set_speaker, ("enabled",)),
            "audiolink.set_volume": (self._set_volume, ("level",)),
            "audiolink.set_audio_routing": (
                self._set_audio_routing,
                ("playback_device", "capture_device"),
            ),
            "audiolink.get_audio_devices": (self._audio_devices, ()),
            "audiolink.get_diagnostics": (self._diagnostics, ()),
            "audiolink.send_signal": (self._send_signal, ("device_id", "payload")),
            # SMS integration
            "audiolink.sms_list_tasks":     (self._sms_list_tasks,     ()),
            "audiolink.sms_cancel_task":    (self._sms_cancel_task,    ("task_id",)),
            "audiolink.sms_send_now":       (self._sms_send_now,       ("device_id", "to_number", "body")),
            "audiolink.sms_set_auto_reply": (self._sms_set_auto_reply, ("enabled",)),
            "audiolink.sms_schedule": (
                self._sms_schedule,
                ("to_number", "message", "scheduled_at"),
            ),
            "audiolink.sms_instruct": (self._sms_instruct, ("instruction",)),
        }

    @property
    def names(self) -> list[str]:
        return sorted(self._table)

    def describe(self) -> list[dict]:
        return [
            {"operation": name, "required_params": list(params)}
            for name, (_, params) in sorted(self._table.items())
        ]

    async def execute(self, operation: str, params: dict | None = None) -> Any:
        entry = self._table.get(operation)
        if entry is None:
            raise ValidationFailed(
                f"unknown audiolink operation: {operation}",
                details={"operation": operation, "available": self.names},
            )
        handler, required = entry
        params = params or {}
        missing = [key for key in required if key not in params]
        if missing:
            raise ValidationFailed(
                "missing required parameters",
                details={"operation": operation, "missing": missing},
            )
        return await handler(**{key: params[key] for key in required})

    # ------------------------------------------------------------------ ops
    async def _get_status(self):
        return await self.service.status()

    async def _start_service(self):
        if self.service.running:
            return {"running": True, "already_running": True, **(await self.service.status())}
        await self.service.start()
        return await self.service.status()

    async def _stop_service(self):
        await self.service.stop()
        return {"running": False}

    async def _discover_devices(self):
        return {"devices": await self.service.discover_devices()}

    async def _paired_devices(self):
        return {"devices": await self.service.paired_devices()}

    async def _pair_device(self, device_id: str):
        return await self.service.create_pairing_request(device_id=device_id)

    async def _connect(self, device_id: str):
        return await self.service.connect(device_id)

    async def _disconnect(self, device_id: str):
        return await self.service.disconnect(device_id)

    async def _start_audio(self, device_id: str | None = None):
        return await self.service.start_audio_session(device_id)

    async def _end_audio(self):
        return await self.service.end_audio_session()

    async def _set_mic(self, enabled: bool):
        return await self.service.set_microphone_enabled(bool(enabled))

    async def _set_speaker(self, enabled: bool):
        return await self.service.set_speaker_enabled(bool(enabled))

    async def _set_volume(self, level: int):
        return await self.service.set_volume(int(level))

    async def _set_audio_routing(self, playback_device: str, capture_device: str):
        return await self.service.set_audio_routing(
            playback_device=str(playback_device),
            capture_device=str(capture_device),
        )

    async def _audio_devices(self):
        return await self.service.get_audio_devices()

    async def _diagnostics(self):
        return await self.service.diagnostics()

    async def _send_signal(self, device_id: str, payload: dict):
        return await self.service.send_signal(device_id, payload)

    # ── SMS operations ────────────────────────────────────────────────────

    async def _sms_list_tasks(self):
        if self.service.task_store is None:
            return {"tasks": []}
        pending = await self.service.task_store.list_pending()
        return {"tasks": [asdict(t) for t in pending]}

    async def _sms_cancel_task(self, task_id: str):
        if self.service.task_store is None:
            return {"cancelled": False, "task_id": task_id}
        ok = await self.service.task_store.cancel(task_id)
        return {"cancelled": ok, "task_id": task_id}

    async def _sms_send_now(self, device_id: str, to_number: str, body: str):
        result = await self.service.send_signal(
            device_id,
            {"type": SMS_SEND, "to_number": to_number, "body": body},
        )
        return {"sent": result.get("sent", False), "to_number": to_number}

    async def _sms_set_auto_reply(self, enabled: bool):
        self.service.config.auto_reply_enabled = bool(enabled)
        return {"auto_reply_enabled": self.service.config.auto_reply_enabled}

    async def _sms_schedule(self, to_number: str, message: str, scheduled_at: str):
        task = await self.service.create_sms_task(
            to_number=str(to_number),
            message=str(message),
            scheduled_at=str(scheduled_at),
        )
        return asdict(task)

    async def _sms_instruct(self, instruction: str):
        if self.service.gbrain is None:
            raise ValidationFailed(
                "gbrain-sms is not configured; set gbrain_url in audiolink.yaml"
            )
        import asyncio

        loop = asyncio.get_running_loop()
        draft_resp = await loop.run_in_executor(
            None, self.service.gbrain.parse_instruction, str(instruction)
        )
        draft = draft_resp.get("task_draft") or {}
        if not draft:
            return {
                "parsed": False,
                "instruction": instruction,
                "error": draft_resp.get("error", "could not parse instruction"),
            }
        task = await self.service.create_sms_task(
            to_number=str(draft.get("number", "")),
            message=str(draft.get("message", "")),
            scheduled_at=str(draft.get("scheduled_at", "")),
            task_type=str(draft.get("type", "sms")),
            label=str(draft.get("label", "")),
        )
        return {"parsed": True, **asdict(task)}
