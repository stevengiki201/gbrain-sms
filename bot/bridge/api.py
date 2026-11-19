"""FastAPI application exposing the AudioLink service.

Two security zones:

* **Public (LAN) endpoints** used by the Android companion during pairing:
  ``/v1/health``, ``/v1/identity``, ``/v1/pairing/*`` and ``/v1/signaling``.
* **Management endpoints** bound to loopback and gated by the admin token in
  ``<data_dir>/admin.token`` (used by the dashboard and GBrain).

All pairings/sessions run over TLS; the service refuses to start without a
certificate.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import os
import re
import secrets
from dataclasses import asdict
from datetime import datetime
import re as _re
from typing import Any, Callable

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from . import __version__
from .errors import AudioLinkError, Forbidden, Unauthorized, Unavailable, ValidationFailed
from .hosting import verify_enrollment_with_website
from .operations import AudioLinkOperations
from .service import AudioLinkService
from .signaling import (
    CONTROL,
    ERROR,
    HELLO,
    HELLO_ACK,
    PING,
    PONG,
    PROTOCOL,
    MEDIA_STOP,
    SDP_ANSWER,
    SDP_OFFER,
    SESSION_END,
    SMS_INBOUND,
    SMS_RESULT,
    SMS_SEND,
    validate_message,
)
from .tasks import ScheduledTask, normalize_scheduled_at
from .discovery import is_loopback
from bot.adb_connector import AdbConnector
from bot.memory_store import normalize_contact

logger = logging.getLogger("gbrain.bridge.api")


async def _maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


# --------------------------------------------------------------------- schemas
class PairingRequestBody(BaseModel):
    device_id: str = Field(min_length=1, max_length=128)
    device_name: str = Field(default="", max_length=128)
    platform: str = Field(default="android", max_length=32)
    app_version: str = Field(default="", max_length=32)


class PairingConfirmBody(BaseModel):
    code: str = Field(min_length=1, max_length=16)


class EnabledBody(BaseModel):
    enabled: bool


class VolumeBody(BaseModel):
    level: int = Field(ge=0, le=100)


class SessionBody(BaseModel):
    device_id: str = Field(min_length=1, max_length=128)


class SmsSendBody(BaseModel):
    to_number: str = Field(min_length=1, max_length=32)
    body: str = Field(min_length=1, max_length=2000)
    device_id: str = Field(default="", max_length=128)


class OperationBody(BaseModel):
    operation: str
    params: dict[str, Any] = Field(default_factory=dict)


class SmsInstructBody(BaseModel):
    instruction: str
    device_id: str = ""


class SmsChatBody(BaseModel):
    message: str
    device_id: str = ""


class SmsTaskBody(BaseModel):
    to_number: str = Field(min_length=1, max_length=64)
    message: str = Field(min_length=1, max_length=4000)
    scheduled_at: str = Field(min_length=1, max_length=64)
    task_type: str = Field(default="sms", max_length=16)
    label: str = Field(default="", max_length=200)


class AutoReplyBody(BaseModel):
    enabled: bool


class HostedEnrollBody(BaseModel):
    token: str = Field(min_length=10, max_length=300)
    device_id: str = Field(min_length=1, max_length=128)
    device_name: str = Field(default="", max_length=128)
    platform: str = Field(default="android", max_length=32)
    app_version: str = Field(default="", max_length=32)


class HostedSmsSendBody(BaseModel):
    website_device_id: str = Field(min_length=1, max_length=128)
    to_number: str = Field(min_length=1, max_length=32)
    body: str = Field(min_length=1, max_length=2000)


def create_app(service: AudioLinkService, app: FastAPI | None = None) -> FastAPI:
    """Register the bridge API on ``app`` (or a new app for standalone use)."""
    if app is None:
        app = FastAPI(
            title="GBrain AudioLink",
            version=__version__,
            description="Two-way PC <-> Android audio communication subsystem",
        )
    app.state.service = service
    app.state.operations = AudioLinkOperations(service)

    @app.exception_handler(AudioLinkError)
    async def _audio_error_handler(_: Request, exc: AudioLinkError) -> JSONResponse:
        return JSONResponse(status_code=exc.http_status, content=exc.to_dict())

    @app.exception_handler(RequestValidationError)
    async def _validation_handler(
        _: Request, exc: RequestValidationError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content={
                "error": {
                    "code": "VALIDATION_FAILED",
                    "message": "request validation failed",
                    "details": {"errors": exc.errors()},
                }
            },
        )

    def require_admin(request: Request) -> None:
        """Loopback-only + admin bearer token."""
        host = request.client.host if request.client else ""
        if not (is_loopback(host) or host in {"localhost", "testclient"}):
            raise Forbidden(
                "management API is only reachable from the local machine",
                code="LOCAL_ONLY",
            )
        if not service.admin_token:
            raise Unavailable("AudioLink service is not running")
        supplied = request.headers.get("x-audiolink-admin", "")
        if not supplied:
            auth = request.headers.get("authorization", "")
            if auth.lower().startswith("bearer "):
                supplied = auth[7:].strip()
        if not supplied or not secrets.compare_digest(supplied, service.admin_token):
            raise Unauthorized("invalid admin token", code="INVALID_ADMIN_TOKEN")

    def require_service_token(request: Request) -> None:
        configured = os.environ.get("GBRAIN_SERVICE_TOKEN", "")
        supplied = request.headers.get("x-gbrain-service", "")
        if not configured or not supplied or not secrets.compare_digest(supplied, configured):
            raise Unauthorized("invalid service token", code="INVALID_SERVICE_TOKEN")

    async def require_device_token(request: Request) -> str:
        """Accept requests from any paired device on the LAN or the local admin token."""

        auth = request.headers.get("authorization", "")
        if not auth.lower().startswith("bearer "):
            raise Unauthorized("missing bearer token", code="UNAUTHORIZED")
        token = auth[7:].strip()
        if not token:
            raise Unauthorized("empty bearer token", code="UNAUTHORIZED")
        if service.admin_token and secrets.compare_digest(token, service.admin_token):
            return "admin"
        if service.registry is None:
            raise Unavailable("AudioLink service is not running")
        devices = await service.registry.list()
        import hashlib as _hl

        token_hash = _hl.sha256(token.encode()).hexdigest()
        for device in devices:
            if device.token_hash and secrets.compare_digest(device.token_hash, token_hash):
                return device.device_id
        raise Unauthorized("unknown device token", code="INVALID_TOKEN")

    admin  = APIRouter(prefix="/v1", dependencies=[Depends(require_admin)])
    device = APIRouter(prefix="/v1")   # authenticated per-request via require_device_token
    public = APIRouter(prefix="/v1")
    hosted = APIRouter(prefix="/v1/hosted", dependencies=[Depends(require_service_token)])

    # ------------------------------------------------------------------ public
    @app.get("/")
    async def root() -> dict:
        return {
            "service": "gbrain-audiolink",
            "version": __version__,
            "protocol": PROTOCOL,
            "docs": "/docs",
            "status": service.running,
        }

    @public.get("/health")
    async def health() -> dict:
        return {
            "status": "ok" if service.running else "starting",
            "service": "gbrain-audiolink",
            "version": __version__,
        }

    @public.get("/identity")
    async def identity() -> dict:
        if service.identity is None:
            raise Unavailable("AudioLink service is not running")
        return {
            "device_id": service.identity.device_id,
            "name": service.identity.name,
            "platform": service.identity.platform,
            "version": __version__,
            "protocol": PROTOCOL,
            "fingerprint": service.identity.fingerprint,
            "pairing_required": True,
        }

    @public.post("/pairing/requests")
    async def pairing_request(body: PairingRequestBody) -> dict:
        request = await service.create_pairing_request(
            device_id=body.device_id,
            device_name=body.device_name,
            platform=body.platform,
            app_version=body.app_version,
        )
        # The code is shown on the desktop only; it must never travel to the phone.
        return {
            "request_id": request["request_id"],
            "expires_at": request["expires_at"],
            "code_length": 6,
            "desktop": {
                "device_id": service.identity.device_id if service.identity else "",
                "name": service.identity.name if service.identity else "",
                "fingerprint": service.identity.fingerprint if service.identity else "",
            },
        }

    @public.post("/pairing/requests/{request_id}/confirm")
    async def pairing_confirm(request_id: str, body: PairingConfirmBody) -> dict:
        return await service.confirm_pairing(request_id, body.code)

    @public.post("/enroll")
    async def hosted_enroll(body: HostedEnrollBody) -> dict:
        website = await asyncio.to_thread(
            verify_enrollment_with_website,
            body.token,
            body.device_id,
            body.device_name,
            body.platform,
            body.app_version,
        )
        return await service.link_hosted_enrollment(
            website=website,
            phone_device_id=body.device_id,
            phone_name=body.device_name,
        )

    # ---------------------------------------------------------------- management
    @admin.get("/status")
    async def status() -> dict:
        return await service.status()

    @admin.post("/service/start")
    async def start_service() -> dict:
        await service.start()
        return await service.status()

    @admin.post("/service/stop")
    async def stop_service() -> dict:
        await service.stop()
        return {"running": False}

    @admin.get("/discovery/devices")
    async def discover_devices() -> dict:
        return {"devices": await service.discover_devices()}

    @admin.get("/devices")
    async def paired_devices() -> dict:
        return {"devices": await service.paired_devices()}

    @admin.delete("/devices/{device_id}")
    async def unpair(device_id: str) -> dict:
        return await service.unpair(device_id)

    @admin.post("/devices/{device_id}/connect")
    async def connect(device_id: str) -> dict:
        return await service.connect(device_id)

    @admin.post("/devices/{device_id}/disconnect")
    async def disconnect(device_id: str) -> dict:
        return await service.disconnect(device_id)

    @admin.get("/pairing/pending")
    async def pending_pairings(include_code: bool = True) -> dict:
        return {"requests": await service.pending_pairings(include_code=include_code)}

    @admin.post("/sessions/start")
    async def start_session(body: SessionBody) -> dict:
        return await service.start_audio_session(body.device_id)

    @admin.post("/sessions/end")
    async def end_session() -> dict:
        return await service.end_audio_session()

    @admin.post("/controls/microphone")
    async def microphone(body: EnabledBody) -> dict:
        return await service.set_microphone_enabled(body.enabled)

    @admin.post("/controls/speaker")
    async def speaker(body: EnabledBody) -> dict:
        return await service.set_speaker_enabled(body.enabled)

    @admin.post("/controls/volume")
    async def volume(body: VolumeBody) -> dict:
        return await service.set_volume(body.level)

    @admin.get("/audio/devices")
    async def audio_devices() -> dict:
        return await service.get_audio_devices()

    @admin.get("/diagnostics")
    async def diagnostics() -> dict:
        return await service.diagnostics()

    @admin.get("/events")
    async def events(limit: int = 50) -> dict:
        return {"events": await service.events(limit=max(1, min(limit, 200)))}

    # ── SMS integration endpoints (device-token auth — reachable from phone) ──

    @device.post("/gbrain/sms/instruct")
    async def sms_instruct(
        body: SmsInstructBody,
        auth_device_id: str = Depends(require_device_token),
    ) -> dict:
        """Accept a natural-language instruction, parse via gbrain-sms, store the task."""
        if service.gbrain is None:
            raise HTTPException(503, detail="GBrain AI backend not configured")
        if service.task_store is None:
            raise HTTPException(503, detail="Task store not initialised")
        loop = asyncio.get_event_loop()
        draft_resp = await loop.run_in_executor(
            None, service.gbrain.parse_instruction, body.instruction
        )
        task_draft = draft_resp.get("task_draft") or {}
        if not task_draft:
            return {
                "parsed": False,
                "instruction": body.instruction,
                "error": draft_resp.get("error", "could not parse instruction"),
            }
        try:
            task = await service.create_sms_task(
                to_number=str(task_draft.get("number", "")),
                message=str(task_draft.get("message", "")),
                scheduled_at=str(task_draft.get("scheduled_at", "")),
                task_type=str(task_draft.get("type", "sms")),
                label=str(task_draft.get("label", "")),
                source_device_id=auth_device_id,
            )
        except AudioLinkError as exc:
            return {
                "parsed": False,
                "instruction": body.instruction,
                "error": exc.message,
            }
        return {
            "parsed": True,
            "task_id": task.task_id,
            "type": task.task_type,
            "label": task.label,
            "scheduled_at": task.scheduled_at,
            "to": task.to_number,
            "message": task.message,
        }

    @device.post("/gbrain/sms/tasks")
    async def sms_create_task(
        body: SmsTaskBody,
        auth_device_id: str = Depends(require_device_token),
    ) -> dict:
        """Create a scheduled SMS/call task with explicit details."""
        task = await service.create_sms_task(
            to_number=body.to_number,
            message=body.message,
            scheduled_at=body.scheduled_at,
            task_type=body.task_type,
            label=body.label,
            source_device_id=auth_device_id,
        )
        return {"scheduled": True, **asdict(task)}

    @device.post("/gbrain/sms/auto-reply")
    async def sms_auto_reply(
        body: AutoReplyBody,
        auth_device_id: str = Depends(require_device_token),
    ) -> dict:
        """Enable/disable auto-replies at runtime from the phone."""
        service.config.auto_reply_enabled = bool(body.enabled)
        service._record(
            "sms.auto_reply_toggled",
            {"enabled": service.config.auto_reply_enabled, "device_id": auth_device_id},
        )
        return {"auto_reply_enabled": service.config.auto_reply_enabled}

    @device.get("/gbrain/sms/tasks")
    async def sms_tasks(
        status: str = "all",
        auth_device_id: str = Depends(require_device_token),
    ) -> dict:
        """List scheduled tasks. status=all|pending|done|cancelled|failed."""
        if service.task_store is None:
            raise HTTPException(503, detail="Task store not initialised")
        wanted = (status or "pending").strip().lower()
        if wanted in {"", "pending", "active"}:
            tasks = await service.task_store.list_pending()
        elif wanted == "all":
            tasks = await service.task_store.list_all()
        elif wanted in {"done", "cancelled", "failed"}:
            tasks = [t for t in await service.task_store.list_all() if t.status == wanted]
        else:
            raise HTTPException(400, detail="status must be all, pending, done, cancelled, or failed")
        return {"tasks": [asdict(t) for t in tasks]}

    @device.delete("/gbrain/sms/tasks/{task_id}")
    async def sms_cancel_task(
        task_id: str,
        auth_device_id: str = Depends(require_device_token),
    ) -> dict:
        """Cancel a pending task by id."""
        if service.task_store is None:
            raise HTTPException(503, detail="Task store not initialised")
        cancelled = await service.task_store.cancel(task_id, reason="cancelled by user")
        if cancelled is None:
            task = await service.task_store.get(task_id)
            if task is None:
                raise HTTPException(404, detail=f"task {task_id!r} not found")
            return {
                "cancelled": False,
                "task_id": task.task_id,
                "status": task.status,
                "message": f"task {task_id!r} is already {task.status}",
            }
        return {
            "cancelled": True,
            "task_id": cancelled.task_id,
            "status": cancelled.status,
            "cancelled_at": cancelled.cancelled_at,
            "cancel_reason": cancelled.cancel_reason,
        }

    @device.post("/gbrain/sms/chat")
    async def sms_chat(
        body: SmsChatBody,
        auth_device_id: str = Depends(require_device_token),
    ) -> dict:
        """Proxy an AI chat message to gbrain-sms; store any embedded task locally."""
        if service.gbrain is None and getattr(service, "bot", None) is None:
            raise HTTPException(503, detail="Gbrain AI backend not configured")
        if service.task_store is None:
            raise HTTPException(503, detail="Task store not initialised")
        effective_device_id = body.device_id or auth_device_id

        cancel_id = _extract_cancel_task_id(body.message)
        if cancel_id:
            cancelled = await service.task_store.cancel(cancel_id, reason="cancelled by gbrain")
            if cancelled is None:
                existing = await service.task_store.get(cancel_id)
                if existing is None:
                    return {
                        "reply": f"I could not find task {cancel_id}.",
                        "cancelled": False,
                        "task": None,
                    }
                return {
                    "reply": f"Task {cancel_id} is already {existing.status}.",
                    "cancelled": False,
                    "task": asdict(existing),
                }
            service._record(
                "task.cancelled_by_gbrain",
                {"task_id": cancelled.task_id, "reason": cancelled.cancel_reason},
            )
            return {
                "reply": f"Task {cancelled.task_id} cancelled by gbrain.",
                "cancelled": True,
                "task": asdict(cancelled),
            }
        send_now = _is_send_now_request(body.message)
        loop = asyncio.get_event_loop()
        if getattr(service, "gbrain", None) is not None:
            result = await loop.run_in_executor(
                None,
                lambda: service.gbrain.ai_chat(
                    effective_device_id, body.message, schedule_local=False
                ),
            )
        else:
            raise HTTPException(503, detail="Gbrain AI client not configured")
        if not result:
            raise HTTPException(502, detail="gbrain-sms did not return a reply")
        task_data = result.get("task")
        if task_data is None and (_looks_like_schedule_request(body.message) or send_now):
            draft_resp = await loop.run_in_executor(
                None, service.gbrain.parse_instruction, body.message
            )
            draft = draft_resp.get("task_draft") or {}
            if not draft and send_now:
                # best-effort extraction from message for "send now ..." requests
                draft = _extract_send_now_draft(body.message)
            if draft:
                if send_now and not _draft_requires_time(body.message):
                    draft["scheduled_at"] = datetime.now().replace(microsecond=0).isoformat()
                result["reply"] = str(result.get("reply", "")).strip()
                task_data = draft
                result["task"] = task_data
        if isinstance(task_data, dict) and task_data.get("scheduled_at"):
            to_number = str(task_data.get("number", "")).strip()
            message_text = str(task_data.get("message", "")).strip()
            task_type = str(task_data.get("type", "sms") or "sms").strip().lower()
            try:
                task = await service.create_sms_task(
                    to_number=to_number,
                    message=message_text,
                    scheduled_at=str(task_data["scheduled_at"]),
                    task_type=task_type,
                    label=str(task_data.get("label", "")),
                    source_device_id=effective_device_id,
                )
            except AudioLinkError as exc:
                logger.warning("AI task rejected: %s", exc.message)
                task_data["scheduled"] = False
                task_data["error"] = exc.message
                if send_now:
                    result["reply"] = f"I could not add the task: {exc.message}."
            else:
                recorded = await _ensure_task_recorded(service, task.task_id)
                if recorded is None:
                    return {
                        "reply": "I tried to add the task but it was not recorded. Please try again.",
                        "task": {"task_id": task.task_id, "status": "missing"},
                    }
                task = recorded
                task_data.update(
                    {
                        "task_id": task.task_id,
                        "scheduled": True,
                        "executor": "bridge",
                        "scheduled_at": task.scheduled_at,
                        "status": task.status,
                    }
                )
                service._record("task.created_via_chat", {"task_id": task.task_id})
                if send_now and task.task_type == "sms":
                    task, dispatch = await _dispatch_task_now(service, task)
                    task_data.update(
                        {
                            "status": task.status if task else "pending",
                            "dispatched": bool(dispatch.get("sent")),
                            "dispatch_error": str(dispatch.get("error", "")),
                        }
                    )
                    if task and task.status == "done":
                        result["reply"] = f"Task {task.task_id} added and completed. SMS sent to {task.to_number}."
                    elif task and task.status == "failed":
                        err = (task.last_error if task else "") or dispatch.get("error") or "unknown error"
                        result["reply"] = f"Task {task.task_id} added but failed to send: {err}."
                    else:
                        err = (task.last_error if task else "") or dispatch.get("error", "")
                        suffix = f" ({err})" if err else ""
                        result["reply"] = f"Task {task.task_id} added and dispatch attempted, but it is still pending{suffix}."
                elif send_now and task.task_type != "sms":
                    result["reply"] = f"Task {task.task_id} added (scheduled call) for {task.to_number} at {task.scheduled_at}."
                else:
                    result["reply"] = f"Task {task.task_id} added for {task.to_number} at {task.scheduled_at}."
        elif task_data is not None:
            task_data["scheduled"] = False
            if send_now:
                result["reply"] = result.get("reply") or "I could not create a send-now task. Please include the phone number and message."
        else:
            if send_now:
                result["reply"] = result.get("reply") or "I could not create a send-now task. Please include the phone number and message."
        return result

    @admin.get("/operations")
    async def operations_list() -> dict:
        return {"operations": app.state.operations.describe()}

    @admin.post("/operations")
    async def operations_execute(body: OperationBody) -> dict:
        result = await app.state.operations.execute(body.operation, body.params)
        return {"operation": body.operation, "ok": True, "result": result}

    @hosted.get("/status")
    async def hosted_status() -> dict:
        return await service.status()

    @hosted.get("/devices")
    async def hosted_devices() -> dict:
        return {"devices": await service.hosted_enrollments()}

    @hosted.get("/events")
    async def hosted_events(website_device_id: str, after: int = 0) -> dict:
        return await service.hosted_events(website_device_id, after)

    @hosted.post("/sms/send")
    async def hosted_sms_send(body: HostedSmsSendBody) -> dict:
        enrollments = await service.hosted_enrollments()
        enrollment = next(
            (
                item
                for item in enrollments
                if item.get("website_device_id") == body.website_device_id
            ),
            None,
        )
        if enrollment is None:
            raise HTTPException(404, detail="hosted device is not enrolled")
        task_id = secrets.token_hex(8)
        result = await service.send_signal(
            str(enrollment["phone_device_id"]),
            {
                "type": SMS_SEND,
                "to_number": body.to_number,
                "body": body.body,
                "task_id": task_id,
            },
        )
        return {**result, "taskId": task_id}

    # -------------------------------------------------------------- SMS inbox / unreplied
    @admin.get("/sms/inbox")
    async def admin_sms_inbox(limit: int = 200, folder: str = "inbox") -> dict:
        bot = getattr(service, "bot", None)
        adb = getattr(bot, "adb", None) if bot is not None else AdbConnector()
        folder = "sent" if folder == "sent" else "inbox"
        msgs = adb._query_sms_folder(folder, limit=max(1, min(limit, 1000)))
        return {
            "folder": folder,
            "count": len(msgs),
            "messages": [
                {
                    "sms_id": m.sms_id,
                    "address": m.address,
                    "body": m.body,
                    "date_ms": m.date_ms,
                    "msg_type": m.msg_type,
                }
                for m in msgs
            ],
        }

    @admin.get("/sms/unreplied")
    async def admin_sms_unreplied(limit: int = 500) -> dict:
        bot = getattr(service, "bot", None)
        adb = getattr(bot, "adb", None) if bot is not None else AdbConnector()
        memory = getattr(bot, "memory", None) if bot is not None else None
        msgs = adb._query_sms_folder("inbox", limit=max(1, min(limit, 2000)))
        latest: dict[str, dict] = {}
        for m in msgs:
            if getattr(m, "msg_type", 1) != 1:
                continue
            key = str(m.address or "").strip()
            prev = latest.get(key)
            if prev is None or int(m.sms_id) > int(prev["sms_id"]):
                latest[key] = {
                    "sms_id": int(m.sms_id),
                    "address": m.address,
                    "body": m.body,
                    "date_ms": int(m.date_ms),
                }
        candidates = []
        for addr, m in latest.items():
            digits = "".join(ch for ch in str(addr) if ch.isdigit())
            if len(digits) < 9:
                continue
            if memory is not None and memory.is_sms_replied(int(m["sms_id"])):
                continue
            awaiting = True
            reason = "latest not replied"
            if memory is not None:
                try:
                    awaiting = memory.conversation_awaiting_reply(addr)
                    reason = "conversation awaiting reply" if awaiting else "conversation not awaiting reply"
                except Exception:
                    awaiting = True
                    reason = "awaiting-check-failed"
            if not awaiting and memory is not None and memory.is_sms_replied(int(m["sms_id"])):
                continue
            # If there's no reply record for latest sms, treat as unreplied candidate.
            candidates.append({**m, "awaiting": awaiting, "reason": reason})
        # prioritize awaiting ones
        candidates.sort(key=lambda x: (0 if x.get("awaiting") else 1, -int(x["sms_id"])) )
        return {"count": len(candidates), "messages": candidates}

    @admin.post("/sms/send")
    async def admin_sms_send(body: SmsSendBody) -> dict:
        dispatch = await service.send_signal(
            body.device_id,
            {"type": SMS_SEND, "to_number": body.to_number, "body": body.body},
        )
        return {"sent": bool(dispatch.get("sent")), "dispatch": dispatch}

    @admin.post("/sms/auto-reply-unreplied")
    async def admin_auto_reply_unreplied(limit: int = 200, max_send: int = 50, dry_run: bool = False) -> dict:
        bot = getattr(service, "bot", None)
        if bot is None:
            raise HTTPException(503, detail="Bot backend not available")
        adb = bot.adb
        memory = bot.memory
        msgs = adb._query_sms_folder("inbox", limit=max(1, min(limit, 2000)))
        # latest per contact
        latest_by_contact: dict[str, object] = {}
        for m in msgs:
            if getattr(m, "msg_type", 1) != 1:
                continue
            digits = "".join(ch for ch in str(m.address or "") if ch.isdigit())
            if len(digits) < 9:
                continue
            contact = normalize_contact(m.address)
            prev = latest_by_contact.get(contact)
            if prev is None or int(m.sms_id) > int(prev.sms_id):
                latest_by_contact[contact] = m

        pending = []
        for contact, m in latest_by_contact.items():
            if memory.is_sms_replied(int(m.sms_id)):
                continue
            # Treat as unreplied if no reply record for latest inbound; prefer awaiting conversations.
            awaiting = memory.conversation_awaiting_reply(contact)
            pending.append((contact, m, awaiting))
        # reply awaiting first, then others whose latest sms has no reply record
        pending.sort(key=lambda x: (0 if x[2] else 1, -int(x[1].sms_id)))

        results = []
        sent = 0
        for contact, m, awaiting in pending:
            if sent >= max(1, min(max_send, 200)):
                break
            item = {
                "sms_id": int(m.sms_id),
                "contact": contact,
                "address": m.address,
                "awaiting": awaiting,
                "body": m.body,
            }
            if dry_run:
                item["action"] = "dry_run"
                results.append(item)
                continue
            try:
                reply = await asyncio.get_event_loop().run_in_executor(
                    None, service.generate_reply, contact, m.body
                )
            except Exception as exc:
                item["action"] = "generate_failed"
                item["error"] = str(exc)
                results.append(item)
                continue
            if not reply:
                item["action"] = "empty_reply"
                results.append(item)
                continue
            # Prefer sending via connected bridge device (no ADB UI automation).
            dispatch = {"sent": False, "error": "no connected device"}
            try:
                online = await service.hub.online_devices()
            except Exception:
                online = []
            if online:
                dispatch = await service.send_signal(
                    online[0]["device_id"],
                    {"type": SMS_SEND, "to_number": contact, "body": reply},
                )
            ok = bool(dispatch.get("sent"))
            item["reply"] = reply
            item["action"] = "sent" if ok else "send_failed"
            item["dispatch"] = dispatch
            if ok:
                try:
                    memory.mark_sms_replied(int(m.sms_id), contact, m.body, reply)
                    memory.mark_sms_processed(int(m.sms_id))
                except Exception as exc:
                    item["record_error"] = str(exc)
                sent += 1
            results.append(item)
        return {
            "scanned_contacts": len(latest_by_contact),
            "pending_contacts": len(pending),
            "sent": sent,
            "dry_run": dry_run,
            "results": results,
        }


    app.include_router(public)
    app.include_router(admin)
    app.include_router(device)
    app.include_router(hosted)

    # ------------------------------------------------------------------ signaling
    @app.websocket("/v1/signaling")
    async def signaling(websocket: WebSocket, device_id: str = "") -> None:
        if service.registry is None:
            await websocket.close(code=1011, reason="service not running")
            return
        auth = websocket.headers.get("authorization", "")
        token = auth[7:].strip() if auth.lower().startswith("bearer ") else ""
        try:
            device = await service.registry.verify_token(device_id, token)
        except AudioLinkError as exc:
            await websocket.close(code=4401, reason=exc.code)
            return

        await websocket.accept()
        peer = await service.hub.attach(device.device_id, device.name, websocket)
        logger.info(
            "signaling peer connected",
            extra={"device_id": device.device_id, "peer": device.name},
        )
        try:
            await websocket.send_json(
                {
                    "type": HELLO_ACK,
                    "protocol": PROTOCOL,
                    "self": {
                        "device_id": service.identity.device_id if service.identity else "",
                        "name": service.identity.name if service.identity else "",
                        "fingerprint": (
                            service.identity.fingerprint if service.identity else ""
                        ),
                    },
                    "device": device.public(),
                    "heartbeat_sec": 15,
                }
            )
            while True:
                try:
                    message = await websocket.receive_json()
                except ValueError:
                    await service.hub.send(
                        device.device_id,
                        {"type": ERROR, "code": "INVALID_MESSAGE", "message": "invalid JSON"},
                    )
                    continue
                await service.hub.record_received(device.device_id)
                try:
                    msg_type = validate_message(message)
                except ValidationFailed as exc:
                    await service.hub.send(device.device_id, exc.to_dict()["error"] | {"type": ERROR})
                    continue

                if msg_type == PING:
                    await service.hub.send(
                        device.device_id,
                        {"type": PONG, "ts": message.get("ts"), "server_ts": __import__("time").time()},
                    )
                elif msg_type == HELLO:
                    await websocket.send_json(
                        {
                            "type": HELLO_ACK,
                            "protocol": PROTOCOL,
                            "device": device.public(),
                            "heartbeat_sec": 15,
                        }
                    )
                elif msg_type == CONTROL:
                    await handle_control(service, device.device_id, message)
                elif msg_type == SESSION_END:
                    try:
                        await service.disconnect(device.device_id)
                    except AudioLinkError:
                        pass
                elif msg_type == MEDIA_STOP:
                    await service.media.end(device.device_id)
                elif msg_type == SMS_INBOUND:
                    asyncio.create_task(
                        handle_sms_inbound(service, device.device_id, message)
                    )
                elif msg_type == SMS_RESULT:
                    await handle_sms_result(service, device.device_id, message)
                elif msg_type == SDP_OFFER:
                    await handle_media_offer(service, device.device_id, device.name, message)
                elif msg_type == SDP_ANSWER:
                    await service.hub.send(
                        device.device_id,
                        {
                            "type": ERROR,
                            "code": "NO_OFFER",
                            "message": (
                                "unexpected SDP answer; the phone is the offerer "
                                "for this media session"
                            ),
                        },
                    )
                elif msg_type == ICE_CANDIDATE:
                    try:
                        await service.media.handle_candidate(
                            device.device_id,
                            {
                                "candidate": message.get("candidate", ""),
                                "sdpMid": message.get("sdp_mid", message.get("sdpMid", "0")),
                                "sdpMLineIndex": message.get(
                                    "sdp_mline_index", message.get("sdpMLineIndex", 0)
                                ),
                            },
                        )
                    except AudioLinkError as exc:
                        await service.hub.send(
                            device.device_id, exc.to_dict()["error"] | {"type": ERROR}
                        )
        except WebSocketDisconnect:
            pass
        except Exception:  # pragma: no cover - transport errors
            logger.exception("signaling loop failed", extra={"device_id": device.device_id})
        finally:
            await service.hub.detach(device.device_id, websocket)
            logger.info("signaling peer disconnected", extra={"device_id": device.device_id})

    return app


_CANCEL_TASK_RE = re.compile(
    r"\bcancel\b.*?\btask\s*#?\s*([A-Za-z0-9_-]+)\b",
    re.IGNORECASE,
)


def _extract_cancel_task_id(text: str) -> str | None:
    match = _CANCEL_TASK_RE.search(text or "")
    return match.group(1) if match else None



_SEND_NOW_RE = re.compile(
    r"\b(send|text|sms|message)\b.*\b(now|right now|immediately|asap|sasa hivi)\b",
    re.IGNORECASE,
)
_PHONE_EXTRACT_RE = re.compile(r"(\+?\d[\d\s\-()]{7,}\d)")


def _is_send_now_request(text: str) -> bool:
    return bool(_SEND_NOW_RE.search(text or "") and _PHONE_EXTRACT_RE.search(text or ""))


def _draft_requires_time(text: str) -> bool:
    lowered = (text or "").lower()
    return bool(
        _TIME_IN_TEXT.search(lowered)
        or "tomorrow" in lowered
        or "today" in lowered
        or "tonight" in lowered
        or re.search(r"\b\d{4}-\d{2}-\d{2}\b", lowered)
    )


async def _ensure_task_recorded(service: AudioLinkService, task_id: str, attempts: int = 3):
    """Retry lookup briefly to ensure task is visible in Tasks section."""
    if service.task_store is None:
        return None
    for _ in range(max(1, attempts)):
        task = await service.task_store.get(task_id)
        if task is not None:
            return task
        await asyncio.sleep(0.2)
    return None


async def _dispatch_task_now(service: AudioLinkService, task):
    dispatch = await service.dispatch_sms_task(task, mark_done_on_dispatch=False, record_attempt=True)
    if dispatch.get("sent"):
        # Phone sends async; wait for sms.result to mark done/failed.
        task = await service.wait_for_task_terminal(task.task_id, timeout=45.0)
    else:
        task = await service.task_store.get(task.task_id) if service.task_store else task
    return task, dispatch

_SCHEDULE_VERB = re.compile(
    r"\b(send|schedule|text|sms|message|remind|call)\b", re.IGNORECASE
)
_PHONE_IN_TEXT = re.compile(r"\+?\d[\d\s-]{7,}\d")
_TIME_IN_TEXT = re.compile(
    r"\b(?:\d{1,2}[:.]\d{2}\s*(?:am|pm)?|\d{1,2}\s*(?:am|pm)|"
    r"today|tomorrow|tonight|morning|evening)\b",
    re.IGNORECASE,
)


def _looks_like_schedule_request(text: str) -> bool:
    """True when a chat message clearly asks to send/schedule something."""
    return bool(
        _SCHEDULE_VERB.search(text)
        and _PHONE_IN_TEXT.search(text)
        and _TIME_IN_TEXT.search(text)
    )


async def handle_sms_inbound(
    service: AudioLinkService, device_id: str, message: dict
) -> None:
    """Handle an inbound SMS signal from the phone.

    When auto-reply is enabled and gbrain-sms is configured, generate a reply
    and signal the phone to send it back to the original sender.
    """
    contact = str(message.get("from_number", "")).strip()
    body = str(message.get("body", "")).strip()
    if not contact or not body:
        logger.warning("sms.inbound missing from_number or body")
        return

    await _maybe_await(service.record_hosted_sms(device_id, message))

    if not service.config.auto_reply_enabled:
        logger.debug("auto_reply disabled — ignoring sms.inbound from %s", device_id)
        return
    bot_backend = vars(service).get("bot")
    if service.gbrain is None and bot_backend is None:
        logger.warning(
            "sms.inbound received but the GBrain AI backend is not configured"
        )
        return

    sms_id = message.get("sms_id")
    try:
        sms_id_int = int(sms_id) if sms_id is not None else None
    except (TypeError, ValueError):
        sms_id_int = None

    if sms_id_int is not None and not service.note_sms_seen(device_id, sms_id_int):
        logger.debug("sms id=%s from %s already handled; skipping", sms_id_int, contact)
        return

    bot_mem = getattr(getattr(service, "bot", None), "memory", None)
    if sms_id_int is not None and bot_mem is not None:
        try:
            if bot_mem.is_sms_replied(sms_id_int) is True:
                logger.info("sms id=%s from %s already replied; skipping", sms_id_int, contact)
                return
        except Exception:
            pass

    # Do not auto-reply to short codes / carrier senders; that can be costly.
    digits = "".join(ch for ch in contact if ch.isdigit())
    if len(digits) < 9:
        logger.info("Skipping auto-reply to non-phone sender %s", contact)
        return

    loop = asyncio.get_event_loop()
    try:
        if bot_backend is not None:
            reply = await loop.run_in_executor(
                None, service.generate_reply, contact, body
            )
        else:
            reply = await loop.run_in_executor(
                None, service.gbrain.generate_reply, contact, body
            )
    except Exception as exc:
        logger.error("generate_reply failed for %s: %s", contact, exc)
        return

    if not reply:
        logger.warning("gbrain returned empty reply for inbound from %s", contact)
        return

    result = await service.send_signal(
        device_id,
        {"type": SMS_SEND, "to_number": contact, "body": reply},
    )
    if result.get("sent"):
        if sms_id_int is not None and bot_mem is not None:
            try:
                bot_mem.mark_sms_replied(sms_id_int, contact, body, reply)
                bot_mem.mark_sms_processed(sms_id_int)
            except Exception as exc:
                logger.warning("Could not record sms_replied for id=%s: %s", sms_id_int, exc)
        service._record(
            "sms.auto_replied",
            {"device_id": device_id, "contact": contact, "sms_id": sms_id_int, "body_preview": body[:40]},
        )
        logger.info("Auto-replied to %s via device %s (sms_id=%s)", contact, device_id, sms_id_int)
    else:
        logger.warning("Auto-reply dispatch failed for %s: %s", contact, result)


async def handle_sms_result(
    service: AudioLinkService, device_id: str, message: dict
) -> None:
    """Record the phone's real SmsManager result; fail a task that did not send."""
    task_id = str(message.get("task_id") or "")
    ok = bool(message.get("ok", False))
    await _maybe_await(service.record_hosted_result(device_id, message))
    service._record(
        "sms.result",
        {
            "device_id": device_id,
            "task_id": task_id,
            "to": str(message.get("to_number", "")),
            "ok": ok,
            "error": str(message.get("error", ""))[:120],
        },
    )
    if task_id and service.task_store is not None:
        task = await service.task_store.get(task_id)
        if not ok:
            await service.task_store.mark_failed(task_id, str(message.get("error", "")))
        elif task is not None and task.status == "pending":
            await service.task_store.mark_done(task_id)


async def handle_media_offer(
    service: AudioLinkService, device_id: str, device_name: str, message: dict
) -> None:
    """Answer a phone's WebRTC offer and send the SDP back over signaling."""
    sdp = message.get("sdp", "")
    try:
        answer = await service.media.handle_offer(device_id, device_name, sdp)
    except AudioLinkError as exc:
        await service.hub.send(device_id, exc.to_dict()["error"] | {"type": ERROR})
        return
    await service.hub.send(
        device_id,
        {
            "type": "sdp.answer",
            "sdp": answer["sdp"],
            "sdp_type": "answer",
            "device_id": device_id,
            "state": answer["state"],
        },
    )
    logger.info(
        "media answer sent",
        extra={"device_id": device_id, "state": answer["state"]},
    )


async def handle_control(
    service: AudioLinkService, device_id: str, message: dict
) -> None:
    action = message.get("action")
    if action == "microphone":
        result = await service.set_microphone_enabled(bool(message.get("enabled", True)))
    elif action == "speaker":
        result = await service.set_speaker_enabled(bool(message.get("enabled", True)))
    elif action == "volume":
        result = await service.set_volume(int(message.get("level", 100)))
    else:
        await service.hub.send(
            device_id,
            {"type": ERROR, "code": "UNKNOWN_CONTROL", "message": f"unknown control action: {action}"},
        )
        return
    await service.hub.send(
        device_id, {"type": "control.state", "action": action, **result}
    )
