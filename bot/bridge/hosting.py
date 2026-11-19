from __future__ import annotations

import asyncio
import json
import os
import ssl
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .errors import Unauthorized

HOSTED_FILE = "hosted.json"


@dataclass
class HostedEnrollment:
    website_device_id: str
    user_id: str
    email: str
    phone_device_id: str
    phone_name: str
    paired_at: str
    role: str = "confirmed"


class HostedStore:
    def __init__(self, data_dir: Path | str) -> None:
        self.path = Path(data_dir).expanduser() / HOSTED_FILE
        self._lock = asyncio.Lock()
        self._loaded = False
        self._enrollments: dict[str, HostedEnrollment] = {}
        self._events: list[dict[str, Any]] = []
        self._seq = 0

    def _load(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        if not self.path.is_file():
            return
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        for record in payload.get("enrollments", {}).values():
            try:
                enrollment = HostedEnrollment(**record)
            except TypeError:
                continue
            self._enrollments[enrollment.website_device_id] = enrollment
        events = payload.get("events", [])
        if isinstance(events, list):
            self._events = [event for event in events if isinstance(event, dict)]
        self._seq = int(payload.get("seq", 0) or 0)

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "seq": self._seq,
            "enrollments": {
                key: asdict(value) for key, value in self._enrollments.items()
            },
            "events": self._events[-2000:],
        }
        tmp = self.path.with_suffix(".json.tmp")
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2, sort_keys=True)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp, self.path)
        finally:
            if tmp.exists():
                tmp.unlink(missing_ok=True)
        os.chmod(self.path, 0o600)

    async def link(
        self,
        *,
        website_device_id: str,
        user_id: str,
        email: str,
        phone_device_id: str,
        phone_name: str,
        role: str,
        paired_at: str,
    ) -> dict[str, Any]:
        async with self._lock:
            self._load()
            enrollment = HostedEnrollment(
                website_device_id=website_device_id,
                user_id=user_id,
                email=email,
                phone_device_id=phone_device_id,
                phone_name=phone_name,
                paired_at=paired_at,
                role=role,
            )
            self._enrollments[website_device_id] = enrollment
            self._save()
            return asdict(enrollment)

    async def list_enrollments(self) -> list[dict[str, Any]]:
        async with self._lock:
            self._load()
            return [asdict(item) for item in self._enrollments.values()]

    async def record_event(
        self,
        phone_device_id: str,
        kind: str,
        payload: dict[str, Any],
    ) -> None:
        async with self._lock:
            self._load()
            matches = [
                item
                for item in self._enrollments.values()
                if item.phone_device_id == phone_device_id
            ]
            if not matches:
                return
            from datetime import datetime, timezone

            created_at = datetime.now(timezone.utc).isoformat()
            for enrollment in matches:
                self._seq += 1
                self._events.append(
                    {
                        "seq": self._seq,
                        "website_device_id": enrollment.website_device_id,
                        "kind": kind,
                        "created_at": created_at,
                        "payload": payload,
                    }
                )
            self._events = self._events[-2000:]
            self._save()

    async def events(self, website_device_id: str, after: int = 0) -> dict[str, Any]:
        async with self._lock:
            self._load()
            selected = [
                event
                for event in self._events
                if event.get("website_device_id") == website_device_id
                and int(event.get("seq", 0)) > max(0, after)
            ]
            return {
                "cursor": max([int(event.get("seq", 0)) for event in selected] or [after]),
                "events": selected,
            }


def verify_enrollment_with_website(
    token: str,
    device_id: str,
    device_name: str,
    platform: str,
    app_version: str,
) -> dict[str, Any]:
    website_url = os.environ.get("GBRAIN_WEBSITE_URL", "https://gbrain.tech").rstrip("/")
    service_token = os.environ.get("GBRAIN_SERVICE_TOKEN", "")
    if not service_token:
        raise Unauthorized("GBRAIN_SERVICE_TOKEN is not configured")
    payload = {
        "token": token,
        "deviceId": device_id,
        "deviceName": device_name,
        "platform": platform,
        "appVersion": app_version,
    }
    request = urllib.request.Request(
        f"{website_url}/api/audiolink/enrollments/verify",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "accept": "application/json",
            "content-type": "application/json",
            "x-service-token": service_token,
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(
            request, timeout=15.0, context=ssl.create_default_context()
        ) as response:
            raw = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:300]
        raise Unauthorized(f"enrollment rejected: {detail}") from exc
    except urllib.error.URLError as exc:
        raise Unauthorized(f"website unavailable: {exc.reason}") from exc
    try:
        parsed = json.loads(raw or "{}")
    except json.JSONDecodeError as exc:
        raise Unauthorized("website returned invalid JSON") from exc
    if not isinstance(parsed, dict) or parsed.get("error"):
        message = parsed.get("error", "enrollment rejected") if isinstance(parsed, dict) else "enrollment rejected"
        raise Unauthorized(str(message))
    return parsed
