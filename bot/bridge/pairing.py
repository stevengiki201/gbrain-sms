"""Short-code pairing between the desktop and an Android phone.

Flow
----
1. The phone requests pairing: ``POST /v1/pairing/requests``.
2. The desktop displays a 6-digit code (dashboard, logs, local API).
3. The user types that code into the phone and confirms.
4. The desktop issues a bearer token exactly once; only its hash is stored.

Codes are single-use, short lived, attempt-limited and rate-limited per device.
"""

from __future__ import annotations

import asyncio
import secrets
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Callable

from .errors import Expired, Forbidden, NotFound, RateLimited, Unauthorized
from .registry import DeviceRegistry, PairedDevice

CODE_DIGITS = 6


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class PairingRequest:
    request_id: str
    device_id: str
    device_name: str
    platform: str
    code: str
    created_at: datetime
    expires_at: datetime
    app_version: str = ""
    attempts_remaining: int = 5
    status: str = "pending"  # pending | approved | expired | cancelled

    def public(self, include_code: bool = False) -> dict:
        data = {
            "request_id": self.request_id,
            "device_id": self.device_id,
            "device_name": self.device_name,
            "platform": self.platform,
            "app_version": self.app_version,
            "status": self.status,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat(),
            "attempts_remaining": self.attempts_remaining,
        }
        if include_code:
            data["code"] = self.code
        return data


@dataclass
class PairingResult:
    device: PairedDevice
    token: str


class PairingManager:
    def __init__(
        self,
        registry: DeviceRegistry,
        *,
        ttl_sec: int = 120,
        max_attempts: int = 5,
        rate_limit: int = 10,
        rate_window_sec: int = 60,
        clock: Callable[[], datetime] = utcnow,
    ) -> None:
        self.registry = registry
        self.ttl_sec = ttl_sec
        self.max_attempts = max_attempts
        self.rate_limit = rate_limit
        self.rate_window_sec = rate_window_sec
        self.clock = clock
        self._requests: dict[str, PairingRequest] = {}
        self._rate: dict[str, list[datetime]] = {}
        self._lock = asyncio.Lock()
        self._listeners: list[Callable[[PairingRequest], None]] = []
        self._last_issued: PairingRequest | None = None

    def add_listener(self, listener: Callable[[PairingRequest], None]) -> None:
        """Subscribe to new pairing requests (used to surface the code)."""
        self._listeners.append(listener)

    async def request_pairing(
        self,
        *,
        device_id: str,
        device_name: str = "",
        platform: str = "android",
        app_version: str = "",
    ) -> PairingRequest:
        if not device_id:
            raise Unauthorized("device_id is required", code="INVALID_DEVICE")
        now = self.clock()
        async with self._lock:
            self._prune(now)
            recent = [
                ts
                for ts in self._rate.get(device_id, [])
                if (now - ts).total_seconds() < self.rate_window_sec
            ]
            if len(recent) >= self.rate_limit:
                raise RateLimited(
                    "too many pairing requests; try again later",
                    details={"retry_after_sec": self.rate_window_sec},
                )
            recent.append(now)
            self._rate[device_id] = recent

            request = PairingRequest(
                request_id=secrets.token_hex(16),
                device_id=device_id,
                device_name=device_name or device_id,
                platform=platform or "unknown",
                app_version=app_version,
                code=f"{secrets.randbelow(10 ** CODE_DIGITS):0{CODE_DIGITS}d}",
                created_at=now,
                expires_at=now + timedelta(seconds=self.ttl_sec),
                attempts_remaining=self.max_attempts,
            )
            self._requests[request.request_id] = request
            self._last_issued = request

        for listener in list(self._listeners):
            try:
                listener(request)
            except Exception:  # pragma: no cover - listener must not break pairing
                pass
        return request

    async def pending(self, include_code: bool = False) -> list[dict]:
        now = self.clock()
        async with self._lock:
            self._prune(now)
            return [
                req.public(include_code=include_code)
                for req in self._requests.values()
                if req.status == "pending"
            ]

    async def confirm(self, request_id: str, code: str) -> PairingResult:
        now = self.clock()
        async with self._lock:
            request = self._requests.get(request_id)
            if request is None:
                raise NotFound(
                    "pairing request not found",
                    code="PAIRING_NOT_FOUND",
                    details={"request_id": request_id},
                )
            if request.status == "cancelled":
                raise Forbidden("pairing request was cancelled", code="PAIRING_CANCELLED")
            if request.status == "approved":
                raise Forbidden(
                    "pairing request was already used", code="PAIRING_ALREADY_USED"
                )
            if request.status == "expired" or now >= request.expires_at:
                request.status = "expired"
                raise Expired("pairing request expired", code="PAIRING_EXPIRED")
            if request.attempts_remaining <= 0:
                request.status = "cancelled"
                raise Forbidden(
                    "too many incorrect pairing codes", code="TOO_MANY_ATTEMPTS"
                )

            supplied = (code or "").strip()
            if not secrets.compare_digest(request.code, supplied):
                request.attempts_remaining -= 1
                if request.attempts_remaining <= 0:
                    request.status = "cancelled"
                    raise Forbidden(
                        "too many incorrect pairing codes", code="TOO_MANY_ATTEMPTS"
                    )
                raise Unauthorized(
                    "incorrect pairing code",
                    code="INVALID_PAIRING_CODE",
                    details={"attempts_remaining": request.attempts_remaining},
                )

            request.status = "approved"
            device, token = await self.registry.pair_device(
                device_id=request.device_id,
                name=request.device_name,
                platform=request.platform,
                app_version=request.app_version,
            )
            return PairingResult(device=device, token=token)

    async def cancel(self, request_id: str) -> None:
        async with self._lock:
            request = self._requests.get(request_id)
            if request is None:
                raise NotFound(
                    "pairing request not found",
                    code="PAIRING_NOT_FOUND",
                    details={"request_id": request_id},
                )
            request.status = "cancelled"

    async def last_issued(self) -> PairingRequest | None:
        return self._last_issued

    def _prune(self, now: datetime) -> None:
        for request_id, request in list(self._requests.items()):
            if request.status == "pending" and now >= request.expires_at:
                request.status = "expired"
            age = (now - request.created_at).total_seconds()
            if request.status != "pending" and age > 900:
                del self._requests[request_id]
