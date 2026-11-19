"""Persistent registry of paired Android devices.

Only a SHA-256 hash of each device token is stored, so a copy of
``devices.json`` cannot be replayed to impersonate a phone. The file is
written atomically with mode ``0600``.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import secrets
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path

from .errors import NotFound, Unauthorized, ValidationFailed

DEVICES_FILE = "devices.json"
TOKEN_PREFIX = "al_"


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def new_token() -> str:
    return TOKEN_PREFIX + secrets.token_urlsafe(32)


@dataclass
class PairedDevice:
    device_id: str
    name: str
    platform: str
    token_hash: str
    paired_at: str
    last_seen_at: str | None = None
    app_version: str = ""
    token_expires_at: str | None = None
    revoked: bool = False
    metadata: dict = field(default_factory=dict)

    def public(self) -> dict:
        """Serialisable view that never leaks the token hash."""
        data = asdict(self)
        data.pop("token_hash", None)
        data.pop("metadata", None)
        return data

    def is_expired(self, now: datetime | None = None) -> bool:
        if not self.token_expires_at:
            return False
        now = now or utcnow()
        try:
            expires = datetime.fromisoformat(self.token_expires_at)
        except ValueError:
            return True
        return expires <= now


class DeviceRegistry:
    """Thread-safe (asyncio) store of paired devices."""

    def __init__(self, data_dir: Path, token_ttl_sec: int = 0) -> None:
        self.path = Path(data_dir).expanduser() / DEVICES_FILE
        self.token_ttl_sec = token_ttl_sec
        self._lock = asyncio.Lock()
        self._devices: dict[str, PairedDevice] = {}
        self._loaded = False

    # ------------------------------------------------------------------ store
    def _load(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        if not self.path.is_file():
            self._devices = {}
            return
        raw = json.loads(self.path.read_text(encoding="utf-8") or "{}")
        self._devices = {}
        for device_id, record in raw.get("devices", {}).items():
            # Tolerate records written by older versions that may be missing
            # fields added later (platform, paired_at, etc.).
            record.setdefault("device_id", device_id)
            record.setdefault("platform", "android")
            record.setdefault("paired_at", "1970-01-01T00:00:00+00:00")
            record.setdefault("token_hash", "")
            record.setdefault("name", device_id)
            # Drop any keys the dataclass doesn't know about
            known = {f.name for f in PairedDevice.__dataclass_fields__.values()}
            clean = {k: v for k, v in record.items() if k in known}
            try:
                self._devices[device_id] = PairedDevice(**clean)
            except TypeError as exc:
                import logging as _log
                _log.getLogger("audiolink.registry").warning(
                    "Skipping malformed device record %s: %s", device_id, exc
                )

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "devices": {d_id: asdict(dev) for d_id, dev in self._devices.items()},
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

    # ------------------------------------------------------------------- read
    async def list(self) -> list[PairedDevice]:
        async with self._lock:
            self._load()
            return sorted(self._devices.values(), key=lambda d: d.paired_at)

    async def get(self, device_id: str) -> PairedDevice:
        async with self._lock:
            self._load()
            device = self._devices.get(device_id)
            if device is None:
                raise NotFound(
                    "device is not paired",
                    details={"device_id": device_id},
                    code="DEVICE_NOT_PAIRED",
                )
            return device

    # ------------------------------------------------------------------ write
    async def pair_device(
        self,
        *,
        device_id: str,
        name: str,
        platform: str,
        app_version: str = "",
        metadata: dict | None = None,
    ) -> tuple[PairedDevice, str]:
        """Create or refresh a device pairing and return its plaintext token."""
        if not device_id or len(device_id) > 128:
            raise ValidationFailed("device_id is required and must be <= 128 chars")
        token = new_token()
        now = utcnow()
        expires = None
        if self.token_ttl_sec > 0:
            expires = (
                datetime.fromtimestamp(now.timestamp() + self.token_ttl_sec, timezone.utc)
                .isoformat()
            )
        async with self._lock:
            self._load()
            existing = self._devices.get(device_id)
            device = PairedDevice(
                device_id=device_id,
                name=name or device_id,
                platform=platform or "unknown",
                token_hash=hash_token(token),
                paired_at=existing.paired_at if existing else now.isoformat(),
                last_seen_at=now.isoformat(),
                app_version=app_version,
                token_expires_at=expires,
                revoked=False,
                metadata=metadata or (existing.metadata if existing else {}),
            )
            self._devices[device_id] = device
            self._save()
        return device, token

    async def verify_token(self, device_id: str, token: str) -> PairedDevice:
        async with self._lock:
            self._load()
            device = self._devices.get(device_id)
            if device is None:
                raise Unauthorized(
                    "device is not paired",
                    code="DEVICE_NOT_PAIRED",
                    details={"device_id": device_id},
                )
            if device.revoked:
                raise Unauthorized("device pairing was revoked", code="DEVICE_REVOKED")
            if device.is_expired():
                raise Unauthorized("device token expired", code="TOKEN_EXPIRED")
            if not token or not secrets.compare_digest(device.token_hash, hash_token(token)):
                raise Unauthorized("invalid device token", code="INVALID_TOKEN")
            device = replace(device, last_seen_at=utcnow().isoformat())
            self._devices[device_id] = device
            self._save()
            return device

    async def touch(self, device_id: str) -> None:
        async with self._lock:
            self._load()
            device = self._devices.get(device_id)
            if device is not None:
                self._devices[device_id] = replace(
                    device, last_seen_at=utcnow().isoformat()
                )
                self._save()

    async def remove(self, device_id: str) -> None:
        async with self._lock:
            self._load()
            if device_id not in self._devices:
                raise NotFound(
                    "device is not paired",
                    details={"device_id": device_id},
                    code="DEVICE_NOT_PAIRED",
                )
            del self._devices[device_id]
            self._save()
