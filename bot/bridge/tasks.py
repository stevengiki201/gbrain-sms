"""Persistent task store for scheduled SMS and call tasks.

Tasks are written to ``~/.local/share/gbrain/audiolink/tasks.json``
atomically (write-to-temp then os.replace) with mode 0600, so they survive
AudioLink restarts. The file uses the same layout pattern as DeviceRegistry.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

logger = logging.getLogger("gbrain.bridge.tasks")

TASKS_FILE = "tasks.json"


def parse_scheduled_at(value: str) -> datetime:
    """Parse an ISO-8601 timestamp into a naive local datetime.

    Models (and people) sometimes append ``Z`` or an offset. Comparing an
    aware timestamp with ``datetime.now()`` raises ``TypeError``, so every
    timestamp is converted to the machine's local wall clock first.
    """
    raw = (value or "").strip()
    if not raw:
        raise ValueError("scheduled_at is required")
    if raw.endswith(("Z", "z")):
        raw = raw[:-1] + "+00:00"
    parsed = datetime.fromisoformat(raw)
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone().replace(tzinfo=None)
    return parsed


def normalize_scheduled_at(value: str) -> str:
    """Return a canonical local ISO-8601 timestamp (seconds precision)."""
    return parse_scheduled_at(value).replace(microsecond=0).isoformat()


@dataclass
class ScheduledTask:
    task_id: str
    task_type: str           # "sms" | "call"
    to_number: str
    message: str
    scheduled_at: str        # ISO-8601, e.g. "2026-11-18T09:00:00"
    label: str = ""
    source_device_id: str = ""
    status: str = "pending"  # pending | done | cancelled | failed
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    completed_at: str = ""
    cancelled_at: str = ""
    cancel_reason: str = ""
    attempts: int = 0
    last_error: str = ""


class TaskStore:
    """Thread-safe (asyncio) persistent store for ScheduledTask objects."""

    def __init__(self, data_dir: Path) -> None:
        self.path = Path(data_dir).expanduser() / TASKS_FILE
        self._lock = asyncio.Lock()
        self._tasks: dict[str, ScheduledTask] = {}
        self._loaded = False

    # ── persistence ──────────────────────────────────────────────────

    def _load(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        if not self.path.is_file():
            self._tasks = {}
            return
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8") or "[]")
            self._tasks = {}
            items = raw if isinstance(raw, list) else []
            allowed = set(ScheduledTask.__dataclass_fields__)
            for item in items:
                if not isinstance(item, dict):
                    continue
                try:
                    data = {k: v for k, v in item.items() if k in allowed}
                    t = ScheduledTask(**data)
                    self._tasks[t.task_id] = t
                except (TypeError, KeyError, ValueError) as exc:
                    logger.debug("Skipping malformed task record: %s", exc)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Could not load tasks.json: %s", exc)
            self._tasks = {}

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = [asdict(t) for t in self._tasks.values()]
        tmp = self.path.with_suffix(".json.tmp")
        fd = os.open(str(tmp), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp, self.path)
        except Exception:
            if tmp.exists():
                tmp.unlink(missing_ok=True)
            raise
        try:
            os.chmod(self.path, 0o600)
        except OSError:
            pass

    # ── public ───────────────────────────────────────────────────────

    async def add(self, task: ScheduledTask) -> None:
        task.scheduled_at = normalize_scheduled_at(task.scheduled_at)
        async with self._lock:
            self._load()
            self._tasks[task.task_id] = task
            self._save()
        logger.info("Task added: %s → %s at %s", task.task_id, task.to_number, task.scheduled_at)

    async def get(self, task_id: str) -> ScheduledTask | None:
        async with self._lock:
            self._load()
            return self._tasks.get(task_id)

    async def list_pending(self) -> list[ScheduledTask]:
        async with self._lock:
            self._load()
            return [t for t in self._tasks.values() if t.status == "pending"]

    async def list_all(self) -> list[ScheduledTask]:
        async with self._lock:
            self._load()
            tasks = list(self._tasks.values())
        tasks.sort(key=lambda t: (t.scheduled_at, t.created_at, t.task_id))
        return tasks

    async def list_due(self) -> list[ScheduledTask]:
        """Return pending tasks whose scheduled_at is in the past or now."""
        now = datetime.now()
        async with self._lock:
            self._load()
            due: list[ScheduledTask] = []
            for task in self._tasks.values():
                if task.status != "pending":
                    continue
                try:
                    when = parse_scheduled_at(task.scheduled_at)
                except (ValueError, TypeError) as exc:
                    logger.warning(
                        "Task %s has an invalid scheduled_at %r: %s",
                        task.task_id,
                        task.scheduled_at,
                        exc,
                    )
                    continue
                if when <= now:
                    due.append(task)
            due.sort(key=lambda t: t.scheduled_at)
            return due

    async def cancel(self, task_id: str, reason: str = "") -> ScheduledTask | None:
        async with self._lock:
            self._load()
            task = self._tasks.get(task_id)
            if task is None or task.status != "pending":
                return None
            task.status = "cancelled"
            task.cancelled_at = datetime.now().isoformat()
            task.cancel_reason = (reason or "").strip()[:200]
            self._save()
        logger.info("Task cancelled: %s%s", task_id, f" ({task.cancel_reason})" if task.cancel_reason else "")
        return task

    async def mark_done(self, task_id: str) -> None:
        async with self._lock:
            self._load()
            task = self._tasks.get(task_id)
            if task is not None and task.status == "pending":
                task.status = "done"
                task.completed_at = datetime.now().isoformat()
                self._save()

    async def mark_failed(self, task_id: str, error: str = "") -> None:
        async with self._lock:
            self._load()
            task = self._tasks.get(task_id)
            if task is not None and task.status == "pending":
                task.status = "failed"
                task.completed_at = datetime.now().isoformat()
                if error:
                    task.last_error = str(error)[:200]
                self._save()
        logger.warning("Task marked failed: %s", task_id)

    async def bump_attempt(
        self, task_id: str, error: str = "", max_attempts: int = 20
    ) -> bool:
        """Record a delivery attempt. Returns True while the task stays pending."""
        async with self._lock:
            self._load()
            task = self._tasks.get(task_id)
            if task is None or task.status != "pending":
                return False
            task.attempts += 1
            task.last_error = (error or "")[:200]
            if task.attempts >= max_attempts:
                task.status = "failed"
                task.completed_at = datetime.now().isoformat()
            self._save()
            return task.status == "pending"
