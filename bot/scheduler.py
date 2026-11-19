"""Async scheduler for delayed SMS replies."""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, time as dt_time
from typing import Awaitable, Callable

logger = logging.getLogger(__name__)

SendCallback = Callable[["ScheduledReply"], Awaitable[bool]]
CallCallback = Callable[["ScheduledCall"], Awaitable[None]]


@dataclass
class ScheduledReply:
    contact: str
    number: str
    body: str
    send_at: datetime
    source: str = "sms"
    task_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    whatsapp_contact_uri: str = ""
    whatsapp_user_id: int | None = None
    whatsapp_chat_uri: str = ""
    task_type: str = "sms"


@dataclass
class ScheduledCall:
    contact: str
    number: str
    message: str
    send_at: datetime
    source: str = "app"
    task_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    speak_delay_sec: float = 6.0
    repeat: int = 1
    task_type: str = "call"


@dataclass
class _QueueItem:
    send_at_mono: float
    order: int
    task: ScheduledReply


class ResponseScheduler:
    def __init__(
        self,
        send_callback: SendCallback,
        dnd_start: str = "",
        dnd_end: str = "",
        call_callback: CallCallback | None = None,
    ) -> None:
        self._send_callback = send_callback
        self._call_callback = call_callback
        self._dnd_start = _parse_time(dnd_start)
        self._dnd_end = _parse_time(dnd_end)
        self._queue: asyncio.PriorityQueue[_QueueItem] = asyncio.PriorityQueue()
        self._tasks: dict[str, ScheduledReply] = {}
        self._order = 0
        self._worker_task: asyncio.Task | None = None
        self._running = False

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._worker_task = asyncio.create_task(self._worker())
        logger.info("Scheduler started")

    async def stop(self, drain: bool = False) -> None:
        self._running = False
        if self._worker_task:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
            self._worker_task = None
        if not drain:
            while not self._queue.empty():
                try:
                    self._queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
            self._tasks.clear()
        logger.info("Scheduler stopped")

    async def schedule(self, task: ScheduledReply) -> str:
        effective = self._apply_dnd(task.send_at)
        task.send_at = effective
        self._tasks[task.task_id] = task
        item = _QueueItem(
            send_at_mono=effective.timestamp(),
            order=self._order,
            task=task,
        )
        self._order += 1
        await self._queue.put(item)
        logger.info(
            "Scheduled reply %s to %s at %s",
            task.task_id,
            task.number,
            effective.isoformat(),
        )
        return task.task_id

    def list_pending(self) -> list[ScheduledReply]:
        return sorted(self._tasks.values(), key=lambda t: t.send_at)

    def cancel(self, task_id: str) -> bool:
        if task_id in self._tasks:
            del self._tasks[task_id]
            logger.info("Cancelled scheduled reply %s", task_id)
            return True
        return False

    def is_dnd_active(self, moment: datetime | None = None) -> bool:
        if not self._dnd_start or not self._dnd_end:
            return False
        now = moment or datetime.now()
        current = now.time()
        start, end = self._dnd_start, self._dnd_end
        if start <= end:
            return start <= current < end
        return current >= start or current < end

    def _apply_dnd(self, send_at: datetime) -> datetime:
        if not self.is_dnd_active(send_at):
            return send_at
        if not self._dnd_end:
            return send_at
        from datetime import timedelta

        adjusted = send_at.replace(
            hour=self._dnd_end.hour,
            minute=self._dnd_end.minute,
            second=0,
            microsecond=0,
        )
        if adjusted <= send_at:
            adjusted += timedelta(days=1)
        logger.info("DND active; deferred send to %s", adjusted.isoformat())
        return adjusted

    async def _worker(self) -> None:
        while self._running:
            try:
                item = await self._queue.get()
            except asyncio.CancelledError:
                break

            task = item.task
            if task.task_id not in self._tasks:
                continue

            delay = item.send_at_mono - time.time()
            if delay > 0:
                try:
                    await asyncio.sleep(delay)
                except asyncio.CancelledError:
                    break

            if task.task_id not in self._tasks:
                continue

            try:
                if getattr(task, "task_type", "sms") == "call":
                    if self._call_callback is not None:
                        await self._call_callback(task)
                        logger.info("Scheduled call %s dispatched", task.task_id)
                    else:
                        logger.warning(
                            "Scheduled call %s has no call_callback; dropping", task.task_id
                        )
                else:
                    success = await self._send_callback(task)
                    if success:
                        logger.info("Scheduled reply %s sent", task.task_id)
                    else:
                        logger.error("Scheduled reply %s failed to send", task.task_id)
            except Exception:
                logger.exception("Error executing scheduled task %s", task.task_id)
            finally:
                self._tasks.pop(task.task_id, None)


def _parse_time(value: str) -> dt_time | None:
    value = (value or "").strip()
    if not value:
        return None
    for fmt in ("%H:%M", "%H:%M:%S"):
        try:
            return datetime.strptime(value, fmt).time()
        except ValueError:
            continue
    return None
