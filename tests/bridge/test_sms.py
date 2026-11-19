"""Integration tests for the AudioLink SMS layer.

Tests cover:
  - sms.inbound signaling validation
  - TaskStore persistence and scheduling
  - auto-reply flow (mocked gbrain)
  - scheduler fire
  - operations table SMS ops
"""

from __future__ import annotations

import asyncio
import tempfile
from dataclasses import asdict
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from bot.bridge.signaling import (
    SMS_INBOUND,
    SMS_RESULT,
    SMS_SEND,
    ValidationFailed,
    validate_message,
)
from bot.bridge.tasks import (
    ScheduledTask,
    TaskStore,
    normalize_scheduled_at,
    parse_scheduled_at,
)


# ── Task 2: signaling validation ─────────────────────────────────────────────

def test_validate_sms_inbound_ok():
    msg_type = validate_message({
        "type": "sms.inbound",
        "sms_id": 1,
        "from_number": "+255712345678",
        "body": "habari",
        "timestamp_ms": 0,
    })
    assert msg_type == SMS_INBOUND


def test_validate_sms_inbound_missing_from_number():
    with pytest.raises(ValidationFailed) as exc_info:
        validate_message({
            "type": "sms.inbound",
            "sms_id": 1,
            "body": "habari",
        })
    assert "from_number" in str(exc_info.value)


def test_validate_sms_inbound_missing_body():
    with pytest.raises(ValidationFailed):
        validate_message({
            "type": "sms.inbound",
            "sms_id": 1,
            "from_number": "+255712345678",
        })


def test_existing_messages_still_validate():
    """Existing message types must not be broken by the new additions."""
    assert validate_message({"type": "ping", "ts": 123}) == "ping"
    assert validate_message({"type": "hello"}) == "hello"
    assert validate_message({"type": "control", "action": "microphone", "enabled": True}) == "control"


# ── Task 5: TaskStore ─────────────────────────────────────────────────────────

@pytest.fixture
def task_store(tmp_path):
    return TaskStore(tmp_path)


def make_task(*, offset_seconds: int = 3600, status: str = "pending") -> ScheduledTask:
    at = (datetime.now() + timedelta(seconds=offset_seconds)).isoformat()
    t = ScheduledTask(
        task_id="test1234",
        task_type="sms",
        to_number="+255712345678",
        message="Hello from test",
        scheduled_at=at,
    )
    t.status = status
    return t


@pytest.mark.asyncio
async def test_task_store_persistence(tmp_path):
    """Task added to store survives a reload from disk."""
    store1 = TaskStore(tmp_path)
    task = make_task()
    await store1.add(task)

    store2 = TaskStore(tmp_path)
    pending = await store2.list_pending()
    assert len(pending) == 1
    assert pending[0].task_id == task.task_id
    assert pending[0].to_number == task.to_number


@pytest.mark.asyncio
async def test_task_store_list_due():
    with tempfile.TemporaryDirectory() as tmp:
        store = TaskStore(Path(tmp))
        past = ScheduledTask(
            task_id="past0001",
            task_type="sms",
            to_number="+255712345678",
            message="past",
            scheduled_at=(datetime.now() - timedelta(seconds=60)).isoformat(),
        )
        future = ScheduledTask(
            task_id="futu0001",
            task_type="sms",
            to_number="+255712345678",
            message="future",
            scheduled_at=(datetime.now() + timedelta(hours=2)).isoformat(),
        )
        await store.add(past)
        await store.add(future)
        due = await store.list_due()
        assert len(due) == 1
        assert due[0].task_id == "past0001"


@pytest.mark.asyncio
async def test_task_store_cancel(task_store):
    task = make_task()
    await task_store.add(task)
    cancelled = await task_store.cancel(task.task_id, reason="test")
    assert cancelled is not None
    assert cancelled.status == "cancelled"
    assert cancelled.cancel_reason == "test"
    pending = await task_store.list_pending()
    assert all(t.task_id != task.task_id for t in pending)


@pytest.mark.asyncio
async def test_task_store_cancel_unknown(task_store):
    cancelled = await task_store.cancel("doesnotexist")
    assert cancelled is None


@pytest.mark.asyncio
async def test_task_store_mark_done(task_store):
    task = make_task()
    await task_store.add(task)
    await task_store.mark_done(task.task_id)
    pending = await task_store.list_pending()
    assert len(pending) == 0
    stored = await task_store.get(task.task_id)
    assert stored is not None and stored.status == "done" and stored.completed_at


# ── Task 6/7: auto-reply flow ─────────────────────────────────────────────────

class FakeHub:
    def __init__(self):
        self.sent: list[tuple[str, dict]] = []

    async def send(self, device_id: str, payload: dict) -> bool:
        self.sent.append((device_id, payload))
        return True

    async def is_online(self, device_id: str) -> bool:
        return True

    async def online_devices(self):
        return [{"device_id": device_id} for device_id, _ in self.sent[-1:]] or \
               [{"device_id": "dev1"}]


class FakeGbrain:
    def __init__(self, reply: str = "Hey!"):
        self._reply = reply
        self.calls: list[tuple[str, str]] = []

    def generate_reply(self, contact: str, body: str, source: str = "sms") -> str:
        self.calls.append((contact, body))
        return self._reply


@pytest.mark.asyncio
async def test_inbound_auto_reply():
    """sms.inbound → gbrain.generate_reply called → sms.send sent to device."""
    from bot.bridge.api import handle_sms_inbound

    hub = FakeHub()
    gbrain = FakeGbrain(reply="Habari sana!")

    # Minimal service mock
    service = MagicMock()
    service.config.auto_reply_enabled = True
    service.gbrain = gbrain
    service._record = MagicMock()

    async def fake_send_signal(device_id, payload):
        hub.sent.append((device_id, payload))
        return {"sent": True, "device_id": device_id}

    service.send_signal = fake_send_signal

    await handle_sms_inbound(
        service,
        "dev1",
        {"type": "sms.inbound", "sms_id": 1, "from_number": "+255712345678", "body": "habari"},
    )

    assert len(hub.sent) == 1
    device_id, payload = hub.sent[0]
    assert device_id == "dev1"
    assert payload["type"] == SMS_SEND
    assert payload["to_number"] == "+255712345678"
    assert payload["body"] == "Habari sana!"
    service._record.assert_called_once()


@pytest.mark.asyncio
async def test_no_reply_when_auto_off():
    """auto_reply_enabled=False → no sms.send signal."""
    from bot.bridge.api import handle_sms_inbound

    hub = FakeHub()
    gbrain = FakeGbrain(reply="Hey!")

    service = MagicMock()
    service.config.auto_reply_enabled = False
    service.gbrain = gbrain
    service._record = MagicMock()
    service.send_signal = AsyncMock()

    await handle_sms_inbound(
        service,
        "dev1",
        {"type": "sms.inbound", "sms_id": 1, "from_number": "+255...", "body": "hi"},
    )

    service.send_signal.assert_not_called()
    assert len(gbrain.calls) == 0


@pytest.mark.asyncio
async def test_no_reply_when_gbrain_none():
    """gbrain not configured → no sms.send signal."""
    from bot.bridge.api import handle_sms_inbound

    service = MagicMock()
    service.config.auto_reply_enabled = True
    service.gbrain = None
    service.send_signal = AsyncMock()

    await handle_sms_inbound(
        service,
        "dev1",
        {"type": "sms.inbound", "sms_id": 1, "from_number": "+255...", "body": "hi"},
    )

    service.send_signal.assert_not_called()


# ── Task 6: scheduler fires due task ─────────────────────────────────────────

@pytest.mark.asyncio
async def test_scheduled_task_fires(tmp_path):
    """A past-due task is signalled via sms.send and marked done."""
    from bot.bridge.service import AudioLinkService

    store = TaskStore(tmp_path)
    past = ScheduledTask(
        task_id="fire0001",
        task_type="sms",
        to_number="+255712345678",
        message="Scheduled hello",
        scheduled_at=(datetime.now() - timedelta(seconds=30)).isoformat(),
        source_device_id="dev1",
    )
    await store.add(past)

    sent_signals: list[dict] = []

    async def fake_send_signal(device_id, payload):
        sent_signals.append({"device_id": device_id, "payload": payload})
        return {"sent": True, "device_id": device_id}

    service = MagicMock()
    service.send_signal = fake_send_signal
    service._record = MagicMock()
    service.task_store = store
    service.dispatch_sms_task = AudioLinkService.dispatch_sms_task.__get__(service, AudioLinkService)
    service.wait_for_task_terminal = AudioLinkService.wait_for_task_terminal.__get__(service, AudioLinkService)

    # Call _fire_task directly (bypass the sleep in _scheduler_loop)
    await AudioLinkService._fire_task(service, past)

    assert len(sent_signals) == 1
    sig = sent_signals[0]
    assert sig["device_id"] == "dev1"
    assert sig["payload"]["type"] == SMS_SEND
    assert sig["payload"]["to_number"] == "+255712345678"
    assert sig["payload"]["body"] == "Scheduled hello"

    # Should now be marked done
    done_tasks = await store.list_pending()
    assert all(t.task_id != "fire0001" for t in done_tasks)


def test_validate_sms_result_ok():
    assert validate_message({
        "type": "sms.result",
        "ok": True,
        "to_number": "+255772691677",
        "task_id": "abc123",
    }) == SMS_RESULT


def test_scheduled_at_normalizes_utc_and_offsets():
    # A Zulu timestamp must become a naive local wall-clock value; otherwise
    # list_due compares aware and naive datetimes and raises TypeError.
    expected = datetime.fromisoformat(
        "2026-11-18T09:00:00+00:00"
    ).astimezone().replace(tzinfo=None)
    assert parse_scheduled_at("2026-11-18T09:00:00Z") == expected
    assert parse_scheduled_at("2026-11-18T12:00:00+03:00") == expected
    normalized = normalize_scheduled_at("2026-11-18T09:00:00Z")
    assert normalized == expected.isoformat()


def test_schedule_intent_detection():
    from bot.bridge.api import _looks_like_schedule_request

    assert _looks_like_schedule_request(
        "at 9:00 pm send hello message to +255772691677"
    )
    assert _looks_like_schedule_request(
        "schedule a hello SMS to +255 772 691 677 tomorrow"
    )
    assert not _looks_like_schedule_request("habari, hujambo?")
    assert not _looks_like_schedule_request("send him a message at 9")


@pytest.mark.asyncio
async def test_admin_sms_send_endpoint(client, service):
    service.hub.online_devices = AsyncMock(return_value=[{"device_id": "dev1"}])
    service.hub.send = AsyncMock(return_value=True)

    response = await client.post(
        "/v1/sms/send",
        json={"to_number": "+255712345678", "body": "Habari from Wi-Fi"},
    )

    assert response.status_code == 200
    assert response.json()["sent"] is True
    device_id, payload = service.hub.send.call_args.args
    assert device_id == "dev1"
    assert payload["type"] == SMS_SEND
    assert payload["to_number"] == "+255712345678"
    assert payload["body"] == "Habari from Wi-Fi"
