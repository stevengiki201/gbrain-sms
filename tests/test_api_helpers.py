from __future__ import annotations

from datetime import datetime, timedelta

from bot.api import (
    _apply_explicit_clock,
    _apply_explicit_type,
    _extract_task,
    _parse_local_datetime,
)


def test_clock_anchors_shifted_model_output():
    now = datetime.now()
    shifted = now.replace(hour=15, minute=45, second=0, microsecond=0)
    result = _apply_explicit_clock("at 12:45 send hello", shifted.isoformat())
    when = datetime.fromisoformat(result)
    assert (when.hour, when.minute) == (12, 45)
    assert when.date() == now.date()


def test_clock_handles_pm_and_tomorrow():
    tomorrow = (datetime.now() + timedelta(days=1)).date()
    result = _apply_explicit_clock(
        "send hello tomorrow at 9:00 pm", "2026-01-01T09:00:00"
    )
    when = datetime.fromisoformat(result)
    assert (when.hour, when.minute) == (21, 0)
    assert when.date() == tomorrow


def test_clock_leaves_unspecified_time_alone():
    assert _apply_explicit_clock("send hello", "2026-11-18T15:00:00") == (
        "2026-11-18T15:00:00"
    )


def test_type_prefers_clear_sms_words():
    draft = _apply_explicit_type("send a text to Sam at 9pm", {"type": "call"})
    assert draft["type"] == "sms"
    draft = _apply_explicit_type("call Sam at 9pm", {"type": "sms"})
    assert draft["type"] == "call"


def test_extract_task_accepts_tags_and_bare_json():
    tagged = '<TASK>{"scheduled_at":"2026-11-18T21:00:00","type":"sms"}</TASK>'
    task, clean = _extract_task(tagged)
    assert task and task["scheduled_at"] == "2026-11-18T21:00:00"
    assert clean == ""

    bare = 'Okay {"scheduled_at":"2026-11-18T21:00:00"}'
    task, _ = _extract_task(bare)
    assert task and task["scheduled_at"] == "2026-11-18T21:00:00"


def test_parse_local_datetime_zulu():
    assert _parse_local_datetime("2026-11-18T09:00:00Z").tzinfo is None
