"""Scheduled auto-answer of incoming phone calls (phase 1).

Polls the device call state and, while a configured time window is active,
answers incoming calls via :class:`bot.call_connector.CallConnector` and
optionally hangs up after a fixed duration. It only reacts on the idle ->
ringing edge so one ring is answered at most once.
"""

from __future__ import annotations

import logging
import re
import time
from datetime import datetime, timedelta

from bot.call_connector import (
    CALL_STATE_IDLE,
    CALL_STATE_OFFHOOK,
    CALL_STATE_RINGING,
    CALL_STATE_UNKNOWN,
    CallConnector,
    CallState,
)

logger = logging.getLogger(__name__)

# Once a one-off `call_answer_at` moment arrives, keep the window open this
# long so a call that comes in shortly after still gets answered.
AT_GRACE_SEC = 300

DEFAULT_GREETING = (
    "Hello, this is Steve. I can't talk right now — please leave a message "
    "after the tone, or send a text."
)


class CallAnswerer:
    def __init__(self, calls: CallConnector, config, voice=None) -> None:
        self.calls = calls
        self.config = config
        self.voice = voice
        self._start = _parse_hhmm(config.call_answer_start)
        self._end = _parse_hhmm(config.call_answer_end)
        self._at = _parse_at(config.call_answer_at)
        self._last_state = CALL_STATE_UNKNOWN
        self._answered_this_ring = False
        self._answered_at_mono: float | None = None
        self._at_fired = False

    # ── Scheduling ───────────────────────────────────────────────────
    def is_active(self, moment: datetime | None = None) -> bool:
        """True when calls should be auto-answered right now."""
        now = moment or datetime.now()

        if self._at is not None:
            if self._at_fired:
                return False
            return self._at <= now < self._at + timedelta(seconds=AT_GRACE_SEC)

        if self._start and self._end:
            current = now.time()
            if self._start <= self._end:
                return self._start <= current < self._end
            # Overnight window, e.g. 22:00 -> 06:00
            return current >= self._start or current < self._end

        # No window configured: active whenever the feature is enabled.
        return True

    # ── Poll loop ────────────────────────────────────────────────────
    def poll(self) -> None:
        state = self.calls.get_state()
        prev, self._last_state = self._last_state, state.state

        if state.state == CALL_STATE_RINGING and prev != CALL_STATE_RINGING:
            self._on_ringing(state)
        elif state.state == CALL_STATE_IDLE and prev in (
            CALL_STATE_RINGING,
            CALL_STATE_OFFHOOK,
        ):
            logger.info("Call ended (previous state: %s)", prev)
            self._reset_ring()
            self._stop_voice("call ended")

        if state.state == CALL_STATE_IDLE:
            self._reset_ring()

        self._maybe_auto_hangup(state)

    def _on_ringing(self, state: CallState) -> None:
        number = state.number
        logger.info("Incoming call ringing from %s", number or "unknown")

        if self._answered_this_ring:
            return
        if not self.is_active():
            logger.info("Call-answer window not active; leaving call unanswered")
            return

        allow = self.config.call_answer_contacts or []
        if allow:
            if not number:
                logger.info("Caller number unknown while allowlist set; ignoring")
                return
            if not any(_numbers_match(number, entry) for entry in allow):
                logger.info("Caller %s not in call_answer_contacts; ignoring", number)
                return

        if self.calls.answer():
            self._answered_this_ring = True
            self._answered_at_mono = time.monotonic()
            self._at_fired = True
            if not self._start_voice(number):
                self._speak_greeting()
        else:
            logger.error("Failed to answer call from %s", number or "unknown")

    def _start_voice(self, number: str | None) -> bool:
        """Start the live voice engine when enabled; False means speak the greeting."""
        if self.voice is None or not getattr(self.config, "voice_enabled", False):
            return False
        if self.voice.start(
            reason=f"incoming call from {number or 'unknown'}",
            contact=number or "",
        ):
            logger.info("Voice engine running for this call")
            return True
        logger.warning("Voice engine could not start; using the fixed greeting")
        return False

    def _stop_voice(self, reason: str) -> None:
        if self.voice is not None and self.voice.is_running():
            self.voice.stop(reason=reason)

    def _speak_greeting(self) -> None:
        if not getattr(self.config, "call_tts_enabled", False):
            return
        greeting = (getattr(self.config, "call_greeting", "") or "").strip()
        if not greeting:
            greeting = DEFAULT_GREETING
        if getattr(self.config, "call_speakerphone", True):
            if not self.calls.set_speakerphone():
                logger.warning(
                    "Could not enable speakerphone; caller may not hear the greeting"
                )
        time.sleep(0.5)
        if not self.calls.speak(greeting):
            logger.warning("Spoken greeting failed (no TTS helper available)")

    def _maybe_auto_hangup(self, state: CallState) -> None:
        limit = float(getattr(self.config, "call_hangup_after_sec", 0) or 0)
        if (
            limit > 0
            and self._answered_at_mono is not None
            and state.state == CALL_STATE_OFFHOOK
            and time.monotonic() - self._answered_at_mono >= limit
        ):
            logger.info("Auto-hangup after %.0fs on call", limit)
            self.calls.hangup()
            self._stop_voice("auto hangup")
            self._answered_at_mono = None

    def _reset_ring(self) -> None:
        self._answered_this_ring = False
        self._answered_at_mono = None

    # ── Status ───────────────────────────────────────────────────────
    def describe(self) -> str:
        state = self.calls.get_state()
        return (
            f"Call state: {state.state} (number: {state.number or '-'}); "
            f"window active: {self.is_active()}"
        )


# ── Parsing helpers ──────────────────────────────────────────────────
def _parse_hhmm(value: str):
    value = (value or "").strip()
    if not value:
        return None
    for fmt in ("%H:%M", "%H:%M:%S"):
        try:
            return datetime.strptime(value, fmt).time()
        except ValueError:
            continue
    logger.warning("Invalid call time '%s' (expected HH:MM)", value)
    return None


def _parse_at(value: str) -> datetime | None:
    value = (value or "").strip()
    if not value:
        return None

    for fmt in ("%Y-%m-%d %H:%M", "%Y-%m-%dT%H:%M"):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue

    now = datetime.now()
    for fmt in ("%H:%M", "%H:%M:%S"):
        try:
            moment = datetime.strptime(value, fmt).time()
        except ValueError:
            continue
        at = now.replace(
            hour=moment.hour,
            minute=moment.minute,
            second=0,
            microsecond=0,
        )
        if at <= now:
            at += timedelta(days=1)
        return at

    logger.warning(
        "Invalid call_answer_at '%s' (expected 'HH:MM' or 'YYYY-MM-DD HH:MM')",
        value,
    )
    return None


def _digits(number: str) -> str:
    return re.sub(r"\D", "", number or "")


def _numbers_match(a: str, b: str) -> bool:
    da, db = _digits(a), _digits(b)
    if not da or not db:
        return False
    if da == db:
        return True
    suffix = min(len(da), len(db), 9)
    return da[-suffix:] == db[-suffix:]
