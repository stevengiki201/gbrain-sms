"""Train Steve with corrected replies that apply to all real contacts."""

from __future__ import annotations

import logging
import re

from bot.agents.situation import detect_situation
from bot.chat_engine import ChatEngine
from bot.memory_store import TrainedReply, normalize_contact

logger = logging.getLogger(__name__)


def _extract_topic(message: str) -> str:
    lower = (message or "").lower()
    for word in (
        "biology", "chemistry", "physics", "math", "maths", "history",
        "geography", "civics", "english", "kiswahili", "exam", "exams",
        "school", "shule",
    ):
        if word in lower:
            return word
    return ""


class TrainingAgent:
    """Save and retrieve trained Q&A pairs for global use."""

    def save_correction(
        self,
        inbound: str,
        reply: str,
        *,
        engine: ChatEngine,
        trained_via_contact: str = "",
        situation: str = "",
        notes: str = "",
    ) -> int:
        inbound = (inbound or "").strip()
        reply = (reply or "").strip()
        if not inbound or not reply:
            raise ValueError("message and reply are required")

        situation = situation or detect_situation(inbound)
        topic = _extract_topic(inbound)
        contact = normalize_contact(trained_via_contact) if trained_via_contact else ""

        trained_id = engine.memory.save_trained_reply(
            inbound,
            reply,
            situation=situation,
            topic=topic,
            notes=notes,
            trained_via_contact=contact,
            apply_globally=True,
        )

        if topic:
            engine.memory.save_profile(
                contact or "+training",
                f"trained_{topic}",
                reply[:200],
                confidence=0.95,
                source="train",
            )

        note = f"When asked about {topic or situation}: keep reply similar to training."
        if topic:
            engine.memory.save_situation_note(contact or "+global", situation, note)

        logger.info(
            "Saved trained reply id=%s situation=%s topic=%s via=%s",
            trained_id,
            situation,
            topic or "-",
            contact or "global",
        )
        return trained_id

    def find_for_message(self, message: str, *, situation: str = "") -> list[TrainedReply]:
        situation = situation or detect_situation(message)
        return self.find_for_message_with_engine(message, situation=situation)

    def find_for_message_with_engine(
        self,
        message: str,
        *,
        engine: ChatEngine,
        situation: str = "",
    ) -> list[TrainedReply]:
        situation = situation or detect_situation(message)
        return engine.memory.find_trained_replies(
            message, situation=situation, limit=3
        )
