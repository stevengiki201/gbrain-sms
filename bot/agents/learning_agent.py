"""Learn contact facts and good reply patterns from conversations."""

from __future__ import annotations

import logging

from bot.agents.ollama_verify import build_thread_text, ollama_extract_json
from bot.agents.situation import detect_situation
from bot.chat_engine import ChatEngine

logger = logging.getLogger(__name__)

_PROFILE_KEYS = frozenset({
    "name",
    "language",
    "tone",
    "relationship",
    "interests",
    "location",
    "occupation",
    "topics",
})


class LearningAgent:
    """Extract and store what Ollama learns about each contact and situation."""

    def learn_from_exchange(
        self,
        contact: str,
        inbound: str,
        reply: str,
        *,
        situation: str,
        route: str = "",
        backend: str = "",
        engine: ChatEngine,
    ) -> None:
        memory = engine.memory
        memory.save_reply_study(
            contact,
            inbound,
            reply,
            situation=situation,
            route=route,
            backend=backend,
        )

        if not engine.check_ollama(force=True):
            logger.debug("Skipping profile learning — Ollama unavailable")
            return

        recent = memory.get_recent_context(contact, limit=8)
        thread = build_thread_text(recent, limit=8)
        existing = memory.get_profile(contact)
        known = "\n".join(f"- {p.key}: {p.value}" for p in existing[:12]) or "(none)"

        prompt = (
            "Study this SMS conversation and extract durable facts about the contact.\n"
            "Return ONLY valid JSON:\n"
            '{"facts": [{"key": "name|language|tone|relationship|interests|location|occupation|topics", "value": "..."}], '
            '"situation_note": "one sentence on how to reply in this situation for this person"}\n\n'
            f"Known already:\n{known}\n\n"
            f"Recent thread:\n{thread}\n\n"
            f"Situation: {situation}\n"
            f"Their message: {inbound}\n"
            f"Our reply that worked: {reply}\n\n"
            "Only include facts clearly supported by the conversation. "
            "Use empty facts array if nothing new."
        )

        data = ollama_extract_json(engine, prompt)
        if not data:
            return

        for item in data.get("facts") or []:
            if not isinstance(item, dict):
                continue
            key = str(item.get("key", "")).strip().lower().replace(" ", "_")
            value = str(item.get("value", "")).strip()
            if key in _PROFILE_KEYS and value:
                memory.save_profile(contact, key, value, confidence=0.75, source="ollama")

        note = str(data.get("situation_note", "")).strip()
        if note:
            memory.save_situation_note(contact, situation, note)

        logger.info(
            "Learned from exchange contact=%s situation=%s facts=%d",
            contact,
            situation,
            len(data.get("facts") or []),
        )

    @staticmethod
    def situation_for_message(message: str) -> str:
        return detect_situation(message)
