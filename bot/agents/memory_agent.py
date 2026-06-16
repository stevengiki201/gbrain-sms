"""Build memory context for Ollama replies and verification."""

from __future__ import annotations

from dataclasses import dataclass, field

from bot.chat_engine import ChatEngine
from bot.language import detect_reply_language
from bot.memory_store import Message, ReplyStudy, TrainedReply


@dataclass
class MemoryContext:
    summary: str
    recent_messages: list[Message]
    preference_lines: list[str]
    profile_lines: list[str] = field(default_factory=list)
    situation: str = "general"
    situation_note: str = ""
    example_replies: list[ReplyStudy] = field(default_factory=list)
    trained_examples: list[TrainedReply] = field(default_factory=list)
    language: str = "en"


class MemoryContextAgent:
    """Load history, learned profile, and situation examples for this contact."""

    def build(
        self,
        contact: str,
        engine: ChatEngine,
        *,
        message: str = "",
        situation: str = "general",
        limit: int = 12,
    ) -> MemoryContext:
        recent = engine.memory.get_recent_context(contact, limit=limit)
        prefs = engine.memory.get_preferences(contact)
        pref_lines = [f"{p.key}: {p.value}" for p in prefs]

        profiles = engine.memory.get_profile(contact)
        profile_lines = [f"{p.key}: {p.value}" for p in profiles[:10]]

        recent_user = [m.body for m in recent if m.role == "user"]
        language = detect_reply_language(message, recent_user) if message else "en"
        for pref in prefs:
            if pref.key == "language" and pref.value in ("sw", "mixed"):
                language = pref.value
                break
        for p in profiles:
            if p.key == "language" and p.value in ("sw", "mixed"):
                language = p.value
                break
        if language == "en" and message:
            if detect_reply_language(message, recent_user) in ("sw", "mixed"):
                language = detect_reply_language(message, recent_user)

        situation_note = engine.memory.get_situation_note(contact, situation) or ""
        examples = engine.memory.get_situation_examples(
            situation, limit=2, contact=contact
        )
        if not examples:
            examples = engine.memory.get_situation_examples(situation, limit=2)

        trained = engine.memory.find_trained_replies(message, situation=situation, limit=3)

        summary_parts: list[str] = []
        if recent:
            summary_parts.append(f"{len(recent)} recent messages on file.")
            last_user = [m for m in recent if m.role == "user"][-3:]
            if last_user:
                topics = "; ".join(m.body[:80] for m in last_user)
                summary_parts.append(f"They recently said: {topics}")
        else:
            summary_parts.append("First conversation with this contact.")

        if profile_lines:
            summary_parts.append("Profile: " + ", ".join(profile_lines[:4]))
        if pref_lines:
            summary_parts.append("Prefs: " + ", ".join(pref_lines))
        if situation_note:
            summary_parts.append(f"Situation tip ({situation}): {situation_note[:80]}")
        if language in ("sw", "mixed"):
            summary_parts.append(f"Reply language: {language}")
        if trained:
            summary_parts.append(f"Trained examples: {len(trained)} match")

        return MemoryContext(
            summary=" ".join(summary_parts),
            recent_messages=recent,
            preference_lines=pref_lines,
            profile_lines=profile_lines,
            situation=situation,
            situation_note=situation_note,
            example_replies=examples,
            trained_examples=trained,
            language=language,
        )
