"""Classify incoming messages into reply situations for routing and learning."""

from __future__ import annotations

from bot.chat_engine import ChatEngine, detect_mood, is_factual_question
from bot.language import SWAHILI_GREETINGS, is_practical_request, is_swahili_message


def detect_situation(message: str) -> str:
    """Return a stable situation tag used for learning and verification."""
    text = (message or "").strip()
    lower = text.lower()
    words = lower.split()

    if not text:
        return "empty"

    if is_factual_question(text):
        return "factual"

    if is_practical_request(text):
        return "request"

    mood = detect_mood(text)
    if mood:
        return f"mood_{mood}"

    if any(g in lower for g in ChatEngine.GOODBYES) or any(
        p in lower for p in ChatEngine.LATER_PHRASES
    ):
        return "goodbye"

    if any(t in lower for t in ChatEngine.THANKS):
        return "thanks"

    if any(w in lower for w in ChatEngine.BUSY_WORDS):
        return "life_update"

    if any(g in lower for g in ChatEngine.GREETINGS) and len(words) <= 6:
        return "greeting"

    if any(g in lower for g in SWAHILI_GREETINGS) or lower in ("vp", "poa"):
        return "greeting"

    if "how are you" in lower or "how r u" in lower:
        return "greeting"

    if is_swahili_message(text) and len(words) <= 5:
        return "greeting"

    if "?" in text and len(words) > 3:
        return "question"

    if len(words) > 8:
        return "substantive"

    return "general"
