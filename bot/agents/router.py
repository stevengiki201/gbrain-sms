"""Route incoming SMS to rules (normal) or Ollama+memory (complex)."""

from __future__ import annotations

from dataclasses import dataclass

from bot.chat_engine import ChatEngine, detect_mood, is_factual_question
from bot.language import is_swahili_message


@dataclass
class RouteDecision:
    route: str  # "normal" | "complex"
    reason: str


class ConversationRouterAgent:
    """Agent 1: decide if a quick rules reply is enough or memory+LLM is needed."""

    def classify(self, message: str, engine: ChatEngine) -> RouteDecision:
        text = (message or "").strip()
        lower = text.lower()
        words = lower.split()

        if not text:
            return RouteDecision("normal", "empty message")

        if is_swahili_message(text):
            return RouteDecision("complex", "kiswahili — use Ollama")

        if detect_mood(text):
            return RouteDecision("complex", "emotional tone detected")

        if any(w in lower for w in ChatEngine.BUSY_WORDS):
            return RouteDecision("complex", "life update / busy")

        if any(p in lower for p in ChatEngine.LATER_PHRASES):
            return RouteDecision("complex", "goodbye / later")

        if any(g in lower for g in ChatEngine.GOODBYES):
            return RouteDecision("complex", "goodbye")

        if len(text) > 120 or len(words) > 25:
            return RouteDecision("complex", "long or detailed message")

        if any(ch in text for ch in ("http://", "https://", "www.")):
            return RouteDecision("complex", "contains link")

        if is_factual_question(text):
            return RouteDecision("complex", "factual / definition question")

        if text.count("?") >= 1 and len(words) > 4:
            return RouteDecision("complex", "question needing context")

        if "?" not in text and len(words) > 4 and any(
            w in lower for w in ("what", "how", "why", "when", "where", "who")
        ):
            return RouteDecision("complex", "question without punctuation")

        if self._is_pure_smalltalk(lower, words):
            return RouteDecision("normal", "casual smalltalk")

        if len(words) > 4:
            return RouteDecision("complex", "substantive statement")

        return RouteDecision("complex", "needs conversational context")

    def _is_pure_smalltalk(self, lower: str, words: list[str]) -> bool:
        """Only true smalltalk gets fast rules replies."""
        if lower in ("ok", "okay", "k", "cool", "nice", "lol", "haha", "yeah", "yep", "nope"):
            return True
        if "how are you" in lower or "how r u" in lower:
            return True
        if lower in ("sup", "wassup", "whats up") or "what's up" in lower:
            return True
        if len(words) <= 3 and any(g in lower for g in ChatEngine.GREETINGS):
            return True
        if len(words) <= 4 and any(t in lower for t in ChatEngine.THANKS):
            return True
        return False
