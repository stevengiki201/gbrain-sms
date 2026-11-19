"""Conversation handling for live calls.

Reuses the configured Ollama endpoint and model, but with a spoken-call
persona: short sentences, no markdown, no emoji, no lists.
"""

from __future__ import annotations

import logging
import re

import httpx

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are speaking on a live telephone call, so everything you write is read aloud.
- Answer in one or two short sentences. Never more than about 40 words.
- No emojis, no markdown, no bullet points, no stage directions, no quotes.
- Sound natural and human: contractions, short reactions, plain words.
- Reply in the same language the caller used — English or Kiswahili.
- If the caller asks a question, answer it directly, then stop.
- If you did not understand, ask a short clarifying question.
- Never mention that you are an AI or a language model."""

FALLBACK_REPLY = "Sorry, I didn't catch that. Could you say it again?"
FALLBACK_REPLY_SW = "Samahani, sikukusikia vizuri. Unaweza kurudia?"

_EMOJI_RE = re.compile(
    "[" "\U0001F300-\U0001FAFF" "\U00002600-\U000027BF" "\U0001F000-\U0001F0FF" "]+",
    flags=re.UNICODE,
)


def clean_for_speech(text: str, limit: int = 240) -> str:
    text = _EMOJI_RE.sub("", text or "")
    text = re.sub(r"[*_`#>|]+", "", text)
    text = re.sub(r"^\s*(?:assistant|steve|bot)\s*:\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip().strip('"').strip()
    if len(text) > limit:
        cut = text[:limit]
        if "." in cut:
            cut = cut.rsplit(".", 1)[0] + "."
        else:
            cut = cut.rstrip() + "..."
        text = cut
    return text


class ConversationSession:
    def __init__(
        self,
        *,
        ollama_url: str,
        model: str,
        bot_name: str = "",
        system_prompt: str = "",
        max_turns: int = 12,
        timeout: float = 60.0,
    ) -> None:
        self.url = ollama_url.rstrip("/")
        self.model = model
        self.bot_name = bot_name.strip()
        self.extra_prompt = system_prompt.strip()
        self.max_turns = max_turns
        self.timeout = timeout
        self.history: list[dict[str, str]] = []
        self.turns = 0

    def reset(self) -> None:
        self.history.clear()
        self.turns = 0

    def _system_message(self) -> str:
        parts = [SYSTEM_PROMPT]
        if self.bot_name:
            parts.append(f"Your name is {self.bot_name}.")
        if self.extra_prompt:
            parts.append(self.extra_prompt)
        return "\n\n".join(parts)

    def reply(self, caller_text: str) -> str:
        caller_text = (caller_text or "").strip()
        if not caller_text:
            return ""

        messages = [{"role": "system", "content": self._system_message()}]
        messages.extend(self.history[-(self.max_turns * 2) :])
        messages.append({"role": "user", "content": caller_text})

        answer = ""
        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(
                    f"{self.url}/api/chat",
                    json={
                        "model": self.model,
                        "messages": messages,
                        "stream": False,
                        "think": False,
                        "options": {
                            "temperature": 0.6,
                            "num_predict": 120,
                            "stop": ["\n\n", "\nCaller:", "\nYou:"],
                        },
                    },
                )
                response.raise_for_status()
                data = response.json()
                answer = (data.get("message") or {}).get("content") or data.get("response") or ""
        except (httpx.HTTPError, ValueError) as exc:
            logger.warning("LLM response failed: %s", exc)

        answer = clean_for_speech(answer)
        if not answer:
            answer = FALLBACK_REPLY_SW if _looks_swahili(caller_text) else FALLBACK_REPLY
            logger.warning("Empty LLM reply; using fallback")
        else:
            logger.info("LLM response generated: %s", answer)

        self.history.append({"role": "user", "content": caller_text})
        self.history.append({"role": "assistant", "content": answer})
        self.turns += 1
        return answer


def _looks_swahili(text: str) -> bool:
    lower = (text or "").lower()
    markers = ("habari", "mambo", "vipi", "asante", "karibu", "nini", "sawa", "ndio", "pole")
    return any(marker in lower for marker in markers)
