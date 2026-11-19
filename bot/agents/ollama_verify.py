"""Shared Ollama calls for verification and learning."""

from __future__ import annotations

import json
import logging
import re

import httpx

from bot.chat_engine import ChatEngine, is_factual_question
from bot.language import verification_language_note, is_practical_request
from bot.memory_store import Message

logger = logging.getLogger(__name__)


def ollama_yes_no(engine: ChatEngine, prompt: str, *, timeout: float = 25.0) -> bool | None:
    try:
        with httpx.Client(timeout=timeout) as client:
            resp = client.post(
                f"{engine.ollama_url}/api/chat",
                json={
                    "model": engine.ollama_model,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False,
                    "think": False,
                    "options": {"temperature": 0.0, "num_predict": 12},
                },
            )
            resp.raise_for_status()
            text = ((resp.json().get("message") or {}).get("content") or "").strip().lower()
            if not text:
                return None
            if text.startswith("yes"):
                return True
            if text.startswith("no"):
                return False
            return "yes" in text.split()[0:2]
    except httpx.HTTPError as exc:
        logger.debug("Ollama yes/no call failed: %s", exc)
        return None


def ollama_extract_json(engine: ChatEngine, prompt: str, *, timeout: float = 45.0) -> dict | None:
    try:
        with httpx.Client(timeout=timeout) as client:
            resp = client.post(
                f"{engine.ollama_url}/api/chat",
                json={
                    "model": engine.ollama_model,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False,
                    "think": False,
                    "options": {"temperature": 0.2, "num_predict": 256},
                },
            )
            resp.raise_for_status()
            text = (resp.json().get("message") or {}).get("content") or ""
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if not match:
                return None
            return json.loads(match.group(0))
    except (httpx.HTTPError, ValueError, json.JSONDecodeError) as exc:
        logger.debug("Ollama JSON extract failed: %s", exc)
        return None


def build_thread_text(context: list[Message], *, limit: int = 6) -> str:
    lines: list[str] = []
    for msg in context[-limit:]:
        role = "Them" if msg.role == "user" else "You"
        lines.append(f"{role}: {msg.body}")
    return "\n".join(lines) if lines else "(no prior messages)"


def verify_chat_fit(
    engine: ChatEngine,
    *,
    incoming: str,
    reply: str,
    context: list[Message],
    situation: str,
    profile_lines: list[str],
    factual: bool = False,
    reply_language: str = "en",
) -> tuple[bool | None, str]:
    """
    Ask Ollama if a proposed reply fits the chat.
    Returns (passed, reason). passed=None when Ollama unavailable.
    """
    thread = build_thread_text(context)
    profile = "\n".join(f"- {line}" for line in profile_lines) if profile_lines else "(none yet)"

    checks = [
        "CRITICALLY IMPORTANT: Reply must address the EXACT topic they just asked about",
        "Reply must NOT introduce unrelated topics (packages, oxygen, breakfast, etc.)",
        "Reply must NOT echo or repeat their message back to them",
        "Addresses their new message appropriately",
        "Fits the recent conversation flow",
        "Sounds like a casual human SMS (not AI, email, or support bot)",
        f"Matches the situation ({situation})",
    ]
    if factual or is_factual_question(incoming):
        checks.append(
            "Answers their knowledge question with real information (not smalltalk like what's up)"
        )
    if situation == "request" or is_practical_request(incoming):
        checks.append(
            "Directly responds to their request or favor (does not ignore it or change topic)"
        )
    lang_note = verification_language_note(reply_language)
    if lang_note:
        checks.append(lang_note.lstrip("- "))

    prompt = (
        "You verify SMS chat replies. Answer ONLY yes or no on the first line.\n"
        "Optionally add one short reason on the second line.\n\n"
        f"Expected reply language: {reply_language}\n"
        f"What we know about this contact:\n{profile}\n\n"
        f"Recent thread:\n{thread}\n\n"
        f"Situation: {situation}\n"
        f"Their message: {incoming}\n"
        f"Proposed reply: {reply}\n\n"
        "The reply must pass ALL checks:\n"
        + "\n".join(f"- {c}" for c in checks)
        + "\n\nIs this reply good to send?"
    )

    try:
        with httpx.Client(timeout=30.0) as client:
            resp = client.post(
                f"{engine.ollama_url}/api/chat",
                json={
                    "model": engine.ollama_model,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False,
                    "think": False,
                    "options": {"temperature": 0.0, "num_predict": 24},
                },
            )
            resp.raise_for_status()
            text = ((resp.json().get("message") or {}).get("content") or "").strip()
            if not text:
                return None, "ollama returned empty"
            lines = text.splitlines()
            first = lines[0].strip().lower()
            reason = lines[1].strip() if len(lines) > 1 else first
            if first.startswith("yes"):
                return True, f"ollama chat-fit: {reason[:120]}"
            if first.startswith("no"):
                return False, f"ollama chat-fit: {reason[:120]}"
            if "yes" in first.split()[0:2]:
                return True, f"ollama chat-fit: {reason[:120]}"
            if "no" in first.split()[0:2]:
                return False, f"ollama chat-fit: {reason[:120]}"
            return None, "ollama answer unclear"
    except httpx.HTTPError as exc:
        logger.debug("Chat-fit verification failed: %s", exc)
        return None, str(exc)
