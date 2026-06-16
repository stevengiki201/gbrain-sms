"""Verification agents — every outbound SMS must pass all checks."""

from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass

from bot.agents.ollama_verify import verify_chat_fit
from bot.chat_engine import ChatEngine, is_factual_question
from bot.language import (
    LANGUAGE_SW,
    SWAHILI_GREETINGS,
    is_practical_request,
    reply_addresses_message,
    reply_has_swahili,
)
from bot.memory_store import Message

logger = logging.getLogger(__name__)

DISMISSIVE_REPLIES = frozenset({
    "yeah?", "what?", "what", "huh?", "huh", "ok", "k", "cool",
})

BUSY_WORDS = (
    "busy", "working", "work", "digging", "farm", "harvest", "groundnut", "job",
)

LATER_WORDS = (
    "later", "check u", "check you", "see you", "see u", "goodbye", "bye", "boe",
)

SMALLTALK_REPLY_PHRASES = (
    "what's up", "whats up", "what is up",
    "what's going on", "what is going on",
    "not much", "just chilling",
)

AI_MARKERS = (
    "as an ai", "language model", "i'm a bot", "i am a bot",
    "as a bot", "openai", "chatgpt", "i cannot help",
    "i can't help", "i'm an assistant", "i am an assistant",
)

# ── Human tone patterns ──────────────────────────────────────────────────
# Things that make a reply sound robotic or unnatural

ROBOTIC_PATTERNS = (
    r"\bi (?:am|have) here (?:to |to help|to assist)\b",
    r"\b(?:please )?(?:do not )?hesitate to (?:ask|contact)\b",
    r"\bhope this (?:helps|message finds you)\b",
    r"\bthank you for (?:your|reaching out|contacting)\b",
    r"\bfor (?:further|more) (?:information|inquiries|questions)\b",
    r"\bif you (?:have any|need|require)\b.*\b(?:please|feel free)\b",
    r"\baccording to\b",
    r"\bin (?:conclusion|summary|addition)\b",
    r"\bit is (?:important|essential|necessary|advisable)\b",
    r"\bI (?:understand|acknowledge) your (?:concern|question|request)\b",
    r"\bhere (?:is|are) (?:a|the|some) (?:information|details|steps)\b",
    r"\blet me (?:know|explain|clarify)\b",
    r"\bI would be (?:happy|glad) to\b",
    r"\bprior to\b",
    r"\bsubsequently\b",
    r"\bpursuant to\b",
    r"\bin the event that\b",
    r"\bat your earliest convenience\b",
)

FORMAL_MARKERS = (
    "dear ", "sincerely", "best regards", "hope this helps",
    "please note", "kindly", "pursuant to", "hereby",
    "hereunder", "aforementioned", "therein",
)

CONVERSATIONAL_FILLERS = (
    "hmm", "haha", "lol", "nah", "yep", "nope",
    "ah", "ooh", "wow", "oh", "ugh", "meh",
    "omg", "bruh", "fam", "ngl", "fr", "smh",
)


@dataclass
class AgentVerdict:
    agent: str
    passed: bool
    reason: str
    ollama_used: bool = False


class BaseVerifier(ABC):
    name: str

    @abstractmethod
    def verify(
        self,
        incoming: str,
        reply: str,
        *,
        context: list[Message],
        engine: ChatEngine,
        factual: bool = False,
        situation: str = "general",
        profile_lines: list[str] | None = None,
        reply_language: str = "en",
    ) -> AgentVerdict:
        ...


class SafetyVerifierAgent(BaseVerifier):
    """Fast pre-check: block unsafe, empty, or bot-revealing text."""

    name = "safety"

    def verify(
        self,
        incoming: str,
        reply: str,
        *,
        context: list[Message],
        engine: ChatEngine,
        factual: bool = False,
        situation: str = "general",
        profile_lines: list[str] | None = None,
        reply_language: str = "en",
    ) -> AgentVerdict:
        text = (reply or "").strip()
        if not text:
            return AgentVerdict(self.name, False, "reply is empty")
        if len(text) > 480:
            return AgentVerdict(self.name, False, "reply too long for SMS")
        lower = text.lower()
        if any(marker in lower for marker in AI_MARKERS):
            return AgentVerdict(self.name, False, "reply reveals AI/bot identity")
        if text.count("\n") > 2:
            return AgentVerdict(self.name, False, "reply looks like formatted text")
        if re.search(r"^[\d\s\W]+$", text):
            return AgentVerdict(self.name, False, "reply has no real words")
        return AgentVerdict(self.name, True, "passed safety checks")


class HumanToneVerifierAgent(BaseVerifier):
    """Check that the reply sounds like a real human texting, not a bot or support agent.

    This verifier catches replies that technically pass safety checks but still
    sound robotic, overly formal, or AI-generated. It uses pattern-based rules
    to detect common AI writing tells.
    """

    name = "human_tone"

    def verify(
        self,
        incoming: str,
        reply: str,
        *,
        context: list[Message],
        engine: ChatEngine,
        factual: bool = False,
        situation: str = "general",
        profile_lines: list[str] | None = None,
        reply_language: str = "en",
    ) -> AgentVerdict:
        text = (reply or "").strip()
        lower = text.lower()
        words = lower.split()

        if not text:
            return AgentVerdict(self.name, False, "empty reply")

        # ── Check for robotic / AI patterns ──────────────────────────────
        for pattern in ROBOTIC_PATTERNS:
            if re.search(pattern, lower):
                return AgentVerdict(
                    self.name, False,
                    f"reply contains robotic pattern: {pattern[:40]}...",
                )

        # ── Check for overly formal language ─────────────────────────────
        for marker in FORMAL_MARKERS:
            if marker in lower:
                return AgentVerdict(
                    self.name, False,
                    f"reply too formal for SMS: '{marker.strip()}'",
                )

        # ── Length sanity for casual SMS ──────────────────────────────────
        if situation != "factual" and len(text) > 280:
            # Casual SMS should be short — allow longer for factual answers
            return AgentVerdict(
                self.name, False,
                f"reply too long for casual SMS ({len(text)} chars, max ~280)",
            )

        # ── Check for sentence structure ──────────────────────────────────
        # Real SMS texts are often fragments, not full grammatical sentences
        sentences = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
        if len(sentences) > 5:
            return AgentVerdict(
                self.name, False,
                f"reply has too many sentences ({len(sentences)}) for casual SMS",
            )

        # ── Detect bullet points or numbered lists ────────────────────────
        if re.search(r"^[\d•\-\*]\s", text, re.MULTILINE):
            return AgentVerdict(self.name, False, "reply contains list formatting")

        # ── Detect ALL CAPS (shouting) ───────────────────────────────────
        if text.isupper() and len(text) > 10:
            return AgentVerdict(self.name, False, "all caps looks unnatural")

        # ── Check for repeated words (AI tell) ───────────────────────────
        for i in range(len(words) - 1):
            if words[i] == words[i + 1] and words[i] not in ("i", "we", "so"):
                return AgentVerdict(
                    self.name, False,
                    f"repeated word '{words[i]}' sounds unnatural",
                )

        # ── Greeting acknowledgment check ────────────────────────────────
        greeting_words = ("hey", "hi", "hello", "morning", "evening", *SWAHILI_GREETINGS)
        if any(g in lower for g in greeting_words):
            if not any(
                w in lower for w in (
                    "hey", "hi", "hello", "morning", "how", "up", "doing",
                    "mambo", "vipi", "habari", "salama", "poa", "ume", "niko",
                )
            ):
                return AgentVerdict(self.name, False, "greeting not acknowledged")

        # ── Dismissive single-word replies ────────────────────────────────
        if lower in ("ok", "k", "cool", "nice", "yeah", "yep", "nope"):
            if len(incoming.strip()) > 20:
                return AgentVerdict(
                    self.name, False,
                    "single-word reply to substantive message",
                )

        # ── Too short for a question ──────────────────────────────────────
        if "?" in incoming and len(text) < 5:
            return AgentVerdict(
                self.name, False,
                "question got too-short answer",
            )

        # ── Check for natural conversational markers ──────────────────────
        # A good casual reply often has contractions, fillers, or warmth
        has_contraction = bool(re.search(r"\b(?:i'm|don't|can't|won't|it's|that's|what's|we're|you're|let's|i've|you've)\b", lower))
        has_filler = any(f in lower for f in CONVERSATIONAL_FILLERS)
        has_warmth = bool(re.search(r"\b(?:thanks|thank|sorry|hey|hi|hello|yeah|sure|cool|nice|awesome|great|love)\b", lower))

        # For longer replies, at least one natural element helps
        if len(text) > 30 and not (has_contraction or has_filler or has_warmth):
            # Not a hard fail, but flag it — some replies are fine without these
            pass  # Allow through but log for monitoring

        return AgentVerdict(self.name, True, "passed human tone checks")


class OllamaChatFitVerifier(BaseVerifier):
    """
    Mandatory Ollama gate: reply must fit the chat, situation, and contact profile.
    Falls back to strict rules only when Ollama is offline.
    """

    name = "chat_fit"

    def verify(
        self,
        incoming: str,
        reply: str,
        *,
        context: list[Message],
        engine: ChatEngine,
        factual: bool = False,
        situation: str = "general",
        profile_lines: list[str] | None = None,
        reply_language: str = "en",
    ) -> AgentVerdict:
        incoming = (incoming or "").strip()
        reply = (reply or "").strip()
        profiles = profile_lines or []

        if not reply:
            return AgentVerdict(self.name, False, "empty reply")

        last_bot = next((m.body.strip() for m in reversed(context) if m.role == "bot"), "")
        if last_bot and reply.lower() == last_bot.lower():
            return AgentVerdict(self.name, False, "duplicate of last sent reply")

        rules = self._rules_check(
            incoming,
            reply,
            factual=factual,
            reply_language=reply_language,
            situation=situation,
        )
        if not rules.passed:
            return rules

        engine.check_ollama(force=True)
        if not engine.check_ollama():
            logger.warning("Ollama offline — using strict rules for chat-fit (degraded)")
            return AgentVerdict(
                self.name,
                rules.passed,
                f"degraded rules-only: {rules.reason}",
                ollama_used=False,
            )

        passed, reason = verify_chat_fit(
            engine,
            incoming=incoming,
            reply=reply,
            context=context,
            situation=situation,
            profile_lines=profiles,
            factual=factual,
            reply_language=reply_language,
        )
        if passed is None:
            logger.warning("Ollama chat-fit unclear — rejecting reply")
            return AgentVerdict(
                self.name, False, reason or "ollama verification unavailable", ollama_used=True
            )

        if passed:
            relevant, rel_reason = reply_addresses_message(
                incoming, reply, situation=situation
            )
            if not relevant:
                return AgentVerdict(self.name, False, rel_reason, ollama_used=True)

        return AgentVerdict(
            self.name,
            passed,
            reason,
            ollama_used=True,
        )

    def _rules_check(
        self,
        incoming: str,
        reply: str,
        *,
        factual: bool = False,
        reply_language: str = "en",
        situation: str = "general",
    ) -> AgentVerdict:
        lower_in = incoming.lower()
        lower_reply = reply.lower().strip()
        reply_words = set(re.findall(r"[a-z']+", lower_reply))

        if reply_language == LANGUAGE_SW:
            if not reply_has_swahili(reply) and len(incoming.strip()) > 3:
                return AgentVerdict(
                    self.name, False, "reply should be in Kiswahili"
                )
        elif reply_language == "mixed":
            if (
                not reply_has_swahili(reply)
                and not any(w in lower_reply for w in ("hey", "ok", "bro", "yeah", "cool"))
                and len(incoming.strip()) > 5
            ):
                return AgentVerdict(self.name, False, "reply should match Kiswahili/English mix")

        if len(incoming.strip()) > 20 and len(reply.strip()) <= 5:
            return AgentVerdict(self.name, False, "reply too short for substantive message")

        if lower_reply in DISMISSIVE_REPLIES and len(incoming.strip()) > 15:
            return AgentVerdict(self.name, False, "dismissive reply to real message")

        if "?" in incoming and len(reply) < 4:
            return AgentVerdict(self.name, False, "question got too-short answer")

        if is_factual_question(incoming) or factual:
            if any(p in lower_reply for p in SMALLTALK_REPLY_PHRASES):
                return AgentVerdict(self.name, False, "smalltalk reply to factual question")
            if len(reply.strip()) < 20:
                return AgentVerdict(
                    self.name, False, "factual question needs a substantive answer"
                )

        greeting_words = ("hey", "hi", "hello", "morning", "evening", *SWAHILI_GREETINGS)
        if any(g in lower_in for g in greeting_words):
            if not any(
                w in lower_reply
                for w in (
                    "hey", "hi", "hello", "morning", "how", "up", "doing",
                    "mambo", "vipi", "habari", "salama", "poa", "ume", "niko",
                )
            ):
                return AgentVerdict(self.name, False, "greeting not acknowledged")

        thanks_words = ("thanks", "thank you", "thx")
        if any(t in lower_in for t in thanks_words):
            if not any(
                w in lower_reply
                for w in ("welcome", "anytime", "no worry", "sure", "of course", "np")
            ):
                return AgentVerdict(self.name, False, "thanks not acknowledged")

        bye_words = ("bye", "goodbye", "good night", "see you", "gn")
        if any(b in lower_in for b in bye_words) or any(
            l in lower_in for l in LATER_WORDS
        ):
            if not any(
                w in lower_reply
                for w in ("bye", "later", "night", "care", "see you", "soon", "catch")
            ):
                return AgentVerdict(self.name, False, "goodbye not acknowledged")

        if any(w in lower_in for w in BUSY_WORDS):
            if not any(
                w in lower_reply
                for w in ("busy", "day", "lot", "work", "hope", "going", "manage", "full", "sounds")
            ):
                return AgentVerdict(self.name, False, "busy message not acknowledged")

        if is_practical_request(incoming) or situation == "request":
            relevant, rel_reason = reply_addresses_message(
                incoming, reply, situation="request"
            )
            if not relevant:
                return AgentVerdict(self.name, False, rel_reason)

        relevant, rel_reason = reply_addresses_message(
            incoming, reply, situation=situation
        )
        if not relevant and len(incoming.strip()) > 25:
            return AgentVerdict(self.name, False, rel_reason)

        formal_markers = (
            "dear ", "sincerely", "best regards",
            "hope this helps", "please note",
        )
        if any(m in lower_reply for m in formal_markers):
            return AgentVerdict(self.name, False, "too formal for SMS")

        if reply.isupper() and len(reply) > 10:
            return AgentVerdict(self.name, False, "all caps looks unnatural")

        if not (2 <= len(reply) <= 320):
            return AgentVerdict(self.name, False, "unnatural length for SMS")

        return AgentVerdict(self.name, True, "passed rules pre-check")


# Backward-compatible aliases used by imports elsewhere
RelevanceVerifierAgent = OllamaChatFitVerifier
