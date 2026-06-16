"""Orchestrate routing, generation, verification, and delivery of SMS replies."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from bot.agents.learning_agent import LearningAgent
from bot.agents.memory_agent import MemoryContext, MemoryContextAgent
from bot.agents.router import ConversationRouterAgent, RouteDecision
from bot.agents.situation import detect_situation
from bot.agents.verifiers import (
    AgentVerdict,
    HumanToneVerifierAgent,
    OllamaChatFitVerifier,
    SafetyVerifierAgent,
)
from bot.chat_engine import ChatEngine, ChatResponse, detect_mood, is_factual_question, parse_schedule_hint
from bot.language import LANGUAGE_MIXED, LANGUAGE_SW

logger = logging.getLogger(__name__)

MAX_GENERATION_ATTEMPTS = 3


@dataclass
class PipelineResponse(ChatResponse):
    route: str = ""
    situation: str = "general"
    memory_summary: str = ""
    verdicts: list[AgentVerdict] = field(default_factory=list)
    attempts: int = 1
    approved: bool = False


class SmsDeliveryPipeline:
    """
    SMS passes through agents before send:
      1. RouterAgent           — normal (rules) vs complex (Ollama + memory)
      2. MemoryContextAgent    — history, learned profile, situation examples
      3. Generator             — draft reply (Ollama when complex/factual)
      4. SafetyVerifier        — block unsafe / bot-like text
      5. HumanToneVerifier     — ensure reply sounds natural and human
      6. OllamaChatFitVerifier — mandatory Ollama check that reply fits the chat
      7. LearningAgent         — after send, store facts and good reply examples
    """

    def __init__(self, engine: ChatEngine) -> None:
        self.engine = engine
        self.router = ConversationRouterAgent()
        self.memory_agent = MemoryContextAgent()
        self.learning_agent = LearningAgent()
        self.verifiers = [
            SafetyVerifierAgent(),
            HumanToneVerifierAgent(),
            OllamaChatFitVerifier(),
        ]

    def process(
        self,
        contact: str,
        user_message: str,
        *,
        source: str = "sms",
    ) -> PipelineResponse:
        topic = detect_mood(user_message)
        schedule_at, cleaned = parse_schedule_hint(user_message)
        message = cleaned or user_message

        route_decision = self.router.classify(message, self.engine)
        situation = detect_situation(message)
        factual = is_factual_question(message)
        if factual and route_decision.route != "complex":
            route_decision = RouteDecision("complex", "factual / knowledge question")

        memory_ctx = self.memory_agent.build(
            contact, self.engine, message=message, situation=situation
        )
        if memory_ctx.language in (LANGUAGE_SW, LANGUAGE_MIXED) and route_decision.route == "normal":
            route_decision = RouteDecision("complex", "kiswahili — use Ollama")
        logger.info(
            "Pipeline route=%s (%s) situation=%s lang=%s factual=%s contact=%s memory=%s",
            route_decision.route,
            route_decision.reason,
            situation,
            memory_ctx.language,
            factual,
            contact,
            memory_ctx.summary[:80],
        )

        last_verdicts: list[AgentVerdict] = []
        used_ollama = False
        for attempt in range(1, MAX_GENERATION_ATTEMPTS + 1):
            draft, used_ollama = self._generate_draft(
                contact,
                message,
                route=route_decision.route,
                attempt=attempt,
                topic=topic,
                factual=factual,
                memory_ctx=memory_ctx,
            )
            draft = draft.strip()
            if not draft:
                draft = self._safe_fallback(message, language=memory_ctx.language)

            verdicts = self._run_verifiers(
                message,
                draft,
                context=memory_ctx.recent_messages,
                factual=factual,
                situation=situation,
                profile_lines=memory_ctx.profile_lines,
                reply_language=memory_ctx.language,
                contact=contact,
                route=route_decision.route,
                attempt=attempt,
            )
            last_verdicts = verdicts
            failed = [v for v in verdicts if not v.passed]

            if not failed:
                logger.info(
                    "Pipeline approved (attempt %d, route=%s, backend=%s): %s",
                    attempt,
                    route_decision.route,
                    "ollama" if used_ollama else self._backend_for_route(route_decision.route),
                    draft[:80],
                )
                for v in verdicts:
                    logger.debug("  [%s] %s — %s", v.agent, "PASS" if v.passed else "FAIL", v.reason)

                return PipelineResponse(
                    text=draft,
                    send_at=schedule_at,
                    topic=topic,
                    backend="ollama" if used_ollama else self._backend_for_route(route_decision.route),
                    route=route_decision.route,
                    situation=situation,
                    memory_summary=memory_ctx.summary,
                    verdicts=verdicts,
                    attempts=attempt,
                    approved=True,
                )

            self._log_failed_verification(
                contact, message, draft, situation, route_decision.route, attempt, verdicts
            )

        fallback = self._safe_fallback(message, language=memory_ctx.language)
        fallback_verdicts = self._run_verifiers(
            message,
            fallback,
            context=memory_ctx.recent_messages,
            factual=factual,
            situation=situation,
            profile_lines=memory_ctx.profile_lines,
            reply_language=memory_ctx.language,
            contact=contact,
            route=route_decision.route,
            attempt=MAX_GENERATION_ATTEMPTS,
        )
        approved = all(v.passed for v in fallback_verdicts)
        if not approved:
            logger.error("Fallback failed verification; using minimal safe reply")
            fallback = (
                self._safe_fallback(message, language=memory_ctx.language)
                if factual
                else self._safe_fallback(message, language=memory_ctx.language)
            )
            fallback_verdicts = self._run_verifiers(
                message,
                fallback,
                context=memory_ctx.recent_messages,
                factual=factual,
                situation=situation,
                profile_lines=memory_ctx.profile_lines,
                reply_language=memory_ctx.language,
                contact=contact,
                route=route_decision.route,
                attempt=MAX_GENERATION_ATTEMPTS,
            )
            approved = all(v.passed for v in fallback_verdicts)

        return PipelineResponse(
            text=fallback,
            send_at=schedule_at,
            topic=topic,
            backend="rules",
            route=route_decision.route,
            situation=situation,
            memory_summary=memory_ctx.summary,
            verdicts=fallback_verdicts or last_verdicts,
            attempts=MAX_GENERATION_ATTEMPTS,
            approved=approved,
        )

    def record_successful_reply(
        self,
        contact: str,
        inbound: str,
        reply: str,
        *,
        situation: str,
        route: str,
        backend: str,
    ) -> None:
        """Store exchange for learning after SMS is delivered."""
        self.learning_agent.learn_from_exchange(
            contact,
            inbound,
            reply,
            situation=situation,
            route=route,
            backend=backend,
            engine=self.engine,
        )

    def _generate_draft(
        self,
        contact: str,
        message: str,
        *,
        route: str,
        attempt: int,
        topic: str | None,
        factual: bool = False,
        memory_ctx: MemoryContext | None = None,
    ) -> tuple[str, bool]:
        swish = memory_ctx and memory_ctx.language in (LANGUAGE_SW, LANGUAGE_MIXED)
        if route == "normal" and not factual and not swish:
            return self.engine.generate_rules_reply(contact, message, topic), False

        ollama_ok = self._ollama_enabled() or factual or swish
        if ollama_ok:
            if factual or swish:
                self.engine.check_ollama(force=True)
            text = self.engine.generate_ollama_reply(
                contact,
                message,
                factual=factual,
                attempt=attempt,
                memory_ctx=memory_ctx,
            )
            if text:
                return text, True

        if factual:
            logger.warning(
                "Ollama unavailable for factual question; using rules fallback"
            )
        elif swish:
            logger.warning(
                "Ollama unavailable for Kiswahili message; using Swahili rules fallback"
            )
        return self.engine.generate_rules_reply(contact, message, topic), False

    def _ollama_enabled(self) -> bool:
        backend = self.engine.reply_backend
        if backend == "rules":
            return False
        if backend == "ollama":
            return self.engine.check_ollama(force=True)
        return self.engine.check_ollama()

    def _backend_for_route(self, route: str) -> str:
        if route == "complex" and self._ollama_enabled():
            return "ollama"
        return "rules"

    def _run_verifiers(
        self,
        incoming: str,
        reply: str,
        *,
        context: list,
        factual: bool = False,
        situation: str = "general",
        profile_lines: list[str] | None = None,
        reply_language: str = "en",
        contact: str = "",
        route: str = "",
        attempt: int = 1,
    ) -> list[AgentVerdict]:
        verdicts: list[AgentVerdict] = []
        for verifier in self.verifiers:
            verdict = verifier.verify(
                incoming,
                reply,
                context=context,
                engine=self.engine,
                factual=factual,
                situation=situation,
                profile_lines=profile_lines,
                reply_language=reply_language,
            )
            verdicts.append(verdict)
            if contact:
                self.engine.memory.log_verification(
                    contact,
                    incoming,
                    reply,
                    situation=situation,
                    route=route,
                    attempt=attempt,
                    verifier=verdict.agent,
                    passed=verdict.passed,
                    reason=verdict.reason,
                    ollama_used=verdict.ollama_used,
                )
            logger.debug(
                "Verifier [%s] %s — %s",
                verdict.agent,
                "PASS" if verdict.passed else "FAIL",
                verdict.reason,
            )
        return verdicts

    def _log_failed_verification(
        self,
        contact: str,
        incoming: str,
        draft: str,
        situation: str,
        route: str,
        attempt: int,
        verdicts: list[AgentVerdict],
    ) -> None:
        failed = [v for v in verdicts if not v.passed]
        if not failed:
            return
        logger.warning(
            "Pipeline attempt %d rejected by %s: %s",
            attempt,
            ", ".join(v.agent for v in failed),
            "; ".join(v.reason for v in failed),
        )

    @staticmethod
    def _safe_fallback(message: str, *, language: str = "en") -> str:
        from bot.language import LANGUAGE_MIXED, LANGUAGE_SW

        lower = (message or "").lower()
        if language in (LANGUAGE_SW, LANGUAGE_MIXED):
            if is_factual_question(message):
                return "Swali zuri — sijui kabisa mwanzo. Ni kwa ajili ya nini?"
            from bot.language import is_practical_request

            if is_practical_request(message) or "mlango" in lower or "fungul" in lower:
                return "Sawa kijana, nakuja kukufungulia."
            if any(g in lower for g in ("habari", "mambo", "vipi", "vp", "hujambo")):
                return "Mambo! Umeshindaje?"
            if any(t in lower for t in ("asante", "ashante")):
                return "Karibu!"
            if any(b in lower for b in ("kwaheri", "baadaye", "usiku")):
                return "Sawa, tutaonana!"
            if "?" in message:
                return "Sijui uhakika — mambo ni vipi?"
            return "Sawa, nieleze zaidi?" if language == LANGUAGE_SW else "Sawa bro, tell me more?"
        if is_factual_question(message):
            return "Hmm good question — I'm not totally sure. What's it for?"
        if any(g in lower for g in ("hey", "hi", "hello", "morning")):
            return "Hey! What's up?"
        if any(t in lower for t in ("thanks", "thank you", "thx")):
            return "Anytime!"
        if any(b in lower for b in ("bye", "goodbye", "good night", "gn", "later", "check u", "boe")):
            return "Alright, talk later!"
        if any(w in lower for w in ("busy", "working", "digging", "farm")):
            return "Sounds like a full day — hope it's going okay."
        if "how are you" in lower:
            return "I'm good, thanks! How about you?"
        if "?" in message:
            return "Honestly not sure — what's going on?"
        return "Got it — tell me more?"
