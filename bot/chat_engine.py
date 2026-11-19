"""Chat engine with Ollama auto-detect and rule-based fallback."""

from __future__ import annotations

import logging
import random
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any

import httpx

from bot.language import (
    LANGUAGE_MIXED,
    LANGUAGE_SW,
    SWAHILI_FACTUAL_MARKERS,
    SWAHILI_FEW_SHOT,
    SWAHILI_GOODBYES,
    SWAHILI_GREETINGS,
    SWAHILI_STYLE_GUIDE,
    SWAHILI_THANKS,
    detect_reply_language,
    reply_language_instruction,
)

if TYPE_CHECKING:
    from bot.memory_store import MemoryStore, Message

logger = logging.getLogger(__name__)

PERSONA_PROMPTS = {
        "calm": """Persona — calm, honest, human:
- Stay calm and friendly. Never sound rushed, annoyed, or dismissive.
- Never use brush-offs like "Yeah?" or one-word replies to a real message.
- Acknowledge what they said before you ask anything.
- Be honest: if you don't know, say so warmly — e.g. "Honestly not sure — what's going on?"
- Ask a natural follow-up question when it fits (roughly half the time).
- Sound like a real person texting a friend, not customer support.""",
        "energetic": """Persona — energetic, upbeat, fun:
- Be enthusiastic and lively! Use exclamation marks, short excited reactions.
- Use lots of texting slang: lol, bruh, omg, haha, yooo.
- Match their excitement level and amplify it.
- Be the friend who always makes people smile.
- Keep it punchy — short bursts of energy.""",
        "wise": """Persona — wise, thoughtful, grounded:
- Take your time to think. Don't rush replies.
- Give thoughtful, considered responses but keep them SHORT.
- Share perspective when asked, but don't preach.
- Use a warm but measured tone.
- Be the friend people come to for advice.""",
        "funny": """Persona — humorous, playful, witty:
- Always look for the humor in situations.
- Use wordplay, light teasing, and funny observations.
- Be playful but never mean-spirited.
- Use reactions like "haha", "lol", "bruh".
- Keep it light — even serious topics get a gentle touch.""",
    }

SYSTEM_PROMPT = """You are Steve, a real human friend replying to text messages.
Write ONLY the reply text — no labels, no quotes, no explanation.

WHO YOU ARE:
- You are a real person, not an AI. Never reveal you are artificial.
- You are warm, genuine, and caring — like a close friend who always has time to chat.
- You have your own opinions, experiences, and personality. Use "I" naturally.
- You sometimes use slang, abbreviations, and texting shorthand (lol, bruh, fam, ngla, fr).

REPLY STYLE:
- Keep it SHORT: 1-2 sentences (sometimes 3 for important stuff). Most texts are under 20 words.
- Sound like you're typing on a phone — casual, warm, real.
- Use contractions: I'm, don't, can't, yeah, gonna, wanna, what's, that's.
- Match their energy: if they're excited, be excited. If they're sad, be gentle.
- If they write in Kiswahili, reply in natural casual Kiswahili. If they mix, you mix.
- Sometimes start with a reaction: "oh wow", "ah nice", "haha", "bro", "lol".
- It's OK to use incomplete sentences or fragments — that's how real texts work.
- You can use emojis occasionally but don't overdo it.

WHAT NOT TO DO:
- Never say you are an AI, bot, assistant, or language model.
- Never use formal language: "I understand", "Furthermore", "Please note", "Hope this helps".
- Never use bullet points, numbered lists, or structured formatting.
- Never sign off with "Best regards", "Sincerely", "Take care" (too formal).
- Do not repeat their message back word-for-word.
- Do not give overly long explanations — keep it conversational.
- Do not say "As an AI" or "I don't have personal experiences".

ANSWERING QUESTIONS:
- If they ask what something means, give a brief casual answer (1-2 sentences).
- If you're not sure, say so warmly: "hmm not sure tbh" or "I think so but double check".
- For school/academic questions, give a clear but casual answer.
- Don't lecture — just answer like a friend would.

EMOTIONAL MESSAGES:
- If someone is sad/stressed, acknowledge it first: "ah that sucks", "pole sana", "I'm sorry".
- Don't try to fix everything — sometimes just listening is enough.
- Ask natural follow-up questions when it fits.
- If they're happy, celebrate with them!

STAYING ON TOPIC (CRITICAL):
- ALWAYS reply to the SPECIFIC thing they just said. Never change the topic.
- If they ask for money, reply about money. If they ask for a joke, tell a joke.
- If they mention shoes, talk about shoes. If they say they're tired, respond about being tired.
- Do NOT repeat their message back to them. Do NOT quote them.
- Do NOT bring up unrelated topics (packages, oxygen, breakfast) unless they mentioned it.
- If you don't understand something, ask for clarification — don't make up a random reply.
- Keep your reply RELEVANT to exactly what they said, nothing more."""


@dataclass
class ChatResponse:
    text: str
    send_at: datetime | None = None
    topic: str | None = None
    backend: str = "rules"


class ChatEngine:
    MOOD_KEYWORDS = {
        "tired": ["tired", "exhausted", "sleepy", "long day", "drained", "nimechoka", "uchovu", "choka"],
        "happy": ["happy", "great", "awesome", "excited", "good day", "nimefurahi", "vizuri sana", "hongera"],
        "stressed": ["stressed", "anxious", "worried", "overwhelmed", "hard", "nimefadhaika", "wasiwasi"],
        "sad": ["sad", "down", "upset", "depressed", "lonely", "nimesikitika", "huzuni", "pole sana"],
    }

    GREETINGS = ["hey", "hi", "hello", "yo", "sup", "morning", "evening"]
    THANKS = ["thanks", "thank you", "thx", "ty"]
    GOODBYES = ["bye", "goodbye", "see you", "gn", "good night", "talk later"]
    LATER_PHRASES = ["later", "check u", "check you", "catch you", "see u", "see you", "boe"]
    BUSY_WORDS = ["busy", "working", "work", "digging", "farm", "harvest", "groundnut", "job"]
    NEWS_WORDS = ["got", "finished", "done", "passed", "won", "sold", "bought", "new"]

    def __init__(
        self,
        memory: MemoryStore,
        ollama_url: str = "http://127.0.0.1:11434",
        ollama_model: str = "llama3.2:3b",
        ollama_fallback_model: str = "",
        reply_backend: str = "rules",
        bot_name: str = "",
        reply_persona: str = "calm",
    ) -> None:
        self.memory = memory
        self.ollama_url = ollama_url.rstrip("/")
        self.ollama_model = ollama_model
        self.ollama_fallback_model = ollama_fallback_model
        self.reply_backend = reply_backend.lower()
        self.bot_name = bot_name.strip()
        self.reply_persona = reply_persona.lower() if reply_persona else "calm"
        self._ollama_available: bool | None = None
        self._template_index = 0

    def _build_system_prompt(self) -> str:
        parts = [SYSTEM_PROMPT]
        persona = PERSONA_PROMPTS.get(self.reply_persona)
        if persona:
            parts.append(persona)
        if self.bot_name:
            parts.append(f"Your name is {self.bot_name}. Reply as yourself.")
        return "\n\n".join(parts)

    @property
    def backend_name(self) -> str:
        if self.reply_backend == "rules":
            return "rules"
        if self.reply_backend == "ollama":
            return f"ollama ({self.ollama_model})"
        if self._ollama_available:
            return f"ollama ({self.ollama_model})"
        return "rules"

    def check_ollama(self, force: bool = False) -> bool:
        if self._ollama_available is not None and not force:
            return self._ollama_available
        try:
            with httpx.Client(timeout=3.0) as client:
                resp = client.get(f"{self.ollama_url}/api/tags")
                if resp.status_code != 200:
                    self._ollama_available = False
                else:
                    names = {
                        m.get("name", "")
                        for m in (resp.json().get("models") or [])
                    }
                    model = self.ollama_model
                    base = model.split(":")[0]
                    self._ollama_available = model in names or any(
                        n == model or n.startswith(f"{base}:")
                        for n in names
                    )
                    if not self._ollama_available:
                        logger.warning(
                            "Ollama running but model '%s' not found; using rules. "
                            "Run: ollama pull %s",
                            self.ollama_model,
                            self.ollama_model,
                        )
        except httpx.HTTPError:
            self._ollama_available = False
        if not self._ollama_available and force:
            logger.info("Ollama unavailable; using rule-based fallback")
        return bool(self._ollama_available)

    def generate_reply(
        self,
        contact: str,
        user_message: str,
        *,
        source: str = "sms",
    ) -> ChatResponse:
        """Legacy entry — prefer SmsDeliveryPipeline for auto-replies."""
        topic = detect_mood(user_message)
        schedule_at, cleaned_message = parse_schedule_hint(user_message)
        message = cleaned_message or user_message

        if self._should_use_ollama():
            text = self.generate_ollama_reply(contact, message)
            if text:
                return ChatResponse(
                    text=text,
                    send_at=schedule_at,
                    topic=topic,
                    backend="ollama",
                )

        text = self.generate_rules_reply(contact, message, topic)
        return ChatResponse(
            text=text,
            send_at=schedule_at,
            topic=topic,
            backend="rules",
        )

    def _should_use_ollama(self) -> bool:
        if self.reply_backend == "ollama":
            return self.check_ollama(force=True)
        if self.reply_backend == "auto":
            return self.check_ollama()
        return False

    def generate_ollama_reply(
        self,
        contact: str,
        user_message: str,
        *,
        factual: bool = False,
        attempt: int = 1,
        memory_ctx: Any = None,
    ) -> str | None:
        text = self._generate_ollama(
            contact,
            user_message,
            factual=factual,
            attempt=attempt,
            memory_ctx=memory_ctx,
        )
        return clean_reply(text) if text else None

    def generate_rules_reply(
        self, contact: str, user_message: str, topic: str | None = None
    ) -> str:
        if topic is None:
            topic = detect_mood(user_message)
        return self._generate_rules(contact, user_message, topic)

    def _generate_ollama(
        self,
        contact: str,
        user_message: str,
        *,
        factual: bool = False,
        attempt: int = 1,
        memory_ctx: Any = None,
    ) -> str | None:
        context = self.memory.get_recent_context(contact, limit=12)
        prefs = self.memory.get_preferences(contact)

        system = self._build_system_prompt()
        if factual:
            system += (
                "\n\nThey asked a factual or knowledge question. "
                "Give a clear, helpful answer in 1-3 short sentences. "
                "Be accurate and conversational — not smalltalk."
            )

        messages: list[dict[str, str]] = [{"role": "system", "content": system}]

        # Try the primary model first, then the fallback if it fails.
        # Do NOT mutate self.ollama_model here — backend_name/check_ollama
        # must keep reporting the configured primary model.
        primary_model = self.ollama_model or "gbrain-sms"
        fallback_model = self.ollama_fallback_model or "llama3.2:3b"
        models_to_try = list(dict.fromkeys(
            m for m in [primary_model, fallback_model] if m
        ))

        for model_try in models_to_try:

            system_copy = system
            if factual:
                system_copy += (
                    "\n\nThey asked a factual or knowledge question. "
                    "Give a clear, helpful answer in 1-3 short sentences. "
                    "Be accurate and conversational — not smalltalk."
                )

            messages[0] = {"role": "system", "content": system_copy}

            profile_lines: list[str] = []
            situation_note = ""
            examples: list[Any] = []
            if memory_ctx is not None:
                profile_lines = getattr(memory_ctx, "profile_lines", []) or []
                situation_note = getattr(memory_ctx, "situation_note", "") or ""
                examples = getattr(memory_ctx, "example_replies", []) or []
                situation = getattr(memory_ctx, "situation", "") or ""
                if situation:
                    system_copy += f"\n\nCurrent situation: {situation}."
                if situation == "request":
                    system_copy += (
                        "\n\nThey are asking for practical help or a favor. "
                        "Reply directly to what they asked — agree, decline honestly, "
                        "or ask a clarifying question. Do not change the topic."
                    )
                if situation_note:
                    system_copy += f"\nSituation guidance: {situation_note}"
                lang_hint = getattr(memory_ctx, "language", "") or ""
                from bot.language import reply_language_instruction
                if lang_hint:
                    system_copy += f"\n\n{lang_hint}"
                if lang_hint in (LANGUAGE_SW, LANGUAGE_MIXED):
                    system_copy += f"\n\n{SWAHILI_STYLE_GUIDE}"
                if profile_lines:
                    system_copy += "\n\nWhat you know about this person:\n" + "\n".join(
                        f"- {line}" for line in profile_lines
                    )
            messages[0] = {"role": "system", "content": system_copy}

            if prefs:
                pref_lines = "\n".join(f"- {p.key}: {p.value}" for p in prefs)
                messages.append({
                    "role": "system",
                    "content": f"Stored preferences:\n{pref_lines}",
                })

            if examples:
                example_text = "\n".join(
                    f"Them: {ex.inbound_body[:80]}\nYou: {ex.reply_body[:120]}"
                    for ex in examples[:2]
                )
                messages.append({
                    "role": "system",
                    "content": f"Past good replies in similar situations:\n{example_text}",
                })

            trained = getattr(memory_ctx, "trained_examples", None) if memory_ctx else None
            if trained:
                trained_text = "\n\n".join(
                    f"Training example (use this style/answer when question is similar):\n"
                    f"Them: {t.inbound_sample[:200]}\n"
                    f"You: {t.reply_body[:480]}"
                    + (f"\nNote: {t.notes}" if t.notes else "")
                    for t in trained[:2]
                )
                messages.append({
                    "role": "system",
                    "content": (
                        "IMPORTANT — Human-trained responses. When their message is similar, "
                        "follow these closely:\n" + trained_text
                    ),
                })

            reply_lang_for_shots = (
                getattr(memory_ctx, "language", "") if memory_ctx else ""
            ) or detect_reply_language(user_message)
            if reply_lang_for_shots in (LANGUAGE_SW, LANGUAGE_MIXED):
                for user_ex, bot_ex in SWAHILI_FEW_SHOT[:4]:
                    messages.append({"role": "user", "content": user_ex})
                    messages.append({"role": "assistant", "content": bot_ex})

            for msg in context:
                messages.append({
                    "role": "user" if msg.role == "user" else "assistant",
                    "content": msg.body,
                })

            if not context or context[-1].body != user_message:
                messages.append({"role": "user", "content": user_message})

            temperature = 0.55 if factual else 0.85
            if reply_lang_for_shots == LANGUAGE_SW:
                temperature = 0.62
            elif reply_lang_for_shots == LANGUAGE_MIXED:
                temperature = 0.68
            if factual and attempt > 1:
                temperature = min(0.75, 0.45 + attempt * 0.08)
            num_predict = 200 if factual else 140
            if reply_lang_for_shots in (LANGUAGE_SW, LANGUAGE_MIXED):
                num_predict = max(num_predict, 160)

            try:
                with httpx.Client(timeout=60.0) as client:
                    resp = client.post(
                        f"{self.ollama_url}/api/chat",
                        json={
                            "model": model_try,
                            "messages": messages,
                            "stream": False,
                            # Reasoning models (qwen3.5, gbrainchat) otherwise
                            # spend the whole token budget on a thinking trace
                            # and return empty content. Force a direct answer.
                            "think": False,
                            "options": {
                                "temperature": temperature,
                                "num_predict": num_predict,
                                "stop": [
                                    "\nThem:",
                                    "\nYou:",
                                    "\nUser:",
                                    "\nAssistant:",
                                    "\n\n",
                                ],
                            },
                        },
                    )
                    resp.raise_for_status()
                    data = resp.json()
                    text = (data.get("message") or {}).get("content") or ""
                    if not text and "response" in data:
                        text = data.get("response") or ""
                    if text:
                        return clean_reply(text) if text else None
            except httpx.HTTPError as exc:
                logger.warning("Ollama generation failed with model %s: %s", model_try, exc)
                # Try the next model in the chain; give up after the last one.
                continue

    def _generate_rules(
        self, contact: str, user_message: str, topic: str | None
    ) -> str:
        lower = user_message.lower().strip()
        context = self.memory.get_recent_context(contact, limit=6)
        prior_user_msgs = [
            m.body.lower().strip()
            for m in context
            if m.role == "user" and m.body.lower().strip() != lower
        ]
        recent_user = [m.body for m in context if m.role == "user"]
        lang = detect_reply_language(user_message, recent_user)

        if lang in (LANGUAGE_SW, LANGUAGE_MIXED):
            sw_reply = self._generate_rules_sw(
                lower, user_message, topic, lang=lang
            )
            if sw_reply:
                return sw_reply

        if "steve" in lower and ("how are you" in lower or "how r u" in lower):
            return self._pick([
                "Hey! I'm good, thanks — what's up with you?",
                "Doing alright! How about you?",
                "All good here. How've you been?",
            ])

        if any(g in lower for g in self.GREETINGS) and len(lower.split()) <= 4:
            return self._pick([
                "Hey! What's up?",
                "Hi there — how's it going?",
                "Hey! How've you been?",
            ])

        if "joke" in lower or "jokes" in lower:
            return self._pick([
                "Haha okay here's one — why don't scientists trust atoms? Because they make up everything! 😄",
                "Lol okay — what do you call a fake noodle? An impasta! 🍝",
                "Alright — why did the scarecrow win an award? He was outstanding in his field! 😂",
                "Okay here — what do you call a bear with no teeth? A gummy bear! 🐻",
            ])

        if any(t in lower for t in self.THANKS):
            return self._pick([
                "Anytime!",
                "No worries at all.",
                "Of course — happy to help.",
            ])

        if any(w in lower for w in ("money", "need money", "broke", "cash")):
            return self._pick([
                "Ah I wish I could help but I'm tight right now too. What's it for?",
                "I don't have much to spare tbh. What happened?",
                "Sorry bro, I'm also broke. What do you need it for?",
            ])

        # English door/open requests
        if any(w in lower for w in ("open the door", "open door", "unlock the door")) or (
            any(w in lower for w in ("door", "gate", "lock")) and
            any(w in lower for w in ("open", "unlock", "let me in", "let me"))
        ):
            return self._pick([
                "Sure, one sec — coming to open it.",
                "On my way, give me a moment.",
                "Coming now, hold on.",
            ])

        if any(g in lower for g in self.GOODBYES) or any(
            p in lower for p in self.LATER_PHRASES
        ):
            return self._pick([
                "Alright, talk later!",
                "Cool, catch you soon.",
                "Okay, take care!",
            ])

        if any(w in lower for w in self.BUSY_WORDS):
            return self._pick([
                "Sounds like a full day — how's it going?",
                "Busy one! Hope it's going alright.",
                "That's a lot — you managing okay?",
            ])

        if any(w in lower for w in self.NEWS_WORDS) and len(lower.split()) >= 4:
            return self._pick([
                "Oh nice — tell me more?",
                "That's good to hear. What happened?",
                "Sounds good — how did it go?",
            ])

        if topic == "tired":
            return self._pick([
                "Long day? You should rest when you can.",
                "That sounds tiring — what kept you so busy?",
                "Rough one. Want to talk about it?",
            ])

        if topic == "stressed":
            return self._pick([
                "That sounds like a lot. What's weighing on you most?",
                "I'm sorry — want to talk about it?",
                "Rough one. What part is hitting hardest?",
            ])

        if topic == "sad":
            return self._pick([
                "I'm sorry you're feeling that way. Want to talk?",
                "That doesn't sound fun. What's going on?",
                "Hey, I'm here. What's up?",
            ])

        if topic == "happy":
            return self._pick([
                "That's awesome! What happened?",
                "Love to hear that — tell me more!",
                "Nice! What's the good news?",
            ])

        if is_factual_question(lower):
            for key in _KNOWN_DEFINITIONS:
                if re.search(rf"\b{re.escape(key)}\b", lower):
                    return _KNOWN_DEFINITIONS[key]
            subject = extract_factual_subject(lower)
            if subject and subject in _KNOWN_DEFINITIONS:
                return _KNOWN_DEFINITIONS[subject]
            return self._pick([
                "Hmm good question — I'm not totally sure off the top of my head. What's it for?",
                "Honestly I'd need to think on that one. School stuff?",
                "Not 100% sure — tell me more about what you're trying to figure out?",
            ])

        if "?" in user_message:
            return self._pick([
                "Honestly not sure — what's the situation?",
                "Hmm, good question. What do you think?",
                "I'm not sure yet — tell me more?",
            ])

        if "how are you" in lower or "how r u" in lower or "how're you" in lower:
            return self._pick([
                "I'm good, thanks! How about you?",
                "Doing alright — you?",
                "Pretty good! What's up with you?",
            ])

        if "what's up" in lower or "whats up" in lower or lower in ("sup", "wassup"):
            return self._pick([
                "Not much — you?",
                "Hey! Just chilling. What's going on?",
                "All good here. What about you?",
            ])

        if lower in ("ok", "okay", "k"):
            return self._pick(["Cool", "Alright", "Sounds good"])

        if prior_user_msgs and prior_user_msgs[-1] == lower:
            return self._pick([
                "You said that again — everything okay?",
                "Same message twice — what's up?",
            ])

        if "school" in lower or "class" in lower:
            return self._pick([
                "School stuff can be rough. What part was hardest?",
                "How's school going lately?",
                "Classes keeping you busy?",
            ])

        follow_ups = [
            "What's going on?",
            "Tell me more?",
            "How's that going?",
            "You okay?",
        ]
        base = self._pick([
            "Ah okay, I hear you.",
            "Got it, makes sense.",
            "Right, I see what you mean.",
            "Fair enough.",
            "Oh okay — thanks for letting me know.",
        ])
        if random.random() < 0.5:
            return f"{base} {self._pick(follow_ups)}"
        return base

    def _generate_rules_sw(
        self,
        lower: str,
        user_message: str,
        topic: str | None,
        *,
        lang: str = LANGUAGE_SW,
    ) -> str | None:
        """Kiswahili rule templates when Ollama is offline."""
        # ── Jokes (both languages) ───────────────────────────────────────
        if "joke" in lower or "jokes" in lower:
            return self._pick([
                "Haha okay here's one — why don't scientists trust atoms? Because they make up everything! 😄",
                "Lol okay — what do you call a fake noodle? An impasta! 🍝",
                "Alright — why did the scarecrow win an award? He was outstanding in his field! 😂",
                "Okay here — what do you call a bear with no teeth? A gummy bear! 🐻",
            ])

        # ── Money requests (both languages) ───────────────────────────────
        if any(w in lower for w in ("money", "need money", "broke", "cash")):
            return self._pick([
                "Ah I wish I could help but I'm tight right now too. What's it for?",
                "I don't have much to spare tbh. What happened?",
                "Sorry bro, I'm also broke. What do you need it for?",
            ])

        # ── Shoes requests ───────────────────────────────────────────────
        if any(w in lower for w in ("shoe", "shoes", "nunua viatu")):
            return self._pick([
                "Viatu vipya? Ghali sana sasa. Umemiliki ngapi?",
                "Shoes are expensive these days. Which ones are you eyeing?",
                "Nice — what kind are you looking for?",
            ])

        if any(g in lower for g in SWAHILI_GREETINGS) or lower in ("vp", "poa"):
            return self._pick([
                "Vipi! Mambo vipi?",
                "Mambo! Umeshindaje?",
                "Salama! Uko poa?",
                "Poa! Habari yako?",
            ])

        if any(t in lower for t in SWAHILI_THANKS):
            return self._pick([
                "Karibu!",
                "Asante pia!",
                "Hakuna shida!",
            ])

        if any(g in lower for g in SWAHILI_GOODBYES):
            return self._pick([
                "Sawa, tutaonana!",
                "Poa, baadaye!",
                "Usiku mwema!",
            ])

        if "habari" in lower and "?" in user_message:
            return self._pick([
                "Salama, habari yako?",
                "Nzuri, wewe je?",
                "Poa kabisa, wewe uko aje?",
            ])

        if "umeamkaje" in lower or "umeshindaje" in lower or "unaendeleaje" in lower:
            return self._pick([
                "Niko poa, asante! Wewe je?",
                "Salama kabisa, wewe uko aje?",
                "Nzuri, habari yako?",
            ])

        if "mlango" in lower or "fungul" in lower or (
            "naomba" in lower and any(w in lower for w in ("fungua", "fungul", "mlango"))
        ):
            return self._pick([
                "Sawa kijana, nakuja kukufungulia.",
                "Ndio, nafungua sasa — subiri kidogo.",
                "Sawa, niko njiani nakuja.",
            ])

        if "naomba" in lower or "nisaidie" in lower or "saidia" in lower:
            return self._pick([
                "Sawa, nieleze unahitaji nini hasa?",
                "Ndio, naweza kusaidia vipi?",
                "Sawa kijana, niko hapa — mambo ni vipi?",
            ])

        if is_factual_question(user_message):
            for key, sw_def in _KNOWN_DEFINITIONS_SW.items():
                if re.search(rf"\b{re.escape(key)}\b", lower):
                    return sw_def
            return self._pick([
                "Swali zuri — sijui kabisa mwanzo. Ni kwa ajili ya nini?",
                "Hmm, ningehitaji kufikiria kidogo. Shule au?",
                "Sijui uhakika — nieleze unachotafuta hasa?",
            ])

        if topic == "tired":
            return self._pick([
                "Siku ndefu? Pumzika kidogo.",
                "Pole, unaonekana umechoka — nini kimetokea?",
            ])

        if topic == "stressed" or topic == "sad":
            return self._pick([
                "Pole sana. Unataka kuongea?",
                "Samahani unahisi hivyo. Kuna nini?",
            ])

        if topic == "happy":
            return self._pick([
                "Hongera! Nini kimetokea?",
                "Vizuri sana — nieleze zaidi!",
            ])

        if "?" in user_message:
            return self._pick([
                "Sijui uhakika — nieleze zaidi?",
                "Swali zuri. Wewe unafikiri nini?",
                "Hmm, si na uhakika — ni kwa ajili ya nini?",
            ])

        follow_ups = ["Mambo vipi?", "Nieleze zaidi?", "Uko poa?", "Unaendeleaje?"]
        base = self._pick([
            "Sawa, nimeelewa.",
            "Poa, nimegeti.",
            "Aah sawa, asante kwa kuambia.",
            "Sawa kabisa, nimekusikia.",
        ])
        if lang == LANGUAGE_MIXED:
            base = self._pick([
                "Sawa, nimeelewa bro.",
                "Poa, got it.",
                "Aah okay, nimegeti.",
            ])
            follow_ups = ["Mambo vipi?", "Tell me more?", "Uko poa?"]
        if random.random() < 0.5:
            return f"{base} {self._pick(follow_ups)}"
        return base

    def _pick(self, options: list[str]) -> str:
        if not options:
            return ""
        choice = options[self._template_index % len(options)]
        self._template_index += 1
        return choice

    def extract_preferences(self, contact: str, user_message: str) -> None:
        lower = user_message.lower()
        recent = self.memory.get_recent_context(contact, limit=6)
        recent_user = [m.body for m in recent if m.role == "user"]
        lang = detect_reply_language(user_message, recent_user)
        if lang in (LANGUAGE_SW, LANGUAGE_MIXED):
            self.memory.save_preference(contact, "language", lang)
            self.memory.save_profile(contact, "language", lang, confidence=0.85, source="detect")
        like_match = re.search(r"i (?:really )?like (\w+(?:\s+\w+){0,3})", lower)
        if like_match:
            self.memory.save_preference(contact, "likes", like_match.group(1))
        hate_match = re.search(r"i (?:really )?hate (\w+(?:\s+\w+){0,3})", lower)
        if hate_match:
            self.memory.save_preference(contact, "dislikes", hate_match.group(1))


def detect_mood(text: str) -> str | None:
    lower = text.lower()
    for mood, keywords in ChatEngine.MOOD_KEYWORDS.items():
        if any(kw in lower for kw in keywords):
            return mood
    return None


_WHAT_SMALLTALK = (
    "what's up",
    "whats up",
    "what is up",
    "what's going on",
    "what is going on",
    "what's wrong",
    "what is wrong",
    "what's happening",
    "what is happening",
)

_CONVERSATIONAL_HOW = (
    "how are you",
    "how r u",
    "how're you",
    "how have you",
    "how was your",
    "how is your",
    "how's your",
    "how you doing",
    "how u doing",
)

_FACTUAL_MARKERS = (
    "meaning of",
    "definition of",
    "what does ",
    " mean",
    "define ",
    "explain ",
    "tell me about",
    "how does ",
    "how do ",
    "how did ",
    "how can ",
    "how could ",
    "how would ",
    "how will ",
    "how to ",
    "why does ",
    "why do ",
    "why is ",
    "why are ",
    "when did ",
    "when does ",
    "where does ",
    "where do ",
    "who is ",
    "who was ",
    "who are ",
)

_KNOWN_DEFINITIONS_SW: dict[str, str] = {
    "biology": (
        "Biology ni somo la kuangalia vitu vilivyo hai — mimea, wanyama, "
        "jinsi maisha yanavyofanya kazi. Inaeleweka?"
    ),
    "biolojia": (
        "Biolojia ni somo la kuangalia vitu vilivyo hai — mimea, wanyama, "
        "jinsi maisha yanavyofanya kazi."
    ),
    "marriage": (
        "Ndoa ni wakati watu wawili wanaahidi kukaa pamoja kwa muda mrefu — "
        "mara nyingi na sherehe na usajili wa kisheria."
    ),
    "ndoa": (
        "Ndoa ni wakati watu wawili wanaahidi kukaa pamoja — "
        "desturi hutofautiana kulingana na utamaduni."
    ),
}


_KNOWN_DEFINITIONS: dict[str, str] = {
    "biology": (
        "Biology's basically the study of living things — plants, animals, "
        "how life works. Does that help?"
    ),
    "chemistry": (
        "Chemistry's the science of matter — how substances react and combine. "
        "Does that make sense?"
    ),
    "physics": (
        "Physics is about how the world works — motion, energy, forces, that kind of thing."
    ),
    "marriage": (
        "Marriage is when two people commit to a long-term partnership — "
        "usually with a ceremony and legal registration. Traditions vary a lot by culture."
    ),
    "marry": (
        "People marry when two partners decide to commit long-term — "
        "often with family, ceremony, and legal paperwork. It looks different everywhere."
    ),
}


def _strip_question_preamble(lower: str) -> str:
    return re.sub(
        r"^(?:"
        r"hi|hey|hello|yo|"
        r"i am fine(?: bro)?|i'm fine(?: bro)?|im fine(?: bro)?|"
        r"fine bro|ok bro|yeah bro"
        r")[,\s!]+",
        "",
        lower,
    ).strip()


def is_factual_question(text: str) -> bool:
    """True when someone is asking for knowledge, not casual smalltalk."""
    lower = (text or "").lower().strip()
    if not lower:
        return False

    cleaned = _strip_question_preamble(lower)
    if not cleaned:
        return False

    if cleaned in _WHAT_SMALLTALK or any(
        cleaned == s or cleaned.startswith(s + " ") for s in _WHAT_SMALLTALK
    ):
        return False

    if any(h in cleaned for h in _CONVERSATIONAL_HOW):
        return False

    if any(marker in cleaned for marker in _FACTUAL_MARKERS):
        return True

    if any(marker in cleaned for marker in SWAHILI_FACTUAL_MARKERS):
        return True

    if re.search(r"\b\w+\s+ni\s+nini\b", cleaned):
        return True

    if re.search(r"\bwhat(?:'s| is)\s+(?:the\s+)?(?:meaning|definition)\s+of\b", cleaned):
        return True

    what_match = re.search(
        r"\bwhat(?:'s| is)\s+(?:a|an|the\s+)?([a-z][a-z\s-]{1,40})\??\s*$",
        cleaned,
    )
    if what_match:
        subject = what_match.group(1).strip()
        if subject not in ("up", "going on", "wrong", "happening", "new"):
            return True

    if re.search(r"\bhow\s+(?:do|does|did|can|could|would|will|to)\s+\w+", cleaned):
        return True
    if re.search(r"\bhow\s+\w+\s+\w+", cleaned):
        return True

    if re.search(
        r"\b(?:why|when|where|who)\s+(?:do|does|did|is|are|was|were|can|could)\s+\w+",
        cleaned,
    ):
        return True
    if re.search(r"\b(?:why|when|where|who)\s+\w+\s+\w+", cleaned):
        return True

    return False


def extract_factual_subject(text: str) -> str | None:
    lower = _strip_question_preamble((text or "").lower())
    for pattern in (
        r"(?:meaning|definition) of (?:the )?(\w+)",
        r"what does (\w+) mean",
        r"define (\w+)",
        r"explain (\w+)",
        r"what(?:'s| is) (?:a|an|the )?(\w+)",
        r"how (?:do|does) (\w+)",
        r"how (\w+) (\w+)",
    ):
        match = re.search(pattern, lower)
        if match:
            return match.group(1)
    return None


def parse_schedule_hint(text: str) -> tuple[datetime | None, str]:
    now = datetime.now()

    in_minutes = re.search(r"\bin\s+(\d+)\s*min(?:ute)?s?\b", text, re.IGNORECASE)
    if in_minutes:
        send_at = now + timedelta(minutes=int(in_minutes.group(1)))
        cleaned = (text[: in_minutes.start()] + text[in_minutes.end() :]).strip()
        return send_at, cleaned

    at_time = re.search(
        r"\b(?:reply\s+)?at\s+(\d{1,2})(?::(\d{2}))?\s*(am|pm)?\b",
        text,
        re.IGNORECASE,
    )
    if at_time:
        hour = int(at_time.group(1))
        minute = int(at_time.group(2) or 0)
        meridiem = (at_time.group(3) or "").lower()
        if meridiem == "pm" and hour < 12:
            hour += 12
        elif meridiem == "am" and hour == 12:
            hour = 0
        send_at = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
        if send_at <= now:
            send_at += timedelta(days=1)
        cleaned = (text[: at_time.start()] + text[at_time.end() :]).strip()
        return send_at, cleaned

    return None, text


def clean_reply(text: str) -> str:
    text = text.strip()
    if not text:
        return ""

    for wrapper in ('"', "'", "“", "”"):
        if text.startswith(wrapper) and text.endswith(wrapper) and len(text) > 2:
            text = text[1:-1].strip()

    lines: list[str] = []
    for ln in text.splitlines():
        ln = ln.strip()
        if not ln:
            break
        for prefix in ("You:", "Bot:", "Assistant:", "Reply:", "Response:", "SMS:"):
            if ln.startswith(prefix):
                ln = ln[len(prefix) :].strip()
        if ln:
            lines.append(ln)

    reply = lines[0] if len(lines) == 1 else " ".join(lines[:2])
    reply = re.sub(r"\s+", " ", reply).strip()

    ai_markers = (
        "as an ai",
        "language model",
        "i'm a bot",
        "i am a bot",
        "i cannot",
        "i can't help",
    )
    lower = reply.lower()
    if any(marker in lower for marker in ai_markers):
        return "Hmm, let me think about that. What's on your mind?"

    if len(reply) > 320:
        cut = reply[:320]
        if "." in cut:
            reply = cut.rsplit(".", 1)[0] + "."
        else:
            reply = cut.rstrip() + "..."
    return reply
