"""Detect Kiswahili in SMS and build language-aware reply instructions."""

from __future__ import annotations

import re

# Common Kiswahili in East African SMS (Kenya / Tanzania)
SWAHILI_WORDS = frozenset({
    "habari", "vipi", "mambo", "salama", "hujambo", "hamjambo", "shikamoo",
    "asante", "ashante", "karibu", "kwaheri", "usiku", "mwema", "ndio", "hapana",
    "sawa", "poa", "safi", "nzuri", "vizuri", "nini", "wapi", "wap", "nipo", "niko",
    "nimeamka", "umeamkaje", "umeshindaje", "unaendeleaje", "sijui", "nafahamu",
    "eleza", "nieleze", "baadaye", "kesho", "leo", "bado", "sasa", "nataka",
    "tafadhali", "samahani", "pole", "hongera", "ndiyo", "sana", "kidogo",
    "mzuri", "mbaya", "kazi", "shule", "nyumbani", "rafiki", "kaka", "dada",
    "baba", "mama", "naweza", "unaweza", "tunaweza", "nina", "una", "tuna",
    "nime", "ume", "tume", "ame", "wame", "kama", "lakini", "pia", "tu", "basi",
    "najua", "fahamu", "maana", "jambo", "vp", "naomba", "mlango", "fungua",
    "kijana", "msichana", "nyumba", "pesa", "nisaidie", "saidia", "nitumie",
    "nakuja", "subiri", "hapo", "huku", "mko", "wako", "yako", "yangu",
    "je", "labda", "bado", "tena", "sana", "kabisa", "tuonane", "niko", "uko",
    "ameamka", "unalala", "umelala", "usingizi", "chakula", "maji", "baridi",
    "joto", "mvua", "safari", "gari", "boda", "simu", "salio", "mkopo", "deni",
    "mama", "baba", "mjomba", "shangazi", "rafiki", "mpenzi", "mpenz", "penzi",
    "nimekumiss", "ninakupenda", "nakupenda", "nimekumiss", "niko", "upo",
    "uko", "wapi", "huku", "hapo", "kule", "hapa", "leo", "kesho", "jana",
    "asubuhi", "mchana", "jioni", "usiku", "saa", "saa", "sijui", "najua",
    "nimechoka", "nimechoka", "pole", "pole", "hongera", "vizuri", "mbaya",
    "hebu", "bas", "eeh", "ee", "aah", "yani", "kweli", "kweli", "sio",
    "siyo", "si", "haina", "noma", "freshi", "fiti", "mshkaji", "msee",
})

SWAHILI_MARKERS = (
    "naomba", "mlango", "fungul", "fungua", "nisaidie", "saidia",
    "nitumie", "nifungulie", "unifungulie", "kijana", "msichana", "nyumbani",
    "habari", "mambo", "vipi", "asante", "tafadhali", "samahani", "nakuja",
    "nime", "ume", "tume", "nina", "una", "sijui", "nafahamu", "nieleze",
    "umeshindaje", "umeamkaje", "unaendeleaje", "habari yako", "mambo vipi",
    "nifanyie", "nitumie", "niletee", "nipe", "nipigie", "nipigie", "wasiliana",
    "nimekumiss", "nakupenda", "ninakupenda", "chakula", "maji", "salio",
    "niko", "uko", "upo", "wapi", "huku", "hapo", "pole sana", "hongera",
)

REQUEST_MARKERS = (
    "naomba", "tafadhali", "please", "can you", "could you", "nisaidie",
    "saidia", "nifungulie", "unifungulie", "nitumie", "niletee", "need you to",
)

REQUEST_TOPIC_WORDS = (
    "mlango", "fungul", "fungua", "door", "nyumba", "pesa", "pesa", "mkopo",
    "msaada", "saidia", "nitumie", "niletee", "kuja", "nakuja",
)

SWAHILI_GREETINGS = (
    "habari", "habari yako", "habari za", "vipi", "mambo", "mambo vipi", "hujambo",
    "salama", "niaje", "vp", "umeamkaje", "umeshindaje", "unaendeleaje",
    "shikamoo", "marahaba", "vipi kk", "mambo kk", "vp kk", "umeamkaje kk",
    "habar", "habari za asubuhi", "habari za mchana", "habari za jioni",
)

SWAHILI_THANKS = ("asante", "ashante", "asante sana", "nashukuru")
SWAHILI_GOODBYES = (
    "kwaheri", "baadaye", "usiku mwema", "lala salama", "tuonane", "tutaonana",
)
SWAHILI_SMALLTALK = ("mambo vipi", "vipi", "habari", "vp", "poa", "salama")

SWAHILI_FACTUAL_MARKERS = (
    "ni nini",
    "maana ya",
    "maana yake",
    "eleza",
    "nieleze",
    "nini maana",
    "kwa nini",
    "kwa nini",
    "je ni",
)

LANGUAGE_SW = "sw"
LANGUAGE_EN = "en"
LANGUAGE_MIXED = "mixed"


def _tokenize_lower(text: str) -> list[str]:
    return re.findall(r"[a-z']+", (text or "").lower())


def detect_reply_language(
    text: str,
    recent_user_messages: list[str] | None = None,
) -> str:
    """
    Return sw | en | mixed for reply language selection.
    Uses current message first, then recent user history if ambiguous.
    """
    lang = _language_from_text(text)
    if lang != LANGUAGE_EN:
        return lang

    if recent_user_messages:
        recent_langs = [_language_from_text(m) for m in recent_user_messages[-4:] if m.strip()]
        swish = sum(1 for lg in recent_langs if lg in (LANGUAGE_SW, LANGUAGE_MIXED))
        if swish >= 1:
            return LANGUAGE_SW if swish >= 2 or lang == LANGUAGE_EN else LANGUAGE_MIXED

    return LANGUAGE_EN


def _language_from_text(text: str) -> str:
    lower = (text or "").lower().strip()
    if not lower:
        return LANGUAGE_EN

    if any(marker in lower for marker in SWAHILI_MARKERS):
        if any(g in lower for g in ("hey", "hi", "hello", "how are you", "bro")):
            return LANGUAGE_MIXED
        return LANGUAGE_SW

    for greeting in SWAHILI_GREETINGS:
        if lower == greeting or lower.startswith(greeting + " ") or lower.endswith(" " + greeting):
            return LANGUAGE_SW

    words = _tokenize_lower(lower)
    if not words:
        return LANGUAGE_EN

    sw_hits = sum(1 for w in words if w in SWAHILI_WORDS)
    sw_marker_hits = sum(1 for m in SWAHILI_FACTUAL_MARKERS if m in lower)

    en_greeting = any(
        g in lower for g in ("hey", "hi", "hello", "what's up", "whats up", "how are you")
    )

    if sw_marker_hits >= 1 or sw_hits >= 2:
        if en_greeting or (sw_hits >= 1 and en_greeting):
            return LANGUAGE_MIXED
        if sw_hits >= 1 and len(words) <= 4:
            return LANGUAGE_SW
        if sw_hits >= 2:
            return LANGUAGE_SW
        return LANGUAGE_MIXED

    if sw_hits == 1:
        if len(words) <= 3:
            return LANGUAGE_SW
        return LANGUAGE_MIXED

    return LANGUAGE_EN


def is_swahili_message(text: str) -> bool:
    return detect_reply_language(text) in (LANGUAGE_SW, LANGUAGE_MIXED)


def reply_language_instruction(lang: str) -> str:
    if lang == LANGUAGE_SW:
        return (
            "IMPORTANT — They wrote in Kiswahili. You MUST reply in natural casual Kiswahili "
            "(Tanzania/Kenya SMS style — like texting a friend).\n"
            "Good examples:\n"
            '- "Mambo vipi! Umeshindaje?"\n'
            '- "Salama, habari yako?"\n'
            '- "Sawa kijana, nakuja sasa."\n'
            '- "Poa, nieleze zaidi?"\n'
            '- "Sijui kabisa — ni kwa ajili ya nini?"\n'
            "Do NOT reply in English. Avoid formal textbook Swahili or long preachy sentences."
        )
    if lang == LANGUAGE_MIXED:
        return (
            "They mix Kiswahili and English (Sheng). Match that mix naturally.\n"
            'Examples: "Poa bro, nimeelewa." / "Sawa, tell me more?" / "Mambo, how\'s it going?"'
        )
    return ""


SWAHILI_STYLE_GUIDE = """Kiswahili SMS style guide:
- Short, warm, casual — 1-2 sentences max
- Use: mambo, vipi, poa, sawa, salama, nakuja, sijui, pole, karibu, ndio, kijana
- OK to use: bro, kk, poa, freshi with Swahili (East African texting)
- Avoid: formal letters, English-only replies, changing topic, AI-sounding phrases
"""

SWAHILI_FEW_SHOT = (
    ("Mambo", "Mambo vipi! Umeshindaje?"),
    ("Habari yako", "Salama, habari yako? Niko poa."),
    ("Vp kk umeamkaje", "Poa bro, nimeamka salama. Wewe je?"),
    ("Naomba unifungulie mlango", "Sawa kijana, nakuja kukufungulia."),
    ("Asante sana", "Karibu!"),
    ("biology ni nini", "Biology ni somo la vitu vilivyo hai — mimea na wanyama. Inaeleweka?"),
    ("Nimechoka leo", "Pole, siku ndefu? Pumzika kidogo."),
)


def reply_has_swahili(text: str) -> bool:
    """True if reply contains recognizable Kiswahili (for verification)."""
    lower = (text or "").lower()
    if any(m in lower for m in SWAHILI_MARKERS):
        return True
    words = set(_tokenize_lower(lower))
    if words & SWAHILI_WORDS:
        return True
    if re.search(r"\b(ni|na|ya|wa|kwa|m|n|u|ku|mu)\b", lower) and len(words) >= 3:
        hits = sum(1 for w in words if w in SWAHILI_WORDS or len(w) > 4)
        if hits >= 2:
            return True
    return False


def verification_language_note(lang: str) -> str:
    if lang == LANGUAGE_SW:
        return "- Reply is in appropriate Kiswahili (matches their language)"
    if lang == LANGUAGE_MIXED:
        return "- Reply matches their Kiswahili/English mix"
    return ""


def is_practical_request(text: str) -> bool:
    lower = (text or "").lower()
    if any(marker in lower for marker in REQUEST_MARKERS):
        return True
    if re.search(r"fungul(?:ie|ia|i)|fungua", lower):
        return True
    return False


def _topic_tokens(text: str) -> set[str]:
    words = set(_tokenize_lower(text))
    topics: set[str] = set()
    lower = (text or "").lower()
    for w in words:
        if len(w) >= 4:
            topics.add(w)
    for marker in REQUEST_TOPIC_WORDS:
        if marker in lower:
            topics.add(marker)
    return topics


def reply_addresses_message(
    incoming: str,
    reply: str,
    *,
    situation: str = "general",
) -> tuple[bool, str]:
    """Rules-based check that reply relates to incoming (catches Ollama mistakes)."""
    lower_in = (incoming or "").lower()
    lower_rep = (reply or "").lower()
    if not lower_rep:
        return False, "empty reply"

    # ── Critical: reject replies that echo the user's own words back ─────
    in_tokens = set(_tokenize_lower(incoming))
    rep_tokens = set(_tokenize_lower(reply))
    if in_tokens and rep_tokens and len(in_tokens) >= 5:
        overlap_ratio = len(in_tokens & rep_tokens) / max(len(in_tokens), 1)
        if overlap_ratio > 0.7:
            return False, "reply echoes user message too closely"

    if situation == "request" or is_practical_request(incoming):
        door_request = "mlango" in lower_in or "fungul" in lower_in
        send_request = (
            ("nitumie" in lower_in or "niletee" in lower_in or "tuma" in lower_in)
            and not door_request
        )
        help_request = (
            ("nisaidie" in lower_in or "saidia" in lower_in)
            and not door_request
            and not send_request
        )

        decline_ok = any(w in lower_rep for w in ("siwezi", "hapana", "pole", "samahani"))

        if door_request:
            if any(
                w in lower_rep
                for w in ("mlango", "fungua", "fungul", "nafungua", "nakuja", "subiri")
            ):
                return True, "door request acknowledged"
            if decline_ok:
                return True, "request declined honestly"
            return False, "reply ignores door/open request"

        if send_request:
            if any(w in lower_rep for w in ("nitumie", "nituma", "tuma", "nakuja", "sawa", "ndio")):
                return True, "send request acknowledged"
            if decline_ok:
                return True, "request declined honestly"
            return False, "reply ignores send request"

        if help_request or "naomba" in lower_in:
            if any(
                w in lower_rep
                for w in ("saidia", "msaada", "naweza", "sawa", "ndio", "nieleze", "nini")
            ):
                return True, "help request acknowledged"
            if decline_ok:
                return True, "request declined honestly"
            return False, "reply ignores their request"

    ack_words = (
        "sawa", "ndio", "okay", "ok", "nakuja", "nitafanya", "basi", "pole",
        "samahani", "siwezi", "hapana", "karibu", "got it", "sure", "yeah",
    )
    in_topics = _topic_tokens(incoming)
    rep_words = set(_tokenize_lower(reply))

    stem_overlap = False
    for topic in in_topics:
        if topic in lower_rep:
            stem_overlap = True
            break
        for rw in rep_words:
            if len(topic) >= 4 and len(rw) >= 4 and topic[:4] == rw[:4]:
                stem_overlap = True
                break

    has_ack = any(a in lower_rep for a in ack_words)

    # For practical requests (even short ones), require relevance.
    if situation == "request" or is_practical_request(incoming):
        if stem_overlap or has_ack:
            return True, "reply relates to message"
        # Catch English door/open requests that slipped past the Swahili path above
        if any(w in lower_in for w in ("door", "open", "unlock", "gate", "let me in")):
            coming_words = ("coming", "one sec", "moment", "hold on", "way", "open")
            if any(w in lower_rep for w in coming_words):
                return True, "door/open request acknowledged"
            return False, "reply does not address open/door request"
        overlap = len(in_topics & rep_words)
        if overlap == 0 and not has_ack:
            return False, "reply unrelated to practical request"

    if len(incoming.strip()) > 30 and in_topics:
        if stem_overlap or has_ack:
            return True, "reply relates to message"
        overlap = len(in_topics & rep_words)
        if overlap == 0 and not has_ack:
            return False, "reply unrelated to message"

    # ── Check that reply doesn't reference topics NOT in the incoming ────
    irrelevant_topic_words = (
        "oxygen", "breakfast", "package", "mafurushi", "jifungua",
        "nyumba", "baraka", "shughuli", "vyumba",
    )
    for word in irrelevant_topic_words:
        if word in lower_rep and word not in lower_in:
            return False, f"reply introduces irrelevant topic: '{word}'"

    return True, "ok"
