"""HTTPS API server for the gbrain-sms Android companion app.

Runs on port 8787 alongside the main polling loop. Uses the same TLS cert/key
as the AudioLink server so the Android app's existing certificate pin already
trusts it — no extra setup on the phone.

Authentication: the phone sends the AudioLink bearer token it received during
pairing. We verify it by computing its SHA-256 hash and comparing against
token_hash values in ~/.local/share/gbrain/audiolink/devices.json.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import secrets
import sqlite3
import time
import urllib.error
import uuid
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Request
from pydantic import BaseModel

if TYPE_CHECKING:
    from bot.main import BotApp

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Auth helpers
# ---------------------------------------------------------------------------

def _hash_token(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def _load_devices(token_file: str) -> dict[str, dict]:
    """Load paired devices from AudioLink devices.json (sibling of admin.token)."""
    devices_path = Path(token_file).expanduser().parent / "devices.json"
    if not devices_path.exists():
        return {}
    try:
        data = json.loads(devices_path.read_text())
        return data.get("devices", {})
    except Exception as exc:
        logger.warning("Could not load devices.json: %s", exc)
        return {}


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class InboundSmsBody(BaseModel):
    sms_id: int
    from_number: str
    body: str
    timestamp_ms: int = 0


class ParseInstructionBody(BaseModel):
    instruction: str


class TaskBody(BaseModel):
    type: str            # "call" or "sms"
    number: str
    scheduled_at: str    # ISO-8601 datetime string
    message: str
    speak_delay_sec: float = 6.0
    repeat: int = 1


class GenerateReplyBody(BaseModel):
    contact: str
    body: str
    source: str = "sms"


class AiChatMessage(BaseModel):
    role: str   # "user" or "assistant"
    text: str


class AiChatBody(BaseModel):
    message: str
    history: list[AiChatMessage] = []   # recent turns for context
    device_id: str = ""
    # AudioLink stores and executes tasks on the phone itself (no ADB). Direct
    # callers that want gbrain-sms to run its own scheduler keep the default.
    schedule_local: bool = True


# ---------------------------------------------------------------------------
# AI assistant DB (per-user chat threads, stored in ai_chats table)
# ---------------------------------------------------------------------------

_AI_DB: "sqlite3.Connection | None" = None
_AI_DB_PATH: "Path | None" = None


def _get_ai_db(config) -> sqlite3.Connection:
    global _AI_DB, _AI_DB_PATH
    path = Path(config.db_path) if Path(config.db_path).is_absolute() \
        else Path(__file__).resolve().parent.parent / config.db_path
    if _AI_DB is None or _AI_DB_PATH != path:
        _AI_DB_PATH = path
        _AI_DB = sqlite3.connect(str(path), check_same_thread=False)
        _AI_DB.row_factory = sqlite3.Row
        _AI_DB.execute("PRAGMA journal_mode=WAL")
        _AI_DB.executescript("""
            CREATE TABLE IF NOT EXISTS ai_chats (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                device_id   TEXT    NOT NULL,
                role        TEXT    NOT NULL,
                text        TEXT    NOT NULL,
                task_json   TEXT,
                timestamp   REAL    NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_ai_chats_device
                ON ai_chats(device_id, timestamp DESC);
        """)
        _AI_DB.commit()
    return _AI_DB


def _save_ai_turn(db, device_id: str, role: str, text: str,
                  task_json: str | None = None) -> int:
    cur = db.execute(
        "INSERT INTO ai_chats (device_id, role, text, task_json, timestamp) "
        "VALUES (?, ?, ?, ?, ?)",
        (device_id, role, text, task_json, time.time()),
    )
    db.commit()
    return cur.lastrowid


def _load_ai_history(db, device_id: str, limit: int = 20) -> list[dict]:
    rows = db.execute(
        "SELECT role, text FROM ai_chats WHERE device_id=? "
        "ORDER BY timestamp DESC LIMIT ?",
        (device_id, limit),
    ).fetchall()
    return [{"role": r["role"], "text": r["text"]} for r in reversed(rows)]


# ---------------------------------------------------------------------------
# Ollama task-parse prompt
# ---------------------------------------------------------------------------

_PARSE_PROMPT = """\
Extract a scheduled task from the instruction below.
Reply with ONLY valid JSON — no explanation, no markdown fences.

Schema:
{{
  "type": "call" or "sms",
  "number": "<phone number in international format if possible>",
  "scheduled_at": "<ISO 8601 datetime, e.g. {today}T08:00:00>",
  "message": "<what to say on the call or the text to send>",
  "confidence": <0.0-1.0>
}}

Today is {today} in the computer's local timezone. Current local time is {now}.
Use exactly the clock time the user says and never convert timezones.
Instruction: {instruction}
JSON:"""

# Models tried in order for task parsing — qwen3.5:2b first (reliable JSON, fast to load)
_PARSE_MODELS = ["qwen3.5:2b", "gbrain-sms:latest", "gbrain:latest", "gbrainchat:latest"]


# ---------------------------------------------------------------------------
# AI assistant system prompt
# ---------------------------------------------------------------------------

_AI_SYSTEM_PROMPT = """\
You are GBrain Assistant, a smart personal AI assistant built into the GBrain phone app.
You help the user manage their calls, SMS messages, and scheduled tasks through natural conversation.

Today is {today}. Current time is {now}.

Your capabilities:
- Schedule phone calls at a specific time with a spoken message
- Schedule SMS messages to be sent at a specific time
- List or cancel existing scheduled tasks
- Answer general questions conversationally

When the user asks you to do something that maps to a task (call someone, send a message,
remind someone, etc.), you MUST:
1. Reply conversationally confirming what you understood
2. Include a JSON block at the very end of your reply, inside <TASK>...</TASK> tags

The JSON format inside <TASK> tags:
{{
  "type": "call" or "sms",
  "number": "<phone number>",
  "scheduled_at": "<ISO 8601, e.g. {today}T08:00:00>",
  "message": "<what to say or send>",
  "label": "<short human-readable label, e.g. Call mom at 8am>"
}}

Rules:
- Only include <TASK> if the user clearly wants to schedule something
- If the time is relative ("in 1 hour", "tomorrow morning"), resolve it to absolute time
- If a phone number is a name ("my mom", "John"), keep the name as the number field — the user will correct it
- If information is missing (no time, no number), ask the user for it instead of guessing
- Keep replies short and friendly — this is a chat, not an email
- You speak both English and Swahili — reply in the same language the user used
"""


def _parse_local_datetime(value: str) -> datetime | None:
    """Parse an ISO timestamp and convert it to the PC's local naive time."""
    raw = str(value or "").strip()
    if not raw:
        return None
    if raw.endswith(("Z", "z")):
        raw = raw[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(raw)
    except ValueError:
        return None
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone().replace(tzinfo=None)
    return parsed


_CLOCK_RE = re.compile(
    r"\b(?:at\s+)?(\d{1,2}):(\d{2})\s*(am|pm)?\b"
    r"|\b(?:at\s+)(\d{1,2})\s*(am|pm)\b",
    re.IGNORECASE,
)


def _apply_explicit_clock(instruction: str, scheduled_at: str) -> str:
    """Anchor a model-produced timestamp to the clock time the user typed.

    Small models occasionally shift times by the UTC offset (e.g. "12:45"
    becomes "15:45" in UTC+3). The user's literal clock time wins.
    """
    match = _CLOCK_RE.search(str(instruction or ""))
    if not match:
        return scheduled_at
    hour = int(match.group(1) or match.group(4) or 0)
    minute = int(match.group(2) or 0)
    marker = (match.group(3) or match.group(5) or "").lower()
    if marker == "pm" and hour < 12:
        hour += 12
    elif marker == "am" and hour == 12:
        hour = 0
    if hour > 23 or minute > 59:
        return scheduled_at

    parsed = _parse_local_datetime(scheduled_at)
    now = datetime.now()
    date = parsed.date() if parsed is not None else now.date()
    lowered = str(instruction or "").lower()
    if "tomorrow" in lowered:
        date = (now + timedelta(days=1)).date()
    elif "today" in lowered or "tonight" in lowered:
        date = now.date()
    candidate = datetime.combine(date, datetime.min.time()).replace(
        hour=hour, minute=minute
    )
    return candidate.isoformat()


_SMS_INTENT_RE = re.compile(
    r"\b(send|sms|text|message|reply|remind)\b", re.IGNORECASE
)
_CALL_INTENT_RE = re.compile(r"\b(call|dial|phone|ring)\b", re.IGNORECASE)


def _apply_explicit_type(instruction: str, draft: dict) -> dict:
    """Let clear words in the instruction decide SMS vs call."""
    text = str(instruction or "")
    if _SMS_INTENT_RE.search(text) and not _CALL_INTENT_RE.search(text):
        draft["type"] = "sms"
    elif _CALL_INTENT_RE.search(text) and not _SMS_INTENT_RE.search(text):
        draft["type"] = "call"
    return draft


_TASK_TAG = re.compile(r"<TASK>\s*(.*?)\s*</TASK>", re.DOTALL | re.IGNORECASE)
_FENCED = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)
_INLINE_TASK = re.compile(r"\{[^{}]*\"scheduled_at\"[^{}]*\}", re.DOTALL)


def _extract_task(reply: str) -> tuple[dict | None, str]:
    """Return (task_json, visible_reply) from a model reply.

    Accepts a <TASK>…</TASK> block (preferred), a fenced JSON block, or a bare
    JSON object that contains ``scheduled_at`` so small models that forget the
    tags still schedule correctly.
    """
    import json as _json

    match = _TASK_TAG.search(reply)
    if match:
        raw = match.group(1).strip()
        clean = (reply[: match.start()] + reply[match.end():]).strip()
    else:
        raw = ""
        clean = reply
    candidates: list[str] = []
    if raw:
        candidates.append(_FENCED.sub("", raw))
    candidates.extend(m.group(0) for m in _INLINE_TASK.finditer(reply))
    for candidate in candidates:
        try:
            data = _json.loads(candidate.strip())
        except (TypeError, ValueError):
            continue
        if isinstance(data, dict) and data.get("scheduled_at"):
            return data, clean
    return None, clean


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def create_gbrain_app(config, bot_app: "BotApp") -> FastAPI:
    app = FastAPI(title="GBrain Server", version="0.1.0")
    app.state.config = config
    app.state.bot_app = bot_app
    app.state.bridge = getattr(bot_app, "bridge", None)

    # -- dependency ----------------------------------------------------------
    async def require_device(request: Request) -> str:
        auth = request.headers.get("authorization", "")
        if not auth.lower().startswith("bearer "):
            raise HTTPException(401, "missing bearer token")
        token = auth[7:].strip()
        token_hash = _hash_token(token)

        # Accept the AudioLink admin token for server-to-server calls
        # (AudioLink calls /generate, /tasks/parse, /ai/chat on behalf of devices).
        admin_token_file = Path(config.audiolink_admin_token_file).expanduser()
        if admin_token_file.is_file():
            try:
                admin_token = admin_token_file.read_text(encoding="utf-8").strip()
                if admin_token and secrets.compare_digest(token, admin_token):
                    return "audiolink-server"
            except OSError:
                pass

        # Accept any paired device token
        devices = _load_devices(config.audiolink_admin_token_file)
        for device_id, device in devices.items():
            stored = device.get("token_hash", "")
            if stored and secrets.compare_digest(token_hash, stored):
                return device_id
        raise HTTPException(401, "unknown device token")

    # ── health ──────────────────────────────────────────────────────────────
    @app.get("/v1/gbrain/health")
    async def health():
        return {"status": "ok", "service": "gbrain-sms", "version": "0.1.0"}

    # ── SMS inbound ─────────────────────────────────────────────────────────
    @app.post("/v1/gbrain/sms/inbound")
    async def inbound_sms(
        body: InboundSmsBody,
        device_id: str = Depends(require_device),
    ):
        bot: BotApp = app.state.bot_app
        if bot.memory.is_sms_replied(body.sms_id):
            return {"status": "already_processed"}
        # Fire-and-forget through the existing pipeline
        asyncio.create_task(
            bot._handle_inbound(
                body.from_number, body.body, source="sms", sms_id=body.sms_id
            )
        )
        return {"status": "queued", "contact": body.from_number}

    # ── SMS history ──────────────────────────────────────────────────────────
    @app.get("/v1/gbrain/sms/history")
    async def sms_history(
        contact: str,
        limit: int = 20,
        device_id: str = Depends(require_device),
    ):
        bot: BotApp = app.state.bot_app
        messages = bot.memory.get_recent_context(contact, limit=limit)
        return {
            "contact": contact,
            "messages": [
                {
                    "role": m.role,
                    "body": m.body,
                    "timestamp": m.timestamp,
                    "source": getattr(m, "source", "sms"),
                }
                for m in messages
            ],
        }

    # ── Contact list ─────────────────────────────────────────────────────────
    @app.get("/v1/gbrain/sms/contacts")
    async def sms_contacts(device_id: str = Depends(require_device)):
        bot: BotApp = app.state.bot_app
        contacts = bot.memory.list_contacts()
        return {"contacts": contacts}

    # ── Reply generation (called by AudioLink) ───────────────────────────
    @app.post("/v1/gbrain/sms/generate")
    async def generate_reply(
        body: GenerateReplyBody,
        device_id: str = Depends(require_device),
    ):
        """Generate a reply for an inbound message. AudioLink calls this; no ADB."""
        bot: BotApp = app.state.bot_app
        loop = asyncio.get_running_loop()
        # Save inbound to memory so pipeline has conversation context
        bot.memory.save_message(body.contact, "user", body.body, source=body.source)
        bot.engine.extract_preferences(body.contact, body.body)
        if bot.config.agent_pipeline_enabled:
            result = await loop.run_in_executor(
                None,
                lambda: bot.pipeline.process(body.contact, body.body, source=body.source),
            )
            if not result.approved:
                raise HTTPException(422, detail="pipeline did not approve a reply")
            reply_text = result.text
        else:
            result = await loop.run_in_executor(
                None,
                lambda: bot.engine.generate_reply(body.contact, body.body, source=body.source),
            )
            reply_text = result.text
        bot.memory.save_message(body.contact, "bot", reply_text, source=body.source)
        return {"reply": reply_text, "contact": body.contact}

    # ── NL instruction → TaskDraft ────────────────────────────────────────
    @app.post("/v1/gbrain/tasks/parse")
    async def parse_task(
        body: ParseInstructionBody,
        device_id: str = Depends(require_device),
    ):
        bot: BotApp = app.state.bot_app
        today = datetime.now().strftime("%Y-%m-%d")
        now = datetime.now().strftime("%H:%M")
        prompt = _PARSE_PROMPT.format(
            today=today, now=now, instruction=body.instruction
        )
        # Try models in order until one returns valid JSON.
        # _PARSE_MODELS are ordered for reliable structured output — do not prepend
        # the SMS persona model (gbrain-sms) since it is not trained for JSON extraction.
        seen_parse: set[str] = set()
        models_to_try = [m for m in _PARSE_MODELS if not (m in seen_parse or seen_parse.add(m))]
        for model in models_to_try:
            payload = json.dumps(
                {
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "think": False,
                    "options": {"temperature": 0.0, "num_predict": 200},
                }
            ).encode()
            req = urllib.request.Request(
                f"{bot.config.ollama_url}/api/generate",
                data=payload,
                headers={"Content-Type": "application/json"},
            )
            try:
                with urllib.request.urlopen(req, timeout=90) as resp:
                    result = json.loads(resp.read())
                raw = result.get("response", "").strip()
                # Strip any thinking trace from reasoning models
                import re as _re_parse
                raw = _re_parse.sub(r"<think>.*?</think>", "", raw, flags=_re_parse.DOTALL).strip()
                raw = _re_parse.sub(r"(?s)^(?:Thinking Process|Thinking)\s*:?\s*\n.+?(?=\{)", "", raw).strip()
                if not raw:
                    logger.debug("parse: model %s returned empty response, trying next", model)
                    continue
                start = raw.find("{")
                end = raw.rfind("}") + 1
                if start >= 0 and end > start:
                    draft = json.loads(raw[start:end])
                    if isinstance(draft, dict) and draft.get("scheduled_at"):
                        draft["scheduled_at"] = _apply_explicit_clock(
                            body.instruction, str(draft["scheduled_at"])
                        )
                        _apply_explicit_type(body.instruction, draft)
                    logger.info("Task parsed by model %s: %s", model, draft)
                    return {"task_draft": draft, "raw_instruction": body.instruction}
                logger.debug("parse: model %s returned no JSON: %r", model, raw[:80])
            except (urllib.error.URLError, json.JSONDecodeError, Exception) as exc:
                logger.warning("Task parse model %s failed: %s", model, exc)
        return {
            "error": "could not parse instruction",
            "raw_instruction": body.instruction,
        }

    # ── Task CRUD ─────────────────────────────────────────────────────────
    @app.post("/v1/gbrain/tasks")
    async def create_task(
        body: TaskBody,
        device_id: str = Depends(require_device),
    ):
        from bot.scheduler import ScheduledCall, ScheduledReply

        bot: BotApp = app.state.bot_app
        try:
            send_at = datetime.fromisoformat(body.scheduled_at)
        except ValueError:
            raise HTTPException(400, f"invalid scheduled_at: {body.scheduled_at!r}")

        if body.type == "sms":
            task = ScheduledReply(
                contact=body.number,
                number=body.number,
                body=body.message,
                send_at=send_at,
                source="app",
            )
        elif body.type == "call":
            task = ScheduledCall(
                contact=body.number,
                number=body.number,
                message=body.message,
                send_at=send_at,
                speak_delay_sec=body.speak_delay_sec,
                repeat=body.repeat,
            )
        else:
            raise HTTPException(400, f"unknown task type: {body.type!r}")

        task_id = await bot.scheduler.schedule(task)
        return {
            "task_id": task_id,
            "scheduled_at": send_at.isoformat(),
            "type": body.type,
        }

    @app.get("/v1/gbrain/tasks")
    async def list_tasks(device_id: str = Depends(require_device)):
        bot: BotApp = app.state.bot_app
        tasks = bot.scheduler.list_pending()
        return {
            "tasks": [
                {
                    "task_id": t.task_id,
                    "type": getattr(t, "task_type", "sms"),
                    "number": t.number,
                    "message": getattr(t, "body", getattr(t, "message", "")),
                    "scheduled_at": t.send_at.isoformat(),
                }
                for t in tasks
            ]
        }

    @app.delete("/v1/gbrain/tasks/{task_id}")
    async def cancel_task(
        task_id: str,
        device_id: str = Depends(require_device),
    ):
        bot: BotApp = app.state.bot_app
        if not bot.scheduler.cancel(task_id):
            raise HTTPException(404, f"task {task_id!r} not found")
        return {"cancelled": task_id}

    # ── AI assistant chat ─────────────────────────────────────────────────
    @app.post("/v1/gbrain/ai/chat")
    async def ai_chat(
        body: AiChatBody,
        device_id: str = Depends(require_device),
    ):
        """
        Conversational AI endpoint. The assistant replies in natural language
        and embeds a <TASK>…</TASK> JSON block when the user wants to schedule
        a call or SMS. The task is automatically stored in the scheduler and
        returned in the response so the app can show a confirmation card.
        """
        bot: BotApp = app.state.bot_app
        db = _get_ai_db(bot.config)
        # AudioLink calls with the phone's id in the body; keep one thread per
        # phone instead of one shared "audiolink-server" thread.
        thread_id = body.device_id or device_id

        today = datetime.now().strftime("%Y-%m-%d")
        now   = datetime.now().strftime("%H:%M")
        system = _AI_SYSTEM_PROMPT.format(today=today, now=now)

        # Build message history: load from DB then append this turn
        history = _load_ai_history(db, thread_id, limit=16)

        # Build Ollama messages array
        messages: list[dict] = [{"role": "system", "content": system}]
        for turn in history:
            messages.append({
                "role": turn["role"],
                "content": turn["text"],
            })
        messages.append({"role": "user", "content": body.message})

        # Save user turn
        _save_ai_turn(db, thread_id, "user", body.message)

        # Try chat-capable models in order
        models_to_try = ["gbrainchat:latest", "gbrain:latest", "qwen3.5:2b",
                         "gbrain-sms:latest", bot.config.ollama_model]
        seen: set[str] = set()
        ordered = [m for m in models_to_try if not (m in seen or seen.add(m))]

        # Build a single-string prompt — all models here use {{ .Prompt }} template
        prompt_parts = [system, ""]
        for turn in history:
            label = "User" if turn["role"] == "user" else "GBrain"
            prompt_parts.append(f"{label}: {turn['text']}")
        prompt_parts.append(f"User: {body.message}")
        prompt_parts.append("GBrain:")
        full_prompt = "\n".join(prompt_parts)

        import re as _re_strip

        reply_text = ""
        for model in ordered:
            payload = json.dumps({
                "model": model,
                "prompt": full_prompt,
                "stream": False,
                "think": False,
                "options": {"temperature": 0.7, "num_predict": 400},
            }).encode()
            req = urllib.request.Request(
                f"{bot.config.ollama_url}/api/generate",
                data=payload,
                headers={"Content-Type": "application/json"},
            )
            try:
                with urllib.request.urlopen(req, timeout=90) as resp:
                    result = json.loads(resp.read())
                # Qwen3 / thinking models put output in "thinking" when response=""
                raw = result.get("response", "").strip()
                if not raw:
                    raw = result.get("thinking", "").strip()
                # Strip any <think>…</think> wrapper
                raw = _re_strip.sub(r"<think>.*?</think>", "", raw,
                                    flags=_re_strip.DOTALL).strip()
                # Strip naked reasoning traces that some Qwen3 Modelfiles leak:
                # "Thinking Process:\n..." or "**Thinking:**\n..." before the answer
                raw = _re_strip.sub(
                    r"(?s)^(?:Thinking Process|Thinking|Internal Monologue)"
                    r"\s*:?\s*\*{0,2}\s*\n.+?(?=\n[A-Z*]|\Z)",
                    "",
                    raw,
                ).strip()
                # If the reply still has a "Response:" or "Answer:" header, strip it
                raw = _re_strip.sub(
                    r"^(?:Response|Answer|GBrain)\s*:\s*",
                    "",
                    raw,
                    flags=_re_strip.IGNORECASE,
                ).strip()
                if raw:
                    reply_text = raw
                    logger.info("AI reply from model %s (%d chars)", model, len(reply_text))
                    break
                logger.debug("AI model %s returned empty", model)
            except Exception as exc:
                logger.warning("AI model %s failed: %s", model, exc)

        if not reply_text:
            reply_text = "Sorry, I could not reach the AI right now. Please try again."

        # Extract a task block from the reply (tagged or bare JSON).
        task_data, clean_reply = _extract_task(reply_text)
        if task_data is not None:
            task_data["scheduled_at"] = _apply_explicit_clock(
                body.message, str(task_data.get("scheduled_at", ""))
            )
            _apply_explicit_type(body.message, task_data)
            sched_at_str = str(task_data.get("scheduled_at", ""))
            send_at = _parse_local_datetime(sched_at_str)
            task_type = "call" if task_data.get("type") == "call" else "sms"
            number = str(task_data.get("number", "")).strip()
            message = str(task_data.get("message", "")).strip()
            if send_at is not None and number and message:
                if body.schedule_local:
                    from bot.scheduler import ScheduledCall, ScheduledReply

                    if task_type == "call":
                        task = ScheduledCall(
                            contact=number, number=number,
                            message=message, send_at=send_at,
                        )
                    else:
                        task = ScheduledReply(
                            contact=number, number=number,
                            body=message, send_at=send_at, source="app",
                        )
                    task_id = await bot.scheduler.schedule(task)
                    task_data["task_id"] = task_id
                    task_data["scheduled"] = True
                    task_data["executor"] = "gbrain-sms"
                    logger.info(
                        "AI created task %s: %s → %s at %s",
                        task_id, task_type, number, sched_at_str,
                    )
                else:
                    # AudioLink owns execution: it stores the task and the
                    # paired phone sends the SMS via SmsManager (no ADB).
                    task_data["task_id"] = uuid.uuid4().hex[:8]
                    task_data["scheduled"] = False
                    task_data["executor"] = "audiolink"
                    task_data["scheduled_at"] = send_at.isoformat()
                    logger.info(
                        "AI task drafted for AudioLink: %s → %s at %s",
                        task_type, number, send_at.isoformat(),
                    )
            else:
                task_data["scheduled"] = False
                task_data["reason"] = (
                    "missing number/message" if not (number and message)
                    else "could not parse scheduled_at"
                )

        # Save assistant turn (clean reply only)
        _save_ai_turn(
            db, thread_id, "assistant", clean_reply,
            task_json=json.dumps(task_data) if task_data else None,
        )

        return {
            "reply": clean_reply,
            "task": task_data,  # None if no task was detected
        }

    @app.get("/v1/gbrain/ai/history")
    async def ai_history(
        limit: int = 30,
        device_id: str = Depends(require_device),
    ):
        """Return the AI chat thread for this device."""
        bot: BotApp = app.state.bot_app
        db = _get_ai_db(bot.config)
        rows = db.execute(
            "SELECT role, text, task_json, timestamp FROM ai_chats "
            "WHERE device_id=? ORDER BY timestamp DESC LIMIT ?",
            (device_id, limit),
        ).fetchall()
        return {
            "messages": [
                {
                    "role":      r["role"],
                    "text":      r["text"],
                    "task":      json.loads(r["task_json"]) if r["task_json"] else None,
                    "timestamp": r["timestamp"],
                }
                for r in reversed(rows)
            ]
        }

    @app.delete("/v1/gbrain/ai/history")
    async def clear_ai_history(device_id: str = Depends(require_device)):
        """Clear the AI chat thread for this device (fresh start)."""
        bot: BotApp = app.state.bot_app
        db = _get_ai_db(bot.config)
        db.execute("DELETE FROM ai_chats WHERE device_id=?", (device_id,))
        db.commit()
        return {"cleared": True}

    bridge = getattr(bot_app, "bridge", None)
    if bridge is not None:
        from bot.bridge.api import create_app as create_bridge_app

        create_bridge_app(bridge, app=app)

    return app


# ---------------------------------------------------------------------------
# Server runner
# ---------------------------------------------------------------------------

async def start_api_server(app: FastAPI, config) -> None:
    cert_path = Path(config.audiolink_ca_file).expanduser()
    key_path = cert_path.parent / "key.pem"

    ssl_kwargs: dict[str, Any] = {}
    if cert_path.exists() and key_path.exists():
        ssl_kwargs = {
            "ssl_certfile": str(cert_path),
            "ssl_keyfile": str(key_path),
        }
        logger.info("GBrain API: TLS enabled (%s)", cert_path)
    else:
        logger.warning(
            "GBrain API: TLS cert/key not found at %s — running HTTP only", cert_path
        )

    cfg = uvicorn.Config(
        app,
        host="0.0.0.0",
        port=config.gbrain_api_port,
        log_level="warning",
        **ssl_kwargs,
    )
    server = uvicorn.Server(cfg)
    logger.info("GBrain API server starting on port %d", config.gbrain_api_port)
    await server.serve()
