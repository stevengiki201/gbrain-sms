"""SQLite conversation memory with simple retrieval."""

from __future__ import annotations

import json
import re
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def normalize_contact(contact: str) -> str:
    """Canonical phone/contact key so +255… and 07… share one thread."""
    raw = (contact or "").strip()
    digits = re.sub(r"\D", "", raw)
    if not digits:
        return raw
    if digits.startswith("255") and len(digits) >= 12:
        return f"+{digits}"
    if digits.startswith("0") and len(digits) >= 10:
        return f"+255{digits[1:]}"
    if raw.startswith("+"):
        return f"+{digits}"
    return raw


def contact_variants(contact: str) -> list[str]:
    raw = (contact or "").strip()
    canonical = normalize_contact(contact)
    digits = re.sub(r"\D", "", canonical)
    variants = {raw, canonical, digits}
    variants.discard("")
    if digits.startswith("255") and len(digits) > 3:
        variants.add(f"+{digits}")
        variants.add("0" + digits[3:])
    return list(variants)


@dataclass
class Message:
    id: int
    contact: str
    role: str
    body: str
    timestamp: float
    source: str = "sms"
    topic: str | None = None


@dataclass
class Preference:
    contact: str
    key: str
    value: str
    updated_at: float


@dataclass
class ContactProfile:
    contact: str
    key: str
    value: str
    confidence: float
    source: str
    updated_at: float


@dataclass
class ReplyStudy:
    id: int
    contact: str
    inbound_body: str
    reply_body: str
    situation: str
    route: str
    backend: str
    created_at: float


@dataclass
class TrainedReply:
    id: int
    inbound_sample: str
    reply_body: str
    situation: str
    topic: str
    notes: str
    trained_via_contact: str
    apply_globally: bool
    created_at: float
    updated_at: float


class MemoryStore:
    SCHEMA = """
    CREATE TABLE IF NOT EXISTS messages (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        contact TEXT NOT NULL,
        role TEXT NOT NULL,
        body TEXT NOT NULL,
        timestamp REAL NOT NULL,
        source TEXT DEFAULT 'sms',
        topic TEXT
    );
    CREATE INDEX IF NOT EXISTS idx_messages_contact_ts
        ON messages(contact, timestamp);

    CREATE TABLE IF NOT EXISTS preferences (
        contact TEXT NOT NULL,
        key TEXT NOT NULL,
        value TEXT NOT NULL,
        updated_at REAL NOT NULL,
        PRIMARY KEY (contact, key)
    );

    CREATE TABLE IF NOT EXISTS processed_sms (
        sms_id INTEGER PRIMARY KEY,
        processed_at REAL NOT NULL
    );

    CREATE TABLE IF NOT EXISTS sms_replies (
        sms_id INTEGER PRIMARY KEY,
        contact TEXT NOT NULL,
        inbound_body TEXT NOT NULL,
        reply_body TEXT NOT NULL,
        replied_at REAL NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_sms_replies_contact
        ON sms_replies(contact, replied_at);

    CREATE TABLE IF NOT EXISTS processed_whatsapp (
        dedup_id TEXT PRIMARY KEY,
        processed_at REAL NOT NULL
    );

    CREATE TABLE IF NOT EXISTS state (
        key TEXT PRIMARY KEY,
        value TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS contact_profiles (
        contact TEXT NOT NULL,
        key TEXT NOT NULL,
        value TEXT NOT NULL,
        confidence REAL NOT NULL DEFAULT 0.7,
        source TEXT NOT NULL DEFAULT 'auto',
        updated_at REAL NOT NULL,
        PRIMARY KEY (contact, key)
    );

    CREATE TABLE IF NOT EXISTS verification_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        contact TEXT NOT NULL,
        inbound_body TEXT NOT NULL,
        draft_reply TEXT NOT NULL,
        situation TEXT,
        route TEXT,
        attempt INTEGER NOT NULL DEFAULT 1,
        verifier TEXT NOT NULL,
        passed INTEGER NOT NULL,
        reason TEXT,
        ollama_used INTEGER NOT NULL DEFAULT 0,
        created_at REAL NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_verification_log_contact
        ON verification_log(contact, created_at DESC);

    CREATE TABLE IF NOT EXISTS reply_study (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        contact TEXT NOT NULL,
        inbound_body TEXT NOT NULL,
        reply_body TEXT NOT NULL,
        situation TEXT NOT NULL,
        route TEXT,
        backend TEXT,
        created_at REAL NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_reply_study_situation
        ON reply_study(situation, created_at DESC);
    CREATE INDEX IF NOT EXISTS idx_reply_study_contact
        ON reply_study(contact, created_at DESC);

    CREATE TABLE IF NOT EXISTS situation_notes (
        contact TEXT NOT NULL,
        situation TEXT NOT NULL,
        note TEXT NOT NULL,
        updated_at REAL NOT NULL,
        PRIMARY KEY (contact, situation)
    );

    CREATE TABLE IF NOT EXISTS trained_replies (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        inbound_sample TEXT NOT NULL,
        reply_body TEXT NOT NULL,
        situation TEXT NOT NULL DEFAULT 'general',
        topic TEXT NOT NULL DEFAULT '',
        notes TEXT NOT NULL DEFAULT '',
        trained_via_contact TEXT NOT NULL DEFAULT '',
        apply_globally INTEGER NOT NULL DEFAULT 1,
        created_at REAL NOT NULL,
        updated_at REAL NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_trained_situation
        ON trained_replies(situation, updated_at DESC);
    """

    def __init__(self, db_path: Path) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.db_path = db_path
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        # Enable WAL mode for better concurrent access and durability
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.executescript(self.SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        # Perform a WAL checkpoint before closing for data integrity
        try:
            self._conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        except Exception:
            pass
        self._conn.close()

    def save_message(
        self,
        contact: str,
        role: str,
        body: str,
        *,
        source: str = "sms",
        topic: str | None = None,
        timestamp: float | None = None,
    ) -> int:
        ts = timestamp if timestamp is not None else time.time()
        contact = normalize_contact(contact)
        cur = self._conn.execute(
            """
            INSERT INTO messages (contact, role, body, timestamp, source, topic)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (contact, role, body, ts, source, topic),
        )
        self._conn.commit()
        return int(cur.lastrowid)

    def get_recent_context(self, contact: str, limit: int = 10) -> list[Message]:
        variants = contact_variants(contact)
        placeholders = ",".join("?" * len(variants))
        rows = self._conn.execute(
            f"""
            SELECT id, contact, role, body, timestamp, source, topic
            FROM messages
            WHERE contact IN ({placeholders})
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            (*variants, limit),
        ).fetchall()
        messages = [self._row_to_message(r) for r in reversed(rows)]
        return messages

    def search_similar(
        self, contact: str, query: str, limit: int = 5
    ) -> list[Message]:
        tokens = _tokenize(query)
        if not tokens:
            return []

        variants = contact_variants(contact)
        placeholders = ",".join("?" * len(variants))
        rows = self._conn.execute(
            f"""
            SELECT id, contact, role, body, timestamp, source, topic
            FROM messages
            WHERE contact IN ({placeholders}) AND role = 'user'
            ORDER BY timestamp DESC
            LIMIT 200
            """,
            tuple(variants),
        ).fetchall()

        scored: list[tuple[int, Message]] = []
        for row in rows:
            body = row["body"] or ""
            body_tokens = set(_tokenize(body))
            overlap = len(body_tokens & tokens)
            if overlap > 0:
                scored.append((overlap, self._row_to_message(row)))

        scored.sort(key=lambda x: (-x[0], -x[1].timestamp))
        return [msg for _, msg in scored[:limit]]

    def save_preference(self, contact: str, key: str, value: str) -> None:
        contact = normalize_contact(contact)
        self._conn.execute(
            """
            INSERT INTO preferences (contact, key, value, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(contact, key) DO UPDATE SET
                value = excluded.value,
                updated_at = excluded.updated_at
            """,
            (contact, key, value, time.time()),
        )
        self._conn.commit()

    def get_preferences(self, contact: str) -> list[Preference]:
        variants = contact_variants(contact)
        placeholders = ",".join("?" * len(variants))
        rows = self._conn.execute(
            f"SELECT contact, key, value, updated_at FROM preferences WHERE contact IN ({placeholders})",
            tuple(variants),
        ).fetchall()
        return [
            Preference(
                contact=r["contact"],
                key=r["key"],
                value=r["value"],
                updated_at=r["updated_at"],
            )
            for r in rows
        ]

    def is_sms_processed(self, sms_id: int) -> bool:
        row = self._conn.execute(
            "SELECT 1 FROM processed_sms WHERE sms_id = ?", (sms_id,)
        ).fetchone()
        return row is not None

    def mark_sms_processed(self, sms_id: int) -> None:
        self._conn.execute(
            "INSERT OR IGNORE INTO processed_sms (sms_id, processed_at) VALUES (?, ?)",
            (sms_id, time.time()),
        )
        self._conn.commit()

    def list_contacts(self) -> list[dict]:
        """Return distinct contacts with last activity and message count, most recent first."""
        rows = self._conn.execute(
            """
            SELECT contact, MAX(timestamp) as last_ts, COUNT(*) as count
            FROM messages
            GROUP BY contact
            ORDER BY last_ts DESC
            LIMIT 100
            """
        ).fetchall()
        return [
            {"contact": r["contact"], "last_ts": r["last_ts"], "count": r["count"]}
            for r in rows
        ]

    def is_sms_replied(self, sms_id: int) -> bool:
        row = self._conn.execute(
            "SELECT 1 FROM sms_replies WHERE sms_id = ?", (sms_id,)
        ).fetchone()
        return row is not None

    def mark_sms_replied(
        self,
        sms_id: int,
        contact: str,
        inbound_body: str,
        reply_body: str,
    ) -> None:
        contact = normalize_contact(contact)
        self._conn.execute(
            """
            INSERT OR REPLACE INTO sms_replies
                (sms_id, contact, inbound_body, reply_body, replied_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (sms_id, contact, inbound_body, reply_body, time.time()),
        )
        self._conn.commit()

    def conversation_awaiting_reply(self, contact: str) -> bool:
        """True when the latest stored turn is from the user (no bot reply yet)."""
        variants = contact_variants(contact)
        placeholders = ",".join("?" * len(variants))
        row = self._conn.execute(
            f"""
            SELECT role FROM messages
            WHERE contact IN ({placeholders})
            ORDER BY timestamp DESC
            LIMIT 1
            """,
            tuple(variants),
        ).fetchone()
        if row is None:
            return True
        return row["role"] == "user"

    def is_whatsapp_processed(self, notif_key: str, text: str) -> bool:
        dedup_id = f"{notif_key}:{text}"
        row = self._conn.execute(
            "SELECT 1 FROM processed_whatsapp WHERE dedup_id = ?", (dedup_id,)
        ).fetchone()
        return row is not None

    def mark_whatsapp_processed(self, notif_key: str, text: str) -> None:
        dedup_id = f"{notif_key}:{text}"
        self._conn.execute(
            "INSERT OR IGNORE INTO processed_whatsapp (dedup_id, processed_at) VALUES (?, ?)",
            (dedup_id, time.time()),
        )
        self._conn.commit()

    def get_state(self, key: str, default: str = "") -> str:
        row = self._conn.execute(
            "SELECT value FROM state WHERE key = ?", (key,)
        ).fetchone()
        return row["value"] if row else default

    def set_state(self, key: str, value: str) -> None:
        self._conn.execute(
            """
            INSERT INTO state (key, value) VALUES (?, ?)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value
            """,
            (key, value),
        )
        self._conn.commit()

    def get_last_seen_sms_id(self) -> int:
        raw = self.get_state("last_seen_sms_id", "0")
        try:
            return int(raw)
        except ValueError:
            return 0

    def set_last_seen_sms_id(self, sms_id: int) -> None:
        self.set_state("last_seen_sms_id", str(sms_id))

    def message_count(self, contact: str | None = None) -> int:
        if contact:
            row = self._conn.execute(
                "SELECT COUNT(*) AS c FROM messages WHERE contact = ?", (contact,)
            ).fetchone()
        else:
            row = self._conn.execute("SELECT COUNT(*) AS c FROM messages").fetchone()
        return int(row["c"])

    def export_json(self, path: Path) -> None:
        rows = self._conn.execute(
            """
            SELECT contact, role, body, timestamp, source, topic
            FROM messages ORDER BY timestamp
            """
        ).fetchall()
        data = [dict(r) for r in rows]
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, ensure_ascii=False)

    # --- Contact learning profiles ---

    def save_profile(
        self,
        contact: str,
        key: str,
        value: str,
        *,
        confidence: float = 0.7,
        source: str = "auto",
    ) -> None:
        contact = normalize_contact(contact)
        key = key.strip().lower().replace(" ", "_")
        value = value.strip()
        if not key or not value:
            return
        row = self._conn.execute(
            "SELECT confidence FROM contact_profiles WHERE contact = ? AND key = ?",
            (contact, key),
        ).fetchone()
        if row and float(row["confidence"]) > confidence:
            confidence = float(row["confidence"])
        self._conn.execute(
            """
            INSERT INTO contact_profiles (contact, key, value, confidence, source, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(contact, key) DO UPDATE SET
                value = excluded.value,
                confidence = excluded.confidence,
                source = excluded.source,
                updated_at = excluded.updated_at
            """,
            (contact, key, value, confidence, source, time.time()),
        )
        self._conn.commit()

    def get_profile(self, contact: str) -> list[ContactProfile]:
        variants = contact_variants(contact)
        placeholders = ",".join("?" * len(variants))
        rows = self._conn.execute(
            f"""
            SELECT contact, key, value, confidence, source, updated_at
            FROM contact_profiles
            WHERE contact IN ({placeholders})
            ORDER BY updated_at DESC
            """,
            tuple(variants),
        ).fetchall()
        return [
            ContactProfile(
                contact=r["contact"],
                key=r["key"],
                value=r["value"],
                confidence=float(r["confidence"]),
                source=r["source"] or "auto",
                updated_at=r["updated_at"],
            )
            for r in rows
        ]

    # --- Training: teach Steve how to reply (applies to all contacts) ---

    def save_trained_reply(
        self,
        inbound_sample: str,
        reply_body: str,
        *,
        situation: str = "general",
        topic: str = "",
        notes: str = "",
        trained_via_contact: str = "",
        apply_globally: bool = True,
    ) -> int:
        inbound_sample = (inbound_sample or "").strip()
        reply_body = (reply_body or "").strip()
        if not inbound_sample or not reply_body:
            raise ValueError("inbound_sample and reply_body required")
        trained_via_contact = normalize_contact(trained_via_contact) if trained_via_contact else ""
        now = time.time()
        cur = self._conn.execute(
            """
            INSERT INTO trained_replies
                (inbound_sample, reply_body, situation, topic, notes,
                 trained_via_contact, apply_globally, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                inbound_sample,
                reply_body,
                situation,
                topic.strip(),
                notes.strip(),
                trained_via_contact,
                1 if apply_globally else 0,
                now,
                now,
            ),
        )
        self._conn.commit()
        return int(cur.lastrowid)

    def update_trained_reply(
        self,
        trained_id: int,
        *,
        reply_body: str | None = None,
        notes: str | None = None,
    ) -> bool:
        row = self._conn.execute(
            "SELECT id FROM trained_replies WHERE id = ?", (trained_id,)
        ).fetchone()
        if not row:
            return False
        if reply_body is not None:
            self._conn.execute(
                "UPDATE trained_replies SET reply_body = ?, updated_at = ? WHERE id = ?",
                (reply_body.strip(), time.time(), trained_id),
            )
        if notes is not None:
            self._conn.execute(
                "UPDATE trained_replies SET notes = ?, updated_at = ? WHERE id = ?",
                (notes.strip(), time.time(), trained_id),
            )
        self._conn.commit()
        return True

    def delete_trained_reply(self, trained_id: int) -> bool:
        cur = self._conn.execute(
            "DELETE FROM trained_replies WHERE id = ?", (trained_id,)
        )
        self._conn.commit()
        return cur.rowcount > 0

    def list_trained_replies(self, *, limit: int = 50) -> list[TrainedReply]:
        rows = self._conn.execute(
            """
            SELECT id, inbound_sample, reply_body, situation, topic, notes,
                   trained_via_contact, apply_globally, created_at, updated_at
            FROM trained_replies
            ORDER BY updated_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return [self._row_to_trained(r) for r in rows]

    def find_trained_replies(
        self,
        message: str,
        *,
        situation: str = "",
        limit: int = 3,
    ) -> list[TrainedReply]:
        rows = self._conn.execute(
            """
            SELECT id, inbound_sample, reply_body, situation, topic, notes,
                   trained_via_contact, apply_globally, created_at, updated_at
            FROM trained_replies
            WHERE apply_globally = 1
            ORDER BY updated_at DESC
            LIMIT 200
            """
        ).fetchall()
        tokens = _tokenize(message)
        if not tokens:
            return []

        scored: list[tuple[float, TrainedReply]] = []
        for row in rows:
            sample = row["inbound_sample"] or ""
            sample_tokens = _tokenize(sample)
            if not sample_tokens:
                continue
            overlap = len(tokens & sample_tokens)
            if overlap == 0:
                continue
            score = overlap / max(len(tokens), len(sample_tokens))
            if situation and row["situation"] == situation:
                score += 0.35
            if row["topic"] and row["topic"].lower() in message.lower():
                score += 0.25
            scored.append((score, self._row_to_trained(row)))

        scored.sort(key=lambda x: (-x[0], -x[1].updated_at))
        return [item for _, item in scored[:limit]]

    def trained_reply_count(self) -> int:
        row = self._conn.execute("SELECT COUNT(*) AS c FROM trained_replies").fetchone()
        return int(row["c"]) if row else 0

    @staticmethod
    def _row_to_trained(row: sqlite3.Row) -> TrainedReply:
        return TrainedReply(
            id=row["id"],
            inbound_sample=row["inbound_sample"],
            reply_body=row["reply_body"],
            situation=row["situation"] or "general",
            topic=row["topic"] or "",
            notes=row["notes"] or "",
            trained_via_contact=row["trained_via_contact"] or "",
            apply_globally=bool(row["apply_globally"]),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def save_situation_note(self, contact: str, situation: str, note: str) -> None:
        contact = normalize_contact(contact)
        note = note.strip()
        if not note:
            return
        self._conn.execute(
            """
            INSERT INTO situation_notes (contact, situation, note, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(contact, situation) DO UPDATE SET
                note = excluded.note,
                updated_at = excluded.updated_at
            """,
            (contact, situation, note, time.time()),
        )
        self._conn.commit()

    def get_situation_note(self, contact: str, situation: str) -> str | None:
        contact = normalize_contact(contact)
        row = self._conn.execute(
            "SELECT note FROM situation_notes WHERE contact = ? AND situation = ?",
            (contact, situation),
        ).fetchone()
        return row["note"] if row else None

    # --- Verification & reply study logs ---

    def log_verification(
        self,
        contact: str,
        inbound_body: str,
        draft_reply: str,
        *,
        situation: str = "",
        route: str = "",
        attempt: int = 1,
        verifier: str,
        passed: bool,
        reason: str = "",
        ollama_used: bool = False,
    ) -> None:
        contact = normalize_contact(contact)
        self._conn.execute(
            """
            INSERT INTO verification_log
                (contact, inbound_body, draft_reply, situation, route, attempt,
                 verifier, passed, reason, ollama_used, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                contact,
                inbound_body,
                draft_reply,
                situation,
                route,
                attempt,
                verifier,
                1 if passed else 0,
                reason[:500],
                1 if ollama_used else 0,
                time.time(),
            ),
        )
        self._conn.commit()

    def save_reply_study(
        self,
        contact: str,
        inbound_body: str,
        reply_body: str,
        *,
        situation: str,
        route: str = "",
        backend: str = "",
    ) -> int:
        contact = normalize_contact(contact)
        cur = self._conn.execute(
            """
            INSERT INTO reply_study
                (contact, inbound_body, reply_body, situation, route, backend, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                contact,
                inbound_body,
                reply_body,
                situation,
                route,
                backend,
                time.time(),
            ),
        )
        self._conn.commit()
        return int(cur.lastrowid)

    def get_situation_examples(
        self, situation: str, *, limit: int = 3, contact: str | None = None
    ) -> list[ReplyStudy]:
        if contact:
            variants = contact_variants(contact)
            placeholders = ",".join("?" * len(variants))
            rows = self._conn.execute(
                f"""
                SELECT id, contact, inbound_body, reply_body, situation, route, backend, created_at
                FROM reply_study
                WHERE situation = ? AND contact IN ({placeholders})
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (situation, *variants, limit),
            ).fetchall()
        else:
            rows = self._conn.execute(
                """
                SELECT id, contact, inbound_body, reply_body, situation, route, backend, created_at
                FROM reply_study
                WHERE situation = ?
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (situation, limit),
            ).fetchall()
        return [self._row_to_reply_study(r) for r in rows]

    def verification_stats(self, contact: str | None = None) -> dict[str, Any]:
        if contact:
            contact = normalize_contact(contact)
            rows = self._conn.execute(
                """
                SELECT verifier, passed, COUNT(*) AS c
                FROM verification_log
                WHERE contact = ?
                GROUP BY verifier, passed
                """,
                (contact,),
            ).fetchall()
        else:
            rows = self._conn.execute(
                """
                SELECT verifier, passed, COUNT(*) AS c
                FROM verification_log
                GROUP BY verifier, passed
                """
            ).fetchall()
        stats: dict[str, Any] = {"by_verifier": {}}
        for r in rows:
            v = r["verifier"]
            stats["by_verifier"].setdefault(v, {"pass": 0, "fail": 0})
            key = "pass" if r["passed"] else "fail"
            stats["by_verifier"][v][key] = int(r["c"])
        return stats

    def learning_counts(self) -> dict[str, int]:
        row = self._conn.execute("SELECT COUNT(*) AS c FROM contact_profiles").fetchone()
        profiles = int(row["c"]) if row else 0
        row = self._conn.execute("SELECT COUNT(*) AS c FROM reply_study").fetchone()
        study = int(row["c"]) if row else 0
        row = self._conn.execute("SELECT COUNT(*) AS c FROM verification_log").fetchone()
        verifications = int(row["c"]) if row else 0
        return {
            "profiles": profiles,
            "reply_study": study,
            "verifications": verifications,
            "trained_replies": self.trained_reply_count(),
        }

    def list_profiles(self, contact: str | None = None) -> list[ContactProfile]:
        if contact:
            return self.get_profile(contact)
        rows = self._conn.execute(
            """
            SELECT contact, key, value, confidence, source, updated_at
            FROM contact_profiles
            ORDER BY updated_at DESC
            LIMIT 50
            """
        ).fetchall()
        return [
            ContactProfile(
                contact=r["contact"],
                key=r["key"],
                value=r["value"],
                confidence=float(r["confidence"]),
                source=r["source"] or "auto",
                updated_at=r["updated_at"],
            )
            for r in rows
        ]

    @staticmethod
    def _row_to_reply_study(row: sqlite3.Row) -> ReplyStudy:
        return ReplyStudy(
            id=row["id"],
            contact=row["contact"],
            inbound_body=row["inbound_body"],
            reply_body=row["reply_body"],
            situation=row["situation"],
            route=row["route"] or "",
            backend=row["backend"] or "",
            created_at=row["created_at"],
        )

    @staticmethod
    def _row_to_message(row: sqlite3.Row) -> Message:
        return Message(
            id=row["id"],
            contact=row["contact"],
            role=row["role"],
            body=row["body"],
            timestamp=row["timestamp"],
            source=row["source"] or "sms",
            topic=row["topic"],
        )


def _tokenize(text: str) -> set[str]:
    words = re.findall(r"[a-z0-9']+", text.lower())
    stop = {
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
        "of", "is", "it", "i", "you", "me", "my", "we", "so", "just", "like",
        "that", "this", "was", "are", "am", "be", "do", "did", "have", "has",
    }
    return {w for w in words if len(w) > 2 and w not in stop}
