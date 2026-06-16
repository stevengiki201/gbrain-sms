"""CLI entry point and main event loop for the USB SMS chatbot."""

from __future__ import annotations

import argparse
import asyncio
import logging
import random
import re
import signal
import sys
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path

from bot.adb_connector import AdbConnector, SmsMessage, WhatsAppNotification
from bot.agents import SmsDeliveryPipeline
from bot.agents.training_agent import TrainingAgent
from bot.chat_engine import ChatEngine, ChatResponse
from bot.config import Config, load_config
from bot.memory_store import MemoryStore, normalize_contact
from bot.network import host_has_network
from bot.scheduler import ResponseScheduler, ScheduledReply

logger = logging.getLogger("bot")


class BotApp:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.memory = MemoryStore(config.resolve_path(config.db_path))
        self.adb = AdbConnector(
            adb_path=config.adb_path,
            sms_uri_authority=config.sms_uri_authority,
            whatsapp_enabled=config.whatsapp_enabled,
            sms_send_strategies=config.sms_send_strategies,
        )
        self.engine = ChatEngine(
            memory=self.memory,
            ollama_url=config.ollama_url,
            ollama_model=config.ollama_model,
            reply_backend=config.reply_backend,
            bot_name=config.bot_name,
            reply_persona=config.reply_persona,
        )
        self.pipeline = SmsDeliveryPipeline(self.engine)
        self.trainer = TrainingAgent()
        self.scheduler = ResponseScheduler(
            send_callback=self._send_scheduled,
            dnd_start=config.dnd_start,
            dnd_end=config.dnd_end,
        )
        self._running = False
        self._last_reply_at: dict[str, float] = {}
        self._reply_in_progress: set[str] = set()
        self._inbox_seeded = False
        self._whatsapp_network_logged = False

    def _should_poll_whatsapp(self) -> bool:
        if not self.config.whatsapp_enabled:
            return False
        if not self.config.whatsapp_require_network:
            return True
        if not host_has_network():
            if not self._whatsapp_network_logged:
                logger.info("WhatsApp polling paused — PC offline")
                self._whatsapp_network_logged = True
            return False
        if not self.adb.device_online():
            if not self._whatsapp_network_logged:
                logger.info("WhatsApp polling paused — phone offline")
                self._whatsapp_network_logged = True
            return False
        if self._whatsapp_network_logged:
            logger.info("Network up — WhatsApp polling enabled")
            self._whatsapp_network_logged = False
        return True

    async def _send_scheduled(self, task: ScheduledReply) -> bool:
        loop = asyncio.get_running_loop()
        if task.source == "whatsapp":
            notif = WhatsAppNotification(
                key="scheduled",
                title=task.contact,
                text="",
                contact=task.contact,
                contact_uri=task.whatsapp_contact_uri,
                user_id=task.whatsapp_user_id,
                chat_uri=task.whatsapp_chat_uri,
            )
            success = await loop.run_in_executor(
                None, self.adb.reply_whatsapp, notif, task.body
            )
        else:
            success = await loop.run_in_executor(
                None, self.adb.send_sms, task.number, task.body
            )
        if success:
            self.memory.save_message(
                task.contact,
                "bot",
                task.body,
                source=task.source,
            )
        return success

    def _save_bot_message(
        self, contact: str, text: str, source: str, topic: str | None
    ) -> None:
        self.memory.save_message(
            contact, "bot", text, source=source, topic=topic
        )

    async def run(self) -> None:
        self._running = True
        await self.scheduler.start()
        self.engine.check_ollama(force=True)

        logger.info("Bot started (backend: %s)", self.engine.backend_name)
        logger.info(
            "Auto-reply: %s | Agent pipeline: %s | Poll: %.1fs | WhatsApp: %s (network-gated: %s)",
            self.config.auto_reply_enabled,
            self.config.agent_pipeline_enabled,
            self.config.poll_interval_sec,
            self.config.whatsapp_enabled,
            self.config.whatsapp_require_network,
        )

        try:
            while self._running:
                if not self.adb.is_connected():
                    logger.warning("Waiting for ADB device...")
                    loop = asyncio.get_running_loop()
                    device = await loop.run_in_executor(
                        None, self.adb.wait_for_device, 30.0
                    )
                    if not device:
                        await asyncio.sleep(self.config.poll_interval_sec)
                        continue

                await self._poll_and_respond()
                await asyncio.sleep(self.config.poll_interval_sec)
        finally:
            await self.scheduler.stop(drain=False)
            self.memory.close()

    async def _poll_and_respond(self) -> None:
        loop = asyncio.get_running_loop()
        last_seen = self.memory.get_last_seen_sms_id()

        if not self._inbox_seeded and last_seen == 0:
            await self._seed_inbox_baseline()
            self._inbox_seeded = True

        new_sms = await loop.run_in_executor(
            None, self.adb.poll_inbox, last_seen
        )

        pending = [s for s in new_sms if not self.memory.is_sms_replied(s.sms_id)]
        latest_by_contact = self._coalesce_latest_sms(pending)

        for sms in new_sms:
            self.memory.set_last_seen_sms_id(max(last_seen, sms.sms_id))
            last_seen = max(last_seen, sms.sms_id)

        for contact, sms in latest_by_contact.items():
            siblings = sorted(
                [s for s in pending if normalize_contact(s.address) == contact],
                key=lambda s: s.sms_id,
            )
            for older in siblings[:-1]:
                self.memory.save_message(contact, "user", older.body, source="sms")
                self.memory.mark_sms_processed(older.sms_id)
            await self._handle_inbound_sms(sms)

        if self._should_poll_whatsapp():
            notifications = await loop.run_in_executor(
                None, self.adb.poll_whatsapp_notifications
            )
            for notif in notifications:
                if self.memory.is_whatsapp_processed(notif.key, notif.text):
                    continue
                self.memory.mark_whatsapp_processed(notif.key, notif.text)
                await self._handle_inbound(
                    notif.contact, notif.text, source="whatsapp", notif=notif
                )

    async def _seed_inbox_baseline(self) -> None:
        """Skip old SMS on first run — only auto-reply to new messages."""
        loop = asyncio.get_running_loop()
        existing = await loop.run_in_executor(
            None, lambda: self.adb.poll_inbox(0, limit=500)
        )
        if not existing:
            return
        max_id = max(m.sms_id for m in existing)
        for sms in existing:
            self.memory.mark_sms_processed(sms.sms_id)
        self.memory.set_last_seen_sms_id(max_id)
        logger.info(
            "Seeded SMS inbox baseline at id %d (%d existing messages skipped)",
            max_id,
            len(existing),
        )

    @staticmethod
    def _coalesce_latest_sms(messages: list[SmsMessage]) -> dict[str, SmsMessage]:
        """One reply per contact per poll — keep only the newest SMS."""
        latest: dict[str, SmsMessage] = {}
        for sms in messages:
            contact = normalize_contact(sms.address)
            current = latest.get(contact)
            if current is None or sms.sms_id > current.sms_id:
                latest[contact] = sms
        return latest

    @staticmethod
    def _is_replyable_phone(address: str) -> bool:
        """Skip short codes / carrier names (e.g. AIRTELMONEY)."""
        digits = re.sub(r"\D", "", address or "")
        return len(digits) >= 9

    async def _handle_inbound_sms(self, sms: SmsMessage) -> None:
        contact = normalize_contact(sms.address)
        body = sms.body

        if self.memory.is_sms_replied(sms.sms_id):
            logger.debug("SMS id=%s already replied; skipping", sms.sms_id)
            self.memory.mark_sms_processed(sms.sms_id)
            return

        if contact in self._reply_in_progress:
            logger.info("Reply already in progress for %s; skipping sms id=%s", contact, sms.sms_id)
            return

        self.memory.save_message(contact, "user", body, source="sms")

        if not self._is_replyable_phone(sms.address):
            logger.info("Skipping auto-reply to non-phone sender: %s", sms.address)
            self.memory.mark_sms_processed(sms.sms_id)
            return

        await self._handle_inbound(
            contact,
            body,
            source="sms",
            sms_id=sms.sms_id,
        )

    async def _handle_inbound(
        self,
        contact: str,
        body: str,
        *,
        source: str = "sms",
        notif: WhatsAppNotification | None = None,
        sms_id: int | None = None,
    ) -> None:
        contact = normalize_contact(contact)
        if not self.config.auto_reply_enabled:
            logger.info("Auto-reply disabled; skipping message from %s", contact)
            return

        if sms_id is not None and self.memory.is_sms_replied(sms_id):
            logger.info("SMS id=%s already has a reply; skipping", sms_id)
            return

        if contact in self._reply_in_progress:
            logger.info("Reply in progress for %s; skipping duplicate trigger", contact)
            return

        if not self._cooldown_ok(contact):
            logger.info("Cooldown active for %s; skipping", contact)
            return

        if source != "sms":
            self.memory.save_message(contact, "user", body, source=source)
        self.engine.extract_preferences(contact, body)

        logger.info(
            "New %s message from %s (sms_id=%s): %s",
            source,
            contact,
            sms_id or "-",
            body[:80],
        )

        self._reply_in_progress.add(contact)
        try:
            loop = asyncio.get_running_loop()
            if self.config.agent_pipeline_enabled:
                pipeline_result = await loop.run_in_executor(
                    None,
                    lambda: self.pipeline.process(contact, body, source=source),
                )
                response = pipeline_result
                logger.info(
                    "Pipeline %s route=%s attempts=%d approved=%s: %s",
                    response.backend,
                    response.route,
                    response.attempts,
                    response.approved,
                    response.text[:80],
                )
                for verdict in response.verdicts:
                    logger.info(
                        "  verifier [%s] %s — %s",
                        verdict.agent,
                        "PASS" if verdict.passed else "FAIL",
                        verdict.reason,
                    )
                if not response.approved:
                    logger.error("Reply not approved by agents; skipping send to %s", contact)
                    return
            else:
                response = await loop.run_in_executor(
                    None,
                    lambda: self.engine.generate_reply(contact, body, source=source),
                )
                logger.info("Generated reply (%s): %s", response.backend, response.text[:80])

            send_at = response.send_at
            if send_at is None and self.config.reply_mode == "scheduled":
                send_at = datetime.now()

            if send_at and send_at > datetime.now():
                task = ScheduledReply(
                    contact=contact,
                    number=contact if source == "sms" else contact,
                    body=response.text,
                    send_at=send_at,
                    source=source,
                    whatsapp_contact_uri=notif.contact_uri if notif else "",
                    whatsapp_user_id=notif.user_id if notif else None,
                    whatsapp_chat_uri=notif.chat_uri if notif else "",
                )
                await self.scheduler.schedule(task)
                logger.info(
                    "Reply queued for %s at %s (%s)",
                    contact,
                    send_at.isoformat(),
                    response.backend,
                )
                return

            await self._human_reply_delay()
            success = await self._deliver_reply(
                contact, response.text, source, notif, topic=response.topic
            )
            if success:
                self._last_reply_at[contact] = asyncio.get_running_loop().time()
                if sms_id is not None:
                    self.memory.mark_sms_replied(sms_id, contact, body, response.text)
                    self.memory.mark_sms_processed(sms_id)
                if self.config.agent_pipeline_enabled and hasattr(response, "situation"):
                    await loop.run_in_executor(
                        None,
                        lambda: self.pipeline.record_successful_reply(
                            contact,
                            body,
                            response.text,
                            situation=getattr(response, "situation", "general"),
                            route=getattr(response, "route", ""),
                            backend=response.backend,
                        ),
                    )
            elif sms_id is not None:
                logger.warning(
                    "Send failed for sms id=%s — will retry on next poll", sms_id
                )
        finally:
            self._reply_in_progress.discard(contact)

    async def _human_reply_delay(self) -> None:
        """Pause briefly so replies feel like a real person typing."""
        lo = self.config.reply_delay_min_sec
        hi = self.config.reply_delay_max_sec
        if hi <= 0:
            return
        delay = random.uniform(max(0.0, lo), max(lo, hi))
        if delay > 0:
            logger.debug("Human reply delay: %.1fs", delay)
            await asyncio.sleep(delay)

    async def _deliver_reply(
        self,
        contact: str,
        text: str,
        source: str,
        notif: WhatsAppNotification | None = None,
        topic: str | None = None,
    ) -> bool:
        loop = asyncio.get_running_loop()
        if source == "whatsapp" and notif is not None:
            success = await loop.run_in_executor(
                None, self.adb.reply_whatsapp, notif, text
            )
        else:
            success = await loop.run_in_executor(
                None, self.adb.send_sms, contact, text
            )

        if success:
            self._save_bot_message(contact, text, source, topic)
            logger.info("Sent reply to %s: %s", contact, text[:80])
        else:
            logger.error("Failed to send reply to %s", contact)
        return success

    def _cooldown_ok(self, contact: str) -> bool:
        loop = asyncio.get_running_loop()
        now = loop.time()
        last = self._last_reply_at.get(contact, 0.0)
        return (now - last) >= self.config.reply_cooldown_sec

    def stop(self) -> None:
        self._running = False


def setup_logging(config: Config) -> None:
    log_path = config.resolve_path(config.log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    level = getattr(logging, config.log_level.upper(), logging.INFO)
    root = logging.getLogger("bot")
    root.setLevel(level)

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(fmt)
    root.addHandler(console)

    file_handler = RotatingFileHandler(
        log_path, maxBytes=2_000_000, backupCount=3, encoding="utf-8"
    )
    file_handler.setFormatter(fmt)
    root.addHandler(file_handler)


def parse_time_arg(value: str) -> datetime:
    for fmt in ("%H:%M", "%Y-%m-%d %H:%M", "%Y-%m-%dT%H:%M"):
        try:
            dt = datetime.strptime(value, fmt)
            if fmt == "%H:%M":
                now = datetime.now()
                dt = now.replace(hour=dt.hour, minute=dt.minute, second=0, microsecond=0)
                if dt <= now:
                    from datetime import timedelta

                    dt += timedelta(days=1)
            return dt
        except ValueError:
            continue
    raise argparse.ArgumentTypeError(f"Invalid time format: {value}")


async def cmd_run(app: BotApp) -> None:
    loop = asyncio.get_running_loop()

    def _signal_handler() -> None:
        logger.info("Shutdown requested")
        app.stop()

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _signal_handler)

    await app.run()


def cmd_status(app: BotApp) -> None:
    devices = app.adb.list_devices()
    app.engine.check_ollama(force=True)

    print("=== USB SMS Chatbot Status ===")
    print(f"ADB devices: {', '.join(devices) if devices else 'none'}")
    print(f"Chat backend: {app.engine.backend_name}")
    print(f"Agent pipeline: {app.config.agent_pipeline_enabled}")
    print(f"Auto-reply: {app.config.auto_reply_enabled}")
    print(f"WhatsApp polling: {app.config.whatsapp_enabled}")
    if app.config.whatsapp_enabled:
        pc_net = host_has_network()
        phone_net = app.adb.device_online() if app.adb.is_connected() else False
        print(f"  PC network: {'online' if pc_net else 'offline'}")
        print(f"  Phone network: {'online' if phone_net else 'offline' if app.adb.is_connected() else 'no device'}")
        print(f"  Polling active: {app._should_poll_whatsapp()}")
    print(f"Messages stored: {app.memory.message_count()}")
    print(f"Last seen SMS id: {app.memory.get_last_seen_sms_id()}")

    stats = app.memory.verification_stats()
    by_v = stats.get("by_verifier", {})
    if by_v:
        print("Verification stats (all contacts):")
        for verifier, counts in by_v.items():
            print(f"  {verifier}: pass={counts.get('pass', 0)} fail={counts.get('fail', 0)}")

    counts = app.memory.learning_counts()
    print(
        f"Learning DB: {counts['profiles']} profile facts, "
        f"{counts['reply_study']} studied replies, "
        f"{counts['trained_replies']} trained replies, "
        f"{counts['verifications']} verification logs"
    )
    if app.config.training_contact:
        print(f"Training contact: {normalize_contact(app.config.training_contact)}")

    pending = app.scheduler.list_pending()
    if pending:
        print(f"Pending scheduled replies: {len(pending)}")
        for task in pending:
            print(f"  [{task.task_id}] {task.number} @ {task.send_at.isoformat()}")
    else:
        print("Pending scheduled replies: 0")

    if app.scheduler.is_dnd_active():
        print(f"DND active ({app.config.dnd_start} - {app.config.dnd_end})")


def cmd_send(app: BotApp, number: str, message: str) -> None:
    if not app.adb.wait_for_device(timeout=10):
        print("No ADB device connected.")
        sys.exit(1)
    success = app.adb.send_sms(number, message)
    if success:
        app.memory.save_message(number, "bot", message)
        print(f"Sent to {number}: {message}")
    else:
        print(f"Failed to send to {number}")
        sys.exit(1)


async def cmd_schedule(app: BotApp, number: str, at: datetime, message: str) -> None:
    await app.scheduler.start()
    task = ScheduledReply(
        contact=number,
        number=number,
        body=message,
        send_at=at,
    )
    task_id = await app.scheduler.schedule(task)
    print(f"Scheduled [{task_id}] to {number} at {at.isoformat()}: {message}")
    print("Press Ctrl+C to exit scheduler worker.")
    try:
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        await app.scheduler.stop(drain=False)


def cmd_history(app: BotApp, contact: str, limit: int) -> None:
    messages = app.memory.get_recent_context(contact, limit=limit)
    if not messages:
        print(f"No history for {contact}")
        return
    for msg in messages:
        ts = datetime.fromtimestamp(msg.timestamp).strftime("%Y-%m-%d %H:%M")
        role = "You" if msg.role == "bot" else "Them"
        print(f"[{ts}] {role}: {msg.body}")


def cmd_cancel(app: BotApp, task_id: str) -> None:
    if app.scheduler.cancel(task_id):
        print(f"Cancelled scheduled reply [{task_id}]")
    else:
        print(f"No pending task with id [{task_id}]")
        sys.exit(1)


def cmd_toggle_auto_reply(config: Config) -> None:
    cfg_path = config.config_path
    if not cfg_path.exists():
        print(f"Config not found at {cfg_path}")
        sys.exit(1)

    import yaml

    with cfg_path.open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}

    current = bool(data.get("auto_reply_enabled", True))
    data["auto_reply_enabled"] = not current

    with cfg_path.open("w", encoding="utf-8") as fh:
        yaml.dump(data, fh, default_flow_style=False)

    state = "enabled" if not current else "disabled"
    print(f"Auto-reply {state}")


def cmd_try_reply(
    app: BotApp,
    message: str,
    contact: str,
    *,
    save: bool = False,
    use_pipeline: bool = True,
) -> None:
    """Run reply generation + verification locally — no SMS sent."""
    app.engine.check_ollama(force=True)

    contact = normalize_contact(contact)
    print("=== Try reply (dry run — nothing sent) ===")
    print(f"Contact: {contact}")
    print(f"Inbound:  {message}")
    print(f"Backend:  {app.engine.backend_name}")
    print(f"Pipeline: {'on' if use_pipeline else 'off'}")
    print()

    if save:
        app.memory.save_message(contact, "user", message, source="test")

    if use_pipeline and app.config.agent_pipeline_enabled:
        result = app.pipeline.process(contact, message, source="test")
        print(f"Route:     {result.route}")
        print(f"Situation: {getattr(result, 'situation', '—')}")
        mem = getattr(result, "memory_summary", "") or ""
        if "Reply language:" in mem or "language:" in mem.lower():
            print(f"Memory:   {mem[:120]}")
        print(f"Approved:  {result.approved}")
        print(f"Attempts:  {result.attempts}")
        print(f"Engine:    {result.backend}")
        print()
        print(f"Reply:\n  {result.text}")
        print()
        if result.verdicts:
            print("Verifiers:")
            for v in result.verdicts:
                ollama = " (ollama)" if getattr(v, "ollama_used", False) else ""
                status = "PASS" if v.passed else "FAIL"
                print(f"  [{v.agent}] {status}{ollama} — {v.reason}")
        if result.memory_summary:
            print(f"\nMemory: {result.memory_summary[:200]}")
        if save and result.approved:
            app.memory.save_message(contact, "bot", result.text, source="test")
            print("\n(Saved test exchange to local DB — use --contact to reload thread context)")
    else:
        response = app.engine.generate_reply(contact, message, source="test")
        print(f"Engine: {response.backend}")
        print(f"\nReply:\n  {response.text}")
        if save:
            app.memory.save_message(contact, "bot", response.text, source="test")

    print("\n--- No SMS was sent ---")


def _default_training_contact(config: Config, contact: str | None) -> str:
    if contact:
        return normalize_contact(contact)
    if config.training_contact.strip():
        return normalize_contact(config.training_contact)
    return "+00000000000"


def cmd_train(
    app: BotApp,
    message: str,
    contact: str | None,
    *,
    correct: str | None = None,
    approve: bool = False,
    notes: str = "",
    save: bool = True,
) -> None:
    """
    Test a reply, then teach Steve the correct answer for ALL contacts.

    Workflow:
      1. train --msg "question"              → see bot draft
      2. train --msg "question" --approve    → save bot reply if good
      3. train --msg "question" --correct "your better answer"
    """
    from bot.agents.situation import detect_situation

    contact = _default_training_contact(app.config, contact)
    situation = detect_situation(message)

    print("=== Train Steve (nothing sent) ===")
    print(f"Training via contact: {contact}")
    print(f"Inbound:  {message}")
    print(f"Situation: {situation}")
    print()

    existing = app.memory.find_trained_replies(message, situation=situation, limit=2)
    if existing:
        print("Existing trained matches:")
        for t in existing:
            print(f"  [#{t.id}] Them: {t.inbound_sample[:60]}...")
            print(f"         You:  {t.reply_body[:80]}...")
        print()

    if correct:
        trained_id = app.trainer.save_correction(
            message,
            correct,
            engine=app.engine,
            trained_via_contact=contact,
            situation=situation,
            notes=notes,
        )
        if save:
            app.memory.save_message(contact, "user", message, source="train")
            app.memory.save_message(contact, "bot", correct, source="train")
        print(f"Saved training #{trained_id} — applies to ALL contacts when similar.")
        print(f"\nCorrect reply stored:\n  {correct[:500]}")
        print("\nTest again:  python -m bot.main try-reply --msg \"...\"")
        return

    app.engine.check_ollama(force=True)
    if save:
        app.memory.save_message(contact, "user", message, source="train")

    result = app.pipeline.process(contact, message, source="train")
    print(f"Bot draft ({result.backend}, approved={result.approved}):")
    print(f"  {result.text}")
    print()

    if approve:
        if not result.approved:
            print("Warning: verifiers did not approve this draft — saving anyway (--approve).")
        trained_id = app.trainer.save_correction(
            message,
            result.text,
            engine=app.engine,
            trained_via_contact=contact,
            situation=situation,
            notes=notes or "approved from train",
        )
        if save:
            app.memory.save_message(contact, "bot", result.text, source="train")
        print(f"Approved and saved as training #{trained_id} for all contacts.")
        return

    print("Next steps:")
    print("  Save bot answer:  train --msg \"...\" --approve")
    print("  Fix answer:       train --msg \"...\" --correct \"your better reply\"")
    print(f"  List training:    train-list")


def cmd_train_list(app: BotApp, limit: int) -> None:
    rows = app.memory.list_trained_replies(limit=limit)
    print(f"=== Trained replies ({len(rows)}) — used for ALL contacts ===\n")
    if not rows:
        print("No training yet. Use: python -m bot.main train --msg \"...\" --correct \"...\"")
        return
    for t in rows:
        print(f"[#{t.id}] situation={t.situation} topic={t.topic or '-'} via={t.trained_via_contact or 'global'}")
        print(f"  Them: {t.inbound_sample[:100]}")
        print(f"  You:  {t.reply_body[:200]}")
        if t.notes:
            print(f"  Note: {t.notes[:80]}")
        print()


def cmd_train_delete(app: BotApp, trained_id: int) -> None:
    if app.memory.delete_trained_reply(trained_id):
        print(f"Deleted training #{trained_id}")
    else:
        print(f"No training found with id #{trained_id}")
        sys.exit(1)


def cmd_study(app: BotApp, contact: str | None, limit: int) -> None:
    """Show learned profiles, situation examples, and verification stats."""
    if contact:
        contact = normalize_contact(contact)
        print(f"=== Learning data for {contact} ===")
        profiles = app.memory.list_profiles(contact)
        stats = app.memory.verification_stats(contact)
    else:
        print("=== Learning data (all contacts) ===")
        profiles = app.memory.list_profiles()
        stats = app.memory.verification_stats()

    counts = app.memory.learning_counts()
    print(
        f"Totals: {counts['profiles']} facts, "
        f"{counts['reply_study']} studied replies, "
        f"{counts['trained_replies']} trained replies, "
        f"{counts['verifications']} verification logs\n"
    )

    if profiles:
        print("Contact profiles:")
        for p in profiles[:limit]:
            print(f"  [{p.contact}] {p.key}: {p.value} (conf={p.confidence:.2f}, {p.source})")
    else:
        print("Contact profiles: (none yet — builds up as Steve chats)")

    by_v = stats.get("by_verifier", {})
    if by_v:
        print("\nVerification pass/fail:")
        for verifier, c in by_v.items():
            print(f"  {verifier}: pass={c.get('pass', 0)} fail={c.get('fail', 0)}")

    if contact:
        for situation in ("factual", "greeting", "goodbye", "question", "general"):
            examples = app.memory.get_situation_examples(
                situation, limit=2, contact=contact
            )
            if examples:
                print(f"\nGood {situation} replies:")
                for ex in examples:
                    print(f"  Them: {ex.inbound_body[:60]}")
                    print(f"  You:  {ex.reply_body[:80]}")


def cmd_export(app: BotApp, path: Path) -> None:
    app.memory.export_json(path)
    print(f"Exported conversation history to {path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="USB SMS Chatbot — local Python bot over ADB"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to config.yaml (default: ./config.yaml)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("run", help="Start the main polling loop")

    sub.add_parser("status", help="Show device, backend, and queue status")

    p_send = sub.add_parser("send", help="Send an SMS immediately")
    p_send.add_argument("--to", required=True, help="Phone number")
    p_send.add_argument("--msg", required=True, help="Message body")

    p_sched = sub.add_parser("schedule", help="Schedule an SMS for a specific time")
    p_sched.add_argument("--to", required=True, help="Phone number")
    p_sched.add_argument(
        "--at", required=True, type=parse_time_arg, help="Time (HH:MM or YYYY-MM-DD HH:MM)"
    )
    p_sched.add_argument("--msg", required=True, help="Message body")

    p_hist = sub.add_parser("history", help="Show conversation history")
    p_hist.add_argument("--contact", required=True, help="Contact phone or name")
    p_hist.add_argument("--limit", type=int, default=20, help="Max messages")

    sub.add_parser("toggle-auto-reply", help="Toggle auto-reply in config.yaml")

    p_cancel = sub.add_parser("cancel", help="Cancel a pending scheduled reply")
    p_cancel.add_argument("--id", required=True, help="Task id from status/schedule output")

    p_export = sub.add_parser("export", help="Export messages to JSON")
    p_export.add_argument("--out", type=Path, default=Path("data/export.json"))

    p_study = sub.add_parser("study", help="Show learning DB and verification stats")
    p_study.add_argument("--contact", default=None, help="Filter by contact")
    p_study.add_argument("--limit", type=int, default=20, help="Max profile rows")

    p_try = sub.add_parser(
        "try-reply",
        help="Test a reply locally (full pipeline, no SMS sent)",
    )
    p_try.add_argument("--msg", required=True, help="Message to simulate receiving")
    p_try.add_argument(
        "--contact",
        default="+00000000000",
        help="Contact for memory context (default: sandbox number, not a real contact)",
    )
    p_try.add_argument(
        "--save",
        action="store_true",
        help="Save this test exchange to the local DB (for thread context tests)",
    )
    p_try.add_argument(
        "--no-pipeline",
        action="store_true",
        help="Skip agent pipeline; use engine only",
    )
    p_try.add_argument(
        "--approve",
        action="store_true",
        help="Save approved bot reply as global training",
    )
    p_try.add_argument(
        "--correct",
        default=None,
        help="Save your corrected reply as global training",
    )

    p_train = sub.add_parser(
        "train",
        help="Test + teach Steve (corrections apply to all real contacts)",
    )
    p_train.add_argument("--msg", required=True, help="Example incoming message")
    p_train.add_argument(
        "--contact",
        default=None,
        help="Training contact for context (default: training_contact in config.yaml)",
    )
    p_train.add_argument(
        "--correct",
        default=None,
        help="Your corrected reply — saved globally for similar questions",
    )
    p_train.add_argument(
        "--approve",
        action="store_true",
        help="Save the bot's generated reply as training",
    )
    p_train.add_argument("--notes", default="", help="Optional note about this training")
    p_train.add_argument(
        "--no-save",
        action="store_true",
        help="Do not save messages to conversation history",
    )

    p_train_list = sub.add_parser("train-list", help="List all trained replies")
    p_train_list.add_argument("--limit", type=int, default=30)

    p_train_del = sub.add_parser("train-delete", help="Delete a trained reply by id")
    p_train_del.add_argument("--id", type=int, required=True)

    p_crawl = sub.add_parser("crawl-swahili", help="Build Kiswahili training corpus")
    p_crawl.add_argument("--limit", type=int, default=200, help="Max Q&A pairs to crawl")
    p_crawl.add_argument("--seed-only", action="store_true", help="Only seed built-in pairs")
    p_crawl.add_argument("--stats", action="store_true", help="Show corpus statistics")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    config = load_config(args.config)
    setup_logging(config)
    app = BotApp(config)

    try:
        if args.command == "run":
            asyncio.run(cmd_run(app))
        elif args.command == "status":
            cmd_status(app)
        elif args.command == "send":
            cmd_send(app, args.to, args.msg)
        elif args.command == "schedule":
            asyncio.run(cmd_schedule(app, args.to, args.at, args.msg))
        elif args.command == "history":
            cmd_history(app, args.contact, args.limit)
        elif args.command == "toggle-auto-reply":
            cmd_toggle_auto_reply(config)
        elif args.command == "cancel":
            cmd_cancel(app, args.id)
        elif args.command == "export":
            cmd_export(app, args.out)
        elif args.command == "study":
            cmd_study(app, args.contact, args.limit)
        elif args.command == "try-reply":
            if args.correct or args.approve:
                cmd_train(
                    app,
                    args.msg,
                    args.contact if args.contact != "+00000000000" else None,
                    correct=args.correct,
                    approve=args.approve,
                    save=not getattr(args, "no_save", False),
                )
            else:
                cmd_try_reply(
                    app,
                    args.msg,
                    args.contact,
                    save=args.save,
                    use_pipeline=not args.no_pipeline,
                )
        elif args.command == "train":
            cmd_train(
                app,
                args.msg,
                args.contact,
                correct=args.correct,
                approve=args.approve,
                notes=args.notes,
                save=not args.no_save,
            )
        elif args.command == "train-list":
            cmd_train_list(app, args.limit)
        elif args.command == "train-delete":
            cmd_train_delete(app, args.id)
        elif args.command == "crawl-swahili":
            cmd_crawl_swahili(app, args.limit, args.seed_only, args.stats)
    finally:
        app.memory.close()


if __name__ == "__main__":
    main()
