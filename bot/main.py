"""CLI entry point and main event loop for the GBrain SMS bot."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import random
import re
import signal
import ssl
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path

from bot.adb_connector import AdbConnector, SmsMessage, WhatsAppNotification
from bot.agents import SmsDeliveryPipeline
from bot.agents.training_agent import TrainingAgent
from bot.call_answerer import CallAnswerer
from bot.call_connector import (
    CALL_STATE_RINGING,
    DEFAULT_OUTGOING_MESSAGE,
    CallConnector,
)
from bot.chat_engine import ChatEngine, ChatResponse
from bot.config import Config, load_config
from bot.memory_store import MemoryStore, normalize_contact
from bot.network import host_has_network
from bot.scheduler import ResponseScheduler, ScheduledReply
from bot.voice import VoiceEngine

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
        self.calls = CallConnector(
            self.adb,
            answer_strategies=config.call_answer_strategies,
            tts_strategies=config.call_tts_strategies,
            tts_helper_action=config.call_tts_helper_action,
            tts_command=config.call_tts_command,
            tts_volume=config.call_tts_volume,
            dial_strategies=config.call_dial_strategies,
            dial_timeout_sec=config.call_dial_timeout_sec,
        )
        self.voice = VoiceEngine(
            config,
            calls=self.calls,
            transcript_sink=self._save_call_transcript,
        )
        self.call_answerer = (
            CallAnswerer(self.calls, config, voice=self.voice)
            if config.call_answer_enabled
            else None
        )
        self.pipeline = SmsDeliveryPipeline(self.engine)
        self.trainer = TrainingAgent()
        self.scheduler = ResponseScheduler(
            send_callback=self._send_scheduled,
            dnd_start=config.dnd_start,
            dnd_end=config.dnd_end,
            call_callback=self._execute_call_task,
        )
        self._running = False
        self._last_reply_at: dict[str, float] = {}
        self._reply_in_progress: set[str] = set()
        self._inbox_seeded = False
        self._whatsapp_network_logged = False
        self.bridge = None
        if config.bridge_enabled:
            from bot.bridge.service import AudioLinkService

            self.bridge = AudioLinkService(self._build_bridge_config())
            self.bridge.bot = self

    def _build_bridge_config(self):
        from bot.bridge.config import AudioLinkConfig

        token_file = Path(self.config.audiolink_admin_token_file).expanduser()
        data_dir = (
            Path(self.config.bridge_data_dir).expanduser()
            if self.config.bridge_data_dir
            else token_file.parent
        )
        data_dir.mkdir(parents=True, exist_ok=True)
        token_file = data_dir / "admin.token"
        cert_file = data_dir / "tls" / "cert.pem"

        self.config.audiolink_admin_token_file = str(token_file)
        self.config.audiolink_ca_file = str(cert_file)
        self.config.audiolink_url = f"https://127.0.0.1:{self.config.gbrain_api_port}"

        kwargs = {
            "host": "0.0.0.0",
            "port": self.config.gbrain_api_port,
            "mdns_enabled": self.config.bridge_mdns_enabled,
            "data_dir": str(data_dir),
            "pairing_code_ttl_sec": self.config.bridge_pairing_code_ttl_sec,
            "gbrain_url": self.config.audiolink_url,
            "gbrain_token_file": str(token_file),
            "auto_reply_enabled": self.config.bridge_auto_reply_enabled,
        }
        if self.config.bridge_device_name:
            kwargs["device_name"] = self.config.bridge_device_name
        bridge_config = AudioLinkConfig(**kwargs)
        bridge_config.audiolink_ca_file = str(cert_file)
        return bridge_config

    def generate_bridge_reply(self, contact: str, body: str, source: str = "sms") -> str:
        """Synchronous reply generation for the bridge signaling path."""
        contact = normalize_contact(contact)
        self.memory.save_message(contact, "user", body, source=source)
        self.engine.extract_preferences(contact, body)

        if self.config.agent_pipeline_enabled:
            response = self.pipeline.process(contact, body, source=source)
            if not response.approved:
                logger.error("Reply not approved by agents; skipping bridge reply to %s", contact)
                return ""
            self.pipeline.record_successful_reply(
                contact,
                body,
                response.text,
                situation=getattr(response, "situation", "general"),
                route=getattr(response, "route", ""),
                backend=response.backend,
            )
        else:
            response = self.engine.generate_reply(contact, body, source=source)

        self.memory.save_message(
            contact, "bot", response.text, source=source, topic=response.topic
        )
        logger.info("Generated bridge reply for %s (%s): %s", contact, response.backend, response.text[:80])
        return response.text

    def _save_call_transcript(self, contact: str, role: str, text: str) -> None:
        try:
            self.memory.save_message(contact, role, text, source="call")
        except Exception as exc:  # transcript history is best-effort
            logger.warning("Could not save call transcript: %s", exc)

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
            success = False
            if self.bridge is not None and self.bridge.running:
                try:
                    result = await self.bridge.send_signal(
                        "",
                        {
                            "type": "sms.send",
                            "to_number": task.number,
                            "body": task.body,
                            "task_id": task.task_id,
                        },
                    )
                    success = bool(result.get("sent"))
                except Exception as exc:
                    logger.warning("Bridge send failed (%s); falling back to ADB", exc)
            if not success:
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

    async def _execute_call_task(self, task) -> None:
        """Push a task.execute call command to the phone via AudioLink signaling.
        Falls back to ADB dial if AudioLink is not reachable."""
        from bot.voice.audiolink_bridge import AudioLinkClient, AudioLinkError
        payload = {
            "type": "task.execute",
            "task_id": task.task_id,
            "action": "call",
            "number": task.number,
            "message": task.message,
            "speak_delay_sec": getattr(task, "speak_delay_sec", 6.0),
            "repeat": getattr(task, "repeat", 1),
        }
        try:
            client = AudioLinkClient.from_config(self.config)
            loop = asyncio.get_running_loop()
            device_id = await loop.run_in_executor(None, client.resolve_device_id)
            await loop.run_in_executor(
                None,
                lambda: client.execute(
                    "audiolink.send_signal",
                    {"device_id": device_id, "payload": payload},
                ),
            )
            logger.info(
                "task.execute sent to phone for call task %s → %s",
                task.task_id,
                task.number,
            )
        except Exception as exc:
            logger.warning(
                "send_signal failed (%s); falling back to ADB dial for task %s",
                exc,
                task.task_id,
            )
            loop = asyncio.get_running_loop()
            if await loop.run_in_executor(None, self.calls.dial, task.number):
                await loop.run_in_executor(
                    None,
                    lambda: self.calls.speak_on_call(
                        task.message,
                        delay=getattr(task, "speak_delay_sec", 6.0),
                        repeat=getattr(task, "repeat", 1),
                    ),
                )

    async def run(self) -> None:
        self._running = True
        await self.scheduler.start()
        self.engine.check_ollama(force=True)

        logger.info("Bot started (backend: %s)", self.engine.backend_name)
        logger.info(
            "SMS transport: %s | Wi-Fi bridge: %s | mDNS: %s | port: %s",
            self.config.sms_transport,
            "enabled" if self.config.bridge_enabled else "disabled",
            "on" if self.config.bridge_mdns_enabled else "off",
            self.config.gbrain_api_port,
        )
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
                if self.config.sms_transport == "adb" and not self.adb.is_connected():
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
        if self.config.sms_transport == "adb":
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

        if self.call_answerer is not None:
            await loop.run_in_executor(None, self.call_answerer.poll)

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

    if app.bridge is not None:
        try:
            await app.bridge.start()
            logger.info("Bridge service started (pairing + signaling + SMS)")
        except Exception as exc:
            logger.exception("Bridge service could not start: %s", exc)

    tasks = [asyncio.create_task(app.run())]
    if app.config.gbrain_api_enabled:
        try:
            from bot.api import create_gbrain_app, start_api_server
            api_app = create_gbrain_app(app.config, app)
            tasks.append(asyncio.create_task(start_api_server(api_app, app.config)))
        except Exception as exc:
            logger.warning("GBrain API server could not start: %s", exc)

    try:
        await asyncio.gather(*tasks)
    finally:
        if app.bridge is not None:
            await app.bridge.stop()


def cmd_bridge_identity(config: Config) -> None:
    import json
    import socket
    from bot.bridge.identity import load_or_create_identity

    token_file = Path(config.audiolink_admin_token_file).expanduser()
    data_dir = (
        Path(config.bridge_data_dir).expanduser()
        if config.bridge_data_dir
        else token_file.parent
    )
    name = config.bridge_device_name or socket.gethostname().split(".")[0] or "gbrain-desktop"
    identity = load_or_create_identity(data_dir, name)
    print(json.dumps({
        "device_id": identity.device_id,
        "name": identity.name,
        "fingerprint": identity.fingerprint,
        "data_dir": str(identity.data_dir),
        "cert": str(identity.cert_path),
        "key": str(identity.key_path),
    }, indent=2))


def _paired_device_count(config: Config) -> int:
    try:
        from bot.bridge.registry import DeviceRegistry

        token_file = Path(config.audiolink_admin_token_file).expanduser()
        data_dir = (
            Path(config.bridge_data_dir).expanduser()
            if config.bridge_data_dir
            else token_file.parent
        )
        return len(asyncio.run(DeviceRegistry(data_dir).list()))
    except Exception:
        return 0


def cmd_status(app: BotApp) -> None:
    devices = app.adb.list_devices()
    app.engine.check_ollama(force=True)

    print("=== GBrain SMS Status ===")
    print(f"SMS transport: {app.config.sms_transport}")
    print(f"Wi-Fi bridge: {'enabled' if app.config.bridge_enabled else 'disabled'}")
    print(f"mDNS discovery: {'enabled' if app.config.bridge_mdns_enabled else 'disabled'}")
    print(f"Bridge port: {app.config.gbrain_api_port}")
    print(
        "Bridge state: "
        + ("running" if app.bridge is not None and app.bridge.running else "stopped")
    )
    print(f"Paired phones: {_paired_device_count(app.config)}")
    print(f"ADB (legacy): {', '.join(devices) if devices else 'not connected'}")
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
    if app.config.call_answer_enabled:
        print("Call auto-answer: enabled")
        print(
            f"  Window: {app.config.call_answer_start or '-'} -> "
            f"{app.config.call_answer_end or '-'}"
            + (
                f" (one-off at {app.config.call_answer_at})"
                if app.config.call_answer_at
                else ""
            )
        )
        if app.call_answerer is not None:
            print(f"  {app.call_answerer.describe()}")
    else:
        print("Call auto-answer: disabled")
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


def cmd_call_status(app: BotApp) -> None:
    state = app.calls.get_state()
    print("=== Call status ===")
    print(f"Auto-answer: {'enabled' if app.config.call_answer_enabled else 'disabled'}")
    print(f"Call state: {state.state}")
    print(f"Incoming number: {state.number or '-'}")
    print(
        f"Window: {app.config.call_answer_start or '-'} -> "
        f"{app.config.call_answer_end or '-'}"
    )
    print(f"One-off at: {app.config.call_answer_at or '-'}")
    if app.call_answerer is not None:
        print(f"Window active now: {app.call_answerer.is_active()}")
    if app.config.call_answer_contacts:
        print(f"Allowlist: {', '.join(app.config.call_answer_contacts)}")
    hangup = app.config.call_hangup_after_sec
    print(f"Hang up after: {hangup:.0f}s" if hangup else "Hang up after: manual")
    print(f"Dial strategies: {', '.join(app.config.call_dial_strategies)}")
    print(f"Voice engine: {'enabled' if app.config.voice_enabled else 'disabled'}")
    print(f"Spoken greeting: {'enabled' if app.config.call_tts_enabled else 'disabled'}")
    if app.config.call_tts_enabled:
        print(f"  TTS strategies: {', '.join(app.config.call_tts_strategies)}")
        print(f"  Speakerphone: {app.config.call_speakerphone}")
        volume = app.config.call_tts_volume
        print(f"  Media volume: {volume if volume else 'unchanged'}")
        greeting = app.config.call_greeting.strip() or "(built-in default)"
        print(f"  Greeting: {greeting}")


def cmd_call_answer_now(app: BotApp, *, dry_run: bool) -> None:
    state = app.calls.get_state()
    print("=== Answer call now ===")
    print(f"Call state: {state.state} (number: {state.number or '-'})")
    if state.state != CALL_STATE_RINGING:
        print("Not currently ringing — nothing to answer.")
        return
    if dry_run:
        print(
            "Would try strategies: "
            + ", ".join(app.config.call_answer_strategies)
        )
        print("(dry run — no keyevents sent)")
        return
    if app.calls.answer():
        print("Answered call.")
    else:
        print("Failed to answer call.")
        sys.exit(1)


def cmd_call_tts_check(app: BotApp) -> None:
    print("=== TTS helper availability ===")
    available = app.calls.tts_availability()
    for name in app.config.call_tts_strategies:
        status = "available" if available.get(name) else "unavailable"
        print(f"  {name}: {status}")
    if not any(available.get(n) for n in app.config.call_tts_strategies):
        print(
            "\nNo TTS helper detected. Install Termux + Termux:API "
            "(termux-tts-speak), a helper app, or a PC TTS engine "
            "(espeak-ng / pico2wave) for pc_tts."
        )
    else:
        print("\nA TTS strategy is ready. It speaks during an active call.")


def cmd_call_dial(
    app: BotApp,
    number: str,
    *,
    speak: str,
    hangup_after: float,
    dry_run: bool,
    delay: float | None = None,
    repeat: int | None = None,
) -> None:
    print("=== Dial ===")
    print(f"Number: {number}")
    print(f"Strategies: {', '.join(app.config.call_dial_strategies)}")
    if dry_run:
        print("(dry run — no call placed)")
        return
    if not app.calls.dial(number):
        print("Failed to place call (no strategy connected).")
        sys.exit(1)
    print("Call connected.")
    message = (speak or app.config.call_outgoing_message).strip()
    if not message:
        message = DEFAULT_OUTGOING_MESSAGE
    wait = app.config.call_outgoing_delay_sec if delay is None else delay
    rounds = app.config.call_outgoing_repeat if repeat is None else repeat
    print(f"Speaking in {wait:.0f}s (x{max(1, rounds)}), over speakerphone...")
    if app.calls.speak_on_call(message, delay=wait, repeat=rounds):
        print("Spoke the message.")
    else:
        print("Could not speak the message.")
    if hangup_after > 0:
        print(f"Hanging up in {hangup_after:.0f}s...")
        time.sleep(hangup_after)
        print("Hung up." if app.calls.hangup() else "Hangup failed.")


def cmd_call_hangup(app: BotApp) -> None:
    if app.voice.is_running():
        app.voice.stop(reason="hangup")
    state = app.calls.get_state()
    if state.state == "idle":
        print("No active call.")
        return
    print("Hung up." if app.calls.hangup() else "Hangup failed.")


def cmd_voice_check(app: BotApp) -> None:
    print("=== Voice engine ===")
    for line in app.voice.status_lines():
        print(line)
    state = app.calls.get_state()
    print(f"Phone call state: {state.state} (number: {state.number or '-'})")
    print(
        "\nAcoustic bridge: put the phone on speakerphone next to this computer "
        "with the PC microphone and speakers unmuted."
    )
    print(
        "Cellular call audio is only carried by the room acoustics — a "
        "non-rooted Android phone does not expose the call stream over ADB."
    )


def cmd_voice_selftest(app: BotApp, seconds: float, reply: bool) -> None:
    print("=== Voice self-test ===")
    print(f"Listening on the PC microphone for {seconds:.0f}s — speak now.")
    result = app.voice.self_test(seconds=seconds, reply=reply)
    print(f"Capture: {result['capture_command']}")
    if result.get("input_device"):
        print(f"Microphone: {result['input_device']} (silence window {result['silence_ms']}ms)")
    print(f"Playback: {result['playback_command']}")
    heard = result["heard"] or []
    print(f"Heard: {' | '.join(heard) if heard else '(nothing recognized)'}")
    if result["reply"]:
        print(f"Reply: {result['reply']}")
        print(f"Played back: {result['played']}")
    errors = result["errors"] or []
    for err in errors:
        print(f"Error: {err}")
    if not heard:
        sys.exit(1)


def cmd_voice_run(app: BotApp, hangup_after: float) -> None:
    state = app.calls.get_state()
    print("=== Voice engine on current call ===")
    print(f"Call state: {state.state} (number: {state.number or '-'})")
    if not app.config.voice_enabled:
        print("voice_enabled is false in config.yaml — enable it first.")
        sys.exit(1)
    if not app.voice.start(reason="manual voice-run", contact=state.number or ""):
        print("Voice engine failed to start — see the log for the reason.")
        sys.exit(1)
    print("Listening. Speak into the phone (speakerphone on). Ctrl+C to stop.")
    started = time.monotonic()
    try:
        while app.voice.is_running():
            if app.calls.get_state().state == "idle":
                print("Call ended.")
                break
            if hangup_after > 0 and time.monotonic() - started >= hangup_after:
                print("Hangup time reached.")
                app.calls.hangup()
                break
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        app.voice.stop(reason="voice-run finished")
        print("Voice engine stopped.")


def _bridge_send(config: Config, number: str, message: str) -> dict:
    token_file = Path(config.audiolink_admin_token_file).expanduser()
    token = token_file.read_text(encoding="utf-8").strip()
    payload = json.dumps({"to_number": number, "body": message}).encode("utf-8")
    request = urllib.request.Request(
        f"https://127.0.0.1:{config.gbrain_api_port}/v1/sms/send",
        data=payload,
        headers={
            "Content-Type": "application/json",
            "X-AudioLink-Admin": token,
        },
        method="POST",
    )
    context = ssl._create_unverified_context()
    with urllib.request.urlopen(request, timeout=10, context=context) as response:
        return json.loads(response.read().decode("utf-8"))


def cmd_send(app: BotApp, number: str, message: str) -> None:
    if app.config.bridge_enabled:
        try:
            result = _bridge_send(app.config, number, message)
        except Exception as exc:
            logger.debug("Wi-Fi bridge send unavailable: %s", exc)
        else:
            if result.get("sent"):
                app.memory.save_message(number, "bot", message)
                print(f"Sent over Wi-Fi to {number}: {message}")
                return
            print(f"Wi-Fi bridge could not send to {number}: {result}")
            sys.exit(1)
    if not app.adb.is_connected():
        print(
            "No Wi-Fi paired phone reachable. Start `python -m bot.main run` "
            "and pair the Android app, or connect ADB for legacy send."
        )
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


def cmd_crawl_swahili(
    app: BotApp, limit: int, seed_only: bool, stats: bool
) -> None:
    from bot.swahili_corpus import (
        corpus_stats,
        crawl_swahili_sources,
        seed_builtin_swahili,
    )

    db_path = app.config.resolve_path(
        app.config.corpus_db_path or app.config.db_path
    )

    if stats:
        s = corpus_stats(db_path)
        print(f"Swahili corpus: {s['total']} entries")
        for cat, count in sorted(s.get("by_category", {}).items()):
            print(f"  {cat}: {count}")
        for src, count in sorted(s.get("by_source", {}).items()):
            print(f"  source={src}: {count}")
        return

    built = seed_builtin_swahili(db_path)
    print(f"Seeded {built} built-in Q&A pairs.")

    if seed_only:
        return

    print(f"Crawling Swahili sources (limit={limit})…")
    inserted = crawl_swahili_sources(db_path, max_total=limit)
    print(f"Crawled {inserted} new entries.")


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
        description="GBrain SMS — local Python bot over Wi-Fi (AudioLink bridge)"
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

    sub.add_parser("call-status", help="Show call state and auto-answer window")

    p_call_now = sub.add_parser(
        "call-answer-now",
        help="Answer a ringing call immediately (phase 1)",
    )
    p_call_now.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would happen without sending keyevents",
    )

    sub.add_parser(
        "call-tts-check",
        help="Report which spoken-greeting TTS helpers are available",
    )

    p_dial = sub.add_parser("call-dial", help="Place an outgoing call (phase 3)")
    p_dial.add_argument("number", help="Phone number to dial")
    p_dial.add_argument(
        "--speak",
        default="",
        help="Message to speak after the call connects",
    )
    p_dial.add_argument(
        "--hangup-after",
        type=float,
        default=0.0,
        help="Seconds to wait before hanging up (0 = stay on the call)",
    )
    p_dial.add_argument(
        "--delay",
        type=float,
        default=None,
        help="Seconds to wait for the callee to answer before speaking",
    )
    p_dial.add_argument(
        "--repeat",
        type=int,
        default=None,
        help="How many times to speak the message",
    )
    p_dial.add_argument(
        "--dry-run",
        action="store_true",
        help="Show strategies without placing a call",
    )

    sub.add_parser("call-hangup", help="Hang up the current call")

    sub.add_parser("voice-check", help="Diagnose the real-time voice engine")
    sub.add_parser("bridge-identity", help="Show bridge device ID and pairing fingerprint")

    p_vselftest = sub.add_parser(
        "voice-selftest",
        help="Record the PC microphone, transcribe, reply, and play back",
    )
    p_vselftest.add_argument(
        "--seconds",
        type=float,
        default=5.0,
        help="How long to listen (default: 5)",
    )
    p_vselftest.add_argument(
        "--no-reply",
        action="store_true",
        help="Only transcribe; skip the LLM reply and playback",
    )

    p_vrun = sub.add_parser(
        "voice-run",
        help="Run the real-time voice engine on the current call",
    )
    p_vrun.add_argument(
        "--hangup-after",
        type=float,
        default=0.0,
        help="Hang up after N seconds (0 = until the call ends)",
    )

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
        elif args.command == "bridge-identity":
            cmd_bridge_identity(config)
        elif args.command == "status":
            cmd_status(app)
        elif args.command == "call-status":
            cmd_call_status(app)
        elif args.command == "call-answer-now":
            cmd_call_answer_now(app, dry_run=args.dry_run)
        elif args.command == "call-tts-check":
            cmd_call_tts_check(app)
        elif args.command == "call-dial":
            cmd_call_dial(
                app,
                args.number,
                speak=args.speak,
                hangup_after=args.hangup_after,
                dry_run=args.dry_run,
                delay=args.delay,
                repeat=args.repeat,
            )
        elif args.command == "call-hangup":
            cmd_call_hangup(app)
        elif args.command == "voice-check":
            cmd_voice_check(app)
        elif args.command == "voice-selftest":
            cmd_voice_selftest(app, args.seconds, not args.no_reply)
        elif args.command == "voice-run":
            cmd_voice_run(app, args.hangup_after)
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
