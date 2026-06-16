"""ADB connector for SMS read/send and optional WhatsApp notifications."""

from __future__ import annotations

import logging
import re
import shlex
import subprocess
import time
import unicodedata
from dataclasses import dataclass
from urllib.parse import quote

logger = logging.getLogger(__name__)

SMS_INBOX_TYPE = 1
DEFAULT_SEND_STRATEGIES = ("respond", "compose", "intent", "isms")
SMS_VERIFY_AUTHORITY = "sms"
SMS_VERIFY_WAIT_SEC = 2.5
ADBSMS_PACKAGE = "dev.gonodono.adbsms"
FALLBACK_SMS_PACKAGES = (
    "com.google.android.apps.messaging",
    "com.android.mms",
    "com.samsung.android.messaging",
)
ISMS_CALLING_PACKAGES = (
    "com.android.mms.service",
    "com.android.mms",
    "com.google.android.apps.messaging",
)
ISMS_SUB_IDS = (0, 1)
# sendTextForSubscriber (API 29+) and legacy transaction indices to try.
ISMS_TRANSACTIONS_MODERN = (5, 6, 7)
ISMS_TRANSACTIONS_LEGACY = (7, 6, 5)


@dataclass
class SmsMessage:
    sms_id: int
    address: str
    body: str
    date_ms: int
    msg_type: int = SMS_INBOX_TYPE


@dataclass
class WhatsAppNotification:
    key: str
    title: str
    text: str
    contact: str
    contact_uri: str = ""
    user_id: int | None = None
    chat_uri: str = ""


class AdbError(Exception):
    """Raised when ADB command fails unexpectedly."""


class AdbConnector:
    def __init__(
        self,
        adb_path: str = "adb",
        sms_uri_authority: str = "adbsms",
        whatsapp_enabled: bool = False,
        sms_send_strategies: list[str] | None = None,
    ) -> None:
        self.adb_path = adb_path
        self.sms_uri_authority = sms_uri_authority
        self.whatsapp_enabled = whatsapp_enabled
        self.sms_send_strategies = _normalize_send_strategies(sms_send_strategies)
        self._connected_device: str | None = None
        self._seen_whatsapp_keys: set[str] = set()
        self._backoff_sec = 1.0
        self._resolved_sms_authority: str | None = None

    def _resolve_sms_authority(self) -> str:
        """Pick adbsms or fall back to content://sms when adbsms sees nothing."""
        if self._resolved_sms_authority is not None:
            return self._resolved_sms_authority

        candidates: list[str] = []
        for auth in (self.sms_uri_authority, "sms"):
            if auth not in candidates:
                candidates.append(auth)

        best = self.sms_uri_authority
        best_count = -1
        for auth in candidates:
            count = self._count_sms_rows(f"content://{auth}/inbox")
            if count > best_count:
                best_count = count
                best = auth

        self._resolved_sms_authority = best
        if best != self.sms_uri_authority:
            logger.warning(
                "adbsms returned no messages; using content://%s instead "
                "(grant READ_SMS to adbsms or set sms_uri_authority: sms)",
                best,
            )
        else:
            logger.info("SMS read authority: content://%s (%d inbox rows)", best, max(best_count, 0))
        return best

    def _count_sms_rows(self, uri: str) -> int:
        try:
            output = self._run_adb(
                ["shell", "content", "query", "--uri", uri, "--projection", "_id"]
            )
        except AdbError:
            return 0
        return sum(1 for line in output.splitlines() if line.strip().startswith("Row:"))

    def _query_sms_folder(
        self,
        folder: str,
        *,
        limit: int = 50,
        authority: str | None = None,
    ) -> list[SmsMessage]:
        auth = authority if authority is not None else self._resolve_sms_authority()
        uri = f"content://{auth}/{folder}"
        cmd = [
            "shell",
            "content",
            "query",
            "--uri",
            uri,
            "--projection",
            "_id:address:body:date:type",
            "--sort",
            "date DESC",
        ]
        try:
            output = self._run_adb(cmd)
        except AdbError as exc:
            logger.debug("SMS query failed for %s: %s", uri, exc)
            return []
        messages = self._parse_sms_rows(output)
        return messages[:limit]

    def list_devices(self) -> list[str]:
        try:
            output = self._run_adb(["devices"], device_serial=None)
        except AdbError:
            return []

        devices: list[str] = []
        for line in output.splitlines()[1:]:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 2 and parts[1] == "device":
                devices.append(parts[0])
        return devices

    def wait_for_device(self, timeout: float | None = None) -> str | None:
        start = time.monotonic()
        while True:
            devices = self.list_devices()
            if devices:
                self._connected_device = devices[0]
                self._backoff_sec = 1.0
                logger.info("ADB device connected: %s", self._connected_device)
                return self._connected_device

            self._connected_device = None
            logger.warning("No ADB device found; retrying in %.0fs", self._backoff_sec)
            if timeout is not None and (time.monotonic() - start) >= timeout:
                return None
            time.sleep(self._backoff_sec)
            self._backoff_sec = min(self._backoff_sec * 2, 30.0)

    def is_connected(self) -> bool:
        devices = self.list_devices()
        if not devices:
            self._connected_device = None
            return False
        self._connected_device = devices[0]
        return True

    def device_online(self) -> bool:
        """True when the phone has network access (for WhatsApp polling)."""
        if not self.is_connected():
            return False
        from bot.network import device_has_network

        return device_has_network(self._run_adb)

    def poll_inbox(
        self,
        last_seen_id: int = 0,
        limit: int = 20,
    ) -> list[SmsMessage]:
        try:
            messages = self._query_sms_folder("inbox", limit=max(limit * 3, 50))
        except AdbError as exc:
            logger.error("Failed to poll SMS inbox: %s", exc)
            return []

        new_messages = [
            m
            for m in messages
            if m.sms_id > last_seen_id and m.msg_type == SMS_INBOX_TYPE
        ]
        new_messages.sort(key=lambda m: m.sms_id)
        return new_messages[:limit]

    def poll_sent(self, limit: int = 10) -> list[SmsMessage]:
        return self._query_sms_folder("sent", limit=limit)

    def _poll_sent_for_verify(self, limit: int = 20) -> list[SmsMessage]:
        """Real sent folder (content://sms) — adbsms mirror may not update on send."""
        messages = self._query_sms_folder("sent", limit=limit, authority=SMS_VERIFY_AUTHORITY)
        if messages:
            return messages
        return self._query_sms_folder("sent", limit=limit)

    def _confirm_sms_delivered(self, number: str, body: str) -> bool:
        """Check sent folder for the message (real delivery proof)."""
        target_digits = _phone_digits(number)
        body_norm = body.strip()
        for attempt in range(3):
            time.sleep(SMS_VERIFY_WAIT_SEC if attempt == 0 else 2.0)
            for msg in self._poll_sent_for_verify(limit=20):
                if not _phones_match(msg.address, target_digits):
                    continue
                if _bodies_match(msg.body, body_norm):
                    logger.debug(
                        "Verified sent SMS id=%s to %s (attempt %d)",
                        msg.sms_id,
                        msg.address,
                        attempt + 1,
                    )
                    return True
        logger.debug(
            "No matching sent SMS for %s (checked last 20 sent messages, 3 attempts)",
            number,
        )
        return False

    def send_sms(self, number: str, body: str) -> bool:
        normalized_body = normalize_message(body)
        normalized_number = normalize_phone(number)

        handlers = {
            "respond": self._send_via_respond_via_message,
            "isms": self._send_via_isms,
            "intent": self._send_via_intent,
            "compose": self._send_via_compose_ui,
            "adbsms": self._send_via_adbsms_compose,
        }

        for name in self.sms_send_strategies:
            handler = handlers.get(name)
            if handler is None:
                logger.warning("Unknown SMS send strategy: %s", name)
                continue

            logger.info("Trying SMS send strategy '%s' for %s", name, normalized_number)
            try:
                if handler(normalized_number, normalized_body):
                    if self._confirm_sms_delivered(normalized_number, normalized_body):
                        logger.info("SMS delivered via %s to %s", name, normalized_number)
                        return True
                    logger.warning(
                        "SMS strategy '%s' reported success but message not in sent folder",
                        name,
                    )
            except Exception:
                logger.exception("SMS send strategy '%s' raised an error", name)

            logger.warning("SMS send strategy '%s' failed for %s", name, normalized_number)

        logger.error("All SMS send strategies failed for %s", normalized_number)
        return False

    def poll_whatsapp_notifications(self) -> list[WhatsAppNotification]:
        if not self.whatsapp_enabled:
            return []

        try:
            output = self._run_adb(["shell", "dumpsys", "notification", "--noredact"])
        except AdbError as exc:
            logger.error("Failed to read notifications: %s", exc)
            return []

        notifications = self._parse_whatsapp_notifications(output)
        fresh: list[WhatsAppNotification] = []
        for notif in notifications:
            dedup = f"{notif.key}:{notif.text}"
            if dedup in self._seen_whatsapp_keys:
                continue
            self._seen_whatsapp_keys.add(dedup)
            fresh.append(notif)
        return fresh

    def reply_whatsapp(self, notif: WhatsAppNotification, message: str) -> bool:
        """Reply via DirectReply when possible, else open chat and type message."""
        body = normalize_message(message)
        if self._reply_via_direct(notif, body):
            return True
        if self._reply_via_chat_ui(notif, body):
            return True
        logger.error("All WhatsApp reply strategies failed for %s", notif.contact)
        return False

    def _reply_via_direct(self, notif: WhatsAppNotification, message: str) -> bool:
        contact_uri = notif.contact_uri
        if not contact_uri:
            contact_id_match = re.search(r"/contacts/(\d+)", notif.key)
            if contact_id_match:
                contact_uri = (
                    f"content://com.whatsapp.provider.contact/contacts/"
                    f"{contact_id_match.group(1)}"
                )
        if not contact_uri:
            logger.debug("No WhatsApp contact URI for direct reply")
            return False

        cmd = [
            "shell",
            "am",
            "startservice",
        ]
        if notif.user_id is not None:
            cmd.extend(["--user", str(notif.user_id)])
        cmd.extend([
            "-a",
            "com.whatsapp.intent.action.DIRECT_REPLY_FROM_MESSAGE",
            "-d",
            contact_uri,
            "-n",
            "com.whatsapp/.notification.DirectReplyService",
            "--es",
            "com.whatsapp.Intent.EXTRA_TEXT",
            message,
        ])

        stdout, stderr, returncode = self._run_adb_shell(cmd)
        output = f"{stdout}\n{stderr}".strip()
        if returncode != 0 or "error:" in output.lower():
            logger.warning("WhatsApp direct reply failed: %s", output[:200])
            return False
        logger.info("WhatsApp direct reply sent to %s", notif.contact)
        return True

    def _reply_via_chat_ui(self, notif: WhatsAppNotification, message: str) -> bool:
        """Open the chat and send via input text (requires screen on / no lock)."""
        if notif.chat_uri:
            cmd = ["shell", "am", "start", "-a", "android.intent.action.VIEW", "-d", notif.chat_uri]
        elif notif.contact_uri:
            cmd = [
                "shell",
                "am",
                "start",
                "-a",
                "android.intent.action.VIEW",
                "-d",
                notif.contact_uri,
            ]
        else:
            phone = _extract_phone_from_contact(notif.contact)
            if not phone:
                return False
            cmd = [
                "shell",
                "am",
                "start",
                "-a",
                "android.intent.action.VIEW",
                "-d",
                f"https://wa.me/{phone}",
            ]

        stdout, stderr, returncode = self._run_adb_shell(cmd)
        output = f"{stdout}\n{stderr}".strip()
        if returncode != 0 or not am_start_succeeded(output):
            logger.warning("WhatsApp chat open failed: %s", output[:200])
            return False

        time.sleep(1.5)
        try:
            self._run_adb(["shell", "input", "text", escape_input_text(message)])
            time.sleep(0.3)
            self._run_adb(["shell", "input", "keyevent", "66"])
        except AdbError as exc:
            logger.warning("WhatsApp UI typing/send failed: %s", exc)
            return False
        logger.info("WhatsApp UI reply sent to %s", notif.contact)
        return True

    def get_android_version(self) -> int | None:
        try:
            output = self._run_adb(["shell", "getprop", "ro.build.version.sdk"])
            return int(output.strip())
        except (AdbError, ValueError):
            return None

    def _functional_sms_packages(self) -> list[str]:
        """Real SMS apps that can send — never adbsms (read-only relay)."""
        pkgs: list[str] = []
        default = self.get_default_sms_package()
        if default == ADBSMS_PACKAGE:
            logger.error(
                "adbsms is the default SMS app and cannot send messages. "
                "On the phone: Settings → Apps → Default apps → SMS app → "
                "choose Google Messages (or your carrier Messages app)."
            )
        elif default:
            pkgs.append(default)
        for pkg in FALLBACK_SMS_PACKAGES:
            if pkg not in pkgs and pkg != ADBSMS_PACKAGE:
                pkgs.append(pkg)
        return pkgs

    def _open_real_sms_compose(
        self,
        number: str,
        body: str | None = None,
    ) -> tuple[bool, str]:
        """Open compose UI in the user's real SMS app (not adbsms)."""
        self._wake_screen()
        sms_uri = f"smsto:{number}"
        if body:
            sms_uri = f"smsto:{number}?body={quote(body, safe='')}"
        for pkg in self._functional_sms_packages():
            cmd = [
                "shell",
                "am",
                "start",
                "-a",
                "android.intent.action.SENDTO",
                "-d",
                sms_uri,
                "-p",
                pkg,
            ]
            if body is not None:
                cmd.extend(["--es", "sms_body", body, "--ez", "exit_on_sent", "true"])
            stdout, stderr, returncode = self._run_adb_shell(cmd)
            output = f"{stdout}\n{stderr}".strip()
            if returncode == 0 and am_start_succeeded(output):
                logger.info("Opened SMS compose in %s", pkg)
                return True, pkg
            logger.debug("SMS compose failed for %s: %s", pkg, output[:160])
        return False, ""

    def _get_screen_size(self) -> tuple[int, int] | None:
        try:
            output = self._run_adb(["shell", "wm", "size"])
        except AdbError:
            return None
        match = re.search(r"Physical size:\s*(\d+)x(\d+)", output)
        if not match:
            return None
        return int(match.group(1)), int(match.group(2))

    def _tap_send_via_ui(self) -> bool:
        """Tap the Send control via UI hierarchy, with coordinate fallback."""
        try:
            self._run_adb(["shell", "uiautomator", "dump", "/sdcard/window_dump.xml"])
            xml = self._run_adb(["shell", "cat", "/sdcard/window_dump.xml"])
        except AdbError as exc:
            logger.debug("UI dump failed: %s", exc)
            xml = ""

        coords = _find_send_button_coords(xml)
        if coords:
            self._run_adb(["shell", "input", "tap", str(coords[0]), str(coords[1])])
            time.sleep(0.4)
            return True

        try:
            self._run_adb(["shell", "input", "keyevent", "111"])
            time.sleep(0.3)
        except AdbError:
            pass

        size = self._get_screen_size()
        if size:
            width, height = size
            self._run_adb([
                "shell", "input", "tap",
                str(int(width * 0.92)),
                str(int(height * 0.94)),
            ])
            time.sleep(0.4)
            return True
        return False

    def _send_via_adbsms_compose(self, number: str, body: str) -> bool:
        """adbsms cannot send unless it is default SMS app (not recommended)."""
        if self.get_default_sms_package() != ADBSMS_PACKAGE:
            logger.info(
                "Skipping adbsms compose — adbsms is read-only and shows "
                "'not functional SMS app'. Use your normal Messages app instead."
            )
            return False
        logger.warning(
            "adbsms is default SMS app but has no outbound send support; "
            "switch default SMS back to Messages and retry."
        )
        return False

    def _send_via_intent(self, number: str, body: str) -> bool:
        ok, pkg = self._open_real_sms_compose(number, body)
        if not ok:
            logger.warning("Intent SMS send failed: no functional SMS app opened")
            return False

        time.sleep(4.0)
        if self._tap_send_only():
            return True
        logger.warning("Auto-send keyevents failed for %s; message may need manual send", pkg)
        return False

    def _send_via_compose_ui(self, number: str, body: str) -> bool:
        ok, pkg = self._open_real_sms_compose(number, body)
        if not ok:
            logger.warning("Compose UI open failed: no functional SMS app")
            return False

        time.sleep(4.0)
        if self._tap_send_only():
            return True
        logger.warning("Auto-send keyevents failed for %s; message may need manual send", pkg)
        return False

    def _tap_send_only(self) -> bool:
        """Press send when message body is already filled by intent extras."""
        try:
            self._tap_send_key()
        except AdbError as exc:
            logger.warning("Send keyevent failed: %s", exc)
            return False
        return True

    def _type_and_send(self, body: str) -> bool:
        """Type message and press send (phone must be unlocked)."""
        try:
            self._run_adb(["shell", "input", "text", escape_input_text(body)])
            time.sleep(0.5)
            self._tap_send_key()
        except AdbError as exc:
            logger.warning("Type and send failed: %s", exc)
            return False
        return True

    def _tap_send_key(self) -> None:
        """Tap Send in the SMS app, then fall back to keyevents."""
        if self._tap_send_via_ui():
            return
        for key in (22, 22, 61, 61, 66):
            self._run_adb(["shell", "input", "keyevent", str(key)])
            time.sleep(0.2)

    def _send_via_isms(self, number: str, body: str) -> bool:
        sdk = self.get_android_version()
        default_pkg = self.get_default_sms_package()
        packages = list(ISMS_CALLING_PACKAGES)
        if default_pkg and default_pkg not in packages:
            packages.insert(0, default_pkg)

        transactions = (
            ISMS_TRANSACTIONS_MODERN if sdk is not None and sdk >= 29 else ISMS_TRANSACTIONS_LEGACY
        )

        for sub_id in ISMS_SUB_IDS:
            for calling_pkg in packages:
                for txn in transactions:
                    cmd = self._build_isms_cmd(
                        sdk, txn, sub_id, calling_pkg, number, body
                    )
                    stdout, stderr, returncode = self._run_adb_shell(cmd)
                    output = f"{stdout}\n{stderr}".strip()
                    if returncode != 0:
                        logger.debug(
                            "isms failed (txn=%s, subId=%s, pkg=%s, rc=%s): %s",
                            txn,
                            sub_id,
                            calling_pkg,
                            returncode,
                            output[:120],
                        )
                        continue
                    if isms_call_succeeded(output):
                        logger.debug(
                            "isms parcel ok (txn=%s, subId=%s, pkg=%s): %s",
                            txn,
                            sub_id,
                            calling_pkg,
                            output.strip()[:120],
                        )
                        return True
                    logger.debug(
                        "isms rejected (txn=%s, subId=%s, pkg=%s): %s",
                        txn,
                        sub_id,
                        calling_pkg,
                        output[:120],
                    )
        return False

    def get_default_sms_package(self) -> str | None:
        stdout, _, returncode = self._run_adb_shell(
            ["shell", "cmd", "role", "get-role-holders", "android.app.role.SMS"]
        )
        if returncode == 0:
            for line in stdout.splitlines():
                pkg = line.strip()
                if pkg and not pkg.startswith("Error") and "." in pkg:
                    return pkg

        stdout, _, returncode = self._run_adb_shell(
            ["shell", "cmd", "role", "list-holders", "android.app.role.SMS"]
        )
        if returncode == 0:
            for line in stdout.splitlines():
                pkg = line.strip()
                if pkg and "." in pkg:
                    return pkg

        stdout, _, returncode = self._run_adb_shell(
            ["shell", "settings", "get", "secure", "sms_default_application"]
        )
        if returncode == 0:
            pkg = stdout.strip()
            if pkg and pkg.lower() != "null" and "." in pkg:
                return pkg

        stdout, _, returncode = self._run_adb_shell(["shell", "dumpsys", "role"])
        if returncode == 0:
            match = re.search(
                r'name=android\.app\.role\.SMS\s+holders=([^\s}\]]+)',
                stdout,
            )
            if match and "." in match.group(1):
                return match.group(1).strip()

        return self._resolve_sendto_package()

    def _resolve_sendto_package(self) -> str | None:
        """Fallback: package that handles smsto: intents (usually Messages)."""
        stdout, _, returncode = self._run_adb_shell([
            "shell", "pm", "query-activities",
            "-a", "android.intent.action.SENDTO",
            "-d", "smsto:0",
        ])
        if returncode != 0:
            return None
        for line in stdout.splitlines():
            match = re.search(r"packageName=([^\s]+)", line)
            if match:
                pkg = match.group(1)
                if pkg != ADBSMS_PACKAGE and "." in pkg:
                    return pkg
        return None

    def _wake_screen(self) -> None:
        try:
            self._run_adb(["shell", "input", "keyevent", "224"])
            time.sleep(0.3)
        except AdbError:
            pass

    def _send_via_respond_via_message(self, number: str, body: str) -> bool:
        """Headless send via RESPOND_VIA_MESSAGE on the real default SMS app."""
        packages = self._functional_sms_packages()
        if not packages:
            return False

        attempts: list[list[str]] = []
        for pkg in packages:
            for extra in ("android.intent.extra.TEXT", "sms_body"):
                attempts.append([
                    "shell", "am", "startservice",
                    "-a", "android.intent.action.RESPOND_VIA_MESSAGE",
                    "-d", f"smsto:{number}",
                    "--es", extra, body,
                    pkg,
                ])

        for cmd in attempts:
            stdout, stderr, returncode = self._run_adb_shell(cmd)
            output = f"{stdout}\n{stderr}".strip()
            if returncode == 0 and "error:" not in output.lower():
                logger.debug("respond_via_message: %s", output[:120] or "ok")
                time.sleep(1.0)
                return True
            logger.debug("respond_via_message failed: %s", output[:120])
        return False

    def _build_isms_cmd(
        self,
        sdk: int | None,
        transaction: int,
        sub_id: int,
        calling_pkg: str,
        number: str,
        body: str,
    ) -> list[str]:
        if sdk is not None and sdk >= 29 and transaction == 5:
            return [
                "shell",
                "service",
                "call",
                "isms",
                str(transaction),
                "i32",
                str(sub_id),
                "s16",
                calling_pkg,
                "s16",
                "null",
                "s16",
                number,
                "s16",
                "null",
                "s16",
                body,
                "s16",
                "null",
                "s16",
                "null",
                "i32",
                "0",
                "i64",
                "0",
            ]
        return [
            "shell",
            "service",
            "call",
            "isms",
            str(transaction),
            "i32",
            str(sub_id),
            "s16",
            calling_pkg,
            "s16",
            number,
            "s16",
            "null",
            "s16",
            body,
            "s16",
            "null",
            "s16",
            "null",
        ]

    def _run_adb_shell(
        self, args: list[str], device_serial: str | None = None
    ) -> tuple[str, str, int]:
        serial = device_serial if device_serial is not None else self._connected_device
        cmd = [self.adb_path]
        if serial:
            cmd.extend(["-s", serial])

        if args and args[0] == "shell" and len(args) > 1:
            # adb joins shell args with spaces; quote each token so multi-word
            # values (e.g. sms_body) are not split by the device shell.
            shell_cmd = " ".join(shlex.quote(part) for part in args[1:])
            cmd.extend(["shell", shell_cmd])
        else:
            cmd.extend(args)

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=30,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise AdbError(f"ADB timeout: {' '.join(cmd)}") from exc
        except OSError as exc:
            raise AdbError(f"ADB not available: {exc}") from exc

        return result.stdout or "", result.stderr or "", result.returncode

    def _run_adb(self, args: list[str], device_serial: str | None = None) -> str:
        stdout, stderr, returncode = self._run_adb_shell(args, device_serial)
        if returncode != 0:
            detail = stderr.strip() or stdout.strip() or f"exit code {returncode}"
            raise AdbError(detail)
        return stdout

    @staticmethod
    def _parse_sms_rows(output: str) -> list[SmsMessage]:
        messages: list[SmsMessage] = []
        for line in output.splitlines():
            line = line.strip()
            if not line.startswith("Row:"):
                continue
            fields = _parse_row_line(line)
            try:
                sms_id = int(fields.get("_id", "0"))
                address = unescape_adb_value(fields.get("address", ""))
                body = unescape_adb_value(fields.get("body", ""))
                date_ms = int(fields.get("date", "0"))
                msg_type = int(fields.get("type", str(SMS_INBOX_TYPE)))
            except ValueError:
                continue
            if not address:
                continue
            messages.append(
                SmsMessage(
                    sms_id=sms_id,
                    address=address,
                    body=body,
                    date_ms=date_ms,
                    msg_type=msg_type,
                )
            )
        return messages

    @staticmethod
    def _parse_whatsapp_notifications(output: str) -> list[WhatsAppNotification]:
        notifications: list[WhatsAppNotification] = []
        blocks = re.split(r"(?=NotificationRecord\()", output)
        for block in blocks:
            pkg_match = re.search(r"NotificationRecord\([^)]*?\bpkg=([\w.]+)", block)
            if not pkg_match or pkg_match.group(1) != "com.whatsapp":
                continue

            key_match = re.search(r"\bkey=([^\s\n]+)", block)
            if not key_match or "|com.whatsapp|" not in key_match.group(1):
                continue

            uid_match = re.search(r"\buid=(\d+)\b", block)
            user_id = int(uid_match.group(1)) if uid_match else None

            title = _extract_notification_field(block, "android.title")
            text = _extract_whatsapp_message_text(block)
            if not text:
                continue

            if _is_system_notification(title, text):
                continue

            contact = title.split("@")[0].strip() if title else "unknown"
            if contact.lower() in ("whatsapp", "messages", ""):
                contact = "unknown"

            contact_uri = ""
            direct_match = re.search(
                r"com\.whatsapp\.intent\.action\.DIRECT_REPLY_FROM_MESSAGE "
                r"dat=(content://com\.whatsapp\.provider\.contact/contacts/\d+)",
                block,
            )
            if direct_match:
                contact_uri = direct_match.group(1)

            chat_uri = ""
            chat_match = re.search(
                r"contentIntent=.*?dat=(whatsapp://[^\s\n]+)",
                block,
                re.DOTALL,
            )
            if chat_match:
                chat_uri = chat_match.group(1)

            # Real WhatsApp message notifications have a reply action or chat intent.
            if not contact_uri and not chat_uri:
                continue

            notifications.append(
                WhatsAppNotification(
                    key=key_match.group(1),
                    title=title,
                    text=text,
                    contact=contact,
                    contact_uri=contact_uri,
                    user_id=user_id,
                    chat_uri=chat_uri,
                )
            )
        return notifications


def _extract_notification_field(block: str, field: str) -> str:
    match = re.search(rf"{re.escape(field)}=String \(([^)]*)\)", block)
    return match.group(1) if match else ""


def _extract_whatsapp_message_text(block: str) -> str:
    text = _extract_notification_field(block, "android.text")
    if text and not _is_summary_text(text):
        return text

    big_text = _extract_notification_field(block, "android.bigText")
    if big_text and not _is_summary_text(big_text):
        return big_text

    lines_match = re.search(r"android\.textLines=String\[\]\(([^)]*)\)", block)
    if lines_match:
        lines = re.findall(r"[^\s,][^,)]*", lines_match.group(1))
        if lines:
            last = lines[-1].strip()
            if last and not _is_summary_text(last):
                return last

    messages_match = re.search(
        r"android\.messages=Parcelable\(\d+\)\s*\(\s*Bundle\[{.*?}\]\s*\)",
        block,
        re.DOTALL,
    )
    if messages_match:
        body_match = re.search(r"text=String \(([^)]*)\)", messages_match.group(0))
        if body_match:
            body = body_match.group(1)
            if body and not _is_summary_text(body):
                return body

    return ""


def _is_system_notification(title: str, text: str) -> bool:
    combined = f"{title} {text}".lower()
    system_markers = (
        "select to speak",
        "usb debugging",
        "usb connected",
        "charging",
        "battery",
        "android system",
        "google play services",
        "accessibility",
        "developer options",
        "file transfer",
        "mtp",
        "adb",
    )
    return any(marker in combined for marker in system_markers)


def _is_summary_text(text: str) -> bool:
    lower = text.lower().strip()
    if re.match(r"^\d+\s+new messages?$", lower):
        return True
    if lower in ("whatsapp", "checking for new messages"):
        return True
    return False


def _extract_phone_from_contact(contact: str) -> str:
    digits = re.sub(r"\D", "", contact)
    return digits if len(digits) >= 9 else ""


def _normalize_send_strategies(strategies: list[str] | None) -> list[str]:
    if not strategies:
        return list(DEFAULT_SEND_STRATEGIES)
    valid = {"respond", "isms", "intent", "compose", "adbsms"}
    normalized = [s for s in strategies if s in valid]
    return normalized or list(DEFAULT_SEND_STRATEGIES)


def am_start_succeeded(output: str) -> bool:
    lower = output.lower()
    if not output.strip():
        return True
    if "error:" in lower:
        return False
    if "unable to resolve intent" in lower:
        return False
    pkg_match = re.search(r"\bpkg=([^\s}]+)", output)
    if pkg_match and "." not in pkg_match.group(1):
        return False
    return True


def isms_call_succeeded(output: str) -> bool:
    if not output.strip():
        return False
    lower = output.lower()
    failure_markers = (
        "permission denial",
        "securityexception",
        "java.lang.",
        "not found",
        "unknown transaction",
        "error",
    )
    if any(marker in lower for marker in failure_markers):
        return False
    if "result: parcel" in lower:
        return True
    return False


def normalize_message(text: str, max_len: int = 1000) -> str:
    normalized = unicodedata.normalize("NFC", text or "")
    cleaned = "".join(ch for ch in normalized if ch >= " " or ch in "\n\t")
    cleaned = cleaned.strip()
    if len(cleaned) > max_len:
        cleaned = cleaned[: max_len - 3] + "..."
    return cleaned


def normalize_phone(number: str) -> str:
    number = (number or "").strip()
    if number.startswith("sms:"):
        number = number[4:]
    return number


def _phone_digits(number: str) -> str:
    return re.sub(r"\D", "", number or "")


def _phones_match(address: str, target_digits: str) -> bool:
    addr_digits = _phone_digits(address)
    if not addr_digits or not target_digits:
        return False
    if addr_digits == target_digits:
        return True
    # Compare significant suffix (handles +255 vs 0 local prefix)
    min_len = min(len(addr_digits), len(target_digits), 9)
    return addr_digits[-min_len:] == target_digits[-min_len:]


def _bodies_match(sent_body: str, expected: str) -> bool:
    sent = (sent_body or "").strip()
    exp = (expected or "").strip()
    if not sent or not exp:
        return False
    if sent == exp or exp in sent or sent in exp:
        return True
    prefix_len = min(20, len(sent), len(exp))
    return prefix_len >= 10 and sent[:prefix_len] == exp[:prefix_len]


def escape_shell_arg(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("'", "\\'")


def escape_input_text(text: str) -> str:
    """Escape text for `adb shell input text` (%s = space, %% = percent)."""
    result = text.replace("\\", "\\\\")
    result = result.replace("%", "%%")
    result = result.replace(" ", "%s")
    for char in "&;|<>(){}[]$\"'":
        result = result.replace(char, f"\\{char}")
    return result


def unescape_adb_value(value: str) -> str:
    value = value.replace("\\n", "\n").replace("\\t", "\t")
    value = re.sub(r"\\(.)", r"\1", value)
    return value


def _find_send_button_coords(xml: str) -> tuple[int, int] | None:
    if not xml:
        return None
    patterns = (
        r'resource-id="Compose:Draft:Send"[^>]*bounds="\[(\d+),(\d+)\]\[(\d+),(\d+)\]"',
        r'content-desc="Send SMS"[^>]*bounds="\[(\d+),(\d+)\]\[(\d+),(\d+)\]"',
        r'bounds="\[(\d+),(\d+)\]\[(\d+),(\d+)\]"[^>]*content-desc="Send SMS"',
        r'content-desc="Send"[^>]*bounds="\[(\d+),(\d+)\]\[(\d+),(\d+)\]"',
    )
    for pattern in patterns:
        for match in re.finditer(pattern, xml):
            x1, y1, x2, y2 = (int(match.group(i)) for i in range(1, 5))
            if x2 > x1 and y2 > y1:
                return (x1 + x2) // 2, (y1 + y2) // 2
    return None


def _parse_row_line(line: str) -> dict[str, str]:
    content = line[len("Row:") :].strip()
    content = re.sub(r"^\d+\s+", "", content)
    match = re.match(
        r"_id=(\d+),\s*address=(.*?),\s*body=(.*),\s*date=(\d+)(?:,\s*type=(\d+))?$",
        content,
    )
    if match:
        fields = {
            "_id": match.group(1),
            "address": match.group(2),
            "body": match.group(3),
            "date": match.group(4),
        }
        if match.group(5):
            fields["type"] = match.group(5)
        return fields

    fields: dict[str, str] = {}
    for part in content.split(", "):
        if "=" not in part:
            continue
        key, val = part.split("=", 1)
        fields[key.strip()] = val.strip()
    return fields
