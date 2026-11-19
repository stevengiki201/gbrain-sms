"""ADB-based phone call detection and control (answer / hang up).

Phase 1: detect incoming calls and answer them through `input keyevent`
injection. There is no call-audio access over plain ADB, so this module does
not speak, play audio, or transcribe — it only picks up and hangs up.
"""

from __future__ import annotations

import logging
import os
import re
import shlex
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass

from bot.adb_connector import AdbConnector, AdbError

logger = logging.getLogger(__name__)

DEFAULT_OUTGOING_MESSAGE = (
    "Hello, this is gbrain, an automated assistant calling from this phone."
)

CALL_STATE_IDLE = "idle"
CALL_STATE_RINGING = "ringing"
CALL_STATE_OFFHOOK = "offhook"
CALL_STATE_UNKNOWN = "unknown"

DEFAULT_ANSWER_STRATEGIES = ("keyevent_call", "headsethook", "ui_tap")

_KEYCODE_WAKEUP = "224"
_KEYCODE_CALL = "5"
_KEYCODE_ENDCALL = "6"
_KEYCODE_HEADSETHOOK = "79"


@dataclass
class CallState:
    state: str = CALL_STATE_UNKNOWN
    number: str = ""


class CallConnector:
    def __init__(
        self,
        adb: AdbConnector,
        answer_strategies: list[str] | None = None,
        tts_strategies: list[str] | None = None,
        tts_helper_action: str = "com.gbrain.sms.SPEAK",
        tts_command: str = "",
        tts_volume: int = 0,
        dial_strategies: list[str] | None = None,
        dial_timeout_sec: float = 20.0,
    ) -> None:
        self.adb = adb
        self.answer_strategies = _normalize_strategies(answer_strategies)
        self.tts_strategies = list(
            tts_strategies or ["termux", "helper_intent", "pc_tts"]
        )
        self.tts_helper_action = tts_helper_action
        self.tts_command = tts_command
        self.tts_volume = tts_volume
        self.dial_strategies = list(
            dial_strategies or ["intent_call", "dialer_keyevent"]
        )
        self.dial_timeout_sec = dial_timeout_sec

    # ── Detection ────────────────────────────────────────────────────
    def get_state(self) -> CallState:
        try:
            output = self.adb._run_adb(["shell", "dumpsys", "telephony.registry"])
        except AdbError as exc:
            logger.debug("telephony.registry dump failed: %s", exc)
            return CallState(CALL_STATE_UNKNOWN, "")

        state = _parse_registry_state(output)
        number = _parse_registry_number(output)
        if state == CALL_STATE_UNKNOWN:
            telecom = self._dump_telecom()
            if telecom:
                state = _parse_telecom_state(telecom)
        return CallState(state, number)

    def _dump_telecom(self) -> str:
        try:
            return self.adb._run_adb(["shell", "dumpsys", "telecom"])
        except AdbError as exc:
            logger.debug("telecom dump failed: %s", exc)
            return ""

    # ── Control ──────────────────────────────────────────────────────
    def answer(self) -> bool:
        self._wake_screen()
        handlers = {
            "keyevent_call": self._answer_keyevent_call,
            "headsethook": self._answer_headsethook,
            "ui_tap": self._answer_ui_tap,
        }
        for name in self.answer_strategies:
            handler = handlers.get(name)
            if handler is None:
                logger.warning("Unknown call-answer strategy: %s", name)
                continue

            logger.info("Trying call-answer strategy '%s'", name)
            try:
                if not handler():
                    logger.debug("Call-answer strategy '%s' did nothing", name)
                    continue
            except AdbError as exc:
                logger.warning("Call-answer strategy '%s' failed: %s", name, exc)
                continue

            if self._wait_for_state(CALL_STATE_OFFHOOK):
                state = self.get_state()
                logger.info(
                    "Answered incoming call from %s via %s",
                    state.number or "unknown",
                    name,
                )
                return True
            logger.warning("Strategy '%s' ran but call did not go off-hook", name)

        logger.error("All call-answer strategies failed")
        return False

    def hangup(self) -> bool:
        for key in (_KEYCODE_ENDCALL, _KEYCODE_HEADSETHOOK):
            try:
                self.adb._run_adb(["shell", "input", "keyevent", key])
            except AdbError as exc:
                logger.warning("Hang-up keyevent %s failed: %s", key, exc)
                continue
            if self._wait_for_state(CALL_STATE_IDLE):
                logger.info("Hung up call via keyevent %s", key)
                return True
        logger.error("Failed to hang up call")
        return False

    def _answer_keyevent_call(self) -> bool:
        self.adb._run_adb(["shell", "input", "keyevent", _KEYCODE_CALL])
        time.sleep(0.8)
        return True

    def _answer_headsethook(self) -> bool:
        self.adb._run_adb(["shell", "input", "keyevent", _KEYCODE_HEADSETHOOK])
        time.sleep(0.8)
        return True

    def _answer_ui_tap(self) -> bool:
        try:
            self.adb._run_adb(["shell", "uiautomator", "dump", "/sdcard/call_dump.xml"])
            xml = self.adb._run_adb(["shell", "cat", "/sdcard/call_dump.xml"])
        except AdbError as exc:
            logger.debug("Call UI dump failed: %s", exc)
            return False

        coords = _find_answer_coords(xml)
        if not coords:
            logger.debug("No answer button found in call UI")
            return False
        self.adb._run_adb(
            ["shell", "input", "tap", str(coords[0]), str(coords[1])]
        )
        time.sleep(0.5)
        return True

    # ── Helpers ──────────────────────────────────────────────────────
    def _wake_screen(self) -> None:
        try:
            self.adb._run_adb(["shell", "input", "keyevent", _KEYCODE_WAKEUP])
            time.sleep(0.2)
        except AdbError:
            pass

    def _wait_for_state(
        self,
        expected: str,
        attempts: int = 4,
        delay: float = 0.6,
    ) -> bool:
        for _ in range(attempts):
            if self.get_state().state == expected:
                return True
            time.sleep(delay)
        return False

    # ── Phase 3: outgoing calls ─────────────────────────────────────
    def dial(self, number: str) -> bool:
        cleaned = _sanitize_number(number)
        if not cleaned:
            logger.warning("Refusing to dial invalid number: %r", number)
            return False
        handlers = {
            "intent_call": self._dial_intent,
            "dialer_keyevent": self._dial_keyevent,
        }
        for name in self.dial_strategies:
            handler = handlers.get(name)
            if handler is None:
                logger.warning("Unknown dial strategy: %s", name)
                continue
            try:
                if handler(cleaned):
                    logger.info("Placed call to %s via '%s'", cleaned, name)
                    return True
                logger.warning("Dial strategy '%s' did not connect", name)
            except (AdbError, OSError) as exc:
                logger.warning("Dial strategy '%s' failed: %s", name, exc)
        return False

    def _dial_attempts(self) -> int:
        return max(3, int(self.dial_timeout_sec / 0.6))

    def _dial_intent(self, number: str) -> bool:
        stdout, stderr, rc = self.adb._run_adb_shell(
            [
                "shell",
                "am",
                "start",
                "-a",
                "android.intent.action.CALL",
                "-d",
                f"tel:{number}",
            ]
        )
        output = f"{stdout}\n{stderr}".lower()
        if rc != 0 or any(
            token in output
            for token in (
                "permission denial",
                "securityexception",
                "error",
                "unable",
            )
        ):
            logger.debug("CALL intent rejected: %s", output[:160])
            return False
        return self._wait_for_state("offhook", attempts=self._dial_attempts())

    def _dial_keyevent(self, number: str) -> bool:
        self.adb._run_adb(
            [
                "shell",
                "am",
                "start",
                "-a",
                "android.intent.action.DIAL",
                "-d",
                f"tel:{number}",
            ]
        )
        time.sleep(1.5)
        self.adb._run_adb(["shell", "input", "keyevent", "5"])
        return self._wait_for_state("offhook", attempts=self._dial_attempts())

    # ── Phase 2: speakerphone + spoken greeting ─────────────────────
    def set_speakerphone(self) -> bool:
        for _ in range(3):
            try:
                self.adb._run_adb(
                    ["shell", "uiautomator", "dump", "/sdcard/call_dump.xml"]
                )
                xml = self.adb._run_adb(
                    ["shell", "cat", "/sdcard/call_dump.xml"]
                )
            except AdbError as exc:
                logger.debug("Speaker UI dump failed: %s", exc)
                time.sleep(0.5)
                continue

            found = _find_speaker_button(xml)
            if not found:
                logger.warning("Speaker button not found in call UI")
                time.sleep(0.5)
                continue
            coords, already_on = found
            if already_on:
                logger.info("Speakerphone already on")
                return True
            try:
                self.adb._run_adb(
                    ["shell", "input", "tap", str(coords[0]), str(coords[1])]
                )
            except AdbError as exc:
                logger.warning("Speaker tap failed: %s", exc)
                return False
            time.sleep(0.4)
            logger.info("Enabled speakerphone")
            return True
        return False

    def speak_on_call(
        self,
        text: str,
        *,
        delay: float = 0.0,
        repeat: int = 1,
    ) -> bool:
        """Speak to the callee over speakerphone, after they answer."""
        text = (text or "").strip()
        if not text:
            return False
        if delay > 0:
            time.sleep(delay)
        if not self.set_speakerphone():
            logger.warning(
                "Could not enable speakerphone; the callee may not hear the message"
            )
        delivered = False
        rounds = max(1, repeat)
        for index in range(rounds):
            if self.speak(text):
                delivered = True
            if index + 1 < rounds:
                time.sleep(3.0)
        return delivered

    def tts_availability(self) -> dict[str, bool]:
        return {
            "termux": self._termux_available(),
            "helper_intent": self._helper_available(),
            "pc_tts": self._pc_tts_available(),
        }

    def speak(self, text: str) -> bool:
        text = (text or "").strip()
        if not text:
            return False
        if self.tts_volume > 0:
            try:
                self.adb._run_adb(
                    ["shell", "media", "volume", "--stream", "3", "--set",
                     str(self.tts_volume)]
                )
            except AdbError as exc:
                logger.debug("Could not raise media volume: %s", exc)

        handlers = {
            "termux": self._speak_termux,
            "helper_intent": self._speak_helper,
            "pc_tts": self._speak_pc,
        }
        for name in self.tts_strategies:
            handler = handlers.get(name)
            if handler is None:
                logger.warning("Unknown TTS strategy: %s", name)
                continue
            try:
                if handler(text):
                    logger.info("Spoke greeting via '%s'", name)
                    return True
            except (AdbError, OSError) as exc:
                logger.warning("TTS strategy '%s' failed: %s", name, exc)
        logger.warning("No TTS strategy could speak the greeting")
        return False

    def _termux_available(self) -> bool:
        try:
            out = self.adb._run_adb(
                ["shell", "sh", "-c", "command -v termux-tts-speak"]
            )
        except AdbError:
            return False
        return bool(out.strip())

    def _speak_termux(self, text: str) -> bool:
        if not self._termux_available():
            return False
        self.adb._run_adb(
            ["shell", "termux-tts-speak", "-e", "com.google.android.tts", text]
        )
        return True

    def _helper_available(self) -> bool:
        try:
            out = self.adb._run_adb(
                ["shell", "pm", "query-services", "-a", self.tts_helper_action]
            )
        except AdbError:
            return False
        return "packageName=" in out

    def _speak_helper(self, text: str) -> bool:
        stdout, stderr, rc = self.adb._run_adb_shell(
            [
                "shell",
                "am",
                "startservice",
                "-a",
                self.tts_helper_action,
                "--es",
                "text",
                text,
            ]
        )
        output = f"{stdout}\n{stderr}".lower()
        if rc != 0 or "error" in output or "unable" in output:
            logger.debug("Helper TTS intent failed: %s", output[:160])
            return False
        return True

    def _prepare_playback_uri(self, remote: str) -> str:
        """Index the pushed file and prefer a content:// URI for playback."""
        file_uri = f"file://{remote}"
        try:
            self.adb._run_adb(
                [
                    "shell",
                    "am",
                    "broadcast",
                    "-a",
                    "android.intent.action.MEDIA_SCANNER_SCAN_FILE",
                    "-d",
                    file_uri,
                ]
            )
            time.sleep(0.6)
            out = self.adb._run_adb(
                [
                    "shell",
                    "content",
                    "query",
                    "--uri",
                    "content://media/external/audio/media",
                    "--projection",
                    "_id",
                    "--where",
                    f"_display_name='{os.path.basename(remote)}'",
                ]
            )
        except AdbError as exc:
            logger.debug("MediaStore indexing failed: %s", exc)
            return file_uri
        match = re.search(r"_id=(\d+)", out)
        if match:
            return f"content://media/external/audio/media/{match.group(1)}"
        return file_uri

    def _resolve_audio_handler(self, uri: str) -> str | None:
        try:
            out = self.adb._run_adb(
                [
                    "shell",
                    "cmd",
                    "package",
                    "resolve-activity",
                    "--brief",
                    "-a",
                    "android.intent.action.VIEW",
                    "-d",
                    uri,
                    "-t",
                    "audio/wav",
                ]
            )
        except AdbError:
            return None
        for line in reversed(
            [part.strip() for part in out.splitlines() if part.strip()]
        ):
            if "/" in line and " " not in line and not line.startswith("priority"):
                return line
        return None

    def _pc_tts_available(self) -> bool:
        if self.tts_command.strip():
            return True
        return any(
            shutil.which(binary)
            for binary in ("espeak-ng", "espeak", "pico2wave")
        )

    def _speak_pc(self, text: str) -> bool:
        wav = self._synthesize_wav(text)
        if not wav:
            return False
        remote = "/sdcard/Download/gbrain_greeting.wav"
        try:
            self.adb._run_adb(["push", wav, remote])
            uri = self._prepare_playback_uri(remote)
            args = ["shell", "am", "start"]
            component = self._resolve_audio_handler(uri)
            if component:
                args.extend(["-n", component])
            args.extend(
                [
                    "-a",
                    "android.intent.action.VIEW",
                    "-d",
                    uri,
                    "-t",
                    "audio/wav",
                    "-f",
                    "0x10000000",
                ]
            )
            stdout, stderr, rc = self.adb._run_adb_shell(args)
            output = f"{stdout}\n{stderr}".lower()
            if rc != 0 or "error" in output:
                logger.debug("PC TTS playback failed: %s", output[:160])
                return False
            return True
        finally:
            try:
                os.unlink(wav)
            except OSError:
                pass

    def _synthesize_wav(self, text: str) -> str | None:
        fd, out = tempfile.mkstemp(prefix="gbrain_greeting_", suffix=".wav")
        os.close(fd)
        try:
            if self.tts_command.strip():
                parts = [
                    token.replace("{out}", out).replace("{text}", text)
                    for token in shlex.split(self.tts_command)
                ]
                subprocess.run(
                    parts, check=True, capture_output=True, timeout=30
                )
            else:
                for binary in ("espeak-ng", "espeak", "pico2wave"):
                    path = shutil.which(binary)
                    if not path:
                        continue
                    subprocess.run(
                        [path, "-w", out, text],
                        check=True,
                        capture_output=True,
                        timeout=30,
                    )
                    break
                else:
                    os.unlink(out)
                    return None
        except (OSError, subprocess.SubprocessError) as exc:
            logger.warning("PC TTS synthesis failed: %s", exc)
            try:
                os.unlink(out)
            except OSError:
                pass
            return None

        if not os.path.exists(out) or os.path.getsize(out) == 0:
            try:
                os.unlink(out)
            except OSError:
                pass
            return None
        return out


def _sanitize_number(number: str) -> str:
    if not number:
        return ""
    cleaned = re.sub(r"[^\d+*#,;]", "", number)
    if cleaned.count("+") > 1:
        return ""
    if "+" in cleaned and not cleaned.startswith("+"):
        return ""
    if not re.search(r"\d", cleaned):
        return ""
    return cleaned


def _normalize_strategies(strategies: list[str] | None) -> list[str]:
    valid = {"keyevent_call", "headsethook", "ui_tap"}
    if not strategies:
        return list(DEFAULT_ANSWER_STRATEGIES)
    normalized = [s for s in strategies if s in valid]
    return normalized or list(DEFAULT_ANSWER_STRATEGIES)


def _parse_registry_state(output: str) -> str:
    match = re.search(r"mCallState=(\d+)", output)
    if not match:
        return CALL_STATE_UNKNOWN
    code = int(match.group(1))
    return {
        0: CALL_STATE_IDLE,
        1: CALL_STATE_RINGING,
        2: CALL_STATE_OFFHOOK,
    }.get(code, CALL_STATE_UNKNOWN)


def _parse_registry_number(output: str) -> str:
    match = re.search(r"mCallIncomingNumber=([^\s]*)", output)
    return match.group(1).strip() if match else ""


def _parse_telecom_state(output: str) -> str:
    """Conservative fallback for ROMs that redact telephony.registry.

    Android's telecom dump keeps historical "SET_RINGING" log lines after a
    call ends, so we only trust an explicit current-state field here — never
    the log section — to avoid answering phantom calls.
    """
    if re.search(r"(?:CallState|call_state|=)\s*[:=]?\s*RINGING", output):
        return CALL_STATE_RINGING
    if re.search(r"(?:CallState|call_state|=)\s*[:=]?\s*OFFHOOK", output):
        return CALL_STATE_OFFHOOK
    if re.search(r"(?:CallState|call_state|=)\s*[:=]?\s*IDLE", output):
        return CALL_STATE_IDLE
    return CALL_STATE_UNKNOWN


def _find_speaker_button(xml: str) -> tuple[tuple[int, int], bool] | None:
    if not xml:
        return None
    for tag in re.findall(r"<node\b[^>]*>", xml):
        label = re.search(r'(?:text|content-desc)="([^"]*)"', tag)
        if not label or "speaker" not in label.group(1).lower():
            continue
        bounds = re.search(r'bounds="\[(\d+),(\d+)\]\[(\d+),(\d+)\]"', tag)
        if not bounds:
            continue
        x1, y1, x2, y2 = (int(bounds.group(i)) for i in range(1, 5))
        if x2 <= x1 or y2 <= y1:
            continue
        text = label.group(1).strip().lower()
        already_on = 'checked="true"' in tag or text.endswith(" on")
        return ((x1 + x2) // 2, (y1 + y2) // 2), already_on
    return None


def _find_answer_coords(xml: str) -> tuple[int, int] | None:
    if not xml:
        return None

    for match in re.finditer(
        r'(?:text|content-desc)="([^"]*)"[^>]*bounds="\[(\d+),(\d+)\]\[(\d+),(\d+)\]"',
        xml,
    ):
        if "answer" in match.group(1).lower():
            x1, y1, x2, y2 = (int(match.group(i)) for i in range(2, 6))
            if x2 > x1 and y2 > y1:
                return (x1 + x2) // 2, (y1 + y2) // 2

    for match in re.finditer(
        r'bounds="\[(\d+),(\d+)\]\[(\d+),(\d+)\]"[^>]*(?:text|content-desc)="([^"]*)"',
        xml,
    ):
        if "answer" in match.group(5).lower():
            x1, y1, x2, y2 = (int(match.group(i)) for i in range(1, 5))
            if x2 > x1 and y2 > y1:
                return (x1 + x2) // 2, (y1 + y2) // 2

    return None
