"""Speech synthesis for the voice engine (Piper, or espeak-ng as fallback).

Piper gives natural voices; espeak-ng is already installed in the repo and
supports Kiswahili, so it is the zero-download default.
"""

from __future__ import annotations

import logging
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import wave
from pathlib import Path

logger = logging.getLogger(__name__)


def _piper_python_ready() -> bool:
    """True when the ``piper`` Python package is importable from the local libs."""
    libs = str(PIPER_LIBS)
    if PIPER_LIBS.is_dir() and libs not in sys.path:
        sys.path.append(libs)
    try:
        import piper  # noqa: F401,PLC0415 - optional dependency
    except Exception:
        return False
    return True

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BUNDLED_ESPEAK = PROJECT_ROOT / "tools" / "tts" / "root" / "usr" / "bin" / "espeak-ng"
BUNDLED_LIB = PROJECT_ROOT / "tools" / "tts" / "root" / "usr" / "lib" / "x86_64-linux-gnu"
BUNDLED_DATA = BUNDLED_LIB / "espeak-ng-data"
OUTPUT_DIR = PROJECT_ROOT / "data" / "voice"
PIPER_LIBS = PROJECT_ROOT / "tools" / "voice" / "pylibs"


class SpeechSynthesizer:
    def __init__(
        self,
        *,
        provider: str = "espeak",
        voice: str = "",
        command: str = "",
        piper_binary: str = "piper",
        piper_model: str = "",
    ) -> None:
        self.provider = (provider or "espeak").lower()
        self.voice = voice.strip()
        self.command = command.strip()
        self.piper_binary = piper_binary or "piper"
        self.piper_model = piper_model.strip()
        self._piper_voice = None
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Diagnostics ──────────────────────────────────────────────────
    def availability(self) -> tuple[bool, str]:
        if self.provider == "piper":
            if not self.piper_model:
                return False, "voice_piper_model is not set"
            if not Path(self.piper_model).is_file():
                return False, f"piper model missing: {self.piper_model}"
            if _piper_python_ready():
                return True, f"piper via python ({self.piper_model})"
            binary = self._piper_binary()
            if not binary:
                return False, f"piper binary not found ({self.piper_binary})"
            return True, f"piper ({self.piper_model})"
        if self.command:
            return True, f"command: {self.command}"
        binary = self._espeak_binary()
        if not binary:
            return False, "espeak-ng not found (tools/tts or PATH)"
        return True, f"espeak-ng ({self._espeak_voice('auto') or 'default'})"

    # ── Synthesis ────────────────────────────────────────────────────
    def synthesize(self, text: str, *, language: str = "auto") -> Path | None:
        text = (text or "").strip()
        if not text:
            return None
        handle, raw_path = tempfile.mkstemp(prefix="gbrain-tts-", suffix=".wav", dir=OUTPUT_DIR)
        os.close(handle)
        out = Path(raw_path)

        if self.provider == "piper":
            ok = self._run_piper(text, out)
        elif self.command:
            ok = self._run_command(text, out)
        else:
            ok = self._run_espeak(text, out, language)

        if not ok or not out.is_file() or out.stat().st_size == 0:
            logger.error("TTS audio generation failed (provider: %s)", self.provider)
            try:
                out.unlink()
            except OSError:
                pass
            return None
        return out

    @staticmethod
    def cleanup(path: Path | None) -> None:
        if path is None:
            return
        try:
            path.unlink()
        except OSError:
            pass

    # ── Engines ──────────────────────────────────────────────────────
    def _run_espeak(self, text: str, out: Path, language: str) -> bool:
        binary = self._espeak_binary()
        if not binary:
            logger.error("espeak-ng is not installed")
            return False
        voice = self._espeak_voice(language)
        argv = [binary, "-w", str(out)]
        if voice:
            argv += ["-v", voice]
        argv.append(text)
        return self._run(argv, "espeak-ng")

    def _run_piper(self, text: str, out: Path) -> bool:
        # The 2023 standalone binary rejects voices whose phoneme map contains
        # multi-codepoint entries (e.g. the Swahili voice's "aɪ"), so prefer the
        # Python package and fall back to the binary only if it is unavailable.
        if self._run_piper_python(text, out):
            return True
        binary = self._piper_binary()
        if not binary:
            logger.error("piper binary not found: %s", self.piper_binary)
            return False
        argv = [
            binary,
            "--model",
            self.piper_model,
            "--output_file",
            str(out),
        ]
        return self._run(argv, "piper", stdin_text=text)

    def _run_piper_python(self, text: str, out: Path) -> bool:
        if not _piper_python_ready():
            return False
        try:
            if self._piper_voice is None:
                from piper import PiperVoice  # noqa: PLC0415 - optional dependency

                self._piper_voice = PiperVoice.load(self.piper_model)
            with wave.open(str(out), "wb") as handle:
                self._piper_voice.synthesize_wav(text, handle)
        except Exception as exc:  # pragma: no cover - depends on local install
            logger.warning("in-process piper failed (%s); trying binary", exc)
            self._piper_voice = None
            return False
        return True

    def _run_command(self, text: str, out: Path) -> bool:
        argv = [
            part.replace("{out}", str(out)).replace("{text}", text)
            for part in shlex.split(self.command)
        ]
        return self._run(argv, "tts command")

    @staticmethod
    def _run(
        argv: list[str],
        label: str,
        *,
        stdin_text: str | None = None,
        timeout: float = 30.0,
    ) -> bool:
        env = os.environ.copy()
        if BUNDLED_LIB.is_dir():
            env["LD_LIBRARY_PATH"] = (
                f"{BUNDLED_LIB}{':' + env['LD_LIBRARY_PATH'] if env.get('LD_LIBRARY_PATH') else ''}"
            )
        if BUNDLED_DATA.is_dir():
            env.setdefault("ESPEAK_DATA_PATH", str(BUNDLED_DATA))
        try:
            result = subprocess.run(
                argv,
                input=stdin_text.encode("utf-8") if stdin_text is not None else None,
                capture_output=True,
                timeout=timeout,
                env=env,
                check=False,
            )
        except subprocess.TimeoutExpired:
            logger.error("%s timed out", label)
            return False
        except OSError as exc:
            logger.error("%s failed: %s", label, exc)
            return False
        if result.returncode != 0:
            logger.error(
                "%s failed (%s): %s",
                label,
                result.returncode,
                result.stderr.decode("utf-8", "replace").strip()[:200],
            )
            return False
        return True

    # ── Locators ─────────────────────────────────────────────────────
    def _espeak_binary(self) -> str | None:
        if BUNDLED_ESPEAK.is_file():
            return str(BUNDLED_ESPEAK)
        return shutil.which("espeak-ng") or shutil.which("espeak")

    def _piper_binary(self) -> str | None:
        candidate = Path(self.piper_binary)
        if candidate.is_file():
            return str(candidate)
        return shutil.which(self.piper_binary)

    def _espeak_voice(self, language: str) -> str:
        if self.voice:
            return self.voice
        language = (language or "auto").lower()
        if language.startswith("sw"):
            return "sw"
        if language.startswith("en"):
            return "en"
        return ""

    def describe(self) -> str:
        ok, detail = self.availability()
        return f"{'ok' if ok else 'unavailable'} — {detail}"
