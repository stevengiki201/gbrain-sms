"""Acoustic audio bridge between the PC sound card and the phone on a call.

A non-rooted Android phone does not expose cellular call audio, so the bridge
is physical: the phone sits on speakerphone next to the computer, the PC
microphone captures the caller, and the PC speakers feed the phone's
microphone. Audio never leaves the room, but the caller does hear it.
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import time
from pathlib import Path

from bot.voice.vad import rms_level

logger = logging.getLogger(__name__)


class AcousticBridge:
    def __init__(
        self,
        *,
        sample_rate: int = 16000,
        input_device: str = "auto",
        output_device: str = "auto",
        chunk_ms: int = 100,
    ) -> None:
        self.sample_rate = sample_rate
        self.input_device = (input_device or "auto").strip()
        self.output_device = (output_device or "auto").strip()
        self.chunk_ms = chunk_ms
        self._proc: subprocess.Popen[bytes] | None = None

    # ── Command discovery ────────────────────────────────────────────
    @property
    def chunk_bytes(self) -> int:
        return int(self.sample_rate * self.chunk_ms / 1000) * 2

    def capture_argv(self) -> list[str] | None:
        if shutil.which("pw-record"):
            argv = [
                "pw-record",
                "--rate",
                str(self.sample_rate),
                "--channels",
                "1",
                "--format",
                "s16",
            ]
            if self.input_device != "auto":
                argv += ["--target", self.input_device]
            argv.append("-")
            return argv
        if shutil.which("arecord"):
            argv = [
                "arecord",
                "-q",
                "-f",
                "S16_LE",
                "-r",
                str(self.sample_rate),
                "-c",
                "1",
                "-t",
                "raw",
            ]
            if self.input_device != "auto":
                argv += ["-D", self.input_device]
            argv.append("-")
            return argv
        return None

    def playback_argv(self, path: Path | str) -> list[str] | None:
        path = str(path)
        if shutil.which("pw-play"):
            argv = ["pw-play"]
            if self.output_device != "auto":
                argv += ["--target", self.output_device]
            argv.append(path)
            return argv
        if shutil.which("aplay"):
            argv = ["aplay", "-q"]
            if self.output_device != "auto":
                argv += ["-D", self.output_device]
            argv.append(path)
            return argv
        return None

    # ── Input discovery ──────────────────────────────────────────────
    def list_sources(self) -> list[dict[str, str]]:
        """PipeWire capture sources as ``{id, name, description}``.

        Sink monitors are dropped: recording them would feed our own playback
        straight back into recognition.
        """
        if not shutil.which("pw-dump"):
            return []
        try:
            dump = subprocess.run(
                ["pw-dump"], capture_output=True, timeout=8, check=False
            )
            nodes = json.loads(dump.stdout or b"[]")
        except (OSError, subprocess.TimeoutExpired, ValueError) as exc:
            logger.debug("Source discovery failed: %s", exc)
            return []
        sources: list[dict[str, str]] = []
        for node in nodes if isinstance(nodes, list) else []:
            props = (node.get("info") or {}).get("props") or {}
            if props.get("media.class") != "Audio/Source":
                continue
            name = str(props.get("node.name", ""))
            if "monitor" in name.lower():
                continue
            sources.append(
                {
                    "id": str(node.get("id", "")),
                    "name": name,
                    "description": str(props.get("node.description") or name),
                }
            )
        return sources

    def record_from(self, target: str, seconds: float) -> bytes:
        """Record ``seconds`` of raw PCM from one source (empty on failure)."""
        argv = [
            "pw-record",
            "--rate",
            str(self.sample_rate),
            "--channels",
            "1",
            "--format",
            "s16",
            "--target",
            target,
            "-",
        ]
        try:
            proc = subprocess.Popen(
                argv, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
            )
        except OSError as exc:
            logger.debug("Cannot record from %s: %s", target, exc)
            return b""
        time.sleep(max(0.1, seconds))
        try:
            proc.terminate()
            pcm, _ = proc.communicate(timeout=3)
        except (subprocess.TimeoutExpired, OSError):
            proc.kill()
            return b""
        return pcm or b""

    def probe_sources(self, *, seconds: float = 0.5) -> list[dict[str, object]]:
        """Short recording from every source, with its measured level."""
        probes: list[dict[str, object]] = []
        for source in self.list_sources():
            pcm = self.record_from(source["id"], seconds)
            probes.append(
                {
                    **source,
                    "level": round(rms_level(pcm), 4),
                    "seconds": round(len(pcm) / 2 / self.sample_rate, 2),
                }
            )
        return probes

    def pick_input_device(self, *, min_level: float = 0.002) -> tuple[str, list[dict[str, object]]]:
        """Pick the source that actually hears the room; ``auto`` if none do."""
        probes = self.probe_sources()
        usable = [p for p in probes if float(p["level"]) >= min_level]
        if not usable:
            return "auto", probes
        best = max(usable, key=lambda p: float(p["level"]))
        return str(best["id"]), probes

    # ── Capture ──────────────────────────────────────────────────────
    def open_capture(self) -> subprocess.Popen[bytes]:
        argv = self.capture_argv()
        if argv is None:
            raise RuntimeError(
                "No audio capture tool found (install pipewire-utils or alsa-utils)"
            )
        logger.info("Audio input connected: %s", " ".join(argv))
        self._proc = subprocess.Popen(
            argv,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
        )
        time.sleep(0.2)
        if self._proc.poll() is not None:
            detail = ""
            if self._proc.stderr is not None:
                detail = self._proc.stderr.read().decode("utf-8", "replace").strip()
            self._proc = None
            raise RuntimeError(
                "audio capture exited immediately" + (f": {detail[:200]}" if detail else "")
            )
        return self._proc

    def read(self, size: int | None = None) -> bytes:
        if self._proc is None or self._proc.stdout is None:
            return b""
        return self._proc.stdout.read(size or self.chunk_bytes)

    def close_capture(self) -> None:
        proc, self._proc = self._proc, None
        if proc is None:
            return
        try:
            proc.terminate()
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            proc.kill()
        except OSError:
            pass
        if proc.stderr is not None:
            try:
                tail = proc.stderr.read().decode("utf-8", "replace").strip()
            except (OSError, ValueError):
                tail = ""
            if tail:
                logger.debug("Audio capture stderr: %s", tail)
        logger.info("Audio input disconnected")

    # ── Playback ─────────────────────────────────────────────────────
    def play(self, wav_path: Path | str, *, timeout: float | None = None) -> bool:
        argv = self.playback_argv(wav_path)
        if argv is None:
            logger.error(
                "No audio playback tool found (install pipewire-utils or alsa-utils)"
            )
            return False
        try:
            result = subprocess.run(
                argv,
                capture_output=True,
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired:
            logger.warning("Audio playback timed out")
            return False
        except OSError as exc:
            logger.error("Audio playback failed: %s", exc)
            return False
        if result.returncode != 0:
            logger.error(
                "Audio playback failed (%s): %s",
                result.returncode,
                result.stderr.decode("utf-8", "replace").strip()[:200],
            )
            return False
        return True

    def play_async(self, wav_path: Path | str) -> subprocess.Popen[bytes] | None:
        """Start playback without waiting (used to cut off our own speech)."""
        argv = self.playback_argv(wav_path)
        if argv is None:
            return None
        try:
            return subprocess.Popen(
                argv,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except OSError as exc:
            logger.error("Audio playback failed: %s", exc)
            return None

    # ── Diagnostics ──────────────────────────────────────────────────
    def check(self) -> dict[str, object]:
        capture = self.capture_argv()
        playback = self.playback_argv("test.wav")
        return {
            "capture_command": " ".join(capture) if capture else None,
            "playback_command": " ".join(playback) if playback else None,
            "sample_rate": self.sample_rate,
            "input_device": self.input_device,
            "output_device": self.output_device,
        }

    def describe(self) -> str:
        capture = self.capture_argv()
        playback = self.playback_argv("test.wav")
        return (
            f"capture: {' '.join(capture) if capture else 'missing'}; "
            f"playback: {' '.join(playback) if playback else 'missing'}"
        )
