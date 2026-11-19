"""Real-time voice engine: caller audio → STT → LLM → TTS → caller.

Audio path (acoustic bridge, because a non-rooted phone exposes no cellular
call audio): PC microphone hears the phone's speakerphone output, and PC
speakers feed the phone's microphone, so the caller hears the reply through
the live call.

While our own speech is playing the capture stream is dropped, so the engine
does not transcribe itself. That makes the bridge half-duplex during replies;
with ``voice_echo_cancellation`` off, loud input can interrupt playback.
"""

from __future__ import annotations

import logging
import queue
import shutil
import threading
import time
from pathlib import Path
from typing import Callable

from bot.voice.audio_bridge import AcousticBridge
from bot.voice.audiolink_bridge import AudioLinkBridge, AudioLinkError
from bot.voice.conversation import ConversationSession
from bot.voice.stt import WhisperTranscriber, write_wav
from bot.voice.tts import OUTPUT_DIR, SpeechSynthesizer
from bot.voice.vad import EndpointDetector, rms_level

logger = logging.getLogger(__name__)

DEFAULT_GREETING = "Hello, this is gbrain. How can I help you?"
MIN_UTTERANCE_SEC = 0.5
SELF_TEST_MAX_SILENCE_MS = 1600

TranscriptSink = Callable[[str, str, str], None]


class VoiceEngine:
    def __init__(self, config, *, calls=None, transcript_sink: TranscriptSink | None = None) -> None:
        self.config = config
        self.calls = calls
        self.transcript_sink = transcript_sink

        rate = int(getattr(config, "voice_sample_rate", 16000) or 16000)
        self.transport = str(getattr(config, "voice_transport", "local") or "local").lower()
        if self.transport == "audiolink":
            self.bridge = AudioLinkBridge(config=config, sample_rate=rate)
        else:
            self.bridge = AcousticBridge(
                sample_rate=rate,
                input_device=getattr(config, "voice_input_device", "auto"),
                output_device=getattr(config, "voice_output_device", "auto"),
            )
        self.transcriber = WhisperTranscriber(
            model_size=getattr(config, "voice_stt_model", "small"),
            model_dir=config.resolve_path(getattr(config, "voice_model_dir", "data/models")),
            language=getattr(config, "voice_language", "auto"),
        )
        self.synth = SpeechSynthesizer(
            provider=getattr(config, "voice_tts_provider", "espeak"),
            voice=getattr(config, "voice_tts_voice", ""),
            command=getattr(config, "voice_tts_command", ""),
            piper_binary=getattr(config, "voice_piper_binary", "piper"),
            piper_model=getattr(config, "voice_piper_model", ""),
        )
        self.session = ConversationSession(
            ollama_url=getattr(config, "ollama_url", "http://127.0.0.1:11434"),
            model=getattr(config, "ollama_model", "llama3.2:3b"),
            bot_name=getattr(config, "bot_name", ""),
            system_prompt=getattr(config, "voice_system_prompt", ""),
        )

        self._stop = threading.Event()
        self._playing = threading.Event()
        self._utterances: "queue.Queue[bytes]" = queue.Queue(maxsize=8)
        self._capture_thread: threading.Thread | None = None
        self._worker_thread: threading.Thread | None = None
        self._play_proc = None
        self._contact = ""
        self._interrupted = False
        self._detector: EndpointDetector | None = None

    # ── State ────────────────────────────────────────────────────────
    def is_running(self) -> bool:
        return self._capture_thread is not None and self._capture_thread.is_alive()

    def availability(self) -> dict[str, object]:
        stt_ok, stt_detail = self.transcriber.availability()
        tts_ok, tts_detail = self.synth.availability()
        info = self.bridge.check()
        info.update(
            {
                "transport": getattr(self, "transport", "local"),
                "enabled": bool(getattr(self.config, "voice_enabled", False)),
                "stt": stt_detail,
                "stt_ok": stt_ok,
                "tts": tts_detail,
                "tts_ok": tts_ok,
                "vad": bool(getattr(self.config, "voice_vad_enabled", True)),
                "interrupt": bool(getattr(self.config, "voice_interrupt_enabled", True)),
                "echo_cancellation": bool(
                    getattr(self.config, "voice_echo_cancellation", True)
                ),
            }
        )
        return info

    def status_lines(self) -> list[str]:
        info = self.availability()
        lines = [
            f"Voice engine: {'enabled' if info['enabled'] else 'disabled'}"
            f"{' (running)' if self.is_running() else ''}",
            f"  Transport: {info.get('transport', 'local')}",
            f"  Capture: {info['capture_command'] or 'missing'}",
            f"  Playback: {info['playback_command'] or 'missing'}",
            f"  STT: {info['stt']}",
            f"  TTS: {info['tts']}",
            f"  Language: {getattr(self.config, 'voice_language', 'auto')}"
            f" | sample rate: {info['sample_rate']}",
            f"  Echo cancellation: {info['echo_cancellation']}"
            f" | interrupt: {info['interrupt']}",
        ]
        return lines

    # ── Lifecycle ────────────────────────────────────────────────────
    def start(
        self,
        *,
        reason: str = "",
        contact: str = "",
        greeting: str | None = None,
    ) -> bool:
        if self.is_running():
            return True
        if not getattr(self.config, "voice_enabled", False):
            logger.warning("Voice engine is disabled (voice_enabled: false)")
            return False

        stt_ok, stt_detail = self.transcriber.availability()
        if not stt_ok:
            logger.error("Cannot start voice engine — %s", stt_detail)
            return False
        tts_ok, tts_detail = self.synth.availability()
        if not tts_ok:
            logger.error("Cannot start voice engine — %s", tts_detail)
            return False
        if self.bridge.capture_argv() is None:
            logger.error(
                "Cannot start voice engine — no capture tool "
                "(install pipewire-utils or alsa-utils)"
            )
            return False

        self._contact = contact or ""
        self._stop.clear()
        self.session.reset()
        self._drain_utterances()

        if self.calls is not None and getattr(self.config, "call_speakerphone", True):
            if not self.calls.set_speakerphone():
                logger.error(
                    "Speakerphone could not be enabled — the caller's voice will not "
                    "reach the phone microphone"
                )

        if hasattr(self.bridge, "start"):
            try:
                self.bridge.start()
            except (AudioLinkError, RuntimeError) as exc:
                logger.error("AudioLink transport could not start: %s", exc)
                return False

        try:
            self.bridge.open_capture()
        except RuntimeError as exc:
            logger.error("Audio input could not be opened: %s", exc)
            return False

        # Calibrate before speaking, so the greeting is not mistaken for room noise.
        if getattr(self.config, "voice_vad_enabled", True):
            self._detector = EndpointDetector(
                sample_rate=self.bridge.sample_rate,
                threshold=float(getattr(self.config, "voice_vad_threshold", 0.012)),
                silence_ms=int(getattr(self.config, "voice_vad_silence_ms", 450)),
                min_speech_ms=int(getattr(self.config, "voice_vad_min_speech_ms", 250)),
            )
            self._calibrate(self._detector)
        else:
            self._detector = None

        self._capture_thread = threading.Thread(
            target=self._capture_loop, name="voice-capture", daemon=True
        )
        self._worker_thread = threading.Thread(
            target=self._worker_loop, name="voice-worker", daemon=True
        )
        self._capture_thread.start()
        self._worker_thread.start()

        threading.Thread(
            target=self.transcriber.preload, name="voice-preload", daemon=True
        ).start()
        logger.info("Voice engine started (%s)", reason or "manual")
        self._speak(greeting if greeting is not None else self._default_greeting())
        return True

    def stop(self, *, reason: str = "") -> None:
        if self._capture_thread is None:
            return
        logger.info("Stopping voice engine (%s)", reason or "call ended")
        self._stop.set()
        self._interrupt_playback()
        self.bridge.close_capture()
        if hasattr(self.bridge, "stop"):
            try:
                self.bridge.stop()
            except (AudioLinkError, RuntimeError) as exc:
                logger.warning("AudioLink transport did not close cleanly: %s", exc)

        for thread in (self._capture_thread, self._worker_thread):
            if thread is not None:
                thread.join(timeout=3.0)
        self._capture_thread = None
        self._worker_thread = None
        self._drain_utterances()
        self._save_transcript_note()
        logger.info("Voice engine stopped")

    # ── Capture ──────────────────────────────────────────────────────
    def _capture_loop(self) -> None:
        use_vad = bool(getattr(self.config, "voice_vad_enabled", True))
        detector = self._detector
        if use_vad and detector is None:
            detector = EndpointDetector(
                sample_rate=self.bridge.sample_rate,
                threshold=float(getattr(self.config, "voice_vad_threshold", 0.012)),
                silence_ms=int(getattr(self.config, "voice_vad_silence_ms", 450)),
                min_speech_ms=int(getattr(self.config, "voice_vad_min_speech_ms", 250)),
            )
            self._calibrate(detector)

        while not self._stop.is_set():
            chunk = self.bridge.read()
            if not chunk:
                if self._stop.is_set():
                    break
                logger.warning("Audio input stream ended")
                break

            if self._playing.is_set():
                if self._should_interrupt(chunk):
                    self._interrupt_playback()
                detector.reset()
                continue

            if use_vad:
                utterance = detector.feed(chunk)
                if utterance:
                    self._enqueue_utterance(utterance)
            elif len(chunk) >= self.bridge.chunk_bytes:
                self._enqueue_utterance(chunk)

    def _calibrate(
        self,
        detector: EndpointDetector,
        *,
        seconds: float = 0.6,
        bridge: AcousticBridge | None = None,
    ) -> None:
        """Measure the room so the VAD threshold clears the noise floor."""
        bridge = bridge or self.bridge
        levels: list[float] = []
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            chunk = bridge.read()
            if not chunk:
                return
            levels.append(rms_level(chunk))
        if not levels:
            return
        levels.sort()
        ambient = levels[len(levels) // 2]
        threshold = detector.calibrate_noise(ambient)
        logger.info("VAD calibrated: room noise %.3f, threshold %.3f", ambient, threshold)

    def _should_interrupt(self, chunk: bytes) -> bool:
        if not getattr(self.config, "voice_interrupt_enabled", True):
            return False
        if getattr(self.config, "voice_echo_cancellation", True):
            # Cannot tell the caller's voice from our own speaker output.
            return False
        threshold = float(getattr(self.config, "voice_vad_threshold", 0.012))
        return rms_level(chunk) >= threshold * 2.5

    def _enqueue_utterance(self, pcm: bytes) -> None:
        seconds = len(pcm) / 2 / self.bridge.sample_rate
        if seconds < MIN_UTTERANCE_SEC:
            logger.debug("Ignoring %.2fs blip", seconds)
            return
        logger.info("Caller speech detected (%.1fs)", seconds)
        try:
            self._utterances.put_nowait(pcm)
        except queue.Full:
            logger.warning("Voice queue full — dropping oldest utterance")
            try:
                self._utterances.get_nowait()
                self._utterances.put_nowait(pcm)
            except queue.Empty:
                pass

    def _drain_utterances(self) -> None:
        while True:
            try:
                self._utterances.get_nowait()
            except queue.Empty:
                return

    # ── Worker ───────────────────────────────────────────────────────
    def _worker_loop(self) -> None:
        while not self._stop.is_set():
            try:
                pcm = self._utterances.get(timeout=0.2)
            except queue.Empty:
                continue

            try:
                text = self.transcriber.transcribe(pcm, self.bridge.sample_rate)
            except Exception as exc:  # model load or decoding failure
                logger.error("Transcription failed: %s", exc)
                continue
            if not text:
                continue
            self._record("caller", text)

            reply = self.session.reply(text)
            if not reply:
                continue
            self._record("bot", reply)
            self._speak(reply)

    # ── Speech ───────────────────────────────────────────────────────
    def _default_greeting(self) -> str:
        return (
            getattr(self.config, "voice_greeting", "") or DEFAULT_GREETING
        ).strip()

    def _speak(self, text: str, *, language: str | None = None) -> bool:
        text = (text or "").strip()
        if not text:
            return False
        language = language or getattr(self.config, "voice_language", "auto")
        wav = self.synth.synthesize(text, language=language)
        if wav is None:
            return False

        self._playing.set()
        proc = self.bridge.play_async(wav)
        if proc is None:
            self._playing.clear()
            self.synth.cleanup(wav)
            return False
        self._play_proc = proc
        try:
            proc.wait(timeout=60)
        except Exception:  # pragma: no cover - defensive
            pass
        finally:
            self._play_proc = None
            self._playing.clear()
            self.synth.cleanup(wav)
            # Let the speaker tail decay so it is not captured as speech.
            time.sleep(0.25)

        interrupted = self._interrupted
        self._interrupted = False
        if proc.returncode not in (0, None) and not interrupted:
            logger.error("Audio playback exited with %s", proc.returncode)
            return False
        if interrupted:
            return False
        logger.info("Audio played to the bridge (PC speakers → phone microphone)")
        return True

    def _interrupt_playback(self) -> None:
        proc = self._play_proc
        if proc is None:
            return
        try:
            proc.terminate()
            proc.wait(timeout=2)
        except Exception:
            pass
        if self._playing.is_set():
            self._interrupted = True
            if not self._stop.is_set():
                logger.info("Caller interruption detected — stopped our playback")

    # ── Transcript ───────────────────────────────────────────────────
    def _record(self, role: str, text: str) -> None:
        if role == "caller":
            logger.info("Transcription completed: %s", text)
        if self.transcript_sink is not None:
            try:
                self.transcript_sink(self._contact or "call", role, text)
            except Exception as exc:  # pragma: no cover - storage must not kill audio
                logger.warning("Transcript save failed: %s", exc)

    def _save_transcript_note(self) -> None:
        if self.session.turns:
            logger.info("Call transcript: %d turns", self.session.turns)

    # ── Diagnostics ──────────────────────────────────────────────────
    def _self_test_bridge(self) -> tuple[AcousticBridge, dict[str, object]]:
        """Pick the microphone that actually hears the room for a self-test."""
        selection: dict[str, object] = {"chosen": None, "probed": []}
        if isinstance(self.bridge, AudioLinkBridge):
            # The input is the call sink monitor, not a room microphone.
            selection["chosen"] = self.bridge.input_device
            return self.bridge, selection
        configured = (self.bridge.input_device or "auto").strip()
        if configured.lower() != "auto":
            selection["chosen"] = configured
            return self.bridge, selection
        if not shutil.which("pw-dump") or not shutil.which("pw-record"):
            return self.bridge, selection

        device, probes = self.bridge.pick_input_device()
        selection["probed"] = probes
        selection["chosen"] = device
        for probe in probes:
            logger.info(
                "Microphone %s (%s): level %.4f over %.1fs",
                probe["id"],
                probe["description"],
                float(probe["level"]),
                float(probe["seconds"]),
            )
        if device == "auto":
            logger.warning(
                "No microphone heard anything while probing; using the system default"
            )
            return self.bridge, selection
        logger.info("Self-test uses microphone %s", device)
        return (
            AcousticBridge(
                sample_rate=self.bridge.sample_rate,
                input_device=device,
                output_device=self.bridge.output_device,
            ),
            selection,
        )

    def self_test(self, *, seconds: float = 5.0, reply: bool = True) -> dict[str, object]:
        """Capture the microphone, transcribe, optionally answer, and play.

        Both the microphone and the end-of-utterance silence window adapt: the
        loudest PipeWire source is used, and the window grows when a captured
        utterance could not be recognised (which usually means it was cut at a
        pause inside a sentence).
        """
        bridge, selection = self._self_test_bridge()
        result: dict[str, object] = {
            "capture_command": bridge.check()["capture_command"],
            "playback_command": self.bridge.check()["playback_command"],
            "input_device": selection["chosen"],
            "probes": selection["probed"],
            "heard": [],
            "reply": "",
            "played": False,
            "errors": [],
        }
        detector = EndpointDetector(
            sample_rate=bridge.sample_rate,
            threshold=float(getattr(self.config, "voice_vad_threshold", 0.012)),
            silence_ms=int(getattr(self.config, "voice_vad_silence_ms", 450)),
            dip_factor=0.35,
        )
        result["silence_ms"] = detector.silence_ms
        try:
            bridge.open_capture()
        except RuntimeError as exc:
            result["errors"].append(str(exc))
            return result

        self._calibrate(detector, bridge=bridge)
        deadline = time.monotonic() + seconds
        heard: list[str] = []
        last_audio: bytes | None = None
        try:
            while time.monotonic() < deadline:
                chunk = bridge.read()
                if not chunk:
                    result["errors"].append("audio input stream ended")
                    break
                utterance = detector.feed(chunk)
                if not utterance:
                    continue
                last_audio = utterance
                try:
                    text = self.transcriber.transcribe(utterance, bridge.sample_rate)
                except Exception as exc:
                    result["errors"].append(f"transcription: {exc}")
                    continue
                if text:
                    heard.append(text)
                    logger.info("Self-test heard: %s", text)
                else:
                    logger.info(
                        "Self-test got %.2fs of audio but no text",
                        len(utterance) / 2 / bridge.sample_rate,
                    )
                    if detector.silence_ms < SELF_TEST_MAX_SILENCE_MS:
                        detector.silence_ms = min(
                            SELF_TEST_MAX_SILENCE_MS, detector.silence_ms + 400
                        )
                        logger.info(
                            "Extending the silence window to %dms", detector.silence_ms
                        )
        finally:
            bridge.close_capture()
        result["silence_ms"] = detector.silence_ms

        result["heard"] = heard
        if last_audio and not heard:
            # Keep the last captured attempt so a failed recognition can be
            # replayed and re-transcribed offline.
            try:
                OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
                path = write_wav(
                    last_audio,
                    Path(OUTPUT_DIR) / "selftest-last.wav",
                    bridge.sample_rate,
                )
                result["last_audio"] = str(path)
                logger.info("Saved unread utterance to %s", path)
            except OSError as exc:
                logger.warning("Could not save self-test audio: %s", exc)
        if reply and heard:
            reply_text = self.session.reply(heard[-1])
            result["reply"] = reply_text
            result["played"] = self._speak(reply_text)
        return result
