"""Whisper speech recognition for the voice engine.

``faster-whisper`` is installed into ``tools/voice/pylibs`` (the system Python
is PEP 668 managed) and loaded lazily on the first utterance so the bot still
starts instantly when voice mode is off.
"""

from __future__ import annotations

import logging
import re
import sys
import tempfile
import time
import wave
from pathlib import Path

logger = logging.getLogger(__name__)

# Mean log-probability above which a forced-language decode is trusted as is.
_GOOD_ENOUGH_LOGPROB = -0.8

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOCAL_PYLIBS = PROJECT_ROOT / "tools" / "voice" / "pylibs"

_LANGUAGE_CODES = {"en": "en", "sw": "sw", "auto": None, "": None}
_HAS_LATIN = re.compile(r"[A-Za-z]{2}")


def _ensure_local_pylibs() -> None:
    if LOCAL_PYLIBS.is_dir():
        path = str(LOCAL_PYLIBS)
        if path not in sys.path:
            sys.path.append(path)


def _to_float32(pcm: bytes):
    """Convert 16-bit LE PCM to the float32 waveform faster-whisper wants."""
    import numpy as np

    usable = len(pcm) - (len(pcm) % 2)
    samples = np.frombuffer(pcm[:usable], dtype="<i2")
    return samples.astype("float32") / 32768.0


def write_wav(pcm: bytes, path: Path, sample_rate: int) -> Path:
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm)
    return path


class WhisperTranscriber:
    def __init__(
        self,
        *,
        model_size: str = "small",
        model_dir: str | Path | None = None,
        language: str = "auto",
        compute_type: str = "int8",
    ) -> None:
        self.model_size = model_size
        self.model_dir = Path(model_dir) if model_dir else PROJECT_ROOT / "data/models"
        self.language = (language or "auto").lower()
        self.compute_type = compute_type
        self._model = None
        self._load_error: str | None = None

    # ── Diagnostics ──────────────────────────────────────────────────
    def availability(self) -> tuple[bool, str]:
        _ensure_local_pylibs()
        try:
            import faster_whisper  # noqa: F401
        except Exception as exc:  # pragma: no cover - depends on install
            return False, f"faster-whisper not installed ({exc})"
        return True, f"faster-whisper ready (model: {self.model_size})"

    # ── Model ────────────────────────────────────────────────────────
    def _load(self):
        if self._model is not None:
            return self._model
        if self._load_error:
            raise RuntimeError(self._load_error)
        _ensure_local_pylibs()
        try:
            from faster_whisper import WhisperModel

            started = time.monotonic()
            self.model_dir.mkdir(parents=True, exist_ok=True)
            target = self.model_size
            local = self.model_dir / f"faster-whisper-{self.model_size}"
            if (local / "model.bin").is_file():
                target = str(local)
            self._model = WhisperModel(
                target,
                device="cpu",
                compute_type=self.compute_type,
                download_root=str(self.model_dir),
            )
            logger.info(
                "Whisper model '%s' loaded in %.1fs (cache: %s)",
                self.model_size,
                time.monotonic() - started,
                self.model_dir,
            )
        except Exception as exc:
            self._load_error = str(exc)
            raise
        return self._model

    def preload(self) -> None:
        """Load the model in the background so the first reply is not delayed."""
        try:
            self._load()
        except Exception as exc:
            logger.error("Whisper preload failed: %s", exc)

    # ── Recognition ──────────────────────────────────────────────────
    def transcribe(self, pcm: bytes, sample_rate: int = 16000) -> str:
        if not pcm:
            return ""
        _ensure_local_pylibs()
        model = self._load()
        language = _LANGUAGE_CODES.get(self.language, self.language or None)

        # Hand faster-whisper a float32 array: it skips PyAV decoding entirely,
        # which also keeps us independent of the system libav version.
        audio = _to_float32(pcm)
        started = time.monotonic()
        text, detected, logprob, confidence = self._run(model, audio, language)

        # Whisper hallucinates when the audio is noisy (for example a speaker a
        # few rooms away): it picks a random language and emits non-Latin text.
        # When the guess is weak, decode Swahili and English and keep whichever
        # the model was more confident about — otherwise a Swahili caller can be
        # answered from a garbled English transcript.
        if language is None and self._needs_retry(detected, text):
            logger.info(
                "Low-confidence detection (%s, p=%.2f); retrying in Swahili and English",
                detected,
                confidence,
            )
            swahili_text, _, swahili_logprob, _ = self._run(model, audio, "sw")
            if swahili_text and swahili_logprob >= _GOOD_ENOUGH_LOGPROB:
                # Most callers here speak Swahili, and a second decode doubles
                # the wait, so accept a confident Swahili result immediately.
                text, detected, logprob = swahili_text, "sw", swahili_logprob
            else:
                candidates = []
                if swahili_text:
                    candidates.append((swahili_logprob, "sw", swahili_text))
                english_text, _, english_logprob, _ = self._run(model, audio, "en")
                if english_text:
                    candidates.append((english_logprob, "en", english_text))
                if candidates:
                    # Higher (less negative) mean log-probability wins; Swahili
                    # wins ties because this bot speaks Swahili first.
                    candidates.sort(
                        key=lambda item: (item[0], item[1] == "sw"), reverse=True
                    )
                    logprob, detected, text = candidates[0]

        logger.info(
            "Transcription completed in %.2fs (%s): %s",
            time.monotonic() - started,
            detected or "auto",
            text or "<empty>",
        )
        return text

    def _run(
        self, model, audio, language: str | None
    ) -> tuple[str, str, float, float]:
        """Decode once; return text, detected language, mean log-prob, confidence."""
        segments, info = model.transcribe(
            audio,
            language=language,
            beam_size=1,
            # Trim non-speech first: Whisper invents sentences when asked to
            # decode silence (very common with speakerphone audio).
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 300},
            condition_on_previous_text=False,
            temperature=0.0,
        )
        parts: list[str] = []
        logprobs: list[float] = []
        for seg in segments:
            piece = seg.text.strip()
            if not piece:
                continue
            # Whisper invents sentences from noise; apply the usual quality
            # gates (high no-speech score, poor log-prob, repetitive text).
            if (
                # Swahili segments often score 0.7–0.8 here but are correct,
                # so this gate only catches near-certain non-speech.
                getattr(seg, "no_speech_prob", 0.0) > 0.85
                or getattr(seg, "avg_logprob", 0.0) < -1.0
                or getattr(seg, "compression_ratio", 0.0) > 2.4
            ):
                logger.debug("Dropped likely hallucination: %s", piece)
                continue
            parts.append(piece)
            logprobs.append(float(getattr(seg, "avg_logprob", 0.0)))
        mean_logprob = sum(logprobs) / len(logprobs) if logprobs else -9.0
        return (
            " ".join(parts).strip(),
            getattr(info, "language", language or ""),
            mean_logprob,
            float(getattr(info, "language_probability", 0.0) or 0.0),
        )

    @staticmethod
    def _needs_retry(detected: str, text: str) -> bool:
        if detected and detected not in ("en", "sw"):
            return True
        if not text:
            return True
        return not _HAS_LATIN.search(text)

    def describe(self) -> str:
        ok, detail = self.availability()
        return f"{'ok' if ok else 'unavailable'} — {detail}"
