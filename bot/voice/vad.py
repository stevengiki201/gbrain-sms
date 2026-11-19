"""Energy-based voice activity detection and utterance endpointing.

The acoustic bridge feeds raw 16-bit PCM; this module decides when the caller
has started and finished speaking so recognition only runs on whole
utterances instead of a continuous stream.
"""

from __future__ import annotations

import math
from array import array

DEFAULT_FRAME_MS = 20


def rms_level(pcm: bytes) -> float:
    """RMS of 16-bit little-endian PCM, normalised to 0..1."""
    usable = len(pcm) - (len(pcm) % 2)
    if usable <= 0:
        return 0.0
    samples = array("h")
    samples.frombytes(pcm[:usable])
    if not samples:
        return 0.0
    total = sum(sample * sample for sample in samples)
    return math.sqrt(total / len(samples)) / 32768.0


class EndpointDetector:
    """Splits a continuous PCM stream into utterances.

    A utterance starts when the level crosses ``threshold``, ends after
    ``silence_ms`` of quiet, and is discarded if it contained less than
    ``min_speech_ms`` of actual speech (coughs, clicks).
    """

    def __init__(
        self,
        *,
        sample_rate: int = 16000,
        frame_ms: int = DEFAULT_FRAME_MS,
        threshold: float = 0.012,
        silence_ms: int = 450,
        min_speech_ms: int = 250,
        max_utterance_ms: int = 20000,
        dip_factor: float = 0.0,
    ) -> None:
        self.sample_rate = sample_rate
        self.frame_ms = frame_ms
        self.threshold = threshold
        # Word gaps dip below the start threshold; keeping the utterance open
        # down to ``threshold * dip_factor`` stops one sentence being split.
        # 0 disables the hysteresis (the original single-threshold behaviour).
        self.dip_factor = max(0.0, min(dip_factor, 0.95))
        self.silence_ms = silence_ms
        self.min_speech_ms = min_speech_ms
        self.max_utterance_ms = max_utterance_ms

        self._frame_bytes = max(2, int(sample_rate * frame_ms / 1000) * 2)
        self._max_bytes = int(sample_rate * max_utterance_ms / 1000) * 2
        self._pending = b""
        self._buffer = bytearray()
        self._speech_ms = 0
        self._quiet_ms = 0
        self.last_level = 0.0
        self.utterances = 0

    # ── Public API ───────────────────────────────────────────────────
    def feed(self, pcm: bytes) -> bytes | None:
        """Add PCM; return a complete utterance when one just ended."""
        if not pcm:
            return None
        self._pending += pcm
        utterance: bytes | None = None
        while len(self._pending) >= self._frame_bytes:
            frame = self._pending[: self._frame_bytes]
            self._pending = self._pending[self._frame_bytes :]
            done = self._process(frame)
            if done and utterance is None:
                utterance = done
        return utterance

    def calibrate_noise(self, ambient_level: float, *, factor: float = 2.5) -> float:
        """Raise the threshold above the measured room noise floor."""
        self.threshold = max(self.threshold, ambient_level * factor)
        return self.threshold

    def reset(self) -> None:
        self._buffer.clear()
        self._speech_ms = 0
        self._quiet_ms = 0
        self._pending = b""

    # ── Internals ────────────────────────────────────────────────────
    @property
    def dip_threshold(self) -> float:
        return self.threshold * self.dip_factor if self.dip_factor else self.threshold

    def _process(self, frame: bytes) -> bytes | None:
        level = rms_level(frame)
        self.last_level = level

        # Start only above the speech threshold; once inside an utterance, a
        # word gap that stays above the dip threshold does not end it.
        active = level >= (self.dip_threshold if self._buffer else self.threshold)
        if active:
            self._buffer.extend(frame)
            self._speech_ms += self.frame_ms
            self._quiet_ms = 0
            if len(self._buffer) >= self._max_bytes:
                return self._finish()
            return None

        if not self._buffer:
            return None

        self._buffer.extend(frame)
        self._quiet_ms += self.frame_ms
        if self._quiet_ms >= self.silence_ms:
            return self._finish()
        return None

    def _finish(self) -> bytes | None:
        pcm = bytes(self._buffer)
        speech_ms = self._speech_ms
        self.reset()
        if speech_ms < self.min_speech_ms:
            return None
        self.utterances += 1
        return pcm
