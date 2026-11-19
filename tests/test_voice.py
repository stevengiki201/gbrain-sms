"""Offline tests for the voice engine building blocks.

Run with: python3 -m unittest tests.test_voice -v
No phone, microphone, or network is required.
"""

from __future__ import annotations

import math
import struct
import unittest
from pathlib import Path

from bot.config import load_config
from bot.voice.audio_bridge import AcousticBridge
from bot.voice.conversation import ConversationSession, clean_for_speech
from bot.voice.tts import SpeechSynthesizer
from bot.voice.vad import EndpointDetector, rms_level


def _tone(seconds: float, rate: int = 16000, amplitude: int = 8000, freq: float = 440.0) -> bytes:
    count = int(rate * seconds)
    return b"".join(
        struct.pack("<h", int(amplitude * math.sin(2 * math.pi * freq * i / rate)))
        for i in range(count)
    )


def _silence(seconds: float, rate: int = 16000) -> bytes:
    return b"\x00\x00" * int(rate * seconds)


class VadTests(unittest.TestCase):
    def test_rms_of_silence_is_zero(self) -> None:
        self.assertEqual(rms_level(_silence(0.2)), 0.0)

    def test_rms_of_tone_is_positive(self) -> None:
        self.assertGreater(rms_level(_tone(0.2)), 0.1)

    def test_speech_then_silence_yields_utterance(self) -> None:
        detector = EndpointDetector(threshold=0.01, silence_ms=300, min_speech_ms=200)
        audio = _silence(0.4) + _tone(0.5) + _silence(0.6)
        utterance = None
        for start in range(0, len(audio), 640):
            found = detector.feed(audio[start : start + 640])
            if found:
                utterance = found
        self.assertIsNotNone(utterance)
        self.assertGreater(len(utterance or b""), 16000 * 0.3)

    def test_short_blip_is_discarded(self) -> None:
        detector = EndpointDetector(threshold=0.01, silence_ms=200, min_speech_ms=250)
        audio = _silence(0.2) + _tone(0.08) + _silence(0.5)
        utterance = None
        for start in range(0, len(audio), 640):
            found = detector.feed(audio[start : start + 640])
            if found:
                utterance = found
        self.assertIsNone(utterance)


    def test_dip_inside_sentence_does_not_split_it(self) -> None:
        detector = EndpointDetector(
            threshold=0.05, silence_ms=300, min_speech_ms=200, dip_factor=0.3
        )
        # A quiet stretch (RMS ~0.02, above 0.015 dip but below the 0.05 start
        # threshold) must not end the utterance the way real word gaps do not.
        audio = (
            _silence(0.1)
            + _tone(0.4)
            + _tone(0.4, amplitude=1000)
            + _tone(0.4)
            + _silence(0.5)
        )
        utterances = []
        for start in range(0, len(audio), 640):
            found = detector.feed(audio[start : start + 640])
            if found:
                utterances.append(found)
        self.assertEqual(len(utterances), 1)
        self.assertGreater(len(utterances[0]) / 2 / 16000, 1.0)


class BridgeInputTests(unittest.TestCase):
    def test_audible_source_wins_and_silent_jack_is_skipped(self) -> None:
        bridge = AcousticBridge()
        bridge.probe_sources = lambda **_: [  # type: ignore[method-assign]
            {"id": "59", "name": "jack", "description": "Headphones Mic", "level": 0.0, "seconds": 0.5},
            {"id": "60", "name": "digital", "description": "Digital Mic", "level": 0.03, "seconds": 0.5},
        ]
        self.assertEqual(bridge.pick_input_device()[0], "60")

    def test_all_silent_falls_back_to_auto(self) -> None:
        bridge = AcousticBridge()
        bridge.probe_sources = lambda **_: [  # type: ignore[method-assign]
            {"id": "59", "name": "jack", "description": "Headphones Mic", "level": 0.0, "seconds": 0.5}
        ]
        self.assertEqual(bridge.pick_input_device()[0], "auto")


class TtsTests(unittest.TestCase):
    def test_espeak_synthesizes_wav(self) -> None:
        synth = SpeechSynthesizer(provider="espeak")
        ok, detail = synth.availability()
        self.assertTrue(ok, detail)
        wav = synth.synthesize("hello from gbrain")
        self.assertIsNotNone(wav)
        assert wav is not None
        self.assertTrue(wav.is_file())
        self.assertGreater(wav.stat().st_size, 1000)
        self.assertEqual(wav.read_bytes()[:4], b"RIFF")
        synth.cleanup(wav)
        self.assertFalse(wav.exists())

    def test_espeak_has_swahili_voice(self) -> None:
        synth = SpeechSynthesizer(provider="espeak")
        wav = synth.synthesize("habari yako", language="sw")
        self.assertIsNotNone(wav)
        assert wav is not None
        self.assertGreater(wav.stat().st_size, 1000)
        synth.cleanup(wav)


class BridgeTests(unittest.TestCase):
    def test_commands_are_discovered(self) -> None:
        bridge = AcousticBridge()
        self.assertIsNotNone(bridge.capture_argv(), "no capture tool installed")
        self.assertIsNotNone(bridge.playback_argv("x.wav"), "no playback tool installed")
        self.assertIn(str(bridge.sample_rate), " ".join(bridge.capture_argv() or []))


class ConversationTests(unittest.TestCase):
    def test_clean_for_speech_strips_markup(self) -> None:
        self.assertEqual(clean_for_speech("**Hello** there 😀"), "Hello there")
        self.assertEqual(clean_for_speech("Assistant: Sure thing."), "Sure thing.")
        self.assertTrue(clean_for_speech("x" * 500).endswith("."))

    def test_session_keeps_context(self) -> None:
        session = ConversationSession(
            ollama_url="http://127.0.0.1:9", model="missing", timeout=0.2
        )
        reply = session.reply("habari yako")
        self.assertTrue(reply)
        self.assertEqual(session.turns, 1)
        self.assertIn("Samahani", reply)


class ConfigTests(unittest.TestCase):
    def test_voice_defaults_load(self) -> None:
        config = load_config()
        self.assertEqual(config.voice_sample_rate, 16000)
        self.assertIsInstance(config.voice_enabled, bool)
        self.assertGreater(config.voice_vad_silence_ms, 0)
        self.assertTrue(hasattr(config, "voice_echo_cancellation"))


class CallLifecycleTests(unittest.TestCase):
    def test_answered_call_starts_voice_and_ending_stops_it(self) -> None:
        from types import SimpleNamespace

        from bot.call_answerer import CallAnswerer
        from bot.call_connector import CallState

        class FakeCalls:
            def __init__(self) -> None:
                self.state = CallState("idle", "")
                self.answered = False

            def get_state(self):
                return self.state

            def answer(self) -> bool:
                self.answered = True
                self.state = CallState("offhook", "0659158158")
                return True

            def hangup(self) -> bool:
                return True

            def set_speakerphone(self) -> bool:
                return True

            def speak(self, text: str) -> bool:
                return True

        class FakeVoice:
            def __init__(self) -> None:
                self.started = None
                self.stopped = False

            def start(self, **kwargs) -> bool:
                self.started = kwargs
                return True

            def is_running(self) -> bool:
                return self.started is not None and not self.stopped

            def stop(self, reason: str = "") -> None:
                self.stopped = True

        config = SimpleNamespace(
            call_answer_start="",
            call_answer_end="",
            call_answer_at="",
            call_answer_contacts=[],
            call_hangup_after_sec=0,
            call_tts_enabled=True,
            call_greeting="",
            call_speakerphone=True,
            voice_enabled=True,
        )
        calls = FakeCalls()
        voice = FakeVoice()
        answerer = CallAnswerer(calls, config, voice=voice)

        calls.state = CallState("ringing", "0659158158")
        answerer.poll()
        self.assertTrue(calls.answered)
        assert voice.started is not None
        self.assertEqual(voice.started["contact"], "0659158158")

        calls.state = CallState("idle", "")
        answerer.poll()
        self.assertTrue(voice.stopped)

    def test_disabled_voice_falls_back_to_greeting(self) -> None:
        from types import SimpleNamespace

        from bot.call_answerer import CallAnswerer
        from bot.call_connector import CallState

        class FakeCalls:
            def __init__(self) -> None:
                self.state = CallState("idle", "")
                self.spoken: list[str] = []

            def get_state(self):
                return self.state

            def answer(self) -> bool:
                self.state = CallState("offhook", "123")
                return True

            def hangup(self) -> bool:
                return True

            def set_speakerphone(self) -> bool:
                return True

            def speak(self, text: str) -> bool:
                self.spoken.append(text)
                return True

        config = SimpleNamespace(
            call_answer_start="",
            call_answer_end="",
            call_answer_at="",
            call_answer_contacts=[],
            call_hangup_after_sec=0,
            call_tts_enabled=True,
            call_greeting="",
            call_speakerphone=True,
            voice_enabled=False,
        )
        calls = FakeCalls()
        answerer = CallAnswerer(calls, config, voice=None)
        calls.state = CallState("ringing", "123")
        answerer.poll()
        self.assertTrue(calls.spoken)


if __name__ == "__main__":
    unittest.main()
