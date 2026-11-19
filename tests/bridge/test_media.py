from __future__ import annotations

import asyncio
import time
from array import array
from fractions import Fraction

import pytest
from av.audio.frame import AudioFrame

from bot.bridge import media as media_mod
from bot.bridge.config import AudioLinkConfig
from bot.bridge.errors import MediaUnavailable
from bot.bridge.media import MediaManager, PcmPlayout, apply_gain, frame_to_pcm


class FakePlayout:
    def __init__(self, target: str = "") -> None:
        self.target = target
        self.frames_played = 0
        self.frames_dropped = 0
        self.volume = 100
        self.enabled = True
        self.closed = False
        self.chunks: list[bytes] = []

    async def start(self) -> None:
        pass

    def push(self, pcm: bytes) -> None:
        self.frames_played += 1
        self.chunks.append(pcm)

    def set_volume(self, level: int) -> None:
        self.volume = level

    def set_enabled(self, enabled: bool) -> None:
        self.enabled = enabled

    async def close(self) -> None:
        self.closed = True

    def status(self) -> dict:
        return {
            "backend": "fake",
            "frames_played": self.frames_played,
            "frames_dropped": self.frames_dropped,
            "volume": self.volume,
            "enabled": self.enabled,
        }


def make_frame(samples: int = 960, layout: str = "mono", fmt: str = "s16") -> AudioFrame:
    frame = AudioFrame(format=fmt, layout=layout, samples=samples)
    frame.sample_rate = 48000
    if fmt == "s16" and layout == "mono":
        frame.planes[0].update(array("h", [100, -100] * (samples // 2)).tobytes())
    return frame


def test_apply_gain_scales_and_mutes() -> None:
    pcm = array("h", [1000, -1000, 32767, -32768]).tobytes()
    assert apply_gain(pcm, 100) == pcm
    assert apply_gain(pcm, 0) == b"\x00" * len(pcm)
    assert apply_gain(pcm, 100, enabled=False) == b"\x00" * len(pcm)
    half = array("h")
    half.frombytes(apply_gain(pcm, 50))
    assert list(half) == [500, -500, 16383, -16384]


def test_frame_conversion() -> None:
    mono = frame_to_pcm(make_frame())
    assert len(mono) == 960 * 2
    assert mono[:4] == array("h", [100, -100]).tobytes()

    stereo = AudioFrame(format="s16", layout="stereo", samples=2)
    stereo.sample_rate = 48000
    stereo.planes[0].update(array("h", [100, 300, -100, -300]).tobytes())
    downmixed = array("h")
    downmixed.frombytes(frame_to_pcm(stereo))
    assert list(downmixed) == [200, -200]

    assert frame_to_pcm(make_frame(fmt="fltp")) == b""


def test_playout_command_and_missing_backend(monkeypatch) -> None:
    monkeypatch.setattr(media_mod, "playout_backend", lambda: "pw-cat")
    command = PcmPlayout(target="alsa_output.test").command()
    assert command[0] == "pw-cat"
    assert "--target" in command and "alsa_output.test" in command
    assert command[-1] == "-"

    monkeypatch.setattr(media_mod, "playout_backend", lambda: "pacat")
    command = PcmPlayout().command()
    assert command[0] == "pacat"
    assert "--format=s16le" in command

    monkeypatch.setattr(media_mod, "playout_backend", lambda: "")
    with pytest.raises(MediaUnavailable) as exc:
        PcmPlayout().command()
    assert exc.value.code == "AUDIO_SERVER_UNAVAILABLE"


def test_playout_queue_drops_old_audio() -> None:
    playout = PcmPlayout.__new__(PcmPlayout)
    playout.volume = 100
    playout.enabled = True
    playout.backend = "pw-cat"
    playout.target = ""
    playout.frames_played = 0
    playout.frames_dropped = 0
    playout.bytes_written = 0
    playout.last_error = ""
    playout._closed = False
    playout._process = None
    playout._pump = None
    playout._queue = asyncio.Queue(maxsize=2)
    for _ in range(4):
        playout.push(b"\x01\x00" * 960)
    assert playout.frames_played == 4
    assert playout.frames_dropped == 2
    assert playout._queue.qsize() == 2


async def test_playout_start_without_tool(monkeypatch) -> None:
    monkeypatch.setattr(media_mod, "playout_backend", lambda: "")
    with pytest.raises(MediaUnavailable):
        await PcmPlayout().start()


async def test_loopback_offer_reaches_playout(monkeypatch) -> None:
    from aiortc import RTCConfiguration, RTCPeerConnection, RTCSessionDescription
    from aiortc.mediastreams import AudioStreamTrack

    class ToneTrack(AudioStreamTrack):
        kind = "audio"

        def __init__(self) -> None:
            super().__init__()
            self._started: float | None = None
            self._sent = 0

        async def recv(self) -> AudioFrame:
            if self._started is None:
                self._started = time.time()
            self._sent += 960
            delay = self._started + self._sent / 48000 - time.time()
            if delay > 0:
                await asyncio.sleep(delay)
            frame = AudioFrame(format="s16", layout="mono", samples=960)
            frame.sample_rate = 48000
            frame.pts = self._sent
            frame.time_base = Fraction(1, 48000)
            frame.planes[0].update(array("h", [5000] * 960).tobytes())
            return frame

    monkeypatch.setattr(media_mod, "playout_backend", lambda: "pw-cat")
    manager = MediaManager(AudioLinkConfig(), playout_factory=FakePlayout)
    pc = RTCPeerConnection(RTCConfiguration(iceServers=[]))
    try:
        pc.addTrack(ToneTrack())
        offer = await pc.createOffer()
        await pc.setLocalDescription(offer)
        deadline = time.monotonic() + 5
        while pc.iceGatheringState != "complete" and time.monotonic() < deadline:
            await asyncio.sleep(0.05)

        answer = await manager.handle_offer(
            "phone-loopback", "Loopback phone", pc.localDescription.sdp
        )
        assert answer["type"] == "answer"
        assert answer["sdp"].startswith("v=0")
        await pc.setRemoteDescription(
            RTCSessionDescription(sdp=answer["sdp"], type="answer")
        )

        playout = manager._sessions["phone-loopback"].playout
        for _ in range(100):
            if playout.frames_played > 0:
                break
            await asyncio.sleep(0.1)
        assert playout.frames_played > 0, "no audio reached the desktop playout"
        status = manager.session_status("phone-loopback")
        assert status is not None
        assert status["frames_received"] > 0
        assert status["level_dbfs"] is not None
    finally:
        await manager.shutdown()
        await pc.close()

    assert manager._sessions == {}


async def test_media_manager_reports_missing_engine(monkeypatch) -> None:
    manager = MediaManager(AudioLinkConfig())
    monkeypatch.setattr(
        manager,
        "_load_aiortc",
        lambda: (_ for _ in ()).throw(MediaUnavailable("not installed")),
    )
    assert manager.available is False
    with pytest.raises(MediaUnavailable):
        await manager.start("phone-x", "Phone")


async def test_answer_without_offer_errors(service) -> None:
    with pytest.raises(MediaUnavailable):
        await service.media.handle_offer("phone-x", "Phone", "")


async def test_offer_without_audio_is_rejected(service) -> None:
    with pytest.raises(MediaUnavailable, match="no audio track"):
        await service.media.handle_offer(
            "phone-1", "Pixel", "v=0\r\no=- 0 0 IN IP4 0.0.0.0\r\ns=-\r\nt=0 0\r\n"
        )
    assert service.media.all_sessions() == []


def test_capture_command_and_missing_tool(monkeypatch) -> None:
    from bot.bridge.media import PcmCapture

    capture = PcmCapture()
    capture.backend = "pw-record"
    command = capture.command()
    assert command[0] == "pw-record"
    assert "--rate" in command and "48000" in command
    assert command[-1] == "-"

    capture.backend = "parec"
    assert capture.command()[0] == "parec"
    assert "--format=s16le" in capture.command()

    capture.backend = ""
    with pytest.raises(MediaUnavailable) as exc:
        capture.command()
    assert exc.value.code == "AUDIO_SERVER_UNAVAILABLE"


class FakeCapture:
    """Real-time paced PCM source used instead of pw-record/parec in tests."""

    def __init__(self, target: str = "") -> None:
        self.target = target
        self.backend = "fake"
        self.running = False
        self.frames_read = 0
        self.last_error = ""

    async def start(self) -> None:
        self.running = True

    async def read_frame(self) -> bytes:
        if not self.running:
            return b""
        await asyncio.sleep(0.02)
        self.frames_read += 1
        return array("h", [2500] * 960).tobytes()

    async def close(self) -> None:
        self.running = False

    def status(self) -> dict:
        return {
            "backend": self.backend,
            "target": self.target,
            "running": self.running,
            "frames_read": self.frames_read,
            "last_error": self.last_error,
        }


async def test_desktop_capture_reaches_a_receive_only_phone(monkeypatch) -> None:
    """M3: the desktop answer carries capture audio, toggled without renegotiation."""
    from aiortc import RTCConfiguration, RTCPeerConnection, RTCSessionDescription

    monkeypatch.setattr(media_mod, "capture_backend", lambda target="": "pw-record")
    manager = MediaManager(
        AudioLinkConfig(),
        playout_factory=FakePlayout,
        capture_factory=FakeCapture,
    )
    pc = RTCPeerConnection(RTCConfiguration(iceServers=[]))
    remote_tracks: list = []

    @pc.on("track")
    def on_track(track) -> None:
        remote_tracks.append(track)

    try:
        pc.addTransceiver("audio", direction="recvonly")
        offer = await pc.createOffer()
        await pc.setLocalDescription(offer)
        deadline = time.monotonic() + 5
        while pc.iceGatheringState != "complete" and time.monotonic() < deadline:
            await asyncio.sleep(0.05)

        answer = await manager.handle_offer(
            "phone-m3", "Receive-only phone", pc.localDescription.sdp
        )
        await pc.setRemoteDescription(
            RTCSessionDescription(sdp=answer["sdp"], type="answer")
        )
        for _ in range(50):
            if remote_tracks:
                break
            await asyncio.sleep(0.05)
        assert remote_tracks, "desktop answer did not include an audio track"
        track = remote_tracks[0]

        # Capture is off by default: only paced silence is sent.
        first = await asyncio.wait_for(track.recv(), timeout=2)
        assert first.samples == 960
        assert max(abs(value) for value in array("h", bytes(first.planes[0]))) == 0
        assert manager.capture_enabled is False

        await manager.set_capture_enabled(True)
        heard = None
        for _ in range(50):
            frame = await asyncio.wait_for(track.recv(), timeout=2)
            if max(abs(value) for value in array("h", bytes(frame.planes[0]))) > 0:
                heard = frame
                break
        assert heard is not None, "desktop capture audio never reached the phone"
        status = manager.session_status("phone-m3")
        assert status is not None
        assert status["capture_enabled"] is True
        assert status["frames_sent"] > 0
        assert status["capture"]["backend"] == "fake"

        await manager.set_capture_enabled(False)
        status = manager.session_status("phone-m3")
        assert status is not None
        assert status["capture_enabled"] is False
    finally:
        await manager.shutdown()
        await pc.close()
