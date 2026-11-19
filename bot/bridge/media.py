"""Desktop WebRTC media engine (phone <-> PC audio).

The phone captures its microphone through the Android WebRTC audio device
module (which applies echo cancellation, noise suppression and automatic gain
control) and sends an Opus track. The desktop answers the offer, receives that
track and feeds decoded 48 kHz mono PCM to a PipeWire/PulseAudio playback
process. The same answer carries a desktop capture track back to the phone
(milestone 3), fed from `pw-record` or `parec`. Desktop capture is off until
the explicit microphone operation enables it; the track sends paced silence
while disabled so no renegotiation is needed.

No audio device is opened until a real, authenticated signaling peer sends an
offer, and every failure is surfaced as a structured error rather than a silent
"connected" session.
"""

from __future__ import annotations

import asyncio
import logging
import shutil
import time
from array import array
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any, Callable

from .config import AudioLinkConfig
from .errors import MediaUnavailable, NotFound

logger = logging.getLogger("gbrain.bridge.media")

SAMPLE_RATE = 48_000
FRAME_SAMPLES = 960  # 20 ms at 48 kHz
FRAME_BYTES = FRAME_SAMPLES * 2
SILENCE_FRAME = b"\x00" * FRAME_BYTES
MAX_QUEUED_FRAMES = 15  # ~300 ms of audio; older audio is dropped to stay live
MAX_QUEUED_CAPTURE_FRAMES = 5  # ~100 ms of desktop capture before dropping
GATHER_TIMEOUT_SEC = 5.0


# --------------------------------------------------------------------- devices
def playout_backend() -> str:
    """Return the command used to play raw PCM, or an empty string."""
    for candidate in ("pw-cat", "pacat"):
        if shutil.which(candidate):
            return candidate
    return ""


def capture_backend(target: str = "") -> str:
    """Return the command used to capture raw PCM, or an empty string.

    ``pw-record --target`` does not resolve PipeWire sink monitors (``x.monitor``)
    and silently falls back to the default microphone, which would send room
    audio instead of the device that was asked for. The Pulse API resolves those
    names correctly, so a monitor target always prefers ``parec``.
    """
    candidates = ("pw-record", "parec")
    if str(target or "").strip().lower().endswith(".monitor"):
        candidates = ("parec", "pw-record")
    for candidate in candidates:
        if shutil.which(candidate):
            return candidate
    return ""


def apply_gain(pcm: bytes, volume: int, enabled: bool = True) -> bytes:
    """Apply a 0-100 volume (and mute) to signed 16-bit little-endian PCM."""
    if not enabled or volume <= 0:
        return b"\x00" * len(pcm)
    if volume >= 100:
        return pcm
    samples = array("h")
    samples.frombytes(pcm)
    gain = volume / 100.0
    for index, value in enumerate(samples):
        samples[index] = int(value * gain)
    return samples.tobytes()


def frame_to_pcm(frame: Any) -> bytes:
    """Convert an ``av.AudioFrame`` to mono s16le PCM.

    aiortc's Opus decoder emits packed s16 frames. Stereo input is downmixed;
    unusual formats are rejected instead of being played as noise.
    """
    fmt = getattr(frame.format, "name", "")
    if fmt != "s16":
        return b""
    planes = frame.planes
    if not planes:
        return b""
    raw = bytes(planes[0])
    channels = len(getattr(frame.layout, "channels", ()) or ())
    if channels <= 1:
        return raw
    interleaved = array("h")
    interleaved.frombytes(raw)
    out = array("h")
    for index in range(0, len(interleaved) - channels + 1, channels):
        total = 0
        for channel in range(channels):
            total += interleaved[index + channel]
        out.append(int(total / channels))
    return out.tobytes()


def rms_dbfs(pcm: bytes) -> float | None:
    """Level of a PCM frame in dBFS, for diagnostics only."""
    if not pcm:
        return None
    samples = array("h")
    samples.frombytes(pcm)
    if not samples:
        return None
    energy = sum(value * value for value in samples) / len(samples)
    if energy <= 0:
        return -90.0
    import math

    return round(20 * math.log10(math.sqrt(energy) / 32768), 1)


class PcmPlayout:
    """Streams PCM to a ``pw-cat``/``pacat`` child process with live buffering."""

    def __init__(self, target: str = "", latency_ms: int = 40) -> None:
        self.backend = playout_backend()
        self.target = target
        self.latency_ms = latency_ms
        self.volume = 100
        self.enabled = True
        self.frames_played = 0
        self.bytes_written = 0
        self.frames_dropped = 0
        self.last_error = ""
        self._queue: asyncio.Queue[bytes | None] = asyncio.Queue(
            maxsize=MAX_QUEUED_FRAMES
        )
        self._process: asyncio.subprocess.Process | None = None
        self._pump: asyncio.Task[None] | None = None
        self._closed = False

    def command(self) -> list[str]:
        if self.backend == "pw-cat":
            command = [
                "pw-cat",
                "--playback",
                "--format",
                "s16",
                "--rate",
                str(SAMPLE_RATE),
                "--channels",
                "1",
                "--latency",
                f"{self.latency_ms}ms",
            ]
            if self.target:
                command += ["--target", self.target]
            return command + ["-"]
        if self.backend == "pacat":
            command = [
                "pacat",
                "--playback",
                "--raw",
                "--format=s16le",
                f"--rate={SAMPLE_RATE}",
                "--channels=1",
                f"--latency-msec={self.latency_ms}",
            ]
            if self.target:
                command += [f"--device={self.target}"]
            return command
        raise MediaUnavailable(
            "no PCM playback tool is installed (need pw-cat or pacat)",
            code="AUDIO_SERVER_UNAVAILABLE",
        )

    async def start(self) -> None:
        command = self.command()
        try:
            self._process = await asyncio.create_subprocess_exec(
                *command,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
        except OSError as exc:
            raise MediaUnavailable(
                "could not start the audio playout process",
                details={"command": command, "error": str(exc)},
                code="AUDIO_SERVER_UNAVAILABLE",
            ) from exc
        self._pump = asyncio.create_task(self._write_loop(), name="audiolink-playout")

    async def _write_loop(self) -> None:
        assert self._process is not None and self._process.stdin is not None
        while True:
            chunk = await self._queue.get()
            if chunk is None:
                return
            try:
                self._process.stdin.write(chunk)
                await self._process.stdin.drain()
                self.bytes_written += len(chunk)
            except (BrokenPipeError, ConnectionResetError, RuntimeError) as exc:
                self.last_error = f"playout process stopped: {exc}"
                logger.warning("audio playout stopped: %s", exc)
                return

    def push(self, pcm: bytes) -> None:
        """Queue one frame, dropping the oldest audio when the sink is behind."""
        if self._closed or not pcm:
            return
        pcm = apply_gain(pcm, self.volume, self.enabled)
        try:
            self._queue.put_nowait(pcm)
        except asyncio.QueueFull:
            try:
                self._queue.get_nowait()
                self.frames_dropped += 1
            except asyncio.QueueEmpty:  # pragma: no cover - race with the pump
                pass
            try:
                self._queue.put_nowait(pcm)
            except asyncio.QueueFull:  # pragma: no cover
                self.frames_dropped += 1
                return
        self.frames_played += 1

    def set_volume(self, level: int) -> None:
        self.volume = max(0, min(100, int(level)))

    def set_enabled(self, enabled: bool) -> None:
        self.enabled = bool(enabled)

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        # The queue can be full when playout fell behind; make room for the
        # sentinel so the pump is told to stop instead of raising from close().
        try:
            self._queue.put_nowait(None)
        except asyncio.QueueFull:
            while True:
                try:
                    self._queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
            try:
                self._queue.put_nowait(None)
            except asyncio.QueueFull:  # pragma: no cover - raced with the pump
                pass
        if self._pump is not None:
            try:
                await asyncio.wait_for(self._pump, timeout=2.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):  # pragma: no cover
                self._pump.cancel()
            self._pump = None
        if self._process is not None:
            if self._process.stdin is not None:
                try:
                    self._process.stdin.close()
                except Exception:  # pragma: no cover
                    pass
            try:
                await asyncio.wait_for(self._process.wait(), timeout=2.0)
            except asyncio.TimeoutError:  # pragma: no cover
                self._process.kill()
                await self._process.wait()
            self._process = None

    def status(self) -> dict:
        return {
            "backend": self.backend,
            "target": self.target,
            "volume": self.volume,
            "enabled": self.enabled,
            "frames_played": self.frames_played,
            "frames_dropped": self.frames_dropped,
            "bytes_written": self.bytes_written,
            "last_error": self.last_error,
        }


class PcmCapture:
    """Reads 48 kHz mono s16 PCM from ``pw-record``/``parec`` on stdout."""

    def __init__(self, target: str = "", latency_ms: int = 40) -> None:
        self.backend = capture_backend(target)
        self.target = target
        self.latency_ms = latency_ms
        self.frames_read = 0
        self.bytes_read = 0
        self.last_error = ""
        self._process: asyncio.subprocess.Process | None = None

    def command(self) -> list[str]:
        if self.backend == "pw-record":
            command = [
                "pw-record",
                "--format",
                "s16",
                "--rate",
                str(SAMPLE_RATE),
                "--channels",
                "1",
                "--latency",
                f"{self.latency_ms}ms",
            ]
            if self.target:
                command += ["--target", self.target]
            return command + ["-"]
        if self.backend == "parec":
            command = [
                "parec",
                "--raw",
                "--format=s16le",
                f"--rate={SAMPLE_RATE}",
                "--channels=1",
                f"--latency-msec={self.latency_ms}",
            ]
            if self.target:
                command += [f"--device={self.target}"]
            return command
        raise MediaUnavailable(
            "no PCM capture tool is installed (need pw-record or parec)",
            code="AUDIO_SERVER_UNAVAILABLE",
        )

    async def start(self) -> None:
        if self._process is not None:
            return
        command = self.command()
        try:
            self._process = await asyncio.create_subprocess_exec(
                *command,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
            )
        except OSError as exc:
            raise MediaUnavailable(
                "could not start the audio capture process",
                details={"command": command, "error": str(exc)},
                code="AUDIO_SERVER_UNAVAILABLE",
            ) from exc

    async def read_frame(self) -> bytes:
        """Return one 20 ms PCM frame, or ``b""`` when capture stopped."""
        process = self._process
        if process is None or process.stdout is None:
            return b""
        try:
            pcm = await process.stdout.readexactly(FRAME_BYTES)
        except asyncio.IncompleteReadError as exc:
            if exc.partial:
                self.last_error = "capture process ended mid-frame"
            return b""
        except (ConnectionResetError, RuntimeError) as exc:
            self.last_error = f"capture process stopped: {exc}"
            return b""
        self.frames_read += 1
        self.bytes_read += len(pcm)
        return pcm

    async def close(self) -> None:
        process = self._process
        if process is None:
            return
        self._process = None
        if process.returncode is None:
            process.terminate()
        try:
            await asyncio.wait_for(process.wait(), timeout=2.0)
        except asyncio.TimeoutError:  # pragma: no cover
            process.kill()
            await process.wait()

    def status(self) -> dict:
        return {
            "backend": self.backend,
            "target": self.target,
            "running": self._process is not None
            and self._process.returncode is None,
            "frames_read": self.frames_read,
            "bytes_read": self.bytes_read,
            "last_error": self.last_error,
        }


# --------------------------------------------------------------------- session
@dataclass
class MediaSessionStatus:
    device_id: str
    state: str = "waiting_for_offer"  # waiting_for_offer|negotiating|connected|failed|closed
    started_at: float = field(default_factory=time.time)
    frames_received: int = 0
    samples_received: int = 0
    frames_dropped: int = 0
    last_frame_at: float = 0.0
    level_dbfs: float | None = None
    frames_sent: int = 0
    samples_sent: int = 0
    level_sent_dbfs: float | None = None
    peer_state: str = "new"
    ice_state: str = "new"
    last_error: str = ""


class DesktopMediaSession:
    def __init__(
        self,
        device_id: str,
        device_name: str,
        playout: PcmPlayout,
        capture: PcmCapture,
        ice_servers: list[dict],
        factory: Any,
        capture_enabled: bool = False,
    ) -> None:
        self.status_data = MediaSessionStatus(device_id=device_id)
        self.device_name = device_name
        self.playout = playout
        self.capture = capture
        self._ice_servers = ice_servers
        self._factory = factory
        self._requested_capture = capture_enabled
        self.capture_track = factory.CaptureTrack(capture)
        self._track_added = False
        self._closed = False
        self._consume_tasks: list[asyncio.Task[None]] = []
        self.pc = self._new_peer_connection()

    def _new_peer_connection(self) -> Any:
        RTCConfiguration = self._factory.RTCConfiguration
        pc = self._factory.RTCPeerConnection(
            RTCConfiguration(iceServers=self._ice_servers)
        )

        @pc.on("track")
        def on_track(track: Any) -> None:
            if track.kind != "audio":
                logger.debug("ignoring %s track from %s", track.kind, self.device_name)
                return
            logger.info("receiving audio from %s", self.device_name)
            task = asyncio.create_task(self._consume(track))
            self._consume_tasks.append(task)

        @pc.on("connectionstatechange")
        def on_connection_state() -> None:
            self.status_data.peer_state = pc.connectionState
            if pc.connectionState == "connected":
                self.status_data.state = "connected"
            elif pc.connectionState in {"failed", "closed"}:
                self.status_data.state = (
                    "closed" if pc.connectionState == "closed" else "failed"
                )

        @pc.on("iceconnectionstatechange")
        def on_ice_state() -> None:
            self.status_data.ice_state = pc.iceConnectionState
            if pc.iceConnectionState == "failed":
                self.status_data.last_error = "ICE connection failed"

        return pc

    async def answer(self, sdp: str) -> str:
        """Apply a remote offer and return the local answer SDP."""
        if self._closed:
            raise MediaUnavailable("media session is closed")
        self.status_data.state = "negotiating"
        try:
            await self.playout.start()
            await self.capture_track.set_enabled(self._requested_capture)
            if not self._track_added:
                self.pc.addTrack(self.capture_track)
                self._track_added = True
            await self.pc.setRemoteDescription(
                self._factory.RTCSessionDescription(sdp=sdp, type="offer")
            )
            answer = await self.pc.createAnswer()
            await self.pc.setLocalDescription(answer)
            await self._wait_for_gathering()
        except Exception as exc:
            self.status_data.state = "failed"
            self.status_data.last_error = str(exc)
            await self.close()
            raise MediaUnavailable(
                "could not negotiate the WebRTC session",
                details={"device_id": self.status_data.device_id, "error": str(exc)},
            ) from exc
        local = self.pc.localDescription
        if local is None:  # pragma: no cover - defensive
            raise MediaUnavailable("no local SDP was produced")
        return local.sdp

    async def _wait_for_gathering(self) -> None:
        deadline = time.monotonic() + GATHER_TIMEOUT_SEC
        while (
            self.pc.iceGatheringState != "complete"
            and time.monotonic() < deadline
            and not self._closed
        ):
            await asyncio.sleep(0.05)

    async def add_candidate(self, candidate: dict) -> None:
        if self._closed or not candidate.get("candidate"):
            return
        init = {
            "candidate": candidate["candidate"],
            "sdpMid": candidate.get("sdpMid", candidate.get("sdp_mid", "0")),
            "sdpMLineIndex": candidate.get(
                "sdpMLineIndex", candidate.get("sdp_mline_index", 0)
            ),
        }
        await self.pc.addIceCandidate(self._factory.RTCIceCandidate(**init))

    async def _consume(self, track: Any) -> None:
        try:
            while not self._closed:
                frame = await track.recv()
                pcm = frame_to_pcm(frame)
                if not pcm:
                    continue
                self.playout.push(pcm)
                self.status_data.frames_received += 1
                self.status_data.samples_received += len(pcm) // 2
                self.status_data.last_frame_at = time.time()
                if self.status_data.frames_received % 25 == 1:
                    self.status_data.level_dbfs = rms_dbfs(pcm)
        except asyncio.CancelledError:  # pragma: no cover - shutdown
            raise
        except Exception as exc:
            if not self._closed:
                logger.info("audio track from %s ended: %s", self.device_name, exc)
                self.status_data.last_error = f"track ended: {exc}"

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for task in self._consume_tasks:
            task.cancel()
        for task in self._consume_tasks:
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass
        self._consume_tasks.clear()
        try:
            await self.pc.close()
        except Exception:  # pragma: no cover - best effort
            logger.debug("peer connection close failed", exc_info=True)
        await self.capture_track.close()
        await self.playout.close()
        if self.status_data.state != "failed":
            self.status_data.state = "closed"

    def status(self) -> dict:
        data = self.status_data
        duration = max(0.0, time.time() - data.started_at)
        age = (
            round(time.time() - data.last_frame_at, 2)
            if data.last_frame_at
            else None
        )
        return {
            "device_id": data.device_id,
            "device_name": self.device_name,
            "state": data.state,
            "transport": "webrtc",
            "duration_sec": round(duration, 1),
            "frames_received": data.frames_received,
            "seconds_received": round(data.samples_received / SAMPLE_RATE, 2),
            "level_dbfs": data.level_dbfs,
            "last_frame_age_sec": age,
            "peer_state": data.peer_state,
            "ice_state": data.ice_state,
            "last_error": data.last_error,
            "playout": self.playout.status(),
            "capture_enabled": self.capture_track.enabled,
            "frames_sent": self.capture_track.frames_sent,
            "seconds_sent": round(
                self.capture_track.samples_sent / SAMPLE_RATE, 2
            ),
            "level_sent_dbfs": self.capture_track.level_dbfs,
            "capture": self.capture_track.status(),
        }

    async def set_capture_enabled(self, enabled: bool) -> None:
        self._requested_capture = enabled
        await self.capture_track.set_enabled(enabled)


# --------------------------------------------------------------------- manager
class MediaManager:
    """Owns desktop media sessions; used by the service and the API."""

    def __init__(
        self,
        config: AudioLinkConfig,
        playout_factory: Callable[..., PcmPlayout] = PcmPlayout,
        capture_factory: Callable[..., PcmCapture] = PcmCapture,
    ) -> None:
        self.config = config
        self._playout_factory = playout_factory
        self._capture_factory = capture_factory
        self._capture_enabled = False
        self._sessions: dict[str, DesktopMediaSession] = {}
        self._aiortc: Any = None
        self._import_error = ""
        self.last_error = ""

    # ------------------------------------------------------------- capability
    def _load_aiortc(self) -> Any:
        if self._aiortc is not None:
            return self._aiortc
        try:
            import aiortc
            from aiortc import (
                MediaStreamTrack,
                RTCConfiguration,
                RTCIceCandidate,
                RTCPeerConnection,
                RTCSessionDescription,
            )
            from av import AudioFrame
        except Exception as exc:
            self._import_error = str(exc)
            raise MediaUnavailable(
                "the WebRTC media engine is not installed (pip install aiortc)",
                details={"error": str(exc)},
            ) from exc

        class _CaptureTrack(MediaStreamTrack):
            """Desktop capture as a paced audio track.

            While capture is disabled the track sends silence at 20 ms cadence,
            so the phone can mute/unmute PC audio by control only and the SDP
            never has to be renegotiated.
            """

            kind = "audio"

            def __init__(self, capture: PcmCapture) -> None:
                super().__init__()
                self.capture = capture
                self.enabled = False
                self.frames_sent = 0
                self.samples_sent = 0
                self.level_dbfs: float | None = None
                self._queue: asyncio.Queue[bytes] = asyncio.Queue(
                    maxsize=MAX_QUEUED_CAPTURE_FRAMES
                )
                self._pump_task: asyncio.Task[None] | None = None
                self._enabled_lock = asyncio.Lock()
                self._pts = 0
                self._deadline = 0.0

            async def set_enabled(self, enabled: bool) -> None:
                enabled = bool(enabled)
                async with self._enabled_lock:
                    if enabled == self.enabled:
                        return
                    if enabled:
                        await self.capture.start()
                        self._queue = asyncio.Queue(
                            maxsize=MAX_QUEUED_CAPTURE_FRAMES
                        )
                        self.enabled = True
                        self._pump_task = asyncio.create_task(
                            self._pump(), name="audiolink-capture"
                        )
                    else:
                        self.enabled = False
                        task = self._pump_task
                        self._pump_task = None
                        if task is not None:
                            task.cancel()
                            try:
                                await task
                            except (asyncio.CancelledError, Exception):
                                pass
                        await self.capture.close()
                        while not self._queue.empty():
                            self._queue.get_nowait()

            async def _pump(self) -> None:
                while self.enabled:
                    pcm = await self.capture.read_frame()
                    if not pcm:
                        return
                    try:
                        self._queue.put_nowait(pcm)
                    except asyncio.QueueFull:
                        try:
                            self._queue.get_nowait()
                            self._queue.put_nowait(pcm)
                        except (asyncio.QueueEmpty, asyncio.QueueFull):
                            pass

            async def recv(self) -> Any:
                pcm = await self._next_pcm()
                frame = AudioFrame(
                    format="s16", layout="mono", samples=FRAME_SAMPLES
                )
                frame.sample_rate = SAMPLE_RATE
                frame.pts = self._pts
                frame.time_base = Fraction(1, SAMPLE_RATE)
                self._pts += FRAME_SAMPLES
                frame.planes[0].update(pcm)
                if self.enabled:
                    self.frames_sent += 1
                    self.samples_sent += FRAME_SAMPLES
                    if self.frames_sent % 25 == 1:
                        self.level_dbfs = rms_dbfs(pcm)
                return frame

            async def _next_pcm(self) -> bytes:
                if self.enabled:
                    try:
                        return await asyncio.wait_for(self._queue.get(), 0.04)
                    except asyncio.TimeoutError:
                        return SILENCE_FRAME
                now = asyncio.get_running_loop().time()
                if self._deadline <= 0:
                    self._deadline = now
                delay = self._deadline - now
                if delay > 0:
                    await asyncio.sleep(delay)
                else:
                    self._deadline = now
                self._deadline += FRAME_SAMPLES / SAMPLE_RATE
                return SILENCE_FRAME

            async def close(self) -> None:
                await self.set_enabled(False)
                await self.capture.close()
                self.stop()

            def status(self) -> dict:
                return {
                    **self.capture.status(),
                    "enabled": self.enabled,
                    "frames_sent": self.frames_sent,
                    "samples_sent": self.samples_sent,
                    "level_dbfs": self.level_dbfs,
                }

        class _Factory:
            pass

        factory = _Factory()
        factory.CaptureTrack = _CaptureTrack
        factory.AudioFrame = AudioFrame
        factory.RTCPeerConnection = RTCPeerConnection
        factory.RTCSessionDescription = RTCSessionDescription
        factory.RTCIceCandidate = RTCIceCandidate
        factory.RTCConfiguration = RTCConfiguration
        factory.version = getattr(aiortc, "__version__", "unknown")
        self._aiortc = factory
        return factory

    @property
    def available(self) -> bool:
        try:
            self._load_aiortc()
        except MediaUnavailable:
            return False
        return bool(playout_backend())

    def diagnostics(self) -> dict:
        try:
            version = self._load_aiortc().version
        except MediaUnavailable:
            version = ""
        return {
            "engine": "aiortc" if version else "",
            "engine_version": version,
            "engine_error": self._import_error or self.last_error,
            "available": self.available,
            "playout_backend": playout_backend(),
            "capture_backend": capture_backend(self.config.capture_device),
            "capture_available": bool(capture_backend(self.config.capture_device)),
            "capture_enabled": self._capture_enabled,
            "codec": "opus/48000 (s16 mono PCM both directions)",
            "sessions": [session.status() for session in self._sessions.values()],
        }

    @property
    def capture_enabled(self) -> bool:
        return self._capture_enabled

    @property
    def capture_available(self) -> bool:
        return bool(capture_backend(self.config.capture_device))

    def session_status(self, device_id: str) -> dict | None:
        session = self._sessions.get(device_id)
        return session.status() if session else None

    def all_sessions(self) -> list[dict]:
        return [session.status() for session in self._sessions.values()]

    # ---------------------------------------------------------------- lifecycle
    def _ice_servers(self) -> list[dict]:
        servers: list[dict] = [
            {"urls": url} for url in self.config.stun_servers if url
        ]
        if self.config.turn_server:
            entry: dict[str, Any] = {"urls": self.config.turn_server}
            if self.config.turn_username:
                entry["username"] = self.config.turn_username
            if self.config.turn_password:
                entry["credential"] = self.config.turn_password
            servers.append(entry)
        return servers

    def _create(self, device_id: str, device_name: str) -> DesktopMediaSession:
        factory = self._load_aiortc()
        if not playout_backend():
            raise MediaUnavailable(
                "no PCM playback tool is installed (need pw-cat or pacat)",
                code="AUDIO_SERVER_UNAVAILABLE",
            )
        playout = self._playout_factory(target=self.config.playback_device)
        capture = self._capture_factory(target=self.config.capture_device)
        session = DesktopMediaSession(
            device_id=device_id,
            device_name=device_name,
            playout=playout,
            capture=capture,
            ice_servers=self._ice_servers(),
            factory=factory,
            capture_enabled=self._capture_enabled,
        )
        self._sessions[device_id] = session
        logger.info("media session created for %s (%s)", device_name, device_id)
        return session

    async def start(self, device_id: str, device_name: str) -> dict:
        session = self._sessions.get(device_id)
        if session is None or session.status_data.state in {"failed", "closed"}:
            if session is not None:
                await session.close()
            session = self._create(device_id, device_name)
            state = "waiting_for_offer"
        else:
            state = session.status_data.state
        return {
            "device_id": device_id,
            "transport": "webrtc",
            "state": state,
            "media": session.status(),
        }

    async def handle_offer(self, device_id: str, device_name: str, sdp: str) -> dict:
        if not sdp or not isinstance(sdp, str):
            raise MediaUnavailable(
                "offer message did not contain an SDP body", details={"device_id": device_id}
            )
        if "m=audio" not in sdp:
            raise MediaUnavailable(
                "offer contains no audio track; refusing to answer an empty session",
                details={"device_id": device_id},
            )
        session = self._sessions.get(device_id)
        if session is None or session.status_data.state in {"failed", "closed"}:
            if session is not None:
                await session.close()
            session = self._create(device_id, device_name)
        answer = await session.answer(sdp)
        return {
            "device_id": device_id,
            "type": "answer",
            "sdp": answer,
            "state": session.status_data.state,
            "media": session.status(),
        }

    async def handle_candidate(self, device_id: str, candidate: dict) -> dict:
        session = self._sessions.get(device_id)
        if session is None:
            raise MediaUnavailable(
                "no media session exists for this device",
                details={"device_id": device_id},
            )
        await session.add_candidate(candidate)
        return {"device_id": device_id, "accepted": True}

    async def end(self, device_id: str) -> dict:
        session = self._sessions.pop(device_id, None)
        if session is None:
            return {"device_id": device_id, "ended": False, "transport": "webrtc"}
        await session.close()
        logger.info("media session ended for %s", device_id)
        return {
            "device_id": device_id,
            "ended": True,
            "transport": "webrtc",
            "seconds_received": round(
                session.status_data.samples_received / SAMPLE_RATE, 2
            ),
            "seconds_sent": round(
                session.capture_track.samples_sent / SAMPLE_RATE, 2
            ),
        }

    async def end_all(self) -> dict:
        results = [await self.end(device_id) for device_id in list(self._sessions)]
        return {"ended": [r for r in results if r["ended"]], "transport": "webrtc"}

    async def shutdown(self) -> None:
        await self.end_all()

    # ---------------------------------------------------------------- controls
    def set_volume(self, level: int) -> None:
        for session in self._sessions.values():
            session.playout.set_volume(level)

    def set_playout_enabled(self, enabled: bool) -> None:
        for session in self._sessions.values():
            session.playout.set_enabled(enabled)

    async def set_capture_enabled(self, enabled: bool) -> dict:
        """Authorize or stop desktop microphone capture for every session."""
        enabled = bool(enabled)
        if enabled and not capture_backend():
            raise MediaUnavailable(
                "no PCM capture tool is installed (need pw-record or parec)",
                code="AUDIO_SERVER_UNAVAILABLE",
            )
        previous = self._capture_enabled
        self._capture_enabled = enabled
        try:
            for session in self._sessions.values():
                await session.set_capture_enabled(enabled)
        except Exception:
            self._capture_enabled = previous
            for session in self._sessions.values():
                try:
                    await session.set_capture_enabled(previous)
                except Exception:  # pragma: no cover - rollback is best effort
                    logger.debug("capture rollback failed", exc_info=True)
            raise
        return {
            "capture_enabled": enabled,
            "capture_backend": capture_backend(),
            "sessions": len(self._sessions),
        }

    def get(self, device_id: str) -> DesktopMediaSession:
        session = self._sessions.get(device_id)
        if session is None:
            raise NotFound(
                "no active media session for this device", details={"device_id": device_id}
            )
        return session
