"""End-to-end test over a real TLS uvicorn server and a real WebSocket client.

Covers: certificate trust, loopback management auth, pairing over HTTPS,
authenticated signaling, heartbeat, session control and media error reporting.
"""

from __future__ import annotations

import asyncio
import json
import ssl
import time
from array import array
from fractions import Fraction

import httpx
import pytest
import uvicorn
import websockets
from websockets.exceptions import InvalidStatus

from aiortc import RTCConfiguration, RTCPeerConnection, RTCSessionDescription
from aiortc.mediastreams import AudioStreamTrack
from av.audio.frame import AudioFrame

from bot.bridge.api import create_app


@pytest.fixture
async def live(service):
    app = create_app(service)
    config = uvicorn.Config(
        app,
        host="127.0.0.1",
        port=0,
        ssl_certfile=str(service.identity.cert_path),
        ssl_keyfile=str(service.identity.key_path),
        log_config=None,
        access_log=False,
    )
    server = uvicorn.Server(config)
    task = asyncio.create_task(server.serve())
    while not server.started:
        await asyncio.sleep(0.02)
    port = server.servers[0].sockets[0].getsockname()[1]
    yield port
    server.should_exit = True
    await asyncio.wait_for(task, timeout=10)


def _ssl_context(service) -> ssl.SSLContext:
    ctx = ssl.create_default_context(cafile=str(service.identity.cert_path))
    return ctx


def _ws_url(port: int, device_id: str) -> str:
    return f"wss://127.0.0.1:{port}/v1/signaling?device_id={device_id}"


async def _pair(service, http: httpx.AsyncClient) -> str:
    start = await http.post(
        "/v1/pairing/requests",
        json={"device_id": "phone-1", "device_name": "Pixel", "platform": "android"},
    )
    request_id = start.json()["request_id"]
    pending = await service.pairing.pending(include_code=True)
    code = pending[0]["code"]
    confirm = await http.post(
        f"/v1/pairing/requests/{request_id}/confirm", json={"code": code}
    )
    assert confirm.status_code == 200
    return confirm.json()["token"]


async def test_tls_pairing_signaling_and_session_control(live, service):
    port = live
    async with httpx.AsyncClient(
        base_url=f"https://127.0.0.1:{port}",
        verify=str(service.identity.cert_path),
    ) as http:
        identity = await http.get("/v1/identity")
        assert identity.status_code == 200
        token = await _pair(service, http)

        async with websockets.connect(
            _ws_url(port, "phone-1"),
            additional_headers={"Authorization": f"Bearer {token}"},
            ssl=_ssl_context(service),
            open_timeout=10,
        ) as ws:
            hello = await asyncio.wait_for(ws.recv(), timeout=5)
            assert '"hello_ack"' in hello

            await ws.send('{"type": "ping", "ts": 42}')
            pong = await asyncio.wait_for(ws.recv(), timeout=5)
            assert '"pong"' in pong and '"ts":42' in pong

            # Media negotiation is explicit about the missing engine.
            await ws.send('{"type": "sdp.offer", "sdp": "v=0"}')
            error = await asyncio.wait_for(ws.recv(), timeout=5)
            assert "MEDIA_UNAVAILABLE" in error

            async with httpx.AsyncClient(
                base_url=f"https://127.0.0.1:{port}",
                verify=str(service.identity.cert_path),
                headers={"X-AudioLink-Admin": service.admin_token},
            ) as admin:
                connect = await admin.post("/v1/devices/phone-1/connect")
                assert connect.status_code == 200
                session = connect.json()
                assert session["state"] == "connected"
                assert session["media_transport"] == "not_negotiated"

                control = await ws.recv()
                assert "control.session" in control

                mic = await admin.post("/v1/controls/microphone", json={"enabled": False})
                assert mic.json()["microphone_enabled"] is False

                disconnect = await admin.post("/v1/devices/phone-1/disconnect")
                assert disconnect.status_code == 200
                assert disconnect.json()["state"] == "ended"

                status = await admin.get("/v1/status")
                assert status.json()["online_devices"][0]["device_id"] == "phone-1"


class FakeCapture:
    """Paced desktop capture stand-in so the TLS test needs no sound server."""

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
        return array("h", [1800] * 960).tobytes()

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


class FakePlayout:
    def __init__(self, target: str = "") -> None:
        self.target = target
        self.frames_played = 0
        self.closed = False

    async def start(self) -> None:
        pass

    def push(self, pcm: bytes) -> None:
        self.frames_played += 1

    def set_volume(self, level: int) -> None:
        pass

    def set_enabled(self, enabled: bool) -> None:
        pass

    async def close(self) -> None:
        self.closed = True

    def status(self) -> dict:
        return {"backend": "fake", "frames_played": self.frames_played}


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
        frame.planes[0].update(array("h", [4000] * 960).tobytes())
        return frame


async def test_signaling_negotiates_real_webrtc_audio(live, service, monkeypatch):
    """Pair, connect, start a session and exchange real Opus audio both ways."""
    port = live
    service.media._playout_factory = FakePlayout
    service.media._capture_factory = FakeCapture
    monkeypatch.setattr(
        "bot.bridge.media.capture_backend", lambda target="": "fake-capture"
    )
    pc = RTCPeerConnection(RTCConfiguration(iceServers=[]))
    remote_tracks: list = []

    @pc.on("track")
    def on_remote_track(track) -> None:
        remote_tracks.append(track)
    try:
        async with httpx.AsyncClient(
            base_url=f"https://127.0.0.1:{port}",
            verify=str(service.identity.cert_path),
        ) as http:
            token = await _pair(service, http)

        async with websockets.connect(
            _ws_url(port, "phone-1"),
            additional_headers={"Authorization": f"Bearer {token}"},
            ssl=_ssl_context(service),
            open_timeout=10,
        ) as ws:
            assert '"hello_ack"' in await asyncio.wait_for(ws.recv(), timeout=5)

            async with httpx.AsyncClient(
                base_url=f"https://127.0.0.1:{port}",
                verify=str(service.identity.cert_path),
                headers={"X-AudioLink-Admin": service.admin_token},
            ) as admin:
                assert (await admin.post("/v1/devices/phone-1/connect")).status_code == 200
                assert '"control.session"' in await asyncio.wait_for(
                    ws.recv(), timeout=5
                )

                started = await admin.post(
                    "/v1/sessions/start", json={"device_id": "phone-1"}
                )
                assert started.status_code == 200, started.text
                body = started.json()
                assert body["transport"] == "webrtc"
                assert body["codec"] == "opus"
                media_start = json.loads(await asyncio.wait_for(ws.recv(), timeout=5))
                assert media_start["type"] == "media.start"
                assert media_start["sample_rate"] == 48000

                pc.addTrack(ToneTrack())
                offer = await pc.createOffer()
                await pc.setLocalDescription(offer)
                deadline = time.monotonic() + 5
                while pc.iceGatheringState != "complete" and time.monotonic() < deadline:
                    await asyncio.sleep(0.05)
                await ws.send(
                    json.dumps(
                        {"type": "sdp.offer", "sdp": pc.localDescription.sdp}
                    )
                )
                answer = json.loads(await asyncio.wait_for(ws.recv(), timeout=10))
                assert answer["type"] == "sdp.answer", answer
                assert "m=audio" in answer["sdp"]
                await pc.setRemoteDescription(
                    RTCSessionDescription(sdp=answer["sdp"], type="answer")
                )

                for _ in range(120):
                    status = (await admin.get("/v1/status")).json()
                    media = status["sessions"][0].get("media")
                    if media and media["frames_received"] > 0:
                        break
                    await asyncio.sleep(0.1)
                assert status["media_transport"] == "webrtc"
                session_media = status["sessions"][0]["media"]
                assert session_media["frames_received"] > 0
                assert session_media["state"] == "connected"
                playout = service.media.get("phone-1").playout
                assert playout.frames_played > 0

                # M3: authorize desktop capture and receive it on the phone.
                assert remote_tracks, "desktop answer carried no audio track"
                phone_audio = remote_tracks[0]
                mic = await admin.post(
                    "/v1/controls/microphone", json={"enabled": True}
                )
                assert mic.status_code == 200, mic.text
                assert mic.json()["capture_enabled"] is True
                heard = False
                for _ in range(60):
                    frame = await asyncio.wait_for(phone_audio.recv(), timeout=2)
                    samples = array("h", bytes(frame.planes[0]))
                    if any(abs(value) > 0 for value in samples):
                        heard = True
                        break
                assert heard, "desktop capture audio never reached the phone"
                status = (await admin.get("/v1/status")).json()
                session_media = status["sessions"][0]["media"]
                assert session_media["capture_enabled"] is True
                assert session_media["frames_sent"] > 0
                assert status["media"]["capture_backend"] == "fake-capture"

                ended = await admin.post("/v1/sessions/end")
                assert ended.status_code == 200
                assert ended.json()["ended"]
                assert '"media.stop"' in await asyncio.wait_for(ws.recv(), timeout=5)
    finally:
        await pc.close()
        await service.media.shutdown()


async def test_signaling_rejects_bad_token(live, service):
    with pytest.raises(InvalidStatus) as exc:
        async with websockets.connect(
            _ws_url(live, "phone-1"),
            additional_headers={"Authorization": "Bearer al_invalid"},
            ssl=_ssl_context(service),
            open_timeout=10,
        ):
            pass
    assert exc.value.response.status_code in {403, 401}


async def test_untrusted_client_cannot_verify_certificate(live, service):
    async with httpx.AsyncClient(
        base_url=f"https://127.0.0.1:{live}", verify=True
    ) as http:
        with pytest.raises(httpx.HTTPError):
            await http.get("/v1/identity")
