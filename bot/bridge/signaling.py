"""Authenticated WebSocket signaling between paired devices.

The desktop is the signaling server. A phone opens
``wss://<desktop>:<port>/v1/signaling`` with a bearer token obtained during
pairing. SDP offers/answers and ICE candidates are relayed as JSON messages;
media itself never passes through this socket.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from .errors import ValidationFailed

logger = logging.getLogger("gbrain.bridge.signaling")

PROTOCOL = "audiolink/1"

# Client -> server
HELLO = "hello"
PING = "ping"
SDP_OFFER = "sdp.offer"
SDP_ANSWER = "sdp.answer"
ICE_CANDIDATE = "ice.candidate"
SESSION_END = "session.end"
CONTROL = "control"
MEDIA_STOP = "media.stop"

# Server -> client
HELLO_ACK = "hello_ack"
PONG = "pong"
ERROR = "error"
PEER_STATE = "peer.state"

# SMS integration (phone -> server: sms.inbound / sms.result;
# server -> phone: sms.send)
SMS_INBOUND = "sms.inbound"
SMS_SEND    = "sms.send"
SMS_RESULT  = "sms.result"

CLIENT_MESSAGES = {
    HELLO,
    PING,
    SDP_OFFER,
    SDP_ANSWER,
    ICE_CANDIDATE,
    SESSION_END,
    CONTROL,
    MEDIA_STOP,
    SMS_INBOUND,
    SMS_RESULT,
}
MAX_MESSAGE_CHARS = 256 * 1024


@dataclass
class Peer:
    device_id: str
    name: str
    connected_at: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    messages_in: int = 0
    messages_out: int = 0
    websocket: Any = None

    def public(self) -> dict:
        return {
            "device_id": self.device_id,
            "name": self.name,
            "connected_at": self.connected_at,
            "last_seen": self.last_seen,
            "messages_in": self.messages_in,
            "messages_out": self.messages_out,
            "online": self.websocket is not None,
        }


def validate_message(message: Any) -> str:
    """Validate the envelope and return its type."""
    if not isinstance(message, dict):
        raise ValidationFailed("signaling message must be a JSON object")
    msg_type = message.get("type")
    if not isinstance(msg_type, str) or not msg_type:
        raise ValidationFailed("signaling message requires a string 'type'")
    if msg_type not in CLIENT_MESSAGES:
        raise ValidationFailed(f"unsupported signaling message type: {msg_type}")
    if len(str(message)) > MAX_MESSAGE_CHARS:
        raise ValidationFailed("signaling message is too large")
    if msg_type == SMS_INBOUND:
        for required_field in ("sms_id", "from_number", "body"):
            if required_field not in message:
                raise ValidationFailed(
                    f"sms.inbound requires '{required_field}'",
                    details={"missing": required_field},
                )
    return msg_type


class SignalingHub:
    def __init__(self) -> None:
        self._peers: dict[str, Peer] = {}
        self._lock = asyncio.Lock()

    async def attach(self, device_id: str, name: str, websocket) -> Peer:
        async with self._lock:
            existing = self._peers.get(device_id)
            if existing and existing.websocket is not None:
                old = existing.websocket
                existing.websocket = websocket
                existing.name = name
                existing.last_seen = time.time()
                peer = existing
            else:
                peer = Peer(device_id=device_id, name=name, websocket=websocket)
                self._peers[device_id] = peer
                old = None
        if old is not None:
            try:
                await old.close(code=4001, reason="replaced by a new connection")
            except Exception:  # pragma: no cover - best effort
                pass
        return peer

    async def detach(self, device_id: str, websocket) -> None:
        async with self._lock:
            peer = self._peers.get(device_id)
            if peer is not None and peer.websocket is websocket:
                peer.websocket = None
                peer.last_seen = time.time()

    async def is_online(self, device_id: str) -> bool:
        async with self._lock:
            peer = self._peers.get(device_id)
            return bool(peer and peer.websocket is not None)

    async def peer(self, device_id: str) -> Peer | None:
        async with self._lock:
            return self._peers.get(device_id)

    async def online_devices(self) -> list[dict]:
        async with self._lock:
            return [
                peer.public()
                for peer in self._peers.values()
                if peer.websocket is not None
            ]

    async def send(self, device_id: str, payload: dict) -> bool:
        async with self._lock:
            peer = self._peers.get(device_id)
            websocket = peer.websocket if peer else None
        if not websocket:
            return False
        try:
            await websocket.send_json(payload)
        except Exception as exc:  # pragma: no cover - transport failure
            logger.debug("send to %s failed: %s", device_id, exc)
            return False
        async with self._lock:
            peer = self._peers.get(device_id)
            if peer:
                peer.messages_out += 1
                peer.last_seen = time.time()
        return True

    async def send_error(self, device_id: str, code: str, message: str) -> bool:
        return await self.send(
            device_id, {"type": ERROR, "code": code, "message": message}
        )

    async def record_received(self, device_id: str) -> None:
        async with self._lock:
            peer = self._peers.get(device_id)
            if peer:
                peer.messages_in += 1
                peer.last_seen = time.time()

    async def shutdown(self) -> None:
        async with self._lock:
            peers = list(self._peers.values())
            self._peers.clear()
        for peer in peers:
            if peer.websocket is not None:
                try:
                    await peer.websocket.close(code=1001, reason="service stopping")
                except Exception:  # pragma: no cover
                    pass
