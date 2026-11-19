"""HTTP client for the gbrain-sms AI server.

AudioLink calls gbrain-sms as a pure reply-generation backend.
All methods are synchronous; call them via ``asyncio.get_event_loop().run_in_executor``
from async contexts.
"""

from __future__ import annotations

import json
import logging
import ssl
import urllib.error
import urllib.request
from pathlib import Path

logger = logging.getLogger("gbrain.bridge.gbrain")


def _expand(path: str) -> str:
    return str(Path(path).expanduser()) if path else ""


class GbrainClient:
    """Calls gbrain-sms as the AI reply-generation server."""

    def __init__(
        self,
        *,
        base_url: str,
        token: str,
        ca_file: str = "",
        timeout: float = 60.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.timeout = timeout
        self._ssl: ssl.SSLContext | None = None
        if ca_file:
            ctx = ssl.create_default_context(cafile=ca_file)
            self._ssl = ctx

    @classmethod
    def from_config(cls, config) -> "GbrainClient":
        """Build a GbrainClient from an AudioLinkConfig instance."""
        token_file = _expand(getattr(config, "gbrain_token_file", ""))
        # Prefer an explicit CA file; fall back to the TLS cert AudioLink
        # already generated at data_dir/tls/cert.pem (same cert gbrain-sms uses).
        ca_file = _expand(getattr(config, "audiolink_ca_file", ""))
        if not ca_file:
            data_path = getattr(config, "data_path", None)
            if data_path is not None:
                candidate = Path(data_path).expanduser() / "tls" / "cert.pem"
                if candidate.exists():
                    ca_file = str(candidate)
        try:
            token = Path(token_file).read_text(encoding="utf-8").strip()
        except OSError as exc:
            raise RuntimeError(
                f"cannot read gbrain token at {token_file!r}: {exc}"
            ) from exc
        if not token:
            raise RuntimeError(f"gbrain token at {token_file!r} is empty")
        base_url = getattr(config, "gbrain_url", "https://127.0.0.1:8788")
        return cls(base_url=base_url, token=token, ca_file=ca_file)

    # ── internal ─────────────────────────────────────────────────────

    def _post(self, path: str, payload: dict) -> dict:
        url = f"{self.base_url}{path}"
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.token}",
            },
        )
        try:
            with urllib.request.urlopen(
                req, context=self._ssl, timeout=self.timeout
            ) as resp:
                return json.loads(resp.read() or b"{}")
        except urllib.error.HTTPError as exc:
            detail = ""
            try:
                body = json.loads(exc.read() or b"{}")
                detail = str(body.get("detail", body.get("error", "")))
            except Exception:
                pass
            logger.warning("gbrain %s returned %s: %s", path, exc.code, detail[:120])
            return {}
        except Exception as exc:
            logger.warning("gbrain %s failed: %s", path, exc)
            return {}

    # ── public ───────────────────────────────────────────────────────

    def generate_reply(self, contact: str, body: str, source: str = "sms") -> str:
        """POST /v1/gbrain/sms/generate → reply text, or '' on any failure."""
        result = self._post(
            "/v1/gbrain/sms/generate",
            {"contact": contact, "body": body, "source": source},
        )
        return result.get("reply", "")

    def parse_instruction(self, instruction: str) -> dict:
        """POST /v1/gbrain/tasks/parse → full response dict (contains task_draft)."""
        # Use a longer timeout — parse may need to load a model
        url = f"{self.base_url}/v1/gbrain/tasks/parse"
        data = json.dumps({"instruction": instruction}).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.token}",
            },
        )
        try:
            with urllib.request.urlopen(req, context=self._ssl, timeout=120.0) as resp:
                return json.loads(resp.read() or b"{}")
        except urllib.error.HTTPError as exc:
            detail = ""
            try:
                body = json.loads(exc.read() or b"{}")
                detail = str(body.get("detail", body.get("error", "")))
            except Exception:
                pass
            logger.warning("gbrain /v1/gbrain/tasks/parse returned %s: %s", exc.code, detail[:120])
            return {}
        except Exception as exc:
            logger.warning("gbrain /v1/gbrain/tasks/parse failed: %s", exc)
            return {}

    def ai_chat(
        self,
        device_id: str,
        message: str,
        *,
        schedule_local: bool = False,
        history: list[dict] | None = None,
    ) -> dict:
        """POST /v1/gbrain/ai/chat → {reply, task} dict.

        ``schedule_local=False`` makes gbrain-sms return a task draft without
        adding it to its own ADB scheduler; AudioLink stores and executes it.
        """
        return self._post(
            "/v1/gbrain/ai/chat",
            {
                "message": message,
                "history": history or [],
                "schedule_local": schedule_local,
                "device_id": device_id,
            },
        )
