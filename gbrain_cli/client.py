from __future__ import annotations

import json
import os
import ssl
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

from .config import local_ca_file


class ApiError(RuntimeError):
    pass


def _request(
    url: str,
    method: str = "GET",
    payload: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    ca_file: str | None = None,
    timeout: float = 20.0,
) -> dict[str, Any]:
    body = None
    request_headers = {"accept": "application/json"}
    if headers:
        request_headers.update(headers)
    if payload is not None:
        body = json.dumps(payload).encode("utf-8")
        request_headers["content-type"] = "application/json"
    request = urllib.request.Request(url, data=body, headers=request_headers, method=method)
    context = ssl.create_default_context(cafile=ca_file) if ca_file else None
    try:
        with urllib.request.urlopen(request, timeout=timeout, context=context) as response:
            raw = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:500]
        raise ApiError(f"{method} {url} returned {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise ApiError(f"{method} {url} failed: {exc.reason}") from exc
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ApiError(f"{method} {url} returned invalid JSON") from exc
    if not isinstance(parsed, dict):
        return {"data": parsed}
    return parsed


class WebsiteClient:
    def __init__(self, base_url: str, token: str | None = None) -> None:
        self.base_url = base_url.rstrip("/")
        self.token = token

    def _headers(self) -> dict[str, str]:
        if not self.token:
            return {}
        return {"x-device-token": self.token}

    def sign_in(
        self,
        email: str,
        password: str,
        name: str,
        fingerprint: str,
    ) -> dict[str, Any]:
        return _request(
            f"{self.base_url}/api/devices/cli-login",
            method="POST",
            payload={
                "email": email,
                "password": password,
                "name": name,
                "fingerprint": fingerprint,
            },
        )

    def service_devices(self) -> list[dict[str, Any]]:
        service_token = os.environ.get("GBRAIN_SERVICE_TOKEN", "")
        if not service_token:
            raise ApiError("GBRAIN_SERVICE_TOKEN is not set")
        response = _request(
            f"{self.base_url}/api/service/devices",
            headers={"x-service-token": service_token},
        )
        devices = response.get("devices")
        return devices if isinstance(devices, list) else []

    def next_command(self, device_id: str) -> dict[str, Any] | None:
        query = urllib.parse.urlencode({"deviceId": device_id})
        response = _request(
            f"{self.base_url}/api/devices/commands/next?{query}",
            headers=self._headers(),
        )
        command = response.get("command")
        return command if isinstance(command, dict) else None

    def post_result(self, command_id: str, status: str, result: dict[str, Any]) -> dict[str, Any]:
        return _request(
            f"{self.base_url}/api/devices/commands/{command_id}/result",
            method="POST",
            payload={"status": status, "result": result},
            headers=self._headers(),
        )

    def post_status(self, status: dict[str, Any]) -> dict[str, Any]:
        return _request(
            f"{self.base_url}/api/devices/status",
            method="POST",
            payload={"status": status},
            headers=self._headers(),
        )

    def post_pairing_code(
        self,
        code: str | None,
        expires_at: str | None,
    ) -> dict[str, Any]:
        return _request(
            f"{self.base_url}/api/devices/pairing-code",
            method="POST",
            payload={"code": code, "expiresAt": expires_at},
            headers=self._headers(),
        )

    def post_sms_cache(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        return _request(
            f"{self.base_url}/api/sms/cache",
            method="POST",
            payload={"messages": messages},
            headers=self._headers(),
        )


class LocalBridgeClient:
    def __init__(self, token: str, base_url: str = "https://127.0.0.1:8787") -> None:
        self.base_url = base_url.rstrip("/")
        self.headers = {"x-audiolink-admin": token}
        self.ca_file = local_ca_file() if self.base_url.startswith("https://") else None

    def _get(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        query = urllib.parse.urlencode(params or {})
        suffix = f"?{query}" if query else ""
        return _request(
            f"{self.base_url}{path}{suffix}",
            headers=self.headers,
            ca_file=self.ca_file,
        )

    def _post(self, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        return _request(
            f"{self.base_url}{path}",
            method="POST",
            payload=payload or {},
            headers=self.headers,
            ca_file=self.ca_file,
        )

    def status(self) -> dict[str, Any]:
        return self._get("/v1/status")

    def pending_pairings(self) -> list[dict[str, Any]]:
        response = self._get("/v1/pairing/pending", {"include_code": "true"})
        requests = response.get("requests")
        return requests if isinstance(requests, list) else []

    def sms_inbox(self, folder: str = "inbox", limit: int = 200) -> list[dict[str, Any]]:
        response = self._get("/v1/sms/inbox", {"folder": folder, "limit": str(limit)})
        messages = response.get("messages")
        return messages if isinstance(messages, list) else []

    def send_sms(self, to: str, message: str) -> dict[str, Any]:
        return self._post("/v1/sms/send", {"to": to, "message": message})

    def tasks(self) -> dict[str, Any]:
        return self._get("/v1/gbrain/sms/tasks")

    def chat(self, message: str) -> dict[str, Any]:
        return self._post("/v1/gbrain/sms/chat", {"message": message})


class HostedBridgeClient:
    def __init__(self, base_url: str, service_token: str) -> None:
        self.base_url = base_url.rstrip("/")
        if not service_token:
            raise ApiError("GBRAIN_SERVICE_TOKEN is not set")
        self.headers = {"x-gbrain-service": service_token}

    def _get(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        query = urllib.parse.urlencode(params or {})
        suffix = f"?{query}" if query else ""
        return _request(f"{self.base_url}{path}{suffix}", headers=self.headers)

    def _post(self, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        return _request(
            f"{self.base_url}{path}",
            method="POST",
            payload=payload or {},
            headers=self.headers,
        )

    def status(self) -> dict[str, Any]:
        return self._get("/v1/hosted/status")

    def devices(self) -> list[dict[str, Any]]:
        response = self._get("/v1/hosted/devices")
        devices = response.get("devices")
        return devices if isinstance(devices, list) else []

    def events(self, website_device_id: str, after: int = 0) -> dict[str, Any]:
        return self._get(
            "/v1/hosted/events",
            {"website_device_id": website_device_id, "after": str(after)},
        )

    def sms_send(
        self,
        website_device_id: str,
        to_number: str,
        body: str,
    ) -> dict[str, Any]:
        return self._post(
            "/v1/hosted/sms/send",
            {
                "website_device_id": website_device_id,
                "to_number": to_number,
                "body": body,
            },
        )
