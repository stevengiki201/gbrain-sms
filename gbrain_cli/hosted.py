from __future__ import annotations

import os
import time
from typing import Any

from .client import ApiError, HostedBridgeClient, WebsiteClient


def _required_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"{name} must be set for hosted mode")
    return value


def _hosted_messages(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    for event in events:
        if event.get("kind") != "sms":
            continue
        payload = event.get("payload") or {}
        messages.append(
            {
                "smsId": f"hosted_{event.get('seq', 0)}",
                "folder": "inbox",
                "address": str(payload.get("from_number", "")),
                "body": str(payload.get("body", "")),
                "dateMs": int(payload.get("timestamp_ms", 0) or 0),
                "msgType": 1,
                "status": "sent",
            }
        )
    return messages


def execute_hosted_command(
    command: dict[str, Any],
    bridge: HostedBridgeClient,
    website_device_id: str,
) -> tuple[str, dict[str, Any]]:
    command_type = command.get("type")
    payload = command.get("payload") or {}
    try:
        if command_type == "bridge.status":
            result = {"ok": True, "status": bridge.status(), "hosted": True}
        elif command_type == "pairing.code":
            result = {"ok": True, "code": None, "hosted": True}
        elif command_type == "sms.history":
            response = bridge.events(website_device_id, 0)
            result = {"ok": True, "messages": _hosted_messages(response.get("events") or [])}
        elif command_type == "sms.send":
            response = bridge.sms_send(
                website_device_id,
                str(payload.get("to", "")),
                str(payload.get("message", "")),
            )
            result = {
                "ok": bool(response.get("sent")),
                "sent": bool(response.get("sent")),
                "hosted": True,
                "taskId": response.get("taskId", ""),
            }
        elif command_type == "tasks.list":
            result = {"ok": True, "tasks": []}
        else:
            result = {"ok": False, "error": f"Unsupported hosted command: {command_type}"}
        return ("done" if result.get("ok") else "failed"), result
    except Exception as exc:
        return "failed", {"ok": False, "error": str(exc)}


def sync_hosted_device(
    website: WebsiteClient,
    bridge: HostedBridgeClient,
    device: dict[str, Any],
    cursors: dict[str, int],
) -> None:
    device_id = str(device["id"])
    response = bridge.events(device_id, cursors.get(device_id, 0))
    cursors[device_id] = int(response.get("cursor", cursors.get(device_id, 0)) or 0)
    messages = _hosted_messages(response.get("events") or [])
    if messages:
        website.post_sms_cache(messages)


def hosted_daemon(args: Any) -> None:
    website_url = _required_env("GBRAIN_WEBSITE_URL")
    bridge_url = _required_env("GBRAIN_AUDIOLINK_BRIDGE_URL")
    service_token = _required_env("GBRAIN_SERVICE_TOKEN")

    service_website = WebsiteClient(website_url)
    bridge = HostedBridgeClient(bridge_url, service_token)
    devices = service_website.service_devices()
    clients: list[tuple[dict[str, Any], WebsiteClient]] = [
        (device, WebsiteClient(website_url, str(device["token"])))
        for device in devices
    ]
    cursors: dict[str, int] = {}
    last_sync: dict[str, float] = {}
    print(f"gbrain-cli hosted daemon started for {len(clients)} hosted device(s)")

    while True:
        try:
            for device, website in clients:
                device_id = str(device["id"])
                try:
                    command = website.next_command(device_id)
                    if command:
                        status, result = execute_hosted_command(command, bridge, device_id)
                        website.post_result(str(command.get("id", "")), status, result)
                        print(f"{device['name']} {command.get('type')}: {status}")
                    now = time.monotonic()
                    if now - last_sync.get(device_id, 0.0) >= args.sync_interval:
                        sync_hosted_device(website, bridge, device, cursors)
                        last_sync[device_id] = now
                except ApiError as exc:
                    print(f"{device['name']} relay error: {exc}")
                except Exception as exc:
                    print(f"{device['name']} daemon error: {exc}")
        except KeyboardInterrupt:
            print("gbrain-cli hosted daemon stopped")
            return
        time.sleep(max(1, args.interval))
