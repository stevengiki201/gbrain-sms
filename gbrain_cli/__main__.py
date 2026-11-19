from __future__ import annotations

import getpass
import os
import time
from typing import Any

from .client import ApiError, LocalBridgeClient, WebsiteClient
from .config import (
    clear_config,
    default_device_name,
    device_fingerprint,
    load_config,
    local_admin_token,
    save_config,
)


def _require_config() -> dict:
    config = load_config()
    required = ("server", "deviceId", "token")
    if any(not config.get(key) for key in required):
        raise RuntimeError("gbrain-cli is not signed in. Run: gbrain-cli sign-in")
    return config


def sign_in(args: Any) -> None:
    email = args.email or input("Email: ").strip()
    password = os.environ.get("GBRAIN_CLI_PASSWORD") or getpass.getpass("Password: ")
    name = args.name or default_device_name()
    fingerprint = device_fingerprint()
    client = WebsiteClient(args.server)
    result = client.sign_in(email, password, name, fingerprint)
    save_config(
        {
            "server": args.server.rstrip("/"),
            "deviceId": result.get("deviceId", ""),
            "token": result.get("token", ""),
            "name": result.get("name", name),
            "fingerprint": fingerprint,
        }
    )
    print(f"Signed in as {name}. Device ID: {result.get('deviceId', '')}")


def sign_out(args: Any) -> None:
    clear_config()
    print("Signed out of gbrain-cli.")


def status(args: Any) -> None:
    config = _require_config()
    print(f"Server: {config['server']}")
    print(f"Device: {config.get('name', '')}")
    print(f"Device ID: {config['deviceId']}")
    print(f"Fingerprint: {config.get('fingerprint', '')}")
    try:
        local = LocalBridgeClient(local_admin_token(), args.local_url)
        local.status()
        print("Local gbrain-sms bridge: reachable")
    except Exception as exc:
        print(f"Local gbrain-sms bridge: unavailable ({exc})")


def _pairing_code(local: LocalBridgeClient) -> dict[str, Any]:
    pending = local.pending_pairings()
    for request in pending:
        code = request.get("code") or request.get("pairing_code")
        if code:
            return {
                "ok": True,
                "code": str(code),
                "expiresAt": request.get("expires_at") or request.get("expiresAt"),
                "requestId": request.get("request_id") or request.get("requestId"),
            }
    return {"ok": True, "code": None, "expiresAt": None}


def _sms_history(local: LocalBridgeClient) -> dict[str, Any]:
    messages: list[dict[str, Any]] = []
    for folder in ("inbox", "sent"):
        for message in local.sms_inbox(folder=folder, limit=200):
            messages.append(
                {
                    "smsId": message.get("sms_id") or message.get("smsId"),
                    "folder": message.get("folder", folder),
                    "address": message.get("address", ""),
                    "body": message.get("body", ""),
                    "dateMs": message.get("date_ms") or message.get("dateMs") or 0,
                    "msgType": message.get("msg_type") or message.get("msgType") or 0,
                }
            )
    return {"ok": True, "messages": messages}


def execute_command(command: dict[str, Any], local: LocalBridgeClient) -> tuple[str, dict[str, Any]]:
    command_type = command.get("type")
    payload = command.get("payload") or {}
    try:
        if command_type == "bridge.status":
            result = {"ok": True, "status": local.status()}
        elif command_type == "pairing.code":
            result = _pairing_code(local)
        elif command_type == "sms.history":
            result = _sms_history(local)
        elif command_type == "sms.send":
            response = local.send_sms(str(payload.get("to", "")), str(payload.get("message", "")))
            result = {"ok": bool(response.get("sent")), **response}
        elif command_type == "tasks.list":
            result = {"ok": True, "tasks": local.tasks()}
        elif command_type == "ai.chat":
            result = {"ok": True, "chat": local.chat(str(payload.get("message", "")))}
        else:
            result = {"ok": False, "error": f"Unsupported command: {command_type}"}
        return ("done" if result.get("ok") else "failed"), result
    except Exception as exc:
        return "failed", {"ok": False, "error": str(exc)}


def sync(website: WebsiteClient, local: LocalBridgeClient) -> None:
    website.post_status({"bridge": "online"})
    pairing = _pairing_code(local)
    website.post_pairing_code(pairing.get("code"), pairing.get("expiresAt"))
    history = _sms_history(local)
    messages = history.get("messages") or []
    if messages:
        website.post_sms_cache(messages)


def daemon(args: Any) -> None:
    config = _require_config()
    website = WebsiteClient(config["server"], config["token"])
    local = LocalBridgeClient(local_admin_token(), args.local_url)
    last_sync = 0.0
    print(f"gbrain-cli daemon started for {config.get('name', config['deviceId'])}")

    while True:
        try:
            command = website.next_command(config["deviceId"])
            if command:
                status, result = execute_command(command, local)
                website.post_result(str(command.get("id", "")), status, result)
                print(f"{command.get('type')}: {status}")
            now = time.monotonic()
            if now - last_sync >= args.sync_interval:
                sync(website, local)
                last_sync = now
        except ApiError as exc:
            print(f"Relay error: {exc}")
        except Exception as exc:
            print(f"Daemon error: {exc}")
        time.sleep(max(1, args.interval))


def main(argv: list[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(prog="gbrain-cli", description="Connect this PC to the GBrain website dashboard.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    sign_in_parser = subparsers.add_parser("sign-in", help="Sign in and register this PC")
    sign_in_parser.add_argument("--email")
    sign_in_parser.add_argument("--name")
    sign_in_parser.add_argument("--server", default="https://gbrain.tech")
    sign_in_parser.set_defaults(handler=sign_in)

    daemon_parser = subparsers.add_parser("daemon", help="Poll for dashboard commands")
    daemon_parser.add_argument("--interval", type=float, default=3.0)
    daemon_parser.add_argument("--sync-interval", type=float, default=30.0)
    daemon_parser.add_argument("--local-url", default="https://127.0.0.1:8787")
    daemon_parser.set_defaults(handler=daemon)

    status_parser = subparsers.add_parser("status", help="Show gbrain-cli configuration")
    status_parser.add_argument("--local-url", default="https://127.0.0.1:8787")
    status_parser.set_defaults(handler=status)

    sign_out_parser = subparsers.add_parser("sign-out", help="Remove stored credentials")
    sign_out_parser.set_defaults(handler=sign_out)

    args = parser.parse_args(argv)
    try:
        args.handler(args)
    except (ApiError, RuntimeError) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
