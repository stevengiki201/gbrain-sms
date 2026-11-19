from __future__ import annotations

import httpx
import pytest


async def test_health_and_identity_are_public(client, service):
    health = await client.get("/v1/health")
    assert health.status_code == 200
    assert health.json()["status"] == "ok"

    identity = await client.get("/v1/identity")
    body = identity.json()
    assert body["device_id"] == service.identity.device_id
    assert body["fingerprint"] == service.identity.fingerprint
    assert body["pairing_required"] is True


async def test_pairing_over_http_returns_token_once(client, service):
    start = await client.post(
        "/v1/pairing/requests",
        json={"device_id": "phone-1", "device_name": "Pixel", "platform": "android"},
    )
    assert start.status_code == 200
    body = start.json()
    assert body["code_length"] == 6
    assert "code" not in body  # never sent to the phone

    pending = await client.get("/v1/pairing/pending")
    code = pending.json()["requests"][0]["code"]

    confirm = await client.post(
        f"/v1/pairing/requests/{body['request_id']}/confirm", json={"code": code}
    )
    assert confirm.status_code == 200
    payload = confirm.json()
    assert payload["token"].startswith("al_")
    assert payload["desktop"]["fingerprint"] == service.identity.fingerprint
    assert payload["device"]["device_id"] == "phone-1"

    devices = await client.get("/v1/devices")
    assert [d["device_id"] for d in devices.json()["devices"]] == ["phone-1"]


async def test_management_api_requires_admin_token(app, service):
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as raw:
        response = await raw.get("/v1/status")
    assert response.status_code == 401
    assert response.json()["error"]["code"] == "INVALID_ADMIN_TOKEN"


async def test_management_api_rejects_non_loopback_clients(app, service):
    transport = httpx.ASGITransport(app=app, client=("192.168.1.50", 4000))
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://testserver",
        headers={"X-AudioLink-Admin": service.admin_token},
    ) as raw:
        response = await raw.get("/v1/status")
    assert response.status_code == 403
    assert response.json()["error"]["code"] == "LOCAL_ONLY"


async def test_operations_endpoint_exposes_gbrain_contract(client, service):
    listed = await client.get("/v1/operations")
    names = {op["operation"] for op in listed.json()["operations"]}
    assert "audiolink.get_status" in names
    assert "audiolink.pair_device" in names
    assert "audiolink.sms_list_tasks" in names
    assert "audiolink.sms_send_now" in names
    assert len(names) == 23

    status = await client.post(
        "/v1/operations", json={"operation": "audiolink.get_status"}
    )
    assert status.status_code == 200
    assert status.json()["result"]["device_id"] == service.identity.device_id

    unknown = await client.post("/v1/operations", json={"operation": "audiolink.nope"})
    assert unknown.status_code == 422
    assert unknown.json()["error"]["code"] == "VALIDATION_FAILED"


async def test_media_session_requires_a_paired_online_device(client, service):
    missing = await client.post(
        "/v1/sessions/start", json={"device_id": "phone-1"}
    )
    assert missing.status_code == 404
    assert missing.json()["error"]["code"] == "DEVICE_NOT_PAIRED"

    await service.registry.pair_device(
        device_id="phone-1", name="Pixel", platform="android"
    )
    offline = await client.post(
        "/v1/sessions/start", json={"device_id": "phone-1"}
    )
    assert offline.status_code == 409
    assert offline.json()["error"]["code"] == "DEVICE_OFFLINE"

    status = (await client.get("/v1/status")).json()
    assert status["media"]["engine"] in {"aiortc", ""}
    assert "playout_backend" in status["media"]
    assert status["media_transport"] in {"webrtc", "unavailable"}


async def test_connect_requires_online_signaling_peer(client, service):
    _, token = await service.registry.pair_device(
        device_id="phone-1", name="Pixel", platform="android"
    )
    response = await client.post("/v1/devices/phone-1/connect")
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "DEVICE_OFFLINE"


async def test_volume_validation(client):
    bad = await client.post("/v1/controls/volume", json={"level": 180})
    assert bad.status_code == 422
    assert bad.json()["error"]["code"] == "VALIDATION_FAILED"
