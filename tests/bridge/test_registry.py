from __future__ import annotations

import json
import stat

import pytest

from bot.bridge.errors import NotFound, Unauthorized
from bot.bridge.registry import DeviceRegistry, hash_token


async def test_pair_verify_remove_roundtrip(tmp_path):
    registry = DeviceRegistry(tmp_path)
    device, token = await registry.pair_device(
        device_id="phone-1", name="Pixel", platform="android", app_version="0.1.0"
    )
    assert token.startswith("al_")
    assert device.token_hash == hash_token(token)
    assert "token_hash" not in device.public()

    verified = await registry.verify_token("phone-1", token)
    assert verified.device_id == "phone-1"

    with pytest.raises(Unauthorized) as exc:
        await registry.verify_token("phone-1", "al_wrong")
    assert exc.value.code == "INVALID_TOKEN"

    await registry.remove("phone-1")
    with pytest.raises(NotFound):
        await registry.get("phone-1")


async def test_registry_file_never_contains_plaintext_token(tmp_path):
    registry = DeviceRegistry(tmp_path)
    _, token = await registry.pair_device(device_id="phone-1", name="Pixel", platform="android")

    payload = (tmp_path / "devices.json").read_text()
    assert token not in payload
    assert hash_token(token) in payload
    assert stat.S_IMODE((tmp_path / "devices.json").stat().st_mode) == 0o600
    assert json.loads(payload)["version"] == 1


async def test_registry_persists_across_instances(tmp_path):
    first = DeviceRegistry(tmp_path)
    _, token = await first.pair_device(device_id="phone-1", name="Pixel", platform="android")

    second = DeviceRegistry(tmp_path)
    device = await second.verify_token("phone-1", token)
    assert device.name == "Pixel"


async def test_revoked_device_is_rejected(tmp_path):
    registry = DeviceRegistry(tmp_path)
    _, token = await registry.pair_device(device_id="p", name="p", platform="android")
    record = registry._devices["p"]
    record.revoked = True
    registry._save()

    with pytest.raises(Unauthorized) as exc:
        await registry.verify_token("p", token)
    assert exc.value.code == "DEVICE_REVOKED"


async def test_repairing_rotates_token(tmp_path):
    registry = DeviceRegistry(tmp_path)
    _, old_token = await registry.pair_device(device_id="p", name="p", platform="android")
    _, new_token = await registry.pair_device(device_id="p", name="p", platform="android")
    assert old_token != new_token

    with pytest.raises(Unauthorized):
        await registry.verify_token("p", old_token)
    await registry.verify_token("p", new_token)
