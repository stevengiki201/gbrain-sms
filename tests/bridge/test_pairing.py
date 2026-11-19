from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from bot.bridge.errors import Expired, Forbidden, NotFound, RateLimited, Unauthorized
from bot.bridge.pairing import PairingManager
from bot.bridge.registry import DeviceRegistry


class FakeClock:
    def __init__(self) -> None:
        self.now = datetime(2026, 1, 1, tzinfo=timezone.utc)

    def __call__(self) -> datetime:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += timedelta(seconds=seconds)


@pytest.fixture
def setup(tmp_path):
    clock = FakeClock()
    registry = DeviceRegistry(tmp_path)
    manager = PairingManager(registry, ttl_sec=60, max_attempts=3, rate_limit=3, clock=clock)
    return manager, registry, clock


async def test_pairing_happy_path_issues_working_token(setup):
    manager, registry, _ = setup
    request = await manager.request_pairing(
        device_id="phone-1", device_name="Pixel", platform="android"
    )
    assert len(request.code) == 6 and request.code.isdigit()

    pending = await manager.pending(include_code=True)
    assert pending[0]["code"] == request.code

    result = await manager.confirm(request.request_id, request.code)
    assert result.device.device_id == "phone-1"
    device = await registry.verify_token("phone-1", result.token)
    assert device.name == "Pixel"


async def test_wrong_code_consumes_attempts_then_locks(setup):
    manager, _, _ = setup
    request = await manager.request_pairing(device_id="p")

    with pytest.raises(Unauthorized) as first:
        await manager.confirm(request.request_id, "000000")
    assert first.value.details["attempts_remaining"] == 2

    with pytest.raises(Unauthorized):
        await manager.confirm(request.request_id, "111111")
    with pytest.raises(Forbidden) as locked:
        await manager.confirm(request.request_id, "222222")
    assert locked.value.code == "TOO_MANY_ATTEMPTS"

    # Even the correct code no longer works after lockout.
    with pytest.raises(Forbidden):
        await manager.confirm(request.request_id, request.code)


async def test_expired_code_is_rejected(setup):
    manager, _, clock = setup
    request = await manager.request_pairing(device_id="p")
    clock.advance(61)
    with pytest.raises(Expired):
        await manager.confirm(request.request_id, request.code)


async def test_code_is_single_use(setup):
    manager, _, _ = setup
    request = await manager.request_pairing(device_id="p")
    await manager.confirm(request.request_id, request.code)
    with pytest.raises(Forbidden) as exc:
        await manager.confirm(request.request_id, request.code)
    assert exc.value.code == "PAIRING_ALREADY_USED"


async def test_rate_limit_blocks_request_flood(setup):
    manager, _, clock = setup
    for _ in range(3):
        await manager.request_pairing(device_id="spammer")
    with pytest.raises(RateLimited):
        await manager.request_pairing(device_id="spammer")
    clock.advance(61)
    await manager.request_pairing(device_id="spammer")


async def test_unknown_request_and_cancel(setup):
    manager, _, _ = setup
    with pytest.raises(NotFound):
        await manager.confirm("missing", "123456")

    request = await manager.request_pairing(device_id="p")
    await manager.cancel(request.request_id)
    with pytest.raises(Forbidden) as exc:
        await manager.confirm(request.request_id, request.code)
    assert exc.value.code == "PAIRING_CANCELLED"
