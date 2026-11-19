from __future__ import annotations

import httpx
import pytest

from bot.bridge.api import create_app
from bot.bridge.config import AudioLinkConfig
from bot.bridge.service import AudioLinkService



@pytest.fixture
def config(tmp_path):
    return AudioLinkConfig(
        data_dir=str(tmp_path),
        mdns_enabled=False,
        host="127.0.0.1",
        port=0,
        device_name="audiolink-test",
        pairing_code_ttl_sec=120,
        pairing_max_attempts=5,
    )


@pytest.fixture
async def service(config):
    svc = AudioLinkService(config)
    await svc.start()
    yield svc
    await svc.stop()


@pytest.fixture
def app(service):
    return create_app(service)


@pytest.fixture
async def client(app, service):
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://testserver",
        headers={"X-AudioLink-Admin": service.admin_token},
    ) as http:
        yield http
