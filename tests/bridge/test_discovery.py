from __future__ import annotations

import socket

from zeroconf.asyncio import AsyncServiceInfo

from bot.bridge.config import SERVICE_TYPE
from bot.bridge.discovery import parse_service_info


def make_info(**overrides):
    props = {
        "id": "desktop-123",
        "name": "Studio PC",
        "host": "studio.local",
        "fp": "AA:BB:CC",
        "v": "1",
    }
    props.update(overrides.pop("properties", {}))
    kwargs = {
        "type_": SERVICE_TYPE,
        "name": f"Studio-123.{SERVICE_TYPE}",
        "addresses": [socket.inet_aton("192.168.1.10")],
        "port": 8787,
        "properties": props,
        "server": "studio.local.",
    }
    kwargs.update(overrides)
    return AsyncServiceInfo(**kwargs)


def test_parse_service_info_maps_txt_records():
    device = parse_service_info(make_info())
    assert device is not None
    assert device.device_id == "desktop-123"
    assert device.name == "Studio PC"
    assert device.port == 8787
    assert device.fingerprint == "AA:BB:CC"
    assert device.addresses == ["192.168.1.10"]
    assert device.base_url == "https://192.168.1.10:8787"


def test_parse_service_info_requires_device_id():
    assert parse_service_info(make_info(properties={"id": ""})) is None


def test_parse_service_info_prefers_ipv4_for_urls():
    info = AsyncServiceInfo(
        SERVICE_TYPE,
        f"x.{SERVICE_TYPE}",
        addresses=[
            socket.inet_pton(socket.AF_INET6, "fe80::1"),
            socket.inet_aton("10.0.0.5"),
        ],
        port=9000,
        properties={"id": "x", "name": "X", "fp": "AB"},
    )
    device = parse_service_info(info)
    assert device is not None
    assert device.base_url == "https://10.0.0.5:9000"
