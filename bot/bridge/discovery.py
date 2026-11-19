"""mDNS discovery of GBrain AudioLink desktops.

The desktop advertises ``_gbrain-alink._tcp.local.`` with a TXT record
carrying its device id, display name, protocol version and TLS certificate
fingerprint. Android browses the same service type, so no manual IP entry is
needed on a local Wi-Fi network.
"""

from __future__ import annotations

import asyncio
import ipaddress
import logging
import socket
import time
from dataclasses import dataclass, field

from zeroconf import IPVersion, ServiceInfo, ServiceStateChange
from zeroconf.asyncio import AsyncServiceBrowser, AsyncServiceInfo, AsyncZeroconf

from .config import PROTOCOL_VERSION, SERVICE_TYPE
from .identity import Identity

logger = logging.getLogger("gbrain.bridge.discovery")


class _AndroidSafeServiceInfo(ServiceInfo):
    """ServiceInfo that never emits NSEC records.

    python-zeroconf adds an NSEC record asserting which address families exist
    whenever a host has only IPv4 (or only IPv6). Android's mDNS decoder
    rejects the record (NetworkStack ``MdnsPacket`` logs "Error while decoding
    multicast packet") and drops the **whole packet**, so the desktop is
    invisible to the companion app even though the announcements are on the
    wire. The address records already tell Android everything it needs, so
    they are published on their own.
    """

    def get_address_and_nsec_records(self, override_ttl: int | None = None) -> set:
        # Same contract, minus the NSEC record Android refuses to parse.
        return set(self.dns_addresses(override_ttl))

    def _get_address_and_nsec_records(self, override_ttl: int | None = None) -> set:
        return self.get_address_and_nsec_records(override_ttl)


def _addresses_from_info(info: AsyncServiceInfo) -> list[str]:
    addresses: list[str] = []
    for packed in info.addresses:
        try:
            if len(packed) == 4:
                addresses.append(socket.inet_ntoa(packed))
            else:
                addresses.append(socket.inet_ntop(socket.AF_INET6, packed))
        except (OSError, ValueError):
            continue
    return sorted(set(addresses))


def _txt(info: AsyncServiceInfo, key: str) -> str:
    raw = info.properties.get(key.encode("utf-8")) or info.properties.get(key)
    if raw is None:
        return ""
    if isinstance(raw, bytes):
        return raw.decode("utf-8", errors="replace")
    return str(raw)


def _usable_http_address(addresses: list[str]) -> str:
    """Prefer IPv4 loopback/LAN addresses over link-local IPv6."""
    for address in addresses:
        if address.startswith("127."):
            return address
    for address in addresses:
        if ":" not in address:
            return address
    return addresses[0] if addresses else ""


@dataclass
class DiscoveredDevice:
    device_id: str
    name: str
    host: str
    port: int
    fingerprint: str
    addresses: list[str] = field(default_factory=list)
    version: int = PROTOCOL_VERSION
    service_name: str = ""
    last_seen: float = field(default_factory=time.time)

    @property
    def base_url(self) -> str:
        address = _usable_http_address(self.addresses) or self.host
        host = f"[{address}]" if ":" in address else address
        return f"https://{host}:{self.port}"

    def public(self) -> dict:
        return {
            "device_id": self.device_id,
            "name": self.name,
            "host": self.host,
            "port": self.port,
            "fingerprint": self.fingerprint,
            "addresses": list(self.addresses),
            "version": self.version,
            "base_url": self.base_url,
            "last_seen": self.last_seen,
        }


def parse_service_info(
    info: AsyncServiceInfo, *, service_type: str = SERVICE_TYPE
) -> DiscoveredDevice | None:
    """Convert an mDNS service record into a discovered device."""
    if not info or not info.addresses or not info.port:
        return None
    device_id = _txt(info, "id")
    if not device_id:
        return None
    fingerprint = _txt(info, "fp")
    try:
        version = int(_txt(info, "v") or PROTOCOL_VERSION)
    except ValueError:
        version = PROTOCOL_VERSION

    addresses = _addresses_from_info(info)
    name = _txt(info, "name") or device_id
    return DiscoveredDevice(
        device_id=device_id,
        name=name,
        host=_txt(info, "host") or (_usable_http_address(addresses)),
        port=info.port,
        fingerprint=fingerprint,
        addresses=addresses,
        version=version,
        service_name=info.name,
    )


class ServiceAdvertiser:
    """Advertises this desktop over mDNS."""

    def __init__(
        self,
        identity: Identity,
        *,
        port: int,
        service_type: str = SERVICE_TYPE,
        name: str | None = None,
    ) -> None:
        self.identity = identity
        self.port = port
        self.service_type = service_type
        self.instance = name or identity.name
        self._zc: AsyncZeroconf | None = None
        self._info: AsyncServiceInfo | None = None

    @property
    def service_name(self) -> str:
        safe = self.instance.replace(".", "-") or "gbrain-audiolink"
        return f"{safe}-{self.identity.device_id[:8]}.{self.service_type}"

    async def start(self) -> None:
        self._zc = AsyncZeroconf()
        info = _AndroidSafeServiceInfo(
            self.service_type,
            self.service_name,
            addresses=[socket.inet_aton(addr) for addr in _local_ipv4_addresses()],
            port=self.port,
            properties={
                "id": self.identity.device_id,
                "name": self.instance,
                "host": socket.gethostname(),
                "fp": self.identity.fingerprint,
                "v": str(PROTOCOL_VERSION),
                "proto": "audiolink",
            },
            server=f"{socket.gethostname().split('.')[0]}.local.",
        )
        self._info = info
        await self._zc.async_register_service(info)
        logger.info(
            "mDNS advertising %s on port %s", self.service_name, self.port
        )

    async def stop(self) -> None:
        if self._zc is None:
            return
        try:
            if self._info is not None:
                await self._zc.async_unregister_service(self._info)
        finally:
            await self._zc.async_close()
            self._zc = None
            self._info = None

    async def update_port(self, port: int) -> None:
        """Re-publish with a new port (used when the API binds port 0)."""
        if self.port == port or self._zc is None:
            self.port = port
            return
        await self.stop()
        self.port = port
        await self.start()


# Interfaces whose addresses a phone on the Wi-Fi LAN cannot reach: loopback,
# container bridges, virtual switches and VPN tunnels. Advertising them makes
# Android's NSD resolve to an unusable address, so they are skipped.
_VIRTUAL_INTERFACE_PREFIXES = (
    "lo",
    "docker",
    "br-",
    "veth",
    "virbr",
    "tap",
    "tun",
    "wg",
    "zt",
    "tailscale",
    "ip6tnl",
)


def _interface_ipv4_addresses() -> list[str]:
    """IPv4 addresses of real network interfaces (Linux ``SIOCGIFADDR``)."""
    try:
        import fcntl
        import struct
    except ImportError:  # pragma: no cover - non-Unix platform
        return []
    try:
        names = [name for _, name in socket.if_nameindex()]
    except OSError:  # pragma: no cover - no interface list
        return []
    addresses: list[str] = []
    for name in names:
        if name.startswith(_VIRTUAL_INTERFACE_PREFIXES):
            continue
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            packed = fcntl.ioctl(
                sock.fileno(), 0x8915, struct.pack("256s", name[:15].encode())
            )[20:24]
        except OSError:
            continue
        finally:
            sock.close()
        address = socket.inet_ntoa(packed)
        if ipaddress.ip_address(address).is_loopback:
            continue
        addresses.append(address)
    return addresses


def _local_ipv4_addresses() -> list[str]:
    """Addresses the phone should dial, preferring the LAN interface."""
    addresses = _interface_ipv4_addresses()
    if not addresses:
        # Fall back to the interface the default route leaves through.
        try:
            probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            try:
                probe.connect(("192.0.2.1", 9))
                addresses.append(probe.getsockname()[0])
            finally:
                probe.close()
        except OSError:
            pass
    if not addresses:  # pragma: no cover - offline machine, keeps the record valid
        addresses = ["127.0.0.1"]
    return sorted(set(addresses))


class DiscoveryBrowser:
    """Browses for AudioLink desktops on the local network."""

    def __init__(
        self,
        *,
        service_type: str = SERVICE_TYPE,
        exclude_device_id: str = "",
        on_update=None,
    ) -> None:
        self.service_type = service_type
        self.exclude_device_id = exclude_device_id
        self.on_update = on_update
        self._zc: AsyncZeroconf | None = None
        self._browser: AsyncServiceBrowser | None = None
        self._devices: dict[str, DiscoveredDevice] = {}
        self._lock = asyncio.Lock()
        self._tasks: set[asyncio.Task] = set()

    @property
    def devices(self) -> list[DiscoveredDevice]:
        return sorted(self._devices.values(), key=lambda d: d.name.lower())

    async def start(self) -> None:
        self._zc = AsyncZeroconf()
        self._browser = AsyncServiceBrowser(
            self._zc.zeroconf, self.service_type, handlers=[self._on_event]
        )
        logger.info("mDNS browsing for %s", self.service_type)

    async def stop(self) -> None:
        if self._zc is None:
            return
        try:
            if self._browser is not None:
                await self._browser.async_cancel()
        finally:
            for task in list(self._tasks):
                task.cancel()
            self._tasks.clear()
            await self._zc.async_close()
            self._zc = None
            self._browser = None

    def _on_event(self, zeroconf, service_type, name, state_change) -> None:
        """zeroconf calls handlers synchronously; schedule the async work."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:  # pragma: no cover - no loop, cannot process
            return
        task = loop.create_task(self._handle(zeroconf, service_type, name, state_change))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _handle(self, zeroconf, service_type, name, state_change) -> None:
        if state_change is ServiceStateChange.Removed:
            async with self._lock:
                removed = [
                    d_id
                    for d_id, dev in self._devices.items()
                    if dev.service_name == name
                ]
                for d_id in removed:
                    self._devices.pop(d_id, None)
            if removed:
                await self._notify()
            return

        if state_change is not ServiceStateChange.Added:
            # Updated records are re-read below; ignore transient transitions.
            pass

        assert self._zc is not None
        try:
            info = await self._zc.async_get_service_info(service_type, name)
        except Exception as exc:  # pragma: no cover - network timing
            logger.debug("mDNS info lookup failed for %s: %s", name, exc)
            return
        device = parse_service_info(info, service_type=service_type)
        if device is None or device.device_id == self.exclude_device_id:
            return
        async with self._lock:
            self._devices[device.device_id] = device
        await self._notify()

    async def _notify(self) -> None:
        if self.on_update is None:
            return
        try:
            result = self.on_update(self.devices)
            if asyncio.iscoroutine(result):
                await result
        except Exception:  # pragma: no cover - listener isolation
            logger.exception("discovery listener failed")


def is_loopback(address: str) -> bool:
    try:
        return ipaddress.ip_address(address.split("%")[0]).is_loopback
    except ValueError:
        return False
