"""Stable device identity and the TLS material used to protect signaling.

The desktop is identified by a persisted UUID plus a self-signed certificate.
The certificate fingerprint is advertised over mDNS and confirmed out-of-band
during pairing, so the phone can pin the exact certificate it paired with.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import ipaddress
import json
import os
import socket
import uuid
from dataclasses import dataclass
from pathlib import Path

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID

from .errors import Unavailable

CERT_FILE = "cert.pem"
KEY_FILE = "key.pem"
IDENTITY_FILE = "identity.json"


def utcnow() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def fingerprint_of_cert_der(der: bytes) -> str:
    """Return a colon-separated uppercase SHA-256 fingerprint."""
    digest = hashlib.sha256(der).hexdigest().upper()
    return ":".join(digest[i : i + 2] for i in range(0, len(digest), 2))


def fingerprint_from_pem(path: Path) -> str:
    cert = x509.load_pem_x509_certificate(Path(path).read_bytes())
    return fingerprint_of_cert_der(cert.public_bytes(serialization.Encoding.DER))


@dataclass(frozen=True)
class Identity:
    device_id: str
    name: str
    data_dir: Path
    cert_path: Path
    key_path: Path
    fingerprint: str
    created_at: str

    @property
    def platform(self) -> str:
        return "linux"


def _atomic_write(path: Path, data: bytes, mode: int = 0o600) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, mode)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
    os.chmod(path, mode)


def load_or_create_identity(data_dir: Path, name: str) -> Identity:
    """Load persisted identity/TLS material, generating it on first run."""
    data_dir = Path(data_dir).expanduser()
    tls_dir = data_dir / "tls"
    cert_path = tls_dir / CERT_FILE
    key_path = tls_dir / KEY_FILE
    identity_path = data_dir / IDENTITY_FILE

    record: dict[str, str] = {}
    if identity_path.is_file():
        record = json.loads(identity_path.read_text(encoding="utf-8"))

    device_id = record.get("device_id") or str(uuid.uuid4())
    created_at = record.get("created_at") or utcnow().isoformat()

    if not cert_path.is_file() or not key_path.is_file():
        cert_pem, key_pem = generate_self_signed_cert(name)
        _atomic_write(cert_path, cert_pem, 0o644)
        _atomic_write(key_path, key_pem, 0o600)

    record = {"device_id": device_id, "created_at": created_at}
    _atomic_write(identity_path, json.dumps(record, indent=2).encode(), 0o600)

    return Identity(
        device_id=device_id,
        name=name,
        data_dir=data_dir,
        cert_path=cert_path,
        key_path=key_path,
        fingerprint=fingerprint_from_pem(cert_path),
        created_at=created_at,
    )


def _local_addresses() -> list[str]:
    addresses = {"127.0.0.1", "::1"}
    try:
        host = socket.gethostname()
        for info in socket.getaddrinfo(host, None):
            addresses.add(info[4][0])
    except OSError:
        pass
    try:
        # No packet is sent; this discovers the primary outbound interface.
        probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            probe.connect(("192.0.2.1", 9))
            addresses.add(probe.getsockname()[0])
        finally:
            probe.close()
    except OSError:
        pass
    return sorted(addresses)


def generate_self_signed_cert(device_name: str) -> tuple[bytes, bytes]:
    """Create a self-signed RSA-2048 certificate valid for this host."""
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    subject = issuer = x509.Name(
        [
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, "GBrain"),
            x509.NameAttribute(NameOID.COMMON_NAME, device_name or "gbrain-audiolink"),
        ]
    )

    san: list[x509.GeneralName] = [x509.DNSName("localhost")]
    if device_name:
        try:
            san.append(x509.DNSName(device_name))
            san.append(x509.DNSName(f"{device_name}.local"))
        except ValueError:
            pass

    ips = []
    for raw in _local_addresses():
        try:
            ips.append(x509.IPAddress(ipaddress.ip_address(raw.split("%")[0])))
        except ValueError:
            continue
    san.extend(ips)

    now = utcnow()
    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - dt.timedelta(minutes=5))
        .not_valid_after(now + dt.timedelta(days=825))
        .add_extension(x509.SubjectAlternativeName(san), critical=False)
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
        .sign(key, hashes.SHA256())
    )

    key_pem = key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    cert_pem = cert.public_bytes(serialization.Encoding.PEM)
    return cert_pem, key_pem


def ensure_tls(identity: Identity) -> tuple[str, str]:
    """Return (cert_path, key_path) for uvicorn, verifying files exist."""
    if not identity.cert_path.is_file() or not identity.key_path.is_file():
        raise Unavailable(
            "TLS material is missing",
            details={"data_dir": str(identity.data_dir)},
        )
    return str(identity.cert_path), str(identity.key_path)
