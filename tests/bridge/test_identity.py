from __future__ import annotations

import stat

from cryptography import x509
from cryptography.hazmat.primitives import serialization

from bot.bridge.identity import load_or_create_identity


def test_identity_is_stable_and_creates_tls_material(tmp_path):
    first = load_or_create_identity(tmp_path, "desk")
    second = load_or_create_identity(tmp_path, "desk")

    assert first.device_id == second.device_id
    assert first.fingerprint == second.fingerprint
    assert first.cert_path.is_file()
    assert first.key_path.is_file()

    key_mode = stat.S_IMODE(first.key_path.stat().st_mode)
    assert key_mode == 0o600

    fingerprint = first.fingerprint
    assert fingerprint.count(":") == 31
    assert fingerprint == fingerprint.upper()


def test_certificate_covers_loopback_for_tls_clients(tmp_path):
    identity = load_or_create_identity(tmp_path, "desk")
    cert = x509.load_pem_x509_certificate(identity.cert_path.read_bytes())
    san = cert.extensions.get_extension_for_class(x509.SubjectAlternativeName).value
    assert "127.0.0.1" in [str(ip) for ip in san.get_values_for_type(x509.IPAddress)]
    assert "localhost" in san.get_values_for_type(x509.DNSName)
    # Certificate must be loadable as DER for pinning on Android.
    assert cert.public_bytes(serialization.Encoding.DER)


def test_separate_data_dirs_get_distinct_identities(tmp_path):
    one = load_or_create_identity(tmp_path / "a", "desk")
    two = load_or_create_identity(tmp_path / "b", "desk")
    assert one.device_id != two.device_id
