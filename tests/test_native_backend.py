from __future__ import annotations

import base64

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from vloop.authorization import CapabilitySigner, CapabilityVerifier, InMemoryNonceStore
from vloop.canonical import canonical_json, digest
from vloop.models import ActionIntent, ActionRule, Effect, Provenance, TaskContract
from vloop.native_backend import (
    NativeBackendUnavailable,
    ed25519_public_key,
    ed25519_sign,
    ed25519_verify,
    native_status,
    reset_native_backend_for_tests,
    sha256_hex,
)


def test_native_backend_fallback_matches_canonical_python_contract() -> None:
    value = {"unicode": "λ", "items": [{"b": 2, "a": 1}]}
    encoded = canonical_json(value).encode("utf-8")
    assert sha256_hex(encoded) == digest(value)
    assert native_status().implementation in {"python", "rust"}


def test_ed25519_backend_contract_is_stable() -> None:
    private_key = b"k" * 32
    public_key = ed25519_public_key(private_key)
    payload = b"canonical V-Loop payload"
    signature = ed25519_sign(private_key, payload)
    expected = base64.urlsafe_b64encode(
        Ed25519PrivateKey.from_private_bytes(private_key).sign(payload)
    ).decode("ascii")
    assert signature == expected
    assert ed25519_verify(public_key, payload, signature)
    assert not ed25519_verify(public_key, b"modified payload", signature)


def test_capability_round_trip_uses_backend_without_api_change() -> None:
    contract = TaskContract("native capability", ("done",), (ActionRule("command.run", Effect.EXECUTE, "/workspace"),))
    intent = ActionIntent(
        "command.run", Effect.EXECUTE, "/workspace/job", {"command": ["/bin/true"]},
        (Provenance.USER,), "native backend test", contract.contract_id, contract.version,
    )
    signer = CapabilitySigner(b"s" * 32)
    from datetime import UTC, datetime, timedelta

    now = datetime.now(UTC)
    capability = signer.issue(
        intent=intent, contract_digest=contract.contract_digest, executor_id="native-test",
        issued_at=now, expires_at=now + timedelta(minutes=1),
    )
    CapabilityVerifier(signer.public_key_bytes, InMemoryNonceStore()).validate_and_consume(
        capability, intent, executor_id="native-test", now=now
    )


def test_required_mode_fails_closed_when_no_extension_is_installed(monkeypatch) -> None:
    try:
        import vloop_native._core  # type: ignore[import-not-found]
    except ModuleNotFoundError:
        monkeypatch.setenv("VLOOP_NATIVE_BACKEND", "required")
        reset_native_backend_for_tests()
        with pytest.raises(NativeBackendUnavailable):
            native_status()
    finally:
        monkeypatch.delenv("VLOOP_NATIVE_BACKEND", raising=False)
        reset_native_backend_for_tests()
