"""Public-key capabilities and durable one-time nonce consumption.

The policy decision point owns an Ed25519 private key.  Executors receive only
the corresponding public key and therefore can verify a capability without
being able to mint one.  Capability consumption is deliberately performed at
the executor enforcement point, immediately before a side effect.
"""

from __future__ import annotations

import base64
import sqlite3
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol
from uuid import uuid4

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey

from .canonical import canonical_json
from .models import ActionIntent, Capability


class CapabilityRejected(PermissionError):
    """An executor rejected a capability before executing a side effect."""


def capability_payload(capability: Capability) -> bytes:
    return canonical_json(
        {
            "capability_id": capability.capability_id,
            "nonce": capability.nonce,
            "intent_digest": capability.intent_digest,
            "contract_digest": capability.contract_digest,
            "executor_id": capability.executor_id,
            "issued_at": capability.issued_at.isoformat(),
            "expires_at": capability.expires_at.isoformat(),
        }
    ).encode("utf-8")


class CapabilitySigner:
    """Policy-side issuer.  Private key material never leaves this object."""

    def __init__(self, private_key: bytes | Ed25519PrivateKey | None = None) -> None:
        if private_key is None:
            self._private_key = Ed25519PrivateKey.generate()
        elif isinstance(private_key, bytes):
            if len(private_key) != 32:
                raise ValueError("Ed25519 private keys must contain exactly 32 bytes")
            self._private_key = Ed25519PrivateKey.from_private_bytes(private_key)
        else:
            self._private_key = private_key

    @property
    def public_key_bytes(self) -> bytes:
        return self._private_key.public_key().public_bytes(
            serialization.Encoding.Raw, serialization.PublicFormat.Raw
        )

    def issue(
        self,
        *,
        intent: ActionIntent,
        contract_digest: str,
        executor_id: str,
        issued_at: datetime,
        expires_at: datetime,
    ) -> Capability:
        if not executor_id.strip():
            raise ValueError("capability needs a non-empty executor audience")
        capability = Capability(
            capability_id=str(uuid4()),
            nonce=str(uuid4()),
            intent_digest=intent.intent_digest,
            contract_digest=contract_digest,
            executor_id=executor_id,
            issued_at=issued_at,
            expires_at=expires_at,
            signature="",
        )
        signature = base64.urlsafe_b64encode(self._private_key.sign(capability_payload(capability))).decode(
            "ascii"
        )
        return Capability(
            capability_id=capability.capability_id,
            nonce=capability.nonce,
            intent_digest=capability.intent_digest,
            contract_digest=capability.contract_digest,
            executor_id=capability.executor_id,
            issued_at=capability.issued_at,
            expires_at=capability.expires_at,
            signature=signature,
        )


class NonceStore(Protocol):
    def consume(self, nonce: str, *, expires_at: datetime) -> bool: ...


class InMemoryNonceStore:
    """Thread-safe development nonce store; use SQLite or a service in production."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._consumed: set[str] = set()

    def consume(self, nonce: str, *, expires_at: datetime) -> bool:
        del expires_at
        with self._lock:
            if nonce in self._consumed:
                return False
            self._consumed.add(nonce)
            return True


class SQLiteNonceStore:
    """Process-safe durable nonce consumption with a uniqueness constraint."""

    def __init__(self, database: str | Path) -> None:
        path = Path(database)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(path, isolation_level=None)
        self._connection.execute("PRAGMA busy_timeout=5000")
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS consumed_capability_nonces (
                nonce TEXT PRIMARY KEY,
                expires_at TEXT NOT NULL,
                consumed_at TEXT NOT NULL
            )
            """
        )

    def consume(self, nonce: str, *, expires_at: datetime) -> bool:
        now = datetime.now(UTC).isoformat()
        try:
            with self._connection:
                self._connection.execute(
                    "INSERT INTO consumed_capability_nonces (nonce, expires_at, consumed_at) VALUES (?, ?, ?)",
                    (nonce, expires_at.isoformat(), now),
                )
        except sqlite3.IntegrityError:
            return False
        return True

    def close(self) -> None:
        self._connection.close()


class CapabilityVerifier:
    """Executor-side public-key verifier and single-use nonce enforcer."""

    def __init__(self, public_key: bytes | Ed25519PublicKey, nonce_store: NonceStore) -> None:
        self._public_key = (
            Ed25519PublicKey.from_public_bytes(public_key)
            if isinstance(public_key, bytes)
            else public_key
        )
        self._nonce_store = nonce_store

    @property
    def nonce_store(self) -> NonceStore:
        """Store inspected by production configuration validation."""

        return self._nonce_store

    @property
    def public_key_bytes(self) -> bytes:
        return self._public_key.public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)

    def validate(
        self,
        capability: Capability,
        intent: ActionIntent,
        *,
        executor_id: str,
        now: datetime | None = None,
    ) -> None:
        current = now or datetime.now(UTC)
        if capability.executor_id != executor_id:
            raise CapabilityRejected("capability audience does not match this executor")
        if capability.expires_at <= current or capability.issued_at > current:
            raise CapabilityRejected("capability is expired or not yet valid")
        if capability.intent_digest != intent.intent_digest:
            raise CapabilityRejected("capability is not bound to this exact intent")
        try:
            signature = base64.urlsafe_b64decode(capability.signature.encode("ascii"))
            self._public_key.verify(signature, capability_payload(capability))
        except (InvalidSignature, ValueError) as exc:
            raise CapabilityRejected("invalid capability signature") from exc

    def validate_and_consume(
        self,
        capability: Capability,
        intent: ActionIntent,
        *,
        executor_id: str,
        now: datetime | None = None,
    ) -> None:
        self.validate(capability, intent, executor_id=executor_id, now=now)
        if not self._nonce_store.consume(capability.nonce, expires_at=capability.expires_at):
            raise CapabilityRejected("capability nonce was already consumed")
