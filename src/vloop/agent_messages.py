"""Immutable, signed, graph-scoped messages between reasoning sessions."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Mapping

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey

from .canonical import canonical_json, digest
from .native_backend import ed25519_sign, ed25519_verify
from .reasoning_sessions import ReasoningSessionStore, SessionRejected


@dataclass(frozen=True, slots=True)
class AgentMessageArtifact:
    sender_node_instance: str
    receiver_node_instance: str
    sender_session_id: str
    receiver_session_id: str
    graph_digest: str
    contract_digest: str
    sequence_number: int
    receiver_sequence_number: int
    payload_digest: str
    provenance_root: str
    issued_at: datetime
    signer_id: str
    communication_edge_id: str
    causal_parent_event_id: str
    signature: str = ""

    def __post_init__(self) -> None:
        if not all((self.sender_node_instance, self.receiver_node_instance, self.sender_session_id, self.receiver_session_id, self.signer_id)):
            raise ValueError("agent messages need sender, receiver, and signer identity")
        if self.sequence_number < 1 or self.receiver_sequence_number < 1:
            raise ValueError("agent messages need positive monotonic sequences")
        if not self.communication_edge_id.strip() or not self.causal_parent_event_id.strip():
            raise ValueError("agent messages need an admitted communication edge and causal parent")
        if self.issued_at.tzinfo is None:
            raise ValueError("agent messages need a timezone-aware timestamp")
        for value in (self.graph_digest, self.contract_digest, self.payload_digest, self.provenance_root):
            if len(value) != 64:
                raise ValueError("agent message bindings must be SHA-256 digests")

    @property
    def artifact_digest(self) -> str:
        return digest(self._payload())

    def payload(self) -> bytes:
        return canonical_json(self._payload()).encode("utf-8")

    def _payload(self) -> dict[str, object]:
        value = asdict(self)
        value["issued_at"] = self.issued_at.isoformat()
        value["signature"] = ""
        return value


class AgentMessageSigner:
    """A deployment-owned signer; an RLM worker never receives this key."""

    def __init__(self, signer_id: str, private_key: bytes | Ed25519PrivateKey | None = None) -> None:
        if not signer_id.strip():
            raise ValueError("message signer id is required")
        self.signer_id = signer_id
        self._key = Ed25519PrivateKey.generate() if private_key is None else (
            Ed25519PrivateKey.from_private_bytes(private_key) if isinstance(private_key, bytes) else private_key
        )

    @property
    def public_key_bytes(self) -> bytes:
        return self._key.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)

    def issue(self, artifact: AgentMessageArtifact) -> AgentMessageArtifact:
        if artifact.signer_id != self.signer_id:
            raise PermissionError("message signer cannot impersonate another issuer")
        return replace(artifact, signature=ed25519_sign(self._key, artifact.payload()))


class AgentMessageVerifier:
    def __init__(self, public_keys: Mapping[str, bytes | Ed25519PublicKey]) -> None:
        self._keys = {
            key_id: Ed25519PublicKey.from_public_bytes(value) if isinstance(value, bytes) else value
            for key_id, value in public_keys.items()
        }

    def validate(self, artifact: AgentMessageArtifact) -> None:
        key = self._keys.get(artifact.signer_id)
        if key is None or not ed25519_verify(key, artifact.payload(), artifact.signature):
            raise PermissionError("agent message signature is invalid")


class AgentMessageStore:
    """Append-only transport that verifies graph and session bindings before storage."""

    def __init__(
        self, database: str | Path, *, sessions: ReasoningSessionStore, verifier: AgentMessageVerifier,
        allowed_edges: frozenset[tuple[str, str]] = frozenset(),
    ) -> None:
        self._connection = sqlite3.connect(Path(database), isolation_level=None, timeout=5.0)
        self._connection.execute("PRAGMA busy_timeout=5000")
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.execute(
            """CREATE TABLE IF NOT EXISTS agent_messages (
                artifact_digest TEXT PRIMARY KEY, sender_session_id TEXT NOT NULL,
                receiver_session_id TEXT NOT NULL, sequence_number INTEGER NOT NULL, receiver_sequence_number INTEGER NOT NULL,
                communication_edge_id TEXT NOT NULL, causal_parent_event_id TEXT NOT NULL,
                payload TEXT NOT NULL, artifact_json TEXT NOT NULL,
                UNIQUE(sender_session_id, receiver_session_id, sequence_number),
                UNIQUE(receiver_session_id, receiver_sequence_number)
            )"""
        )
        self._sessions = sessions
        self._verifier = verifier
        self._allowed_edges = allowed_edges

    def send(self, artifact: AgentMessageArtifact, payload: str) -> AgentMessageArtifact:
        if digest(payload) != artifact.payload_digest:
            raise PermissionError("message payload does not match its declared digest")
        self._verifier.validate(artifact)
        sender = self._sessions.get(artifact.sender_session_id)
        receiver = self._sessions.get(artifact.receiver_session_id)
        if (sender.run_id, sender.contract_digest, sender.root_graph_digest or sender.graph_digest) != (
            receiver.run_id, receiver.contract_digest, receiver.root_graph_digest or receiver.graph_digest,
        ):
            raise SessionRejected("cross-run, cross-contract, or cross-root-graph agent messaging is forbidden")
        if (artifact.sender_node_instance, artifact.receiver_node_instance, artifact.contract_digest, artifact.graph_digest) != (
            sender.node_instance_id, receiver.node_instance_id, sender.contract_digest,
            sender.root_graph_digest or sender.graph_digest,
        ):
            raise SessionRejected("agent message is not bound to its graph node instances")
        direct_parent_child = sender.parent_session_id == receiver.session_id or receiver.parent_session_id == sender.session_id
        explicit_edge = (sender.node_instance_id, receiver.node_instance_id) in self._allowed_edges
        if not direct_parent_child and not explicit_edge:
            raise SessionRejected("agent message is not permitted by the graph communication ACL")
        expected = self._connection.execute(
            "SELECT COALESCE(MAX(sequence_number), 0) + 1 FROM agent_messages WHERE sender_session_id = ? AND receiver_session_id = ?",
            (sender.session_id, receiver.session_id),
        ).fetchone()[0]
        if artifact.sequence_number != expected:
            raise PermissionError("agent message sequence is not monotonic")
        expected_receiver = self._connection.execute(
            "SELECT COALESCE(MAX(receiver_sequence_number), 0) + 1 FROM agent_messages WHERE receiver_session_id = ?",
            (receiver.session_id,),
        ).fetchone()[0]
        if artifact.receiver_sequence_number != expected_receiver:
            raise PermissionError("agent message receiver sequence is not monotonic")
        self._connection.execute(
            "INSERT INTO agent_messages VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                artifact.artifact_digest, sender.session_id, receiver.session_id, artifact.sequence_number,
                artifact.receiver_sequence_number, artifact.communication_edge_id, artifact.causal_parent_event_id,
                payload, json.dumps(_encode(artifact), sort_keys=True),
            ),
        )
        return artifact

    def inbox(self, receiver_session_id: str) -> tuple[tuple[AgentMessageArtifact, str], ...]:
        self._sessions.get(receiver_session_id)
        rows = self._connection.execute(
            "SELECT artifact_json, payload FROM agent_messages WHERE receiver_session_id = ? ORDER BY receiver_sequence_number", (receiver_session_id,)
        )
        return tuple((_decode(row[0]), row[1]) for row in rows)


def _encode(value: AgentMessageArtifact) -> dict[str, object]:
    result = asdict(value)
    result["issued_at"] = value.issued_at.isoformat()
    return result


def _decode(value: str) -> AgentMessageArtifact:
    data = json.loads(value)
    data["issued_at"] = datetime.fromisoformat(data["issued_at"])
    return AgentMessageArtifact(**data)
