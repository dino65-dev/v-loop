"""Immutable, signed handoffs between graph-runtime nodes.

The runtime never needs to trust conversational state to advance a workflow:
each handoff names its producing node instance, exact inputs, workspace
generation, policy authority and validator outcomes.  The typed payloads are
small enough to use as cache keys while the referenced artifacts remain in
deployment-owned storage.
"""

from __future__ import annotations

import base64
from dataclasses import asdict, dataclass, replace
from enum import StrEnum
from typing import Mapping

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey

from .canonical import canonical_json, digest


class ArtifactType(StrEnum):
    PLAN = "plan"
    PREPARED_OPERATION = "prepared-operation"
    WORKSPACE_TRANSITION = "workspace-transition"
    EXECUTION_RESULT = "execution-result"
    EVALUATION_RECEIPT = "evaluation-receipt"
    COMPLETION_PROOF = "completion-proof"
    MEMORY_CLAIM = "memory-claim"


def _sha256(value: str, *, label: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{label} must be a SHA-256 hex digest")


@dataclass(frozen=True, slots=True)
class ProofCarryingArtifact:
    artifact_type: ArtifactType
    schema_digest: str
    producer_node_instance_id: str
    graph_digest: str
    contract_digest: str
    workspace_generation: int
    authority_class: str
    input_artifact_digests: Mapping[str, str]
    validator_results: Mapping[str, str]
    payload_digest: str
    signer_id: str
    signature: str = ""

    def __post_init__(self) -> None:
        if not self.authority_class.strip() or not self.signer_id.strip() or self.workspace_generation < 0:
            raise ValueError("proof-carrying artifacts need authority, signer, and non-negative generation")
        for name, value in (
            ("schema", self.schema_digest),
            ("producer node instance", self.producer_node_instance_id),
            ("graph", self.graph_digest),
            ("contract", self.contract_digest),
            ("payload", self.payload_digest),
        ):
            _sha256(value, label=name)
        if any(not name.strip() for name in self.input_artifact_digests) or any(
            _sha256(value, label="input artifact") for value in self.input_artifact_digests.values()
        ):
            raise ValueError("proof-carrying artifact inputs are invalid")
        if not self.validator_results or any(not name.strip() or value not in {"pass", "fail", "inconclusive"} for name, value in self.validator_results.items()):
            raise ValueError("proof-carrying artifacts require closed validator results")

    @property
    def artifact_digest(self) -> str:
        return digest({**asdict(self), "signature": ""})

    def payload(self) -> bytes:
        return canonical_json({**asdict(self), "signature": ""}).encode("utf-8")


@dataclass(frozen=True, slots=True)
class WorkspaceTransition:
    """A strictly ordered workspace state edge, not a mutable workspace label."""

    artifact: ProofCarryingArtifact
    parent_snapshot_digest: str
    output_snapshot_digest: str
    operation_id: str
    changed_paths: tuple[str, ...]
    artifact_manifest_digest: str
    supervisor_receipt_digest: str

    def __post_init__(self) -> None:
        if self.artifact.artifact_type is not ArtifactType.WORKSPACE_TRANSITION:
            raise ValueError("workspace transitions require a workspace-transition artifact")
        if not self.operation_id.strip() or not self.changed_paths:
            raise ValueError("workspace transitions need an operation and changed paths")
        for path in self.changed_paths:
            if not path.startswith("/") or "/../" in f"/{path.strip('/')}/" or path.endswith("/.."):
                raise ValueError("workspace transition paths must be normalized absolute paths")
        for label, value in (
            ("parent snapshot", self.parent_snapshot_digest),
            ("output snapshot", self.output_snapshot_digest),
            ("artifact manifest", self.artifact_manifest_digest),
            ("supervisor receipt", self.supervisor_receipt_digest),
        ):
            _sha256(value, label=label)
        if self.parent_snapshot_digest == self.output_snapshot_digest:
            raise ValueError("workspace transition must advance to a new snapshot")
        if self.artifact.input_artifact_digests.get("parent_snapshot") != self.parent_snapshot_digest:
            raise ValueError("workspace transition does not bind its parent snapshot input")
        expected_payload = digest(
            {
                "parent_snapshot_digest": self.parent_snapshot_digest,
                "output_snapshot_digest": self.output_snapshot_digest,
                "operation_id": self.operation_id,
                "changed_paths": self.changed_paths,
                "artifact_manifest_digest": self.artifact_manifest_digest,
                "supervisor_receipt_digest": self.supervisor_receipt_digest,
            }
        )
        if self.artifact.payload_digest != expected_payload:
            raise ValueError("workspace transition fields are not covered by the signed artifact payload")


class ArtifactSigner:
    """Deployment-owned signing boundary for proof-carrying artifacts."""

    def __init__(self, private_key: bytes | Ed25519PrivateKey | None = None, *, signer_id: str) -> None:
        if not signer_id.strip():
            raise ValueError("artifact signer needs an identity")
        self.signer_id = signer_id
        self._key = Ed25519PrivateKey.generate() if private_key is None else (
            Ed25519PrivateKey.from_private_bytes(private_key) if isinstance(private_key, bytes) else private_key
        )

    @property
    def public_key_bytes(self) -> bytes:
        return self._key.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)

    def sign(self, artifact: ProofCarryingArtifact) -> ProofCarryingArtifact:
        if artifact.signer_id != self.signer_id:
            raise ValueError("artifact signer differs from artifact identity")
        return replace(artifact, signature=base64.urlsafe_b64encode(self._key.sign(artifact.payload())).decode("ascii"))


class ArtifactVerifier:
    def __init__(self, public_keys: Mapping[str, bytes | Ed25519PublicKey]) -> None:
        self._keys = {
            name: Ed25519PublicKey.from_public_bytes(value) if isinstance(value, bytes) else value
            for name, value in public_keys.items()
        }

    def validate(self, artifact: ProofCarryingArtifact) -> None:
        key = self._keys.get(artifact.signer_id)
        if key is None:
            raise PermissionError("artifact signer is not trusted")
        try:
            key.verify(base64.urlsafe_b64decode(artifact.signature.encode("ascii")), artifact.payload())
        except (InvalidSignature, ValueError) as exc:
            raise PermissionError("proof-carrying artifact signature is invalid") from exc
