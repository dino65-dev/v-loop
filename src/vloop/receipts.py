"""Signed receipts emitted by protected evaluators and VM supervisors.

The executor and guest are untrusted.  A receipt is accepted only after a
public-key verification and exact binding to the run, intent, contract, whole
artifact manifest, primary artifact, and an evaluator policy owned by the
deployment.  Schema v1 is retained only for explicitly configured development
paths; production policy requires schema v2.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime, timedelta
from typing import Any, Mapping
from uuid import uuid4

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey

from .canonical import canonical_json, digest


class ReceiptRejected(PermissionError):
    """A verifier rejected an unauthenticated or incorrectly bound receipt."""


def artifact_manifest_digest(artifact_digests: Mapping[str, str]) -> str:
    """Hash the complete named artifact set, not a selected executor value."""

    if not artifact_digests or any(
        not isinstance(name, str) or not name or not isinstance(value, str) or not value
        for name, value in artifact_digests.items()
    ):
        raise ValueError("artifact manifest must contain non-empty string names and digests")
    return digest(dict(artifact_digests))


@dataclass(frozen=True, slots=True)
class ReceiptPolicy:
    """Immutable allowlist for a protected evaluator receipt type.

    The deployment pins signer key identifiers, evaluator/test artifacts, and
    a minimum receipt schema.  Key rotation is represented by the allowed-key
    set while revocation advances the minimum epoch without trusting receipt
    metadata alone.
    """

    receipt_type: str
    allowed_key_ids: frozenset[str]
    allowed_evaluator_images: frozenset[str]
    allowed_test_suites: frozenset[str]
    minimum_schema_version: int = 2
    minimum_revocation_epoch: int = 0
    required_contract_digest: str | None = None
    workspace_snapshot_schema: str | None = None
    workspace_exclusion_policy_digests: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        if not self.receipt_type.strip():
            raise ValueError("receipt policy needs a receipt type")
        if self.minimum_schema_version < 2:
            raise ValueError("production receipt policies require schema v2 or newer")
        if self.minimum_revocation_epoch < 0:
            raise ValueError("revocation epoch cannot be negative")
        if self.workspace_snapshot_schema is not None and not self.workspace_snapshot_schema.strip():
            raise ValueError("workspace snapshot schema cannot be blank")
        if any(not value.strip() for value in self.workspace_exclusion_policy_digests):
            raise ValueError("workspace exclusion-policy digests cannot be blank")
        for values, label in (
            (self.allowed_key_ids, "key ids"),
            (self.allowed_evaluator_images, "evaluator images"),
            (self.allowed_test_suites, "test suites"),
        ):
            if not values or any(not value.strip() for value in values):
                raise ValueError(f"receipt policy needs non-empty allowed {label}")

    @property
    def policy_digest(self) -> str:
        return digest(
            {
                "receipt_type": self.receipt_type,
                "allowed_key_ids": sorted(self.allowed_key_ids),
                "allowed_evaluator_images": sorted(self.allowed_evaluator_images),
                "allowed_test_suites": sorted(self.allowed_test_suites),
                "minimum_schema_version": self.minimum_schema_version,
                "minimum_revocation_epoch": self.minimum_revocation_epoch,
                "required_contract_digest": self.required_contract_digest,
                "workspace_snapshot_schema": self.workspace_snapshot_schema,
                "workspace_exclusion_policy_digests": sorted(self.workspace_exclusion_policy_digests),
            }
        )


@dataclass(frozen=True, slots=True)
class ReceiptKeyTrustEntry:
    """Verifier-owned lifecycle and scope for one evaluator signing key."""

    key_id: str
    valid_from: datetime
    valid_until: datetime
    receipt_types: frozenset[str]
    evaluator_images: frozenset[str]
    revoked_at: datetime | None = None

    def __post_init__(self) -> None:
        if not self.key_id.strip() or not self.receipt_types or not self.evaluator_images:
            raise ValueError("receipt key trust needs key id, receipt types, and evaluator images")
        if self.valid_from.tzinfo is None or self.valid_until.tzinfo is None or self.valid_until <= self.valid_from:
            raise ValueError("receipt key trust interval is invalid")
        if self.revoked_at is not None and self.revoked_at.tzinfo is None:
            raise ValueError("receipt revocation time must be timezone-aware")


@dataclass(frozen=True, slots=True)
class EvaluationReceipt:
    receipt_type: str
    run_id: str
    intent_digest: str
    candidate_artifact_digest: str
    evaluator_image_digest: str
    test_suite_digest: str
    result: str
    nonce: str
    issued_at: datetime
    expires_at: datetime
    claims: Mapping[str, Any] = field(default_factory=dict)
    schema_version: int = 2
    key_id: str = "default"
    contract_digest: str = ""
    artifact_manifest_digest: str = ""
    primary_artifact_name: str = ""
    primary_artifact_digest: str = ""
    workspace_snapshot_digest: str = ""
    dependency_lock_digest: str = ""
    toolchain_digest: str = ""
    environment_digest: str = ""
    verifier_policy_digest: str = ""
    revocation_epoch: int = 0
    signature: str = ""

    def __post_init__(self) -> None:
        if self.schema_version not in {1, 2}:
            raise ValueError("unsupported receipt schema")
        if self.result not in {"pass", "fail", "inconclusive"}:
            raise ValueError("invalid receipt result")
        for value in (
            self.receipt_type,
            self.run_id,
            self.intent_digest,
            self.candidate_artifact_digest,
            self.evaluator_image_digest,
            self.test_suite_digest,
            self.nonce,
            self.key_id,
        ):
            if not value.strip():
                raise ValueError("receipt has a required blank field")
        if self.issued_at.tzinfo is None or self.expires_at.tzinfo is None:
            raise ValueError("receipt timestamps must be timezone-aware")
        if self.schema_version >= 2:
            for value in (
                self.contract_digest,
                self.artifact_manifest_digest,
                self.primary_artifact_name,
                self.primary_artifact_digest,
                self.workspace_snapshot_digest,
                self.dependency_lock_digest,
                self.toolchain_digest,
                self.environment_digest,
                self.verifier_policy_digest,
            ):
                if not value.strip():
                    raise ValueError("schema v2 receipt has a required blank binding")
            if self.primary_artifact_digest != self.candidate_artifact_digest:
                raise ValueError("candidate digest must be the primary artifact digest")
        if self.revocation_epoch < 0:
            raise ValueError("revocation epoch cannot be negative")

    def payload(self) -> bytes:
        return canonical_json(
            {
                "schema_version": self.schema_version,
                "receipt_type": self.receipt_type,
                "run_id": self.run_id,
                "intent_digest": self.intent_digest,
                "candidate_artifact_digest": self.candidate_artifact_digest,
                "evaluator_image_digest": self.evaluator_image_digest,
                "test_suite_digest": self.test_suite_digest,
                "result": self.result,
                "nonce": self.nonce,
                "issued_at": self.issued_at.isoformat(),
                "expires_at": self.expires_at.isoformat(),
                "claims": dict(self.claims),
                "key_id": self.key_id,
                "contract_digest": self.contract_digest,
                "artifact_manifest_digest": self.artifact_manifest_digest,
                "primary_artifact_name": self.primary_artifact_name,
                "primary_artifact_digest": self.primary_artifact_digest,
                "workspace_snapshot_digest": self.workspace_snapshot_digest,
                "dependency_lock_digest": self.dependency_lock_digest,
                "toolchain_digest": self.toolchain_digest,
                "environment_digest": self.environment_digest,
                "verifier_policy_digest": self.verifier_policy_digest,
                "revocation_epoch": self.revocation_epoch,
            }
        ).encode("utf-8")

    def as_mapping(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "receipt_type": self.receipt_type,
            "run_id": self.run_id,
            "intent_digest": self.intent_digest,
            "candidate_artifact_digest": self.candidate_artifact_digest,
            "evaluator_image_digest": self.evaluator_image_digest,
            "test_suite_digest": self.test_suite_digest,
            "result": self.result,
            "nonce": self.nonce,
            "issued_at": self.issued_at.isoformat(),
            "expires_at": self.expires_at.isoformat(),
            "claims": dict(self.claims),
            "key_id": self.key_id,
            "contract_digest": self.contract_digest,
            "artifact_manifest_digest": self.artifact_manifest_digest,
            "primary_artifact_name": self.primary_artifact_name,
            "primary_artifact_digest": self.primary_artifact_digest,
            "workspace_snapshot_digest": self.workspace_snapshot_digest,
            "dependency_lock_digest": self.dependency_lock_digest,
            "toolchain_digest": self.toolchain_digest,
            "environment_digest": self.environment_digest,
            "verifier_policy_digest": self.verifier_policy_digest,
            "revocation_epoch": self.revocation_epoch,
            "signature": self.signature,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "EvaluationReceipt":
        return cls(
            receipt_type=str(value["receipt_type"]),
            run_id=str(value["run_id"]),
            intent_digest=str(value["intent_digest"]),
            candidate_artifact_digest=str(value["candidate_artifact_digest"]),
            evaluator_image_digest=str(value["evaluator_image_digest"]),
            test_suite_digest=str(value["test_suite_digest"]),
            result=str(value["result"]),
            nonce=str(value["nonce"]),
            issued_at=datetime.fromisoformat(str(value["issued_at"])),
            expires_at=datetime.fromisoformat(str(value["expires_at"])),
            claims=dict(value.get("claims", {})),
            schema_version=int(value.get("schema_version", 1)),
            key_id=str(value.get("key_id", "default")),
            contract_digest=str(value.get("contract_digest", "")),
            artifact_manifest_digest=str(value.get("artifact_manifest_digest", "")),
            primary_artifact_name=str(value.get("primary_artifact_name", "")),
            primary_artifact_digest=str(value.get("primary_artifact_digest", "")),
            workspace_snapshot_digest=str(value.get("workspace_snapshot_digest", "")),
            dependency_lock_digest=str(value.get("dependency_lock_digest", "")),
            toolchain_digest=str(value.get("toolchain_digest", "")),
            environment_digest=str(value.get("environment_digest", "")),
            verifier_policy_digest=str(value.get("verifier_policy_digest", "")),
            revocation_epoch=int(value.get("revocation_epoch", 0)),
            signature=str(value["signature"]),
        )


class ReceiptSigner:
    """Private-key component that belongs only to a protected evaluator."""

    def __init__(
        self,
        private_key: bytes | Ed25519PrivateKey | None = None,
        *,
        key_id: str = "default",
    ) -> None:
        if not key_id.strip():
            raise ValueError("receipt signer needs a key id")
        self._key_id = key_id
        self._key = (
            Ed25519PrivateKey.generate()
            if private_key is None
            else Ed25519PrivateKey.from_private_bytes(private_key)
            if isinstance(private_key, bytes)
            else private_key
        )

    @property
    def key_id(self) -> str:
        return self._key_id

    @property
    def public_key_bytes(self) -> bytes:
        return self._key.public_key().public_bytes(
            serialization.Encoding.Raw, serialization.PublicFormat.Raw
        )

    def issue(
        self,
        *,
        receipt_type: str,
        run_id: str,
        intent_digest: str,
        candidate_artifact_digest: str,
        evaluator_image_digest: str,
        test_suite_digest: str,
        result: str,
        contract_digest: str = "",
        artifact_digests: Mapping[str, str] | None = None,
        primary_artifact_name: str = "",
        workspace_snapshot_digest: str = "",
        dependency_lock_digest: str = "",
        toolchain_digest: str = "",
        environment_digest: str = "",
        verifier_policy_digest: str = "",
        workspace_snapshot: Any | None = None,
        revocation_epoch: int = 0,
        claims: Mapping[str, Any] | None = None,
        ttl: timedelta = timedelta(minutes=10),
        now: datetime | None = None,
        schema_version: int = 2,
    ) -> EvaluationReceipt:
        issued_at = now or datetime.now(UTC)
        claims_payload = dict(claims or {})
        if workspace_snapshot is not None:
            required_attributes = (
                "schema_version",
                "workspace_snapshot_digest",
                "dependency_lock_digest",
                "toolchain_digest",
                "environment_digest",
            )
            if any(not isinstance(getattr(workspace_snapshot, name, None), str) or not getattr(workspace_snapshot, name) for name in required_attributes):
                raise ValueError("workspace snapshot is not canonical snapshot data")
            if workspace_snapshot_digest and workspace_snapshot_digest != workspace_snapshot.workspace_snapshot_digest:
                raise ValueError("workspace snapshot digest conflicts with canonical snapshot")
            workspace_snapshot_digest = workspace_snapshot.workspace_snapshot_digest
            dependency_lock_digest = workspace_snapshot.dependency_lock_digest
            toolchain_digest = workspace_snapshot.toolchain_digest
            environment_digest = workspace_snapshot.environment_digest
            claims_payload["workspace_snapshot_schema"] = workspace_snapshot.schema_version
            claims_payload["workspace_snapshot_exclusion_policy_digest"] = workspace_snapshot.exclusion_policy_digest
        manifest_digest = ""
        primary_digest = ""
        if schema_version >= 2:
            if artifact_digests is None or not primary_artifact_name:
                raise ValueError("schema v2 issuance needs artifact_digests and a primary artifact name")
            manifest_digest = artifact_manifest_digest(artifact_digests)
            try:
                primary_digest = artifact_digests[primary_artifact_name]
            except KeyError as exc:
                raise ValueError("primary artifact is absent from the artifact manifest") from exc
            if primary_digest != candidate_artifact_digest:
                raise ValueError("candidate digest must match the declared primary artifact")
        unsigned = EvaluationReceipt(
            receipt_type=receipt_type,
            run_id=run_id,
            intent_digest=intent_digest,
            candidate_artifact_digest=candidate_artifact_digest,
            evaluator_image_digest=evaluator_image_digest,
            test_suite_digest=test_suite_digest,
            result=result,
            nonce=str(uuid4()),
            issued_at=issued_at,
            expires_at=issued_at + ttl,
            claims=claims_payload,
            schema_version=schema_version,
            key_id=self._key_id,
            contract_digest=contract_digest,
            artifact_manifest_digest=manifest_digest,
            primary_artifact_name=primary_artifact_name,
            primary_artifact_digest=primary_digest,
            workspace_snapshot_digest=workspace_snapshot_digest,
            dependency_lock_digest=dependency_lock_digest,
            toolchain_digest=toolchain_digest,
            environment_digest=environment_digest,
            verifier_policy_digest=verifier_policy_digest,
            revocation_epoch=revocation_epoch,
        )
        return replace(
            unsigned,
            signature=base64.urlsafe_b64encode(self._key.sign(unsigned.payload())).decode("ascii"),
        )


class ReceiptVerifier:
    """Verifier-side public-key check with policy and authoritative-state binding."""

    def __init__(
        self,
        public_key: bytes | Ed25519PublicKey | Mapping[str, bytes | Ed25519PublicKey],
        *,
        policy: ReceiptPolicy | None = None,
        key_id: str = "default",
        trust_entries: Mapping[str, ReceiptKeyTrustEntry] | None = None,
        maximum_ttl: timedelta = timedelta(minutes=15),
        maximum_age: timedelta = timedelta(minutes=15),
        clock_skew: timedelta = timedelta(seconds=30),
    ) -> None:
        if isinstance(public_key, Mapping):
            self._keys = {
                name: Ed25519PublicKey.from_public_bytes(value) if isinstance(value, bytes) else value
                for name, value in public_key.items()
            }
        else:
            self._keys = {
                key_id: Ed25519PublicKey.from_public_bytes(public_key) if isinstance(public_key, bytes) else public_key
            }
        if not self._keys or any(not name.strip() for name in self._keys):
            raise ValueError("receipt verifier needs keyed public verification material")
        self.policy = policy
        self._trust_entries = dict(trust_entries or {})
        if self._trust_entries and set(self._trust_entries) != set(self._keys):
            raise ValueError("receipt trust entries must cover exactly the configured keys")
        self._maximum_ttl = maximum_ttl
        self._maximum_age = maximum_age
        self._clock_skew = clock_skew
        if policy is not None and policy.allowed_key_ids.difference(self._keys):
            raise ValueError("receipt policy names keys unavailable to this verifier")

    @property
    def has_verifier_owned_trust(self) -> bool:
        return bool(self._trust_entries)

    def validate(
        self,
        receipt: EvaluationReceipt,
        *,
        receipt_type: str,
        run_id: str,
        intent_digest: str,
        artifact_digests: Mapping[str, str],
        contract_digest: str | None = None,
        now: datetime | None = None,
    ) -> None:
        current = now or datetime.now(UTC)
        if receipt.receipt_type != receipt_type:
            raise ReceiptRejected("receipt type does not match verifier")
        if receipt.run_id != run_id or receipt.intent_digest != intent_digest:
            raise ReceiptRejected("receipt is not bound to this run and intent")
        if (
            receipt.expires_at <= current
            or receipt.issued_at > current + self._clock_skew
            or current - receipt.issued_at > self._maximum_age
            or receipt.expires_at - receipt.issued_at > self._maximum_ttl
        ):
            raise ReceiptRejected("receipt is expired or not yet valid")
        key = self._keys.get(receipt.key_id)
        if key is None:
            raise ReceiptRejected("receipt key is not trusted")
        try:
            signature = base64.urlsafe_b64decode(receipt.signature.encode("ascii"))
            key.verify(signature, receipt.payload())
        except (InvalidSignature, ValueError) as exc:
            raise ReceiptRejected("invalid receipt signature") from exc

        trust = self._trust_entries.get(receipt.key_id)
        if trust is not None:
            if current < trust.valid_from or current >= trust.valid_until or (
                trust.revoked_at is not None and current >= trust.revoked_at
            ):
                raise ReceiptRejected("receipt signing key is not currently trusted")
            if receipt.receipt_type not in trust.receipt_types:
                raise ReceiptRejected("receipt type is not authorized for this signing key")
            if receipt.evaluator_image_digest not in trust.evaluator_images:
                raise ReceiptRejected("evaluator identity is not authorized for this signing key")

        if self.policy is not None:
            if receipt.receipt_type != self.policy.receipt_type:
                raise ReceiptRejected("receipt does not match its evaluator policy")
            if receipt.schema_version < self.policy.minimum_schema_version:
                raise ReceiptRejected("receipt schema is below the policy minimum")
            if receipt.key_id not in self.policy.allowed_key_ids:
                raise ReceiptRejected("receipt signer is not allowed by policy")
            if receipt.evaluator_image_digest not in self.policy.allowed_evaluator_images:
                raise ReceiptRejected("evaluator image is not allowed by policy")
            if receipt.test_suite_digest not in self.policy.allowed_test_suites:
                raise ReceiptRejected("test suite is not allowed by policy")
            # ``revocation_epoch`` is retained for audit compatibility only.
            # Revocation is determined from the verifier-owned trust entry;
            # a compromised signer can otherwise mint any epoch it chooses.
            if receipt.verifier_policy_digest != self.policy.policy_digest:
                raise ReceiptRejected("receipt is bound to a different evaluator policy")
            if (
                self.policy.workspace_snapshot_schema is not None
                and receipt.claims.get("workspace_snapshot_schema") != self.policy.workspace_snapshot_schema
            ):
                raise ReceiptRejected("receipt was not generated by the required canonical snapshot service")
            if (
                self.policy.workspace_exclusion_policy_digests
                and receipt.claims.get("workspace_snapshot_exclusion_policy_digest")
                not in self.policy.workspace_exclusion_policy_digests
            ):
                raise ReceiptRejected("receipt used an unapproved workspace exclusion policy")

        expected_contract = contract_digest or (self.policy.required_contract_digest if self.policy else None)
        if expected_contract is not None and receipt.contract_digest != expected_contract:
            raise ReceiptRejected("receipt is not bound to this contract")
        if self.policy and self.policy.required_contract_digest and receipt.contract_digest != self.policy.required_contract_digest:
            raise ReceiptRejected("receipt is not bound to the policy contract")
        if receipt.schema_version >= 2:
            expected_manifest = artifact_manifest_digest(artifact_digests)
            if receipt.artifact_manifest_digest != expected_manifest:
                raise ReceiptRejected("receipt artifact manifest does not match executor artifacts")
            if artifact_digests.get(receipt.primary_artifact_name) != receipt.primary_artifact_digest:
                raise ReceiptRejected("receipt primary artifact does not match executor artifacts")
            if receipt.primary_artifact_digest != receipt.candidate_artifact_digest:
                raise ReceiptRejected("receipt candidate is not the authoritative primary artifact")
        elif receipt.candidate_artifact_digest not in set(artifact_digests.values()):
            raise ReceiptRejected("development receipt candidate digest is not an executor artifact")
