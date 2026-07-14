"""Signed receipts emitted by protected evaluators and VM supervisors.

Guest/executor output is untrusted. A receipt is useful only when its Ed25519
signature, run/intent binding, actual artifact digest, evaluator identity, and
expiry are all checked by verifier-side code.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any, Mapping
from uuid import uuid4

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey

from .canonical import canonical_json


class ReceiptRejected(PermissionError):
    """A verifier rejected an unauthenticated or incorrectly bound receipt."""


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
    schema_version: int = 1
    signature: str = ""

    def __post_init__(self) -> None:
        if self.schema_version != 1:
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
        ):
            if not value.strip():
                raise ValueError("receipt has a required blank field")
        if self.issued_at.tzinfo is None or self.expires_at.tzinfo is None:
            raise ValueError("receipt timestamps must be timezone-aware")

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
            signature=str(value["signature"]),
        )


class ReceiptSigner:
    """Private-key component that belongs only to a protected evaluator."""

    def __init__(self, private_key: bytes | Ed25519PrivateKey | None = None) -> None:
        self._key = (
            Ed25519PrivateKey.generate()
            if private_key is None
            else Ed25519PrivateKey.from_private_bytes(private_key)
            if isinstance(private_key, bytes)
            else private_key
        )

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
        claims: Mapping[str, Any] | None = None,
        ttl: timedelta = timedelta(minutes=10),
        now: datetime | None = None,
    ) -> EvaluationReceipt:
        issued_at = now or datetime.now(UTC)
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
            claims=dict(claims or {}),
        )
        return EvaluationReceipt(
            receipt_type=unsigned.receipt_type,
            run_id=unsigned.run_id,
            intent_digest=unsigned.intent_digest,
            candidate_artifact_digest=unsigned.candidate_artifact_digest,
            evaluator_image_digest=unsigned.evaluator_image_digest,
            test_suite_digest=unsigned.test_suite_digest,
            result=unsigned.result,
            nonce=unsigned.nonce,
            issued_at=unsigned.issued_at,
            expires_at=unsigned.expires_at,
            claims=unsigned.claims,
            signature=base64.urlsafe_b64encode(self._key.sign(unsigned.payload())).decode("ascii"),
        )


class ReceiptVerifier:
    """Verifier-side public-key check with run/intent/artifact binding."""

    def __init__(self, public_key: bytes | Ed25519PublicKey) -> None:
        self._key = Ed25519PublicKey.from_public_bytes(public_key) if isinstance(public_key, bytes) else public_key

    def validate(
        self,
        receipt: EvaluationReceipt,
        *,
        receipt_type: str,
        run_id: str,
        intent_digest: str,
        artifact_digests: Mapping[str, str],
        now: datetime | None = None,
    ) -> None:
        current = now or datetime.now(UTC)
        if receipt.receipt_type != receipt_type:
            raise ReceiptRejected("receipt type does not match verifier")
        if receipt.run_id != run_id or receipt.intent_digest != intent_digest:
            raise ReceiptRejected("receipt is not bound to this run and intent")
        if receipt.expires_at <= current or receipt.issued_at > current:
            raise ReceiptRejected("receipt is expired or not yet valid")
        if receipt.candidate_artifact_digest not in set(artifact_digests.values()):
            raise ReceiptRejected("receipt candidate digest is not an executor artifact")
        try:
            signature = base64.urlsafe_b64decode(receipt.signature.encode("ascii"))
            self._key.verify(signature, receipt.payload())
        except (InvalidSignature, ValueError) as exc:
            raise ReceiptRejected("invalid receipt signature") from exc
