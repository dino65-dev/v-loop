"""Producer-owned signed attestations for graph facts.

The controller may transport an attestation to the scheduler, but it cannot
turn a boolean or a dictionary into completion of an externally owned node.
The wire format deliberately mirrors the useful subset of an in-toto/DSSE
statement: typed subject, typed predicate, payload digest, signer identity,
and an authenticated envelope.  A deployment can replace these Ed25519 test
keys with SPIFFE/SVID-backed signers without changing the graph protocol.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import Mapping, Protocol
from uuid import uuid4

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey

from .canonical import canonical_json, digest
from .native_backend import ed25519_sign, ed25519_verify


class CompletionResult(StrEnum):
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    INDETERMINATE = "indeterminate"


@dataclass(frozen=True, slots=True)
class WorkloadIdentity:
    """A workload principal, normally a SPIFFE ID in production."""

    subject: str
    roles: frozenset[str]
    valid_from: datetime
    valid_until: datetime
    revoked_at: datetime | None = None

    def __post_init__(self) -> None:
        if not self.subject.startswith("spiffe://") or not self.roles:
            raise ValueError("workload identities need a SPIFFE subject and one or more roles")
        if self.valid_from.tzinfo is None or self.valid_until.tzinfo is None or self.valid_until <= self.valid_from:
            raise ValueError("workload identity validity is invalid")
        if self.revoked_at is not None and self.revoked_at.tzinfo is None:
            raise ValueError("workload identity revocation time must be timezone-aware")

    def active(self, now: datetime) -> bool:
        return self.valid_from <= now < self.valid_until and (self.revoked_at is None or now < self.revoked_at)


@dataclass(frozen=True, slots=True)
class ValidatedNodeCompletion:
    """An authenticated transition fact emitted by a node's exclusive owner."""

    graph_digest: str
    contract_digest: str
    run_id: str
    template_node_id: str
    node_instance_id: str
    producer_identity: str
    producer_role: str
    artifact_digest: str
    validator_policy_digest: str
    result: CompletionResult
    issued_at: datetime
    expires_at: datetime
    nonce: str
    input_artifact_digests: Mapping[str, str]
    facts: Mapping[str, str] = field(default_factory=dict)
    authority_refs: tuple[str, ...] = ()
    evidence_refs: tuple[str, ...] = ()
    transparency_log_proof: str = ""
    signature: str = ""

    def __post_init__(self) -> None:
        if not all((self.run_id.strip(), self.template_node_id.strip(), self.node_instance_id, self.producer_identity.startswith("spiffe://"), self.producer_role.strip(), self.nonce.strip())):
            raise ValueError("node completions need run, node, producer, role, and nonce identities")
        for value in (
            self.graph_digest,
            self.contract_digest,
            self.node_instance_id,
            self.artifact_digest,
            self.validator_policy_digest,
            *self.input_artifact_digests.values(),
        ):
            if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
                raise ValueError("node completion digests must be SHA-256 hex")
        if self.issued_at.tzinfo is None or self.expires_at.tzinfo is None or self.expires_at <= self.issued_at:
            raise ValueError("node completion validity is invalid")
        if len(self.authority_refs) != len(set(self.authority_refs)) or len(self.evidence_refs) != len(set(self.evidence_refs)):
            raise ValueError("node completion references must be unique")

    @property
    def completion_digest(self) -> str:
        return digest(self._payload_mapping())

    def payload(self) -> bytes:
        return canonical_json(self._payload_mapping()).encode("utf-8")

    def _payload_mapping(self) -> dict:
        value = asdict(self)
        value["issued_at"] = self.issued_at.isoformat()
        value["expires_at"] = self.expires_at.isoformat()
        value["signature"] = ""
        return value


@dataclass(frozen=True, slots=True)
class NodeOwnershipPolicy:
    """Deployment-owned map from graph template to workload role."""

    graph_digest: str
    template_roles: Mapping[str, str]
    policy_digest: str = ""

    def __post_init__(self) -> None:
        if len(self.graph_digest) != 64 or not self.template_roles or any(
            not node.strip() or not role.strip() for node, role in self.template_roles.items()
        ):
            raise ValueError("node ownership policies need graph-bound non-empty role assignments")
        expected = digest({"graph_digest": self.graph_digest, "template_roles": dict(self.template_roles)})
        if self.policy_digest and self.policy_digest != expected:
            raise ValueError("node ownership policy digest does not match its assignments")
        object.__setattr__(self, "policy_digest", expected)

    def required_role(self, template_node_id: str) -> str:
        try:
            return self.template_roles[template_node_id]
        except KeyError as exc:
            raise PermissionError("node has no authorised producer role") from exc


class CompletionSigner:
    """Private signing endpoint owned by one workload, never by the scheduler."""

    def __init__(self, identity: WorkloadIdentity, private_key: bytes | Ed25519PrivateKey | None = None) -> None:
        self.identity = identity
        self._key = Ed25519PrivateKey.generate() if private_key is None else (
            Ed25519PrivateKey.from_private_bytes(private_key) if isinstance(private_key, bytes) else private_key
        )

    @property
    def public_key_bytes(self) -> bytes:
        return self._key.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)

    def issue(
        self,
        *,
        graph_digest: str,
        contract_digest: str,
        run_id: str,
        template_node_id: str,
        node_instance_id: str,
        producer_role: str,
        artifact_digest: str,
        validator_policy_digest: str,
        result: CompletionResult = CompletionResult.SUCCEEDED,
        input_artifact_digests: Mapping[str, str] = {},
        facts: Mapping[str, str] = {},
        authority_refs: tuple[str, ...] = (),
        evidence_refs: tuple[str, ...] = (),
        transparency_log_proof: str = "",
        now: datetime | None = None,
        ttl: timedelta = timedelta(minutes=10),
    ) -> ValidatedNodeCompletion:
        issued = now or datetime.now(UTC)
        unsigned = ValidatedNodeCompletion(
            graph_digest=graph_digest, contract_digest=contract_digest, run_id=run_id,
            template_node_id=template_node_id, node_instance_id=node_instance_id,
            producer_identity=self.identity.subject, producer_role=producer_role,
            artifact_digest=artifact_digest, validator_policy_digest=validator_policy_digest,
            result=result, issued_at=issued, expires_at=issued + ttl, nonce=str(uuid4()),
            input_artifact_digests=dict(input_artifact_digests), facts={key: str(value) for key, value in facts.items()},
            authority_refs=authority_refs, evidence_refs=evidence_refs, transparency_log_proof=transparency_log_proof,
        )
        return replace(unsigned, signature=ed25519_sign(self._key, unsigned.payload()))


class CompletionVerifier:
    """Verifies workload identity, role ownership, expiry and DSSE-like signature."""

    def __init__(
        self,
        identities: Mapping[str, WorkloadIdentity],
        public_keys: Mapping[str, bytes | Ed25519PublicKey],
        ownership: NodeOwnershipPolicy,
        *,
        maximum_ttl: timedelta = timedelta(minutes=15),
        clock_skew: timedelta = timedelta(seconds=30),
    ) -> None:
        if set(identities) != set(public_keys):
            raise ValueError("completion trust identities and keys must match exactly")
        self.identities = dict(identities)
        self.keys = {
            subject: Ed25519PublicKey.from_public_bytes(key) if isinstance(key, bytes) else key
            for subject, key in public_keys.items()
        }
        self.ownership = ownership
        self.maximum_ttl = maximum_ttl
        self.clock_skew = clock_skew

    def validate(
        self,
        completion: ValidatedNodeCompletion,
        *,
        expected_graph_digest: str,
        expected_contract_digest: str,
        expected_run_id: str,
        expected_template_node_id: str,
        expected_node_instance_id: str,
        now: datetime | None = None,
    ) -> None:
        current = now or datetime.now(UTC)
        if (
            completion.graph_digest != expected_graph_digest
            or completion.contract_digest != expected_contract_digest
            or completion.run_id != expected_run_id
            or completion.template_node_id != expected_template_node_id
            or completion.node_instance_id != expected_node_instance_id
        ):
            raise PermissionError("node completion is bound to another graph, contract, run, or instance")
        identity = self.identities.get(completion.producer_identity)
        key = self.keys.get(completion.producer_identity)
        if identity is None or key is None or not identity.active(current):
            raise PermissionError("node completion producer workload is not trusted and active")
        if completion.producer_role not in identity.roles or completion.producer_role != self.ownership.required_role(expected_template_node_id):
            raise PermissionError("node completion producer does not own this graph node")
        if (
            completion.expires_at <= current
            or completion.issued_at > current + self.clock_skew
            or completion.expires_at - completion.issued_at > self.maximum_ttl
        ):
            raise PermissionError("node completion is expired or has invalid lifetime")
        if not ed25519_verify(key, completion.payload(), completion.signature):
            raise PermissionError("node completion signature is invalid")


class CompletionClient(Protocol):
    """Remote service API used by controllers; scheduler receives only proofs."""

    def complete(self, **kwargs: object) -> ValidatedNodeCompletion: ...


class DevelopmentCompletionFabric:
    """Explicit local-development stand-in for independently deployed services.

    It is intentionally constructed outside ``DurableGraphScheduler``.  The
    controller receives only its ``complete`` client surface; production
    deployments replace it with mTLS/SPIFFE service clients.
    """

    def __init__(self, *, graph_digest: str, template_roles: Mapping[str, str], now: datetime | None = None) -> None:
        issued = now or datetime.now(UTC)
        roles = frozenset(template_roles.values())
        self.ownership = NodeOwnershipPolicy(graph_digest, dict(template_roles))
        self._signers: dict[str, CompletionSigner] = {}
        identities: dict[str, WorkloadIdentity] = {}
        keys: dict[str, bytes] = {}
        for role in roles:
            subject = f"spiffe://vloop.local/{role}"
            identity = WorkloadIdentity(subject, frozenset({role}), issued - timedelta(minutes=1), issued + timedelta(days=1))
            signer = CompletionSigner(identity)
            self._signers[role] = signer
            identities[subject] = identity
            keys[subject] = signer.public_key_bytes
        self.verifier = CompletionVerifier(identities, keys, self.ownership)

    def complete(self, **kwargs: object) -> ValidatedNodeCompletion:
        template_node_id = str(kwargs["template_node_id"])
        role = self.ownership.required_role(template_node_id)
        return self._signers[role].issue(producer_role=role, **kwargs)  # type: ignore[arg-type]
