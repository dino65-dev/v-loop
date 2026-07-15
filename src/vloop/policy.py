"""Deterministic, default-deny action authorization."""

from __future__ import annotations

import base64
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
import re
import sqlite3
from typing import Any, Iterable, Mapping
from uuid import uuid4
from pathlib import Path
import threading

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey

from .authorization import CapabilitySigner, CapabilityVerifier, InMemoryNonceStore
from .canonical import canonical_json
from .models import ActionIntent, Capability, Effect, Provenance, TaskContract


class PolicyDenied(PermissionError):
    """The action is outside the currently authorized task envelope."""


_SENSITIVE_ARGUMENT_KEY = re.compile(
    r"(?:api.?key|token|password|secret|credential|authorization|cookie)", re.I
)
_TOKEN_LIKE_VALUE = re.compile(r"(?:bearer\s+|(?:sk|rk|ghp)[_-])[A-Za-z0-9._-]{6,}", re.I)


@dataclass(frozen=True, slots=True)
class Approval:
    """Development-only in-process approval; production requires a receipt."""

    intent_digest: str
    approver: str
    approved_at: datetime


class ApprovalRejected(PermissionError):
    """An approval receipt is unsigned, expired, or scoped to another action."""


@dataclass(frozen=True, slots=True)
class ApprovalTrustEntry:
    """Verifier-owned binding from one signing key to one human principal."""

    key_id: str
    subject: str
    roles: frozenset[str]
    valid_from: datetime
    valid_until: datetime
    revoked_at: datetime | None = None

    def __post_init__(self) -> None:
        if not self.key_id.strip() or not self.subject.strip() or not self.roles:
            raise ValueError("approval trust entry needs a key, subject, and roles")
        if self.valid_from.tzinfo is None or self.valid_until.tzinfo is None:
            raise ValueError("approval trust entry times must be timezone-aware")
        if self.valid_until <= self.valid_from:
            raise ValueError("approval trust interval is invalid")
        if self.revoked_at is not None and self.revoked_at.tzinfo is None:
            raise ValueError("approval revocation time must be timezone-aware")


class ApprovalConsumptionStore:
    def consume(self, approval_id: str, *, intent_digest: str, executor_id: str) -> bool: ...


class InMemoryApprovalConsumptionStore:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._consumed: set[tuple[str, str, str]] = set()

    def consume(self, approval_id: str, *, intent_digest: str, executor_id: str) -> bool:
        key = (approval_id, intent_digest, executor_id)
        with self._lock:
            if key in self._consumed:
                return False
            self._consumed.add(key)
            return True


class SQLiteApprovalConsumptionStore:
    """Durable one-time approval consumption at the policy decision point."""

    def __init__(self, database: str | Path) -> None:
        path = Path(database)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(path, isolation_level=None, timeout=5.0)
        self._connection.execute("PRAGMA busy_timeout=5000")
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS consumed_approvals (
                approval_id TEXT NOT NULL,
                intent_digest TEXT NOT NULL,
                executor_id TEXT NOT NULL,
                consumed_at TEXT NOT NULL,
                PRIMARY KEY (approval_id, intent_digest, executor_id)
            )
            """
        )

    def consume(self, approval_id: str, *, intent_digest: str, executor_id: str) -> bool:
        try:
            self._connection.execute(
                "INSERT INTO consumed_approvals VALUES (?, ?, ?, ?)",
                (approval_id, intent_digest, executor_id, datetime.now(UTC).isoformat()),
            )
        except sqlite3.IntegrityError:
            return False
        return True

    def close(self) -> None:
        self._connection.close()


@dataclass(frozen=True, slots=True)
class SignedApprovalReceipt:
    key_id: str
    approval_id: str
    approver: str
    scope: str
    intent_digest: str
    contract_digest: str
    executor_id: str
    issued_at: datetime
    expires_at: datetime
    signature: str

    def __post_init__(self) -> None:
        if not all(
            value.strip()
            for value in (
                self.key_id,
                self.approval_id,
                self.approver,
                self.scope,
                self.intent_digest,
                self.contract_digest,
                self.executor_id,
            )
        ):
            raise ValueError("approval receipt has a required blank field")
        if self.scope != "action.execute":
            raise ValueError("approval receipt has an unsupported scope")
        if self.issued_at.tzinfo is None or self.expires_at.tzinfo is None:
            raise ValueError("approval receipt timestamps must be timezone-aware")

    def payload(self) -> bytes:
        return canonical_json(
            {
                "key_id": self.key_id,
                "approval_id": self.approval_id,
                "approver": self.approver,
                "scope": self.scope,
                "intent_digest": self.intent_digest,
                "contract_digest": self.contract_digest,
                "executor_id": self.executor_id,
                "issued_at": self.issued_at.isoformat(),
                "expires_at": self.expires_at.isoformat(),
            }
        ).encode("utf-8")


class ApprovalSigner:
    """Private-key approver service; planners/controllers never receive this key."""

    def __init__(self, private_key: bytes | Ed25519PrivateKey | None = None, *, key_id: str = "default") -> None:
        if not key_id.strip():
            raise ValueError("approval signer needs a key id")
        self._key_id = key_id
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

    def approve(
        self,
        *,
        intent: ActionIntent,
        contract: TaskContract,
        approver: str,
        executor_id: str,
        ttl: timedelta = timedelta(minutes=10),
        now: datetime | None = None,
    ) -> SignedApprovalReceipt:
        issued_at = now or datetime.now(UTC)
        unsigned = SignedApprovalReceipt(
            key_id=self._key_id,
            approval_id=str(uuid4()),
            approver=approver,
            scope="action.execute",
            intent_digest=intent.intent_digest,
            contract_digest=contract.contract_digest,
            executor_id=executor_id,
            issued_at=issued_at,
            expires_at=issued_at + ttl,
            signature="",
        )
        return SignedApprovalReceipt(
            key_id=unsigned.key_id,
            approval_id=unsigned.approval_id,
            approver=unsigned.approver,
            scope=unsigned.scope,
            intent_digest=unsigned.intent_digest,
            contract_digest=unsigned.contract_digest,
            executor_id=unsigned.executor_id,
            issued_at=unsigned.issued_at,
            expires_at=unsigned.expires_at,
            signature=base64.urlsafe_b64encode(self._key.sign(unsigned.payload())).decode("ascii"),
        )


class ApprovalVerifier:
    """Policy-side public key verifier for narrowly scoped human approvals."""

    def __init__(
        self,
        public_keys: Mapping[str, bytes | Ed25519PublicKey],
        *,
        allowed_approvers: frozenset[str] = frozenset(),
        trust_entries: Mapping[str, ApprovalTrustEntry] | None = None,
        consumption_store: ApprovalConsumptionStore | None = None,
        maximum_ttl: timedelta = timedelta(minutes=15),
        maximum_age: timedelta = timedelta(minutes=15),
        clock_skew: timedelta = timedelta(seconds=30),
    ) -> None:
        if not public_keys or (not allowed_approvers and not trust_entries):
            raise ValueError("approval verifier needs keys and approver trust")
        self._keys = {
            key_id: Ed25519PublicKey.from_public_bytes(key) if isinstance(key, bytes) else key
            for key_id, key in public_keys.items()
        }
        self._allowed_approvers = allowed_approvers
        self._trust_entries = dict(trust_entries or {})
        if self._trust_entries and set(self._trust_entries) != set(self._keys):
            raise ValueError("approval trust entries must cover exactly the configured keys")
        self._consumption_store = consumption_store or InMemoryApprovalConsumptionStore()
        self._maximum_ttl = maximum_ttl
        self._maximum_age = maximum_age
        self._clock_skew = clock_skew

    @property
    def has_verifier_owned_trust(self) -> bool:
        return bool(self._trust_entries)

    @property
    def consumption_store(self) -> ApprovalConsumptionStore:
        return self._consumption_store

    def validate(
        self,
        receipt: SignedApprovalReceipt,
        *,
        intent: ActionIntent,
        contract: TaskContract,
        executor_id: str,
        now: datetime | None = None,
    ) -> None:
        current = now or datetime.now(UTC)
        trust = self._trust_entries.get(receipt.key_id)
        if trust is not None:
            if trust.subject != receipt.approver or "security-reviewer" not in trust.roles:
                raise ApprovalRejected("approval key is not bound to this approver identity")
            if current < trust.valid_from or current >= trust.valid_until or (
                trust.revoked_at is not None and current >= trust.revoked_at
            ):
                raise ApprovalRejected("approval signing key is not currently trusted")
        elif receipt.approver not in self._allowed_approvers:
            raise ApprovalRejected("approval signer identity is not allowed")
        if receipt.intent_digest != intent.intent_digest or receipt.contract_digest != contract.contract_digest:
            raise ApprovalRejected("approval is not bound to this intent and contract")
        if receipt.executor_id != executor_id:
            raise ApprovalRejected("approval is not bound to this executor")
        if (
            receipt.expires_at <= current
            or receipt.issued_at > current + self._clock_skew
            or current - receipt.issued_at > self._maximum_age
            or receipt.expires_at - receipt.issued_at > self._maximum_ttl
        ):
            raise ApprovalRejected("approval is expired or not yet valid")
        key = self._keys.get(receipt.key_id)
        if key is None:
            raise ApprovalRejected("approval key is not trusted")
        try:
            key.verify(base64.urlsafe_b64decode(receipt.signature.encode("ascii")), receipt.payload())
        except (InvalidSignature, ValueError) as exc:
            raise ApprovalRejected("invalid approval signature") from exc
        if not self._consumption_store.consume(
            receipt.approval_id, intent_digest=intent.intent_digest, executor_id=executor_id
        ):
            raise ApprovalRejected("approval receipt was already consumed")


class PolicyUseCounterStore:
    """Atomic consumption of all overlapping action-rule budgets."""

    def consume(self, limits: tuple[tuple[tuple[str, str, str, str], int | None], ...]) -> bool:
        raise NotImplementedError


class InMemoryPolicyUseCounterStore(PolicyUseCounterStore):
    """Development counter store; production uses ``SQLitePolicyUseCounterStore``."""

    def __init__(self) -> None:
        self._counts: dict[tuple[str, str, str, str], int] = {}
        self._lock = threading.Lock()

    def consume(self, limits: tuple[tuple[tuple[str, str, str, str], int | None], ...]) -> bool:
        with self._lock:
            if any(limit is not None and self._counts.get(key, 0) >= limit for key, limit in limits):
                return False
            for key, _limit in limits:
                self._counts[key] = self._counts.get(key, 0) + 1
            return True


class SQLitePolicyUseCounterStore(PolicyUseCounterStore):
    """Durable process-safe counter state for policy action-rule budgets."""

    def __init__(self, database: str | Path) -> None:
        path = Path(database)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(path, isolation_level=None, timeout=5.0)
        self._connection.execute("PRAGMA busy_timeout=5000")
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS policy_use_counters (
                contract_digest TEXT NOT NULL,
                tool TEXT NOT NULL,
                effect TEXT NOT NULL,
                target_prefix TEXT NOT NULL,
                use_count INTEGER NOT NULL,
                PRIMARY KEY (contract_digest, tool, effect, target_prefix)
            )
            """
        )

    def consume(self, limits: tuple[tuple[tuple[str, str, str, str], int | None], ...]) -> bool:
        self._connection.execute("BEGIN IMMEDIATE")
        try:
            for key, limit in limits:
                row = self._connection.execute(
                    """
                    SELECT use_count FROM policy_use_counters
                    WHERE contract_digest = ? AND tool = ? AND effect = ? AND target_prefix = ?
                    """,
                    key,
                ).fetchone()
                if limit is not None and row is not None and row[0] >= limit:
                    self._connection.execute("ROLLBACK")
                    return False
            for key, _limit in limits:
                self._connection.execute(
                    """
                    INSERT INTO policy_use_counters
                    (contract_digest, tool, effect, target_prefix, use_count)
                    VALUES (?, ?, ?, ?, 1)
                    ON CONFLICT(contract_digest, tool, effect, target_prefix)
                    DO UPDATE SET use_count = use_count + 1
                    """,
                    key,
                )
            self._connection.execute("COMMIT")
            return True
        except Exception:
            self._connection.execute("ROLLBACK")
            raise

    def close(self) -> None:
        self._connection.close()


class PolicyGate:
    """Reference monitor for concrete action intents.

    It belongs in a separate service/process from both planner and executor.
    """

    def __init__(
        self,
        contract: TaskContract,
        *,
        signing_key: bytes | None = None,
        capability_ttl: timedelta = timedelta(minutes=5),
        approval_verifier: ApprovalVerifier | None = None,
        use_counter_store: PolicyUseCounterStore | None = None,
    ) -> None:
        self.contract = contract
        self._signer = CapabilitySigner(signing_key)
        self._ttl = capability_ttl
        self._use_counter_store = use_counter_store or InMemoryPolicyUseCounterStore()
        self._legacy_nonce_store = InMemoryNonceStore()
        self._approval_verifier = approval_verifier

    @property
    def capability_public_key(self) -> bytes:
        """Public verifier key for the separately deployed executor service."""

        return self._signer.public_key_bytes

    @property
    def signed_approval_verifier(self) -> ApprovalVerifier | None:
        return self._approval_verifier

    @property
    def use_counter_store(self) -> PolicyUseCounterStore:
        return self._use_counter_store

    def authorize(
        self,
        intent: ActionIntent,
        *,
        executor_id: str,
        approvals: Iterable[Approval | SignedApprovalReceipt] = (),
        now: datetime | None = None,
    ) -> Capability:
        now = now or datetime.now(UTC)
        if self.contract.is_expired(now):
            raise PolicyDenied("contract expired")
        if intent.contract_id != self.contract.contract_id or intent.contract_version != self.contract.version:
            raise PolicyDenied("intent references a different contract version")
        if intent.tool in self.contract.forbidden_actions:
            raise PolicyDenied(f"forbidden tool: {intent.tool}")
        if self.contract.require_argument_provenance and not intent.has_complete_argument_provenance:
            missing = sorted(set(intent.arguments).difference(intent.argument_provenance_graph))
            raise PolicyDenied(f"action needs a complete provenance DAG for every argument: {missing}")
        if self._contains_inline_secret(intent.arguments):
            raise PolicyDenied("inline secrets are forbidden; use an executor-owned credential boundary")
        matching_rules = [
            rule
            for rule in self.contract.allowed_actions
            if rule.tool == intent.tool
            and rule.effect == intent.effect
            and self._target_matches(intent.target, rule.target_prefix)
        ]
        if not matching_rules:
            raise PolicyDenied("no action rule authorizes this tool/effect/target")
        self._validate_argument_rules(intent, matching_rules)
        approval_required = any(rule.approval_required for rule in matching_rules)
        untrusted_arguments = tuple(
            name
            for name in intent.arguments
            if {
                Provenance.UNTRUSTED_RETRIEVAL,
                Provenance.TOOL_OUTPUT,
            }.intersection(intent.provenance_for_argument(name))
        )
        tainted_high_impact = (
            {
                Provenance.UNTRUSTED_RETRIEVAL,
                Provenance.TOOL_OUTPUT,
            }.intersection(intent.provenance)
            or untrusted_arguments
        ) and intent.effect in {
            Effect.WRITE,
            Effect.EXECUTE,
            Effect.NETWORK,
            Effect.DELETE,
            Effect.PUBLISH,
        }
        always_approval = intent.effect in {Effect.DELETE, Effect.PUBLISH}
        requires_approval = always_approval or tainted_high_impact or approval_required
        approved = self._is_approved(intent, approvals, executor_id=executor_id) if requires_approval else False
        if tainted_high_impact and not approved:
            raise PolicyDenied("tainted high-impact action or argument requires explicit approval")
        if always_approval and not approved:
            raise PolicyDenied(f"{intent.effect.value} always requires explicit approval")
        if approval_required and not approved:
            raise PolicyDenied("this action requires explicit approval")
        limits = tuple(
            (
                (self.contract.contract_digest, rule.tool, rule.effect.value, rule.target_prefix),
                rule.max_uses,
            )
            for rule in matching_rules
        )
        if not self._use_counter_store.consume(limits):
            raise PolicyDenied("action rule use budget exhausted")
        expires_at = now + self._ttl
        return self._signer.issue(
            intent=intent,
            contract_digest=self.contract.contract_digest,
            executor_id=executor_id,
            issued_at=now,
            expires_at=expires_at,
        )

    def validate_and_consume(
        self, capability: Capability, intent: ActionIntent, now: datetime | None = None
    ) -> None:
        """Compatibility helper for single-process tests only.

        Production controllers must not call this.  The executor receives the
        public key and consumes the nonce immediately before the side effect.
        """

        if capability.contract_digest != self.contract.contract_digest:
            raise PolicyDenied("capability belongs to another contract")
        try:
            CapabilityVerifier(self.capability_public_key, self._legacy_nonce_store).validate_and_consume(
                capability,
                intent,
                executor_id=capability.executor_id,
                now=now,
            )
        except PermissionError as exc:
            raise PolicyDenied(str(exc)) from exc

    def _is_approved(
        self,
        intent: ActionIntent,
        approvals: Iterable[Approval | SignedApprovalReceipt],
        *,
        executor_id: str,
    ) -> bool:
        for approval in approvals:
            if isinstance(approval, SignedApprovalReceipt):
                if self._approval_verifier is None:
                    continue
                try:
                    self._approval_verifier.validate(
                        approval,
                        intent=intent,
                        contract=self.contract,
                        executor_id=executor_id,
                    )
                except ApprovalRejected:
                    continue
                return True
            if self._approval_verifier is None and approval.intent_digest == intent.intent_digest:
                return True
        return False

    @staticmethod
    def _validate_argument_rules(intent: ActionIntent, matching_rules) -> None:
        """Intersect every matching server-owned semantic argument constraint."""

        for rule in matching_rules:
            allowed_names = {constraint.name for constraint in rule.argument_rules}
            if not rule.allow_unlisted_arguments:
                extras = set(intent.arguments).difference(allowed_names)
                if extras:
                    raise PolicyDenied(f"action has unlisted arguments: {sorted(extras)}")
            for constraint in rule.argument_rules:
                if constraint.name not in intent.arguments:
                    if constraint.required:
                        raise PolicyDenied(f"required argument is missing: {constraint.name}")
                    continue
                error = constraint.validate(intent.arguments[constraint.name])
                if error:
                    raise PolicyDenied(error)

    @staticmethod
    def _target_matches(target: str, prefix: str) -> bool:
        normalized_prefix = prefix.rstrip("/") or "/"
        return normalized_prefix == "/" or target == normalized_prefix or target.startswith(
            normalized_prefix + "/"
        )

    @classmethod
    def _contains_inline_secret(cls, value: Any, *, key: str = "") -> bool:
        if _SENSITIVE_ARGUMENT_KEY.search(key) and value is not None and value != "" and value is not False:
            return True
        if isinstance(value, Mapping):
            return any(cls._contains_inline_secret(item, key=str(item_key)) for item_key, item in value.items())
        if isinstance(value, (tuple, list, set, frozenset)):
            return any(cls._contains_inline_secret(item, key=key) for item in value)
        return isinstance(value, str) and bool(_TOKEN_LIKE_VALUE.search(value))
