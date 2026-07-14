"""Deterministic, default-deny action authorization."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
import re
from typing import Any, Iterable, Mapping

from .authorization import CapabilitySigner, CapabilityVerifier, InMemoryNonceStore
from .models import ActionIntent, Capability, Effect, Provenance, TaskContract


class PolicyDenied(PermissionError):
    """The action is outside the currently authorized task envelope."""


_SENSITIVE_ARGUMENT_KEY = re.compile(
    r"(?:api.?key|token|password|secret|credential|authorization|cookie)", re.I
)
_TOKEN_LIKE_VALUE = re.compile(r"(?:bearer\s+|(?:sk|rk|ghp)[_-])[A-Za-z0-9._-]{6,}", re.I)


@dataclass(frozen=True, slots=True)
class Approval:
    intent_digest: str
    approver: str
    approved_at: datetime


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
    ) -> None:
        self.contract = contract
        self._signer = CapabilitySigner(signing_key)
        self._ttl = capability_ttl
        self._uses: dict[tuple[str, str, str], int] = {}
        self._legacy_nonce_store = InMemoryNonceStore()

    @property
    def capability_public_key(self) -> bytes:
        """Public verifier key for the separately deployed executor service."""

        return self._signer.public_key_bytes

    def authorize(
        self,
        intent: ActionIntent,
        *,
        executor_id: str,
        approvals: Iterable[Approval] = (),
        now: datetime | None = None,
    ) -> Capability:
        now = now or datetime.now(UTC)
        if self.contract.is_expired(now):
            raise PolicyDenied("contract expired")
        if intent.contract_id != self.contract.contract_id or intent.contract_version != self.contract.version:
            raise PolicyDenied("intent references a different contract version")
        if intent.tool in self.contract.forbidden_actions:
            raise PolicyDenied(f"forbidden tool: {intent.tool}")
        if self._contains_inline_secret(intent.arguments):
            raise PolicyDenied("inline secrets are forbidden; use an executor-owned credential boundary")
        if intent.effect in {Effect.DELETE, Effect.PUBLISH} and not self._is_approved(intent, approvals):
            raise PolicyDenied(f"{intent.effect.value} always requires explicit approval")

        matching_rules = [
            rule
            for rule in self.contract.allowed_actions
            if rule.tool == intent.tool
            and rule.effect == intent.effect
            and self._target_matches(intent.target, rule.target_prefix)
        ]
        if not matching_rules:
            raise PolicyDenied("no action rule authorizes this tool/effect/target")
        approval_required = any(rule.approval_required for rule in matching_rules)
        if Provenance.UNTRUSTED_RETRIEVAL in intent.provenance and intent.effect in {
            Effect.WRITE,
            Effect.NETWORK,
            Effect.DELETE,
            Effect.PUBLISH,
        } and not self._is_approved(intent, approvals):
            raise PolicyDenied("tainted high-impact action requires explicit approval")
        if approval_required and not self._is_approved(intent, approvals):
            raise PolicyDenied("this action requires explicit approval")
        use_keys = [(rule.tool, rule.effect.value, rule.target_prefix) for rule in matching_rules]
        if any(
            rule.max_uses is not None and self._uses.get(use_key, 0) >= rule.max_uses
            for rule, use_key in zip(matching_rules, use_keys, strict=True)
        ):
            raise PolicyDenied("action rule use budget exhausted")
        for use_key in use_keys:
            self._uses[use_key] = self._uses.get(use_key, 0) + 1
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

    def _is_approved(self, intent: ActionIntent, approvals: Iterable[Approval]) -> bool:
        return any(approval.intent_digest == intent.intent_digest for approval in approvals)

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
