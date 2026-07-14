"""Deterministic, default-deny action authorization."""

from __future__ import annotations

import base64
import hashlib
import hmac
import secrets
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Iterable
from uuid import uuid4

from .canonical import canonical_json
from .models import ActionIntent, Capability, Effect, Provenance, TaskContract


class PolicyDenied(PermissionError):
    """The action is outside the currently authorized task envelope."""


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
        self._key = signing_key or secrets.token_bytes(32)
        self._ttl = capability_ttl
        self._uses: dict[tuple[str, str, str], int] = {}
        self._consumed_capabilities: set[str] = set()

    def authorize(
        self,
        intent: ActionIntent,
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
        rule = matching_rules[0]
        if Provenance.UNTRUSTED_RETRIEVAL in intent.provenance and intent.effect in {
            Effect.WRITE,
            Effect.NETWORK,
            Effect.DELETE,
            Effect.PUBLISH,
        } and not self._is_approved(intent, approvals):
            raise PolicyDenied("tainted high-impact action requires explicit approval")
        if rule.approval_required and not self._is_approved(intent, approvals):
            raise PolicyDenied("this action requires explicit approval")
        use_key = (rule.tool, rule.effect.value, rule.target_prefix)
        if rule.max_uses is not None and self._uses.get(use_key, 0) >= rule.max_uses:
            raise PolicyDenied("action rule use budget exhausted")
        self._uses[use_key] = self._uses.get(use_key, 0) + 1

        capability_id = str(uuid4())
        expires_at = now + self._ttl
        signature = self._sign(capability_id, intent.intent_digest, expires_at)
        return Capability(
            capability_id=capability_id,
            intent_digest=intent.intent_digest,
            contract_digest=self.contract.contract_digest,
            expires_at=expires_at,
            signature=signature,
        )

    def validate_and_consume(
        self, capability: Capability, intent: ActionIntent, now: datetime | None = None
    ) -> None:
        now = now or datetime.now(UTC)
        if capability.capability_id in self._consumed_capabilities:
            raise PolicyDenied("capability was already consumed")
        if capability.expires_at <= now:
            raise PolicyDenied("capability expired")
        if capability.contract_digest != self.contract.contract_digest:
            raise PolicyDenied("capability belongs to another contract")
        if capability.intent_digest != intent.intent_digest:
            raise PolicyDenied("capability is not bound to this exact intent")
        expected = self._sign(capability.capability_id, capability.intent_digest, capability.expires_at)
        if not hmac.compare_digest(expected, capability.signature):
            raise PolicyDenied("invalid capability signature")
        self._consumed_capabilities.add(capability.capability_id)

    def _is_approved(self, intent: ActionIntent, approvals: Iterable[Approval]) -> bool:
        return any(approval.intent_digest == intent.intent_digest for approval in approvals)

    @staticmethod
    def _target_matches(target: str, prefix: str) -> bool:
        normalized_prefix = prefix.rstrip("/") or "/"
        return normalized_prefix == "/" or target == normalized_prefix or target.startswith(
            normalized_prefix + "/"
        )

    def _sign(self, capability_id: str, intent_digest: str, expires_at: datetime) -> str:
        payload = canonical_json(
            {
                "capability_id": capability_id,
                "intent_digest": intent_digest,
                "expires_at": expires_at.isoformat(),
                "contract_digest": self.contract.contract_digest,
            }
        )
        return base64.urlsafe_b64encode(
            hmac.new(self._key, payload.encode("utf-8"), hashlib.sha256).digest()
        ).decode("ascii")
