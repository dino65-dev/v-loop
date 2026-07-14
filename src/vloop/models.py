"""Immutable data models crossing the controller, gate, and verifier."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import PurePosixPath
from typing import Any, Mapping
from uuid import uuid4

from .canonical import digest


class Effect(StrEnum):
    READ = "read"
    WRITE = "write"
    EXECUTE = "execute"
    NETWORK = "network"
    DELETE = "delete"
    PUBLISH = "publish"


class Provenance(StrEnum):
    USER = "user"
    TRUSTED_REPOSITORY = "trusted-repository"
    UNTRUSTED_RETRIEVAL = "untrusted-retrieval"
    TOOL_OUTPUT = "tool-output"
    VERIFIED_MEMORY = "verified-memory"


class CheckStatus(StrEnum):
    PASS = "pass"
    FAIL = "fail"
    INCONCLUSIVE = "inconclusive"


class LoopDecision(StrEnum):
    ACCEPT = "accept"
    REPAIR = "repair"
    REPLAN = "replan"
    ESCALATE = "escalate"
    STOP = "stop"


@dataclass(frozen=True, slots=True)
class ActionRule:
    tool: str
    effect: Effect
    target_prefix: str
    approval_required: bool = False
    max_uses: int | None = None


@dataclass(frozen=True, slots=True)
class TaskContract:
    """Machine-readable task authority, not a credential."""

    goal: str
    success_conditions: tuple[str, ...]
    allowed_actions: tuple[ActionRule, ...]
    forbidden_actions: tuple[str, ...] = ()
    maximum_iterations: int = 8
    maximum_tool_calls: int = 32
    expires_at: datetime | None = None
    contract_id: str = field(default_factory=lambda: str(uuid4()))
    version: int = 1

    def __post_init__(self) -> None:
        if self.maximum_iterations < 1 or self.maximum_tool_calls < 1:
            raise ValueError("budgets must be positive")
        if not self.success_conditions:
            raise ValueError("a contract needs at least one success condition")
        if not self.allowed_actions:
            raise ValueError("a contract needs at least one allowed action")

    @property
    def contract_digest(self) -> str:
        return digest(self)

    def is_expired(self, now: datetime | None = None) -> bool:
        return self.expires_at is not None and (now or datetime.now(UTC)) >= self.expires_at


@dataclass(frozen=True, slots=True)
class ActionIntent:
    """Normalized proposed action before any execution."""

    tool: str
    effect: Effect
    target: str
    arguments: Mapping[str, Any]
    provenance: tuple[Provenance, ...]
    explanation: str
    contract_id: str
    contract_version: int
    idempotency_key: str = field(default_factory=lambda: str(uuid4()))

    def __post_init__(self) -> None:
        if not self.target.startswith("/"):
            raise ValueError("target must be an absolute resource path")
        target_path = PurePosixPath(self.target)
        if any(part in {".", ".."} for part in target_path.parts):
            raise ValueError("target may not contain traversal segments")
        if not self.provenance:
            raise ValueError("intent provenance is required")
        if not self.explanation.strip():
            raise ValueError("intent explanation is required")

    @property
    def intent_digest(self) -> str:
        return digest(self)


@dataclass(frozen=True, slots=True)
class Capability:
    capability_id: str
    nonce: str
    intent_digest: str
    contract_digest: str
    executor_id: str
    issued_at: datetime
    expires_at: datetime
    signature: str


@dataclass(frozen=True, slots=True)
class ExecutionObservation:
    success: bool
    exit_code: int | None
    stdout: str
    stderr: str
    artifact_digests: Mapping[str, str] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class CheckResult:
    name: str
    status: CheckStatus
    evidence: Mapping[str, Any]
    message: str = ""


@dataclass(frozen=True, slots=True)
class VerificationReport:
    correctness: CheckStatus
    policy: CheckStatus
    evidence: CheckStatus
    quality: CheckStatus
    checks: tuple[CheckResult, ...]

    @property
    def accepted(self) -> bool:
        return (
            self.correctness is CheckStatus.PASS
            and self.policy is CheckStatus.PASS
            and self.evidence is CheckStatus.PASS
            and self.quality is CheckStatus.PASS
        )
