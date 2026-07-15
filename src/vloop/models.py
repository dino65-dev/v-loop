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


class ArgumentKind(StrEnum):
    STRING = "string"
    INTEGER = "integer"
    BOOLEAN = "boolean"
    PATH = "path"
    ARGV = "argv"
    ENUM = "enum"


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
    WAITING = "waiting"
    STOP = "stop"


@dataclass(frozen=True, slots=True)
class ArgumentRule:
    """Server-owned semantics for one named action argument."""

    name: str
    kind: ArgumentKind
    required: bool = False
    allowed_values: tuple[str, ...] = ()
    maximum_length: int | None = None
    minimum: int | None = None
    maximum: int | None = None

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("argument rule name is required")
        if self.kind is ArgumentKind.ENUM and not self.allowed_values:
            raise ValueError("enum argument rules need allowed values")
        if self.kind is not ArgumentKind.ENUM and self.allowed_values:
            raise ValueError("allowed values are only valid for enum arguments")
        if self.maximum_length is not None and self.maximum_length < 1:
            raise ValueError("argument maximum length must be positive")
        if self.minimum is not None and self.maximum is not None and self.minimum > self.maximum:
            raise ValueError("argument minimum exceeds maximum")

    def validate(self, value: Any) -> str | None:
        if self.kind is ArgumentKind.STRING:
            valid = isinstance(value, str)
            length = len(value) if valid else 0
        elif self.kind is ArgumentKind.INTEGER:
            valid = isinstance(value, int) and not isinstance(value, bool)
            length = 0
        elif self.kind is ArgumentKind.BOOLEAN:
            valid = isinstance(value, bool)
            length = 0
        elif self.kind is ArgumentKind.PATH:
            valid = isinstance(value, str) and value.startswith("/") and not any(
                part in {".", ".."} for part in PurePosixPath(value).parts
            )
            length = len(value) if isinstance(value, str) else 0
        elif self.kind is ArgumentKind.ARGV:
            valid = isinstance(value, list) and bool(value) and all(
                isinstance(item, str) and "\x00" not in item for item in value
            )
            length = len(value) if isinstance(value, list) else 0
        else:  # enum
            valid = isinstance(value, str) and value in self.allowed_values
            length = len(value) if isinstance(value, str) else 0
        if not valid:
            return f"argument {self.name!r} does not satisfy {self.kind.value} semantics"
        if self.maximum_length is not None and length > self.maximum_length:
            return f"argument {self.name!r} exceeds maximum length"
        if self.kind is ArgumentKind.INTEGER:
            if self.minimum is not None and value < self.minimum:
                return f"argument {self.name!r} is below the allowed minimum"
            if self.maximum is not None and value > self.maximum:
                return f"argument {self.name!r} exceeds the allowed maximum"
        return None


@dataclass(frozen=True, slots=True)
class ActionRule:
    tool: str
    effect: Effect
    target_prefix: str
    approval_required: bool = False
    max_uses: int | None = None
    argument_rules: tuple[ArgumentRule, ...] = ()
    allow_unlisted_arguments: bool = True

    def __post_init__(self) -> None:
        if not self.tool.strip() or not self.target_prefix.startswith("/"):
            raise ValueError("action rules need a tool and absolute target prefix")
        if self.max_uses is not None and self.max_uses < 1:
            raise ValueError("action rule max uses must be positive")
        names = [rule.name for rule in self.argument_rules]
        if len(names) != len(set(names)):
            raise ValueError("action rule argument names must be unique")


@dataclass(frozen=True, slots=True)
class TaskContract:
    """Machine-readable task authority, not a credential."""

    goal: str
    success_conditions: tuple[str, ...]
    allowed_actions: tuple[ActionRule, ...]
    forbidden_actions: tuple[str, ...] = ()
    required_verifiers: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
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
        allowed_categories = {"correctness", "policy", "evidence", "quality"}
        unknown_categories = set(self.required_verifiers).difference(allowed_categories)
        if unknown_categories:
            raise ValueError(f"unknown verifier categories: {sorted(unknown_categories)}")
        required_names: list[str] = []
        for category, names in self.required_verifiers.items():
            if not names or any(not isinstance(name, str) or not name.strip() for name in names):
                raise ValueError(f"required verifier category {category!r} needs non-empty check names")
            required_names.extend(names)
        if len(required_names) != len(set(required_names)):
            raise ValueError("a required verifier check may belong to only one category")

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
    argument_provenance: Mapping[str, tuple[Provenance, ...]] = field(default_factory=dict)

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
        unknown_provenance = set(self.argument_provenance).difference(self.arguments)
        if unknown_provenance:
            raise ValueError("argument provenance refers to an unknown argument")
        for argument, provenance in self.argument_provenance.items():
            if not provenance or any(not isinstance(value, Provenance) for value in provenance):
                raise ValueError(f"argument {argument!r} needs non-empty provenance")

    @property
    def intent_digest(self) -> str:
        return digest(self)

    def provenance_for_argument(self, name: str) -> tuple[Provenance, ...]:
        return self.argument_provenance.get(name, self.provenance)


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
