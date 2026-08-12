"""Typed input/output protocol for the untrusted RLM intelligence plane."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Mapping

from .canonical import digest
from .programmable_context import ContextAuthority


def _require_digest(value: str, label: str) -> None:
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise ValueError(f"{label} must be a SHA-256 digest")


RLM_PROTOCOL_VERSION = "vloop.rlm.v2"


@dataclass(frozen=True, slots=True)
class ModelUsageReceipt:
    """Trusted accounting fact returned by the registered model endpoint."""

    provider_request_id: str
    request_digest: str
    model_digest: str
    call_label: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    started_at: datetime
    finished_at: datetime
    provider_reported: bool

    def __post_init__(self) -> None:
        if not self.provider_request_id.strip() or self.call_label not in {"plan", "synthesis"}:
            raise ValueError("model usage receipts need a provider id and known call label")
        for value, label in ((self.request_digest, "request"), (self.model_digest, "model")):
            _require_digest(value, label)
        if min(self.prompt_tokens, self.completion_tokens, self.total_tokens) < 0:
            raise ValueError("model usage cannot be negative")
        if self.total_tokens != self.prompt_tokens + self.completion_tokens:
            raise ValueError("model usage total does not match prompt plus completion")
        if self.started_at.tzinfo is None or self.finished_at.tzinfo is None or self.finished_at < self.started_at:
            raise ValueError("model usage receipt timing is invalid")

    @property
    def receipt_digest(self) -> str:
        value = asdict(self)
        value["started_at"] = self.started_at.isoformat()
        value["finished_at"] = self.finished_at.isoformat()
        return digest(value)


@dataclass(frozen=True, slots=True)
class RLMReasoningRequest:
    run_id: str
    contract_digest: str
    graph_digest: str
    node_instance_id: str
    context_manifest_digest: str
    allowed_context_handles: tuple[str, ...]
    maximum_recursive_calls: int
    maximum_tokens: int
    timeout_seconds: int
    model_digest: str
    harness_digest: str
    session_id: str
    protocol_version: str = RLM_PROTOCOL_VERSION

    def __post_init__(self) -> None:
        if self.protocol_version != RLM_PROTOCOL_VERSION:
            raise ValueError("RLM request protocol version is not supported")
        if not self.run_id.strip() or not self.node_instance_id.strip() or not self.session_id.strip() or not self.allowed_context_handles:
            raise ValueError("RLM request needs run, node, session, and context handles")
        if len(self.allowed_context_handles) != len(set(self.allowed_context_handles)) or any(not value.startswith("context://") for value in self.allowed_context_handles):
            raise ValueError("RLM request context handles must be unique context URIs")
        if min(self.maximum_recursive_calls, self.maximum_tokens, self.timeout_seconds) < 1:
            raise ValueError("RLM request budgets must be positive")
        for value, label in (
            (self.contract_digest, "contract"), (self.graph_digest, "graph"), (self.context_manifest_digest, "context manifest"),
            (self.model_digest, "model"), (self.harness_digest, "harness"),
        ):
            _require_digest(value, label)

    @property
    def request_digest(self) -> str:
        return digest(asdict(self))


@dataclass(frozen=True, slots=True)
class ActionProposal:
    """Untyped-to-authority candidate only; it is never a capability or intent."""

    tool: str
    effect: str
    target: str
    arguments: Mapping[str, object]
    rationale: str
    authority_ceiling: ContextAuthority

    def __post_init__(self) -> None:
        if not all((self.tool.strip(), self.effect.strip(), self.target.strip(), self.rationale.strip())):
            raise ValueError("action proposals need descriptive fields")
        if len(self.tool) > 160 or len(self.effect) > 64 or len(self.target) > 4_096 or len(self.rationale) > 8_000:
            raise ValueError("action proposal text exceeds protocol bounds")
        _validate_json_value(dict(self.arguments), depth=0)

    @property
    def proposal_digest(self) -> str:
        return digest(asdict(self))


@dataclass(frozen=True, slots=True)
class ChildSessionProposal:
    objective: str
    context_handles: tuple[str, ...]
    token_budget: int
    call_budget: int

    def __post_init__(self) -> None:
        if not self.objective.strip() or min(self.token_budget, self.call_budget) < 1:
            raise ValueError("child session proposal needs objective and positive budgets")
        if len(self.objective) > 8_000 or not self.context_handles or len(self.context_handles) != len(set(self.context_handles)):
            raise ValueError("child session proposal is not bounded")
        if any(not value.startswith("context://") for value in self.context_handles):
            raise ValueError("child session handles must be context URIs")


@dataclass(frozen=True, slots=True)
class RLMWorkerOutput:
    program: Mapping[str, object]
    context_reads: tuple[str, ...]
    final_summary: str
    candidate_actions: tuple[ActionProposal, ...] = ()
    child_sessions: tuple[ChildSessionProposal, ...] = ()
    token_usage: int = 0
    model_calls: int = 0
    usage_receipts: tuple[ModelUsageReceipt, ...] = ()
    protocol_version: str = RLM_PROTOCOL_VERSION

    def __post_init__(self) -> None:
        if self.protocol_version != RLM_PROTOCOL_VERSION or not self.final_summary.strip() or self.token_usage < 0 or self.model_calls < 0:
            raise ValueError("RLM worker output is invalid")
        if len(self.final_summary) > 16_000 or len(self.context_reads) != len(set(self.context_reads)):
            raise ValueError("RLM worker output exceeds protocol bounds")
        if any(not value.startswith("context://") for value in self.context_reads):
            raise ValueError("RLM worker output has an invalid context handle")
        if len({item.call_label for item in self.usage_receipts}) != len(self.usage_receipts):
            raise ValueError("model usage call labels must be unique")
        if self.usage_receipts:
            if self.token_usage != sum(item.total_tokens for item in self.usage_receipts) or self.model_calls != len(self.usage_receipts):
                raise ValueError("worker usage totals must match usage receipts")
        elif self.token_usage or self.model_calls:
            raise ValueError("non-zero RLM usage requires model usage receipts")


@dataclass(frozen=True, slots=True)
class ReasoningArtifact:
    request_digest: str
    program_digest: str
    context_reads: tuple[str, ...]
    child_session_refs: tuple[str, ...]
    derived_artifacts: tuple[str, ...]
    candidate_actions: tuple[ActionProposal, ...]
    token_usage: int
    model_calls: int
    final_summary: str
    authority_ceiling: ContextAuthority
    usage_receipt_digests: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for value in (self.request_digest, self.program_digest):
            _require_digest(value, "reasoning artifact")
        if not self.final_summary.strip() or min(self.token_usage, self.model_calls) < 0:
            raise ValueError("reasoning artifact is invalid")

    @property
    def artifact_digest(self) -> str:
        return digest(asdict(self))


def _validate_json_value(value: object, *, depth: int) -> None:
    """Reject non-JSON, excessive-depth, non-finite, and oversized arguments."""

    if depth > 8:
        raise ValueError("JSON arguments exceed maximum nesting depth")
    if value is None or isinstance(value, (bool, int, str)):
        if isinstance(value, str) and len(value) > 16_000:
            raise ValueError("JSON argument string exceeds protocol bounds")
        return
    if isinstance(value, float):
        if value != value or value in {float("inf"), float("-inf")}:
            raise ValueError("JSON argument numbers must be finite")
        return
    if isinstance(value, list):
        if len(value) > 128:
            raise ValueError("JSON argument arrays exceed protocol bounds")
        for item in value:
            _validate_json_value(item, depth=depth + 1)
        return
    if isinstance(value, dict):
        if len(value) > 128 or any(not isinstance(key, str) or len(key) > 256 for key in value):
            raise ValueError("JSON argument objects exceed protocol bounds")
        for item in value.values():
            _validate_json_value(item, depth=depth + 1)
        return
    raise ValueError("arguments must be JSON-compatible values")
