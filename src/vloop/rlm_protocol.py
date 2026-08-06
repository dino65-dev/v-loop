"""Typed input/output protocol for the untrusted RLM intelligence plane."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping

from .canonical import digest
from .programmable_context import ContextAuthority


def _require_digest(value: str, label: str) -> None:
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise ValueError(f"{label} must be a SHA-256 digest")


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

    def __post_init__(self) -> None:
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


@dataclass(frozen=True, slots=True)
class RLMWorkerOutput:
    program: Mapping[str, object]
    context_reads: tuple[str, ...]
    final_summary: str
    candidate_actions: tuple[ActionProposal, ...] = ()
    child_sessions: tuple[ChildSessionProposal, ...] = ()
    token_usage: int = 0
    model_calls: int = 0

    def __post_init__(self) -> None:
        if not self.final_summary.strip() or self.token_usage < 0 or self.model_calls < 0:
            raise ValueError("RLM worker output is invalid")


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

    def __post_init__(self) -> None:
        for value in (self.request_digest, self.program_digest):
            _require_digest(value, "reasoning artifact")
        if not self.final_summary.strip() or min(self.token_usage, self.model_calls) < 0:
            raise ValueError("reasoning artifact is invalid")

    @property
    def artifact_digest(self) -> str:
        return digest(asdict(self))
