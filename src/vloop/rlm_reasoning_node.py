"""Graph-bound coordinator for the untrusted RLM reasoning worker."""

from __future__ import annotations

from dataclasses import dataclass
from uuid import uuid4

from .canonical import digest
from .programmable_context import ContextAuthority, ProgrammableContextStore
from .reasoning_sessions import ReasoningSessionStore
from .rlm_protocol import RLMReasoningRequest, ReasoningArtifact
from .rlm_worker import RLMWorker


@dataclass(frozen=True, slots=True)
class RLMNodePolicy:
    enabled: bool = False
    maximum_child_sessions: int = 4
    child_timeout_seconds: int = 1_800

    def __post_init__(self) -> None:
        if min(self.maximum_child_sessions, self.child_timeout_seconds) < 1:
            raise ValueError("RLM node policy limits must be positive")


class RLMReasoningNode:
    """Returns advisory artifacts only; graph completion remains externally signed."""

    def __init__(self, worker: RLMWorker, sessions: ReasoningSessionStore, policy: RLMNodePolicy = RLMNodePolicy()) -> None:
        self._worker = worker
        self._sessions = sessions
        self._policy = policy

    def execute(self, request: RLMReasoningRequest, context: ProgrammableContextStore) -> ReasoningArtifact:
        if not self._policy.enabled:
            raise PermissionError("RLM reasoning node is disabled pending matched-budget evaluation")
        manifest = context.manifest(allowed_handles=request.allowed_context_handles)
        if manifest.manifest_digest != request.context_manifest_digest:
            raise PermissionError("RLM node received a context manifest mismatch")
        session = self._sessions.require_binding(
            request.session_id, run_id=request.run_id, contract_digest=request.contract_digest,
            graph_digest=request.graph_digest, node_instance_id=request.node_instance_id,
            context_root_digest=request.context_manifest_digest,
        )
        output = self._worker.run(request, context)
        if output.token_usage > request.maximum_tokens or output.model_calls > request.maximum_recursive_calls:
            raise PermissionError("RLM worker exceeded the graph-owned resource budget")
        if set(output.context_reads).difference(request.allowed_context_handles):
            raise PermissionError("RLM worker read an unadmitted context handle")
        children = output.child_sessions[: self._policy.maximum_child_sessions]
        if len(output.child_sessions) > self._policy.maximum_child_sessions:
            raise PermissionError("RLM worker exceeded the child session limit")
        reserved_tokens = output.token_usage + sum(item.token_budget for item in children)
        reserved_calls = output.model_calls + sum(item.call_budget for item in children)
        if reserved_tokens > session.remaining_token_budget or reserved_calls > session.remaining_call_budget:
            raise PermissionError("RLM worker and child sessions exceed the parent recursive budget")
        child_refs: list[str] = []
        for proposal in children:
            if set(proposal.context_handles).difference(request.allowed_context_handles):
                raise PermissionError("RLM child proposal escaped its admitted context")
            child = self._sessions.spawn_child(
                session.session_id, child_node_instance_id=digest({"parent": request.node_instance_id, "spawn": str(uuid4())}),
                token_budget=proposal.token_budget, call_budget=proposal.call_budget,
            )
            child_refs.append(child.session_id)
        self._sessions.consume(session.session_id, tokens=output.token_usage, calls=output.model_calls)
        ceilings = [context.read(handle, allowed_handles=request.allowed_context_handles).authority_ceiling for handle in output.context_reads]
        authority = min(ceilings, default=ContextAuthority.UNTRUSTED)
        return ReasoningArtifact(
            request_digest=request.request_digest, program_digest=digest(dict(output.program)),
            context_reads=output.context_reads, child_session_refs=tuple(child_refs),
            derived_artifacts=(), candidate_actions=output.candidate_actions, token_usage=output.token_usage,
            model_calls=output.model_calls, final_summary=output.final_summary, authority_ceiling=authority,
        )
