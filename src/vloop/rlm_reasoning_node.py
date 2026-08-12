"""Graph-bound coordinator for the untrusted RLM reasoning worker."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .canonical import digest
from .programmable_context import ContextAuthority, ProgrammableContextStore
from .reasoning_sessions import ChildSessionAdmission, ReasoningSessionStore
from .rlm_protocol import ChildSessionProposal, RLMReasoningRequest, ReasoningArtifact
from .rlm_worker import RLMWorker


@dataclass(frozen=True, slots=True)
class RLMNodePolicy:
    enabled: bool = False
    maximum_child_sessions: int = 4
    child_timeout_seconds: int = 1_800
    require_provider_usage: bool = True

    def __post_init__(self) -> None:
        if min(self.maximum_child_sessions, self.child_timeout_seconds) < 1:
            raise ValueError("RLM node policy limits must be positive")


class ChildAdmissionProvider(Protocol):
    """Phase-2 boundary: reserve GraphIR children before durable admission."""

    def reserve_children(
        self,
        request: RLMReasoningRequest,
        *,
        parent_artifact_digest: str,
        proposals: tuple[ChildSessionProposal, ...],
        context: ProgrammableContextStore,
    ) -> tuple[ChildSessionAdmission, ...]: ...


class RLMReasoningNode:
    """Returns advisory artifacts only; graph completion remains externally signed."""

    def __init__(
        self, worker: RLMWorker, sessions: ReasoningSessionStore, policy: RLMNodePolicy = RLMNodePolicy(),
        child_admitter: ChildAdmissionProvider | None = None,
    ) -> None:
        self._worker = worker
        self._sessions = sessions
        self._policy = policy
        self._child_admitter = child_admitter

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
        self._validate_usage(request, output)
        if output.token_usage > request.maximum_tokens or output.model_calls > request.maximum_recursive_calls:
            raise PermissionError("RLM worker exceeded the graph-owned resource budget")
        if set(output.context_reads).difference(request.allowed_context_handles):
            raise PermissionError("RLM worker read an unadmitted context handle")
        children = output.child_sessions
        if len(output.child_sessions) > self._policy.maximum_child_sessions:
            raise PermissionError("RLM worker exceeded the child session limit")
        for proposal in children:
            if set(proposal.context_handles).difference(request.allowed_context_handles):
                raise PermissionError("RLM child proposal escaped its admitted context")
        parent_artifact_digest = digest({
            "request": request.request_digest, "program": dict(output.program), "context_reads": output.context_reads,
            "summary": output.final_summary, "usage": tuple(item.receipt_digest for item in output.usage_receipts),
        })
        if children and self._child_admitter is None:
            raise PermissionError("child proposals require a GraphIR reservation provider")
        atomic_admitter = getattr(self._child_admitter, "admit_step", None)
        if children and callable(atomic_admitter):
            realised_children = atomic_admitter(
                request=request, parent_session=session, parent_artifact_digest=parent_artifact_digest,
                output=output, proposals=children, context=context,
            )
        else:
            admissions = () if not children else self._child_admitter.reserve_children(
                request, parent_artifact_digest=parent_artifact_digest, proposals=children, context=context,
            )
            self._validate_admissions(children, admissions, parent_artifact_digest, context)
            _updated, realised_children = self._sessions.admit_reasoning_step(
                session.session_id, token_usage=output.token_usage, call_usage=output.model_calls,
                children=admissions, state={"program_digest": digest(dict(output.program)), "summary": output.final_summary},
            )
        ceilings = [context.read(handle, allowed_handles=request.allowed_context_handles).authority_ceiling for handle in output.context_reads]
        authority = min(ceilings, default=ContextAuthority.UNTRUSTED)
        return ReasoningArtifact(
            request_digest=request.request_digest, program_digest=digest(dict(output.program)),
            context_reads=output.context_reads, child_session_refs=tuple(child.session_id for child in realised_children),
            derived_artifacts=(), candidate_actions=output.candidate_actions, token_usage=output.token_usage,
            model_calls=output.model_calls, final_summary=output.final_summary, authority_ceiling=authority,
            usage_receipt_digests=tuple(item.receipt_digest for item in output.usage_receipts),
        )

    def _validate_usage(self, request: RLMReasoningRequest, output) -> None:
        if output.protocol_version != "vloop.rlm.v2":
            raise PermissionError("RLM worker used an unsupported protocol version")
        if output.model_calls and not output.usage_receipts:
            raise PermissionError("RLM worker omitted mandatory model usage receipts")
        if len({receipt.provider_request_id for receipt in output.usage_receipts}) != len(output.usage_receipts):
            raise PermissionError("model usage receipts must have unique provider request ids")
        for receipt in output.usage_receipts:
            if receipt.request_digest != request.request_digest or receipt.model_digest != request.model_digest:
                raise PermissionError("model usage receipt belongs to another request or model")
            if self._policy.require_provider_usage and not receipt.provider_reported:
                raise PermissionError("model provider usage is mandatory for budget-gated RLM experiments")
            if (receipt.finished_at - receipt.started_at).total_seconds() > request.timeout_seconds:
                raise PermissionError("model usage receipt exceeded the admitted deadline")
        if output.usage_receipts:
            elapsed = max(item.finished_at for item in output.usage_receipts) - min(item.started_at for item in output.usage_receipts)
            if elapsed.total_seconds() > request.timeout_seconds:
                raise PermissionError("RLM request exceeded the admitted aggregate deadline")
        if output.token_usage > request.maximum_tokens or output.model_calls > request.maximum_recursive_calls:
            raise PermissionError("RLM worker exceeded the graph-owned resource budget")

    @staticmethod
    def _validate_admissions(
        proposals: tuple[ChildSessionProposal, ...], admissions: tuple[ChildSessionAdmission, ...],
        parent_artifact_digest: str, context: ProgrammableContextStore,
    ) -> None:
        if len(proposals) != len(admissions):
            raise PermissionError("GraphIR admission count does not match child proposals")
        for proposal, admission in zip(proposals, admissions, strict=True):
            expected_manifest = context.manifest(allowed_handles=proposal.context_handles).manifest_digest
            if (
                admission.objective != proposal.objective or admission.token_budget != proposal.token_budget
                or admission.call_budget != proposal.call_budget or admission.context_manifest_digest != expected_manifest
                or admission.allowed_context_handles != proposal.context_handles
                or admission.parent_artifact_digest != parent_artifact_digest
            ):
                raise PermissionError("child GraphIR admission does not exactly bind the proposal context and budget")
