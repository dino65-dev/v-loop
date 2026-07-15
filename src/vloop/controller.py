"""Verified adaptive-loop state machine."""

from __future__ import annotations

from dataclasses import asdict, replace
from datetime import UTC, datetime
from typing import Iterable, Mapping, Protocol
from uuid import uuid4

from .canonical import digest
from .completion import ActionSafetyReport, EvidenceAccumulator, FinalVerifier, TaskCompletionReport
from .context import ContextPackage, ContextTrust
from .evaluation import ProtectedEvaluationOrchestrator
from .executor import Executor
from .ledger import EvidenceLedger
from .memory import MemoryCandidateProducer, VerifiedMemoryCommitter
from .models import (
    ActionIntent,
    ArgumentProvenance,
    ArgumentProvenanceNode,
    CheckResult,
    CheckStatus,
    ExecutionObservation,
    LoopDecision,
    Provenance,
    TaskContract,
    VerificationReport,
)
from .neural_verifier import ShadowNeuralVerifier
from .policy import Approval, PolicyDenied, PolicyGate, SignedApprovalReceipt
from .probes import ProbeReport, ProtectedProbeRunner
from .repair import RepairController
from .run_state import RunCheckpoint, RunPhase, RunStateStore
from .verifiers import HybridVerifier


class Planner(Protocol):
    def propose(self, *, contract: TaskContract, history: tuple[dict, ...]) -> ActionIntent: ...


class ContextProvider(Protocol):
    """Provides a provenance-labelled, bounded package for an iteration."""

    def build(self, *, contract: TaskContract, history: tuple[dict, ...]) -> ContextPackage: ...


class ContextualPlanner(Protocol):
    def propose_with_context(
        self,
        *,
        contract: TaskContract,
        history: tuple[dict, ...],
        context: ContextPackage,
    ) -> ActionIntent: ...


class EffectReconciler(Protocol):
    """Deployment-owned reconciliation of an uncertain previously-started effect."""

    def reconcile(
        self,
        *,
        run_id: str,
        contract: TaskContract,
        intent: ActionIntent,
        executor_id: str,
    ) -> ExecutionObservation: ...


class VerifiedLoop:
    """Small-step controller with deterministic stop and acceptance behavior."""

    def __init__(
        self,
        *,
        contract: TaskContract,
        planner: Planner,
        gate: PolicyGate,
        executor: Executor,
        verifier: HybridVerifier,
        ledger: EvidenceLedger,
        shadow_verifier: ShadowNeuralVerifier | None = None,
        repair_controller: RepairController | None = None,
        final_verifier: FinalVerifier | None = None,
        probe_runner: ProtectedProbeRunner | None = None,
        context_provider: ContextProvider | None = None,
        memory_committer: VerifiedMemoryCommitter | None = None,
        memory_candidate_producer: MemoryCandidateProducer | None = None,
        state_store: RunStateStore | None = None,
        run_id: str | None = None,
        effect_reconciler: EffectReconciler | None = None,
        evaluation_orchestrator: ProtectedEvaluationOrchestrator | None = None,
    ) -> None:
        self.contract = contract
        self.planner = planner
        self.gate = gate
        self.executor = executor
        self.verifier = verifier
        self.ledger = ledger
        self.shadow_verifier = shadow_verifier
        self.repair_controller = repair_controller or RepairController()
        self.final_verifier = final_verifier
        self.probe_runner = probe_runner
        self.context_provider = context_provider
        self.memory_committer = memory_committer
        self.memory_candidate_producer = memory_candidate_producer
        self.state_store = state_store
        self.effect_reconciler = effect_reconciler
        self.evaluation_orchestrator = evaluation_orchestrator
        self.run_id = run_id or str(uuid4())
        self._evidence = EvidenceAccumulator(self.run_id)
        self._history: list[dict] = []
        self._seen_failures: set[tuple[str, str]] = set()
        self._tool_calls = 0
        self._checkpoint: RunCheckpoint | None = None

    def run(self, approvals: Iterable[Approval | SignedApprovalReceipt] = ()) -> LoopDecision:
        resumed_terminal = self._restore_or_start()
        if resumed_terminal is not None:
            return resumed_terminal
        start_iteration = self._checkpoint.next_iteration if self._checkpoint is not None else 1
        for iteration in range(start_iteration, self.contract.maximum_iterations + 1):
            if self._tool_calls >= self.contract.maximum_tool_calls:
                return self._terminal(LoopDecision.STOP, "tool-call-budget-exhausted")
            reconciled = self._checkpoint is not None and self._checkpoint.phase is RunPhase.RECONCILED_EFFECT
            if reconciled:
                intent = self._checkpoint.pending_intent
                observation = self._checkpoint.reconciled_observation
                assert intent is not None and observation is not None  # checkpoint invariant
                execution_event_hash = self.ledger.append(
                    "execution.reconciled",
                    {
                        "run_id": self.run_id,
                        "iteration": iteration,
                        "intent_digest": intent.intent_digest,
                        "executor_id": self._checkpoint.executor_id,
                        "success": observation.success,
                        "exit_code": observation.exit_code,
                        "artifact_digests": dict(observation.artifact_digests),
                    },
                )
            elif self._checkpoint is not None and self._checkpoint.phase in {
                RunPhase.PENDING_AUTHORIZATION,
                RunPhase.AWAITING_APPROVAL,
            }:
                intent = self._checkpoint.pending_intent
                assert intent is not None  # checkpoint invariant
                self.ledger.append(
                    "intent.resumed",
                    {"run_id": self.run_id, "iteration": iteration, "intent_digest": intent.intent_digest},
                )
            else:
                intent = self._propose()
                self.ledger.append(
                    "intent.proposed",
                    {
                        "run_id": self.run_id,
                        "iteration": iteration,
                        "intent_digest": intent.intent_digest,
                        "tool": intent.tool,
                        "effect": intent.effect.value,
                        "target": intent.target,
                        "provenance": [value.value for value in intent.provenance],
                    },
                )
                self._checkpoint_pending_authorization(iteration, intent)
            if not reconciled:
                try:
                    capability = self.gate.authorize(
                        intent,
                        executor_id=self.executor.executor_id,
                        approvals=approvals,
                    )
                except PolicyDenied as exc:
                    reason = str(exc)
                    self._history.append({"intent": intent.intent_digest, "failure": reason})
                    self.ledger.append(
                        "intent.denied",
                        {"run_id": self.run_id, "intent_digest": intent.intent_digest, "reason": reason},
                    )
                    if "requires explicit approval" in reason or "this action requires explicit approval" in reason:
                        return self._await_approval(iteration, intent, reason)
                    return self._terminal(LoopDecision.ESCALATE, "policy-denied")

                self._tool_calls += 1
                # Persist before the effect starts. A process death after this
                # transition is never retried by the controller; the executor's
                # idempotency/supervisor record must be reconciled first.
                self._checkpoint_pending_effect(iteration, intent)
                self.ledger.append(
                    "execution.started",
                    {"run_id": self.run_id, "iteration": iteration, "intent_digest": intent.intent_digest},
                )
                binder = getattr(self.executor, "bind_run", None)
                if callable(binder):
                    binder(self.run_id, self.contract.contract_digest)
                observation = self.executor.execute(intent, capability)
                execution_event_hash = self.ledger.append(
                    "execution.observed",
                    {
                        "run_id": self.run_id,
                        "intent_digest": intent.intent_digest,
                        "capability_id": capability.capability_id,
                        "executor_id": capability.executor_id,
                        "success": observation.success,
                        "exit_code": observation.exit_code,
                        "artifact_digests": dict(observation.artifact_digests),
                        "metadata": dict(observation.metadata),
                    },
                )
            if self.evaluation_orchestrator is not None:
                try:
                    bundle = self.evaluation_orchestrator.evaluate(
                        run_id=self.run_id,
                        contract=self.contract,
                        intent=intent,
                        observation=observation,
                    )
                    existing_receipts = observation.metadata.get("evaluator_receipts", {})
                    merged_receipts = (
                        dict(existing_receipts) if isinstance(existing_receipts, Mapping) else {}
                    )
                    merged_receipts.update(bundle.evaluator_receipts)
                    observation = replace(
                        observation,
                        metadata={
                            **observation.metadata,
                            "evaluator_receipts": merged_receipts,
                            "workspace_snapshot_digest": bundle.workspace_snapshot.workspace_snapshot_digest,
                            "workspace_snapshot_schema": bundle.workspace_snapshot.schema_version,
                            "workspace_exclusion_policy_digest": bundle.workspace_snapshot.exclusion_policy_digest,
                        },
                    )
                    self.ledger.append(
                        "evaluation.completed",
                        {
                            "run_id": self.run_id,
                            "intent_digest": intent.intent_digest,
                            "workspace_snapshot_digest": bundle.workspace_snapshot.workspace_snapshot_digest,
                            "receipt_types": sorted(bundle.evaluator_receipts),
                        },
                    )
                except Exception as exc:
                    self.ledger.append(
                        "evaluation.unavailable",
                        {
                            "run_id": self.run_id,
                            "intent_digest": intent.intent_digest,
                            "error_type": type(exc).__name__,
                        },
                    )
            try:
                report = self.verifier.verify(
                    self.contract,
                    observation,
                    run_id=self.run_id,
                    intent=intent,
                )
            except Exception as exc:
                report = VerificationReport(
                    CheckStatus.INCONCLUSIVE,
                    CheckStatus.INCONCLUSIVE,
                    CheckStatus.INCONCLUSIVE,
                    CheckStatus.INCONCLUSIVE,
                    (
                        CheckResult(
                            "verification-runtime",
                            CheckStatus.INCONCLUSIVE,
                            {"error_type": type(exc).__name__},
                            "verification runtime failed closed",
                        ),
                    ),
                )
            # Probes protect every action that could contribute to final
            # completion, not just reports that already satisfy every task
            # criterion. This closes the multi-step acceptance bypass.
            if (
                (report.accepted or self._safe_for_criterion_progress(observation, report))
                and self.probe_runner is not None
            ):
                preaccept_probes = self._run_probes(
                    intent,
                    observation,
                    report,
                    "pre-accept",
                    force=True,
                )
                if preaccept_probes is not None:
                    report = self._merge_probe_report(report, preaccept_probes)
            self._evidence.append(intent=intent, observation=observation, report=report)
            verification_event_hash = self.ledger.append(
                "verification.completed",
                {
                    "run_id": self.run_id,
                    "intent_digest": intent.intent_digest,
                    "accepted": report.accepted,
                    "correctness": report.correctness.value,
                    "policy": report.policy.value,
                    "evidence": report.evidence.value,
                    "quality": report.quality.value,
                    "checks": [asdict(check) for check in report.checks],
                },
            )
            diagnostic = self._record_shadow_diagnostic(intent, observation, report)
            # An action can safely contribute one task criterion without also
            # producing every other whole-task receipt.  Missing criteria are
            # progress, not an action failure. Hard verifier failures still
            # follow the repair path below.
            if (
                not report.accepted
                and self.final_verifier is not None
                and self._safe_for_criterion_progress(observation, report)
            ):
                final_check, _final_event_hash = self._verify_final_goal(report)
                if final_check.status is CheckStatus.PASS:
                    self._commit_verified_memory(
                        report=report,
                        final_check=final_check,
                        evidence_refs=(execution_event_hash, verification_event_hash, _final_event_hash),
                    )
                    return self._terminal(LoopDecision.ACCEPT, "verified-success")
                self._history.append(
                    {
                        "intent": intent.intent_digest,
                        "task_progress": True,
                        "final_status": final_check.status.value,
                        "criterion_statuses": dict(final_check.evidence.get("condition_statuses", {})),
                    }
                )
                self.ledger.append(
                    "criterion.progressed",
                    {
                        "run_id": self.run_id,
                        "intent_digest": intent.intent_digest,
                        "final_status": final_check.status.value,
                    },
                )
                self._checkpoint_ready(iteration + 1)
                continue
            if report.accepted:
                final_check, final_event_hash = self._verify_final_goal(report)
                if final_check.status is CheckStatus.PASS:
                    self._commit_verified_memory(
                        report=report,
                        final_check=final_check,
                        evidence_refs=(
                            execution_event_hash,
                            verification_event_hash,
                            final_event_hash,
                        ),
                    )
                    return self._terminal(LoopDecision.ACCEPT, "verified-success")

                failure_key = ("final-goal", final_check.status.value)
                self.ledger.append(
                    "repair.directive",
                    {
                        "run_id": self.run_id,
                        "intent_digest": intent.intent_digest,
                        "decision": LoopDecision.REPLAN.value,
                        "stage": "replan",
                        "reason": "final-goal-not-satisfied",
                        "advisory_stage": None,
                        "evidence_gaps": (),
                    },
                )
                if failure_key in self._seen_failures:
                    return self._terminal(LoopDecision.ESCALATE, "repeated-final-goal-failure")
                self._seen_failures.add(failure_key)
                self._history.append(
                    {
                        "intent": intent.intent_digest,
                        "final_goal": final_check.status.value,
                        "final_goal_message": final_check.message[:240],
                    }
                )
                self._checkpoint_ready(iteration + 1)
                continue

            directive = self.repair_controller.direct(report, diagnostic)
            self.ledger.append(
                "repair.directive",
                {
                    "run_id": self.run_id,
                    "intent_digest": intent.intent_digest,
                    "decision": directive.decision.value,
                    "stage": directive.stage,
                    "reason": directive.reason,
                    "advisory_stage": directive.advisory_stage,
                    "evidence_gaps": directive.evidence_gaps,
                },
            )
            if directive.decision is LoopDecision.ESCALATE:
                return self._terminal(LoopDecision.ESCALATE, directive.reason)
            probe_report = self._run_probes(intent, observation, report, directive.stage)
            failure_key = (self._failure_signature(intent), directive.stage)
            if failure_key in self._seen_failures:
                return self._terminal(LoopDecision.ESCALATE, "repeated-identical-failure")
            self._seen_failures.add(failure_key)
            self._history.append(
                {
                    "intent": intent.intent_digest,
                    "repair_stage": directive.stage,
                    "repair_reason": directive.reason,
                    "advisory_stage": directive.advisory_stage,
                    "evidence_gaps": directive.evidence_gaps,
                    "checks": [(check.name, check.status.value) for check in report.checks],
                    "probes": (
                        [(check.name, check.status.value) for check in probe_report.results]
                        if probe_report is not None
                        else []
                    ),
                }
            )
            self._checkpoint_ready(iteration + 1)
        return self._terminal(LoopDecision.STOP, "iteration-budget-exhausted")

    def _restore_or_start(self) -> LoopDecision | None:
        """Restore a safe checkpoint, never a potentially executed effect."""

        if self.state_store is None:
            self.ledger.append(
                "run.started",
                {"run_id": self.run_id, "contract_digest": self.contract.contract_digest},
            )
            return None
        checkpoint = self.state_store.load(self.run_id)
        if checkpoint is None:
            checkpoint = RunCheckpoint(
                run_id=self.run_id,
                contract_digest=self.contract.contract_digest,
                phase=RunPhase.READY,
                next_iteration=1,
                tool_calls=0,
                history=(),
                seen_failures=(),
                evidence=self._evidence.snapshot(),
            )
            self._checkpoint = self.state_store.save(checkpoint)
            self.ledger.append(
                "run.started",
                {"run_id": self.run_id, "contract_digest": self.contract.contract_digest},
            )
            return None
        if checkpoint.contract_digest != self.contract.contract_digest:
            raise ValueError("persisted run state belongs to a different task contract")
        if checkpoint.next_iteration > self.contract.maximum_iterations + 1:
            raise ValueError("persisted run state exceeds the contract iteration budget")
        self._checkpoint = checkpoint
        self._tool_calls = checkpoint.tool_calls
        self._history = list(checkpoint.history)
        self._seen_failures = set(checkpoint.seen_failures)
        self._evidence = EvidenceAccumulator.from_snapshot(checkpoint.evidence)
        if checkpoint.phase is RunPhase.TERMINAL:
            try:
                return LoopDecision(checkpoint.terminal_decision or "")
            except ValueError as exc:
                raise ValueError("persisted terminal decision is invalid") from exc
        if checkpoint.phase is RunPhase.PENDING_EFFECT:
            self._save_checkpoint(
                phase=RunPhase.RECONCILIATION_REQUIRED,
                next_iteration=checkpoint.next_iteration,
                pending_intent=checkpoint.pending_intent,
                executor_id=checkpoint.executor_id,
            )
            self.ledger.append(
                "run.resume.blocked",
                {
                    "run_id": self.run_id,
                    "reason": "pending-effect-requires-reconciliation",
                    "intent_digest": checkpoint.pending_intent.intent_digest if checkpoint.pending_intent else "",
                },
            )
            return LoopDecision.WAITING
        if checkpoint.phase is RunPhase.RECONCILIATION_REQUIRED:
            self.ledger.append(
                "run.resume.blocked",
                {
                    "run_id": self.run_id,
                    "reason": "reconciliation-required",
                    "intent_digest": checkpoint.pending_intent.intent_digest if checkpoint.pending_intent else "",
                },
            )
            return LoopDecision.WAITING
        self.ledger.append(
            "run.resumed",
            {
                "run_id": self.run_id,
                "next_iteration": checkpoint.next_iteration,
                "phase": checkpoint.phase.value,
            },
        )
        return None

    def _save_checkpoint(
        self,
        *,
        phase: RunPhase,
        next_iteration: int,
        pending_intent: ActionIntent | None = None,
        executor_id: str = "",
        reconciled_observation: ExecutionObservation | None = None,
        terminal_decision: LoopDecision | None = None,
        terminal_reason: str | None = None,
    ) -> None:
        if self.state_store is None:
            return
        if self._checkpoint is None:  # pragma: no cover - internal lifecycle invariant
            raise RuntimeError("run checkpoint was not initialized")
        checkpoint = RunCheckpoint(
            run_id=self.run_id,
            contract_digest=self.contract.contract_digest,
            phase=phase,
            next_iteration=next_iteration,
            tool_calls=self._tool_calls,
            history=tuple(self._history),
            seen_failures=tuple(sorted(self._seen_failures)),
            evidence=self._evidence.snapshot(),
            pending_intent=pending_intent,
            executor_id=executor_id,
            reconciled_observation=reconciled_observation,
            terminal_decision=terminal_decision.value if terminal_decision is not None else None,
            terminal_reason=terminal_reason,
            revision=self._checkpoint.revision,
        )
        self._checkpoint = self.state_store.save(checkpoint)

    def _checkpoint_pending_authorization(self, iteration: int, intent: ActionIntent) -> None:
        self._save_checkpoint(
            phase=RunPhase.PENDING_AUTHORIZATION,
            next_iteration=iteration,
            pending_intent=intent,
            executor_id=self.executor.executor_id,
        )

    def _checkpoint_pending_effect(self, iteration: int, intent: ActionIntent) -> None:
        self._save_checkpoint(
            phase=RunPhase.PENDING_EFFECT,
            next_iteration=iteration,
            pending_intent=intent,
            executor_id=self.executor.executor_id,
        )

    def _checkpoint_ready(self, next_iteration: int) -> None:
        self._save_checkpoint(phase=RunPhase.READY, next_iteration=next_iteration)

    def _await_approval(self, iteration: int, intent: ActionIntent, reason: str) -> LoopDecision:
        """Persist the exact action as resumable reviewer work, not terminal state."""

        self._save_checkpoint(
            phase=RunPhase.AWAITING_APPROVAL,
            next_iteration=iteration,
            pending_intent=intent,
            executor_id=self.executor.executor_id,
        )
        self.ledger.append(
            "run.awaiting-approval",
            {
                "run_id": self.run_id,
                "iteration": iteration,
                "intent_digest": intent.intent_digest,
                "executor_id": self.executor.executor_id,
                "reason": reason,
            },
        )
        return LoopDecision.WAITING

    def resume_with_approval(self, approval: Approval | SignedApprovalReceipt) -> LoopDecision:
        """Resume only the exact persisted intent after an external approval."""

        if self.state_store is None:
            raise RuntimeError("approval resume requires a durable run-state store")
        checkpoint = self.state_store.load(self.run_id)
        if checkpoint is None or checkpoint.phase is not RunPhase.AWAITING_APPROVAL:
            raise RuntimeError("run is not awaiting approval")
        return self.run((approval,))

    def reconcile_effect(self) -> LoopDecision:
        """Ask the deployment-owned reconciler for a signed effect outcome."""

        if self.state_store is None or self.effect_reconciler is None:
            raise RuntimeError("effect reconciliation requires durable state and a trusted reconciler")
        checkpoint = self.state_store.load(self.run_id)
        if checkpoint is None or checkpoint.phase is not RunPhase.RECONCILIATION_REQUIRED:
            raise RuntimeError("run does not require effect reconciliation")
        intent = checkpoint.pending_intent
        assert intent is not None  # checkpoint invariant
        observation = self.effect_reconciler.reconcile(
            run_id=self.run_id,
            contract=self.contract,
            intent=intent,
            executor_id=checkpoint.executor_id,
        )
        self._checkpoint = checkpoint
        self._save_checkpoint(
            phase=RunPhase.RECONCILED_EFFECT,
            next_iteration=checkpoint.next_iteration,
            pending_intent=intent,
            executor_id=checkpoint.executor_id,
            reconciled_observation=observation,
        )
        self.ledger.append(
            "effect.reconciled",
            {
                "run_id": self.run_id,
                "intent_digest": intent.intent_digest,
                "executor_id": checkpoint.executor_id,
                "success": observation.success,
                "exit_code": observation.exit_code,
            },
        )
        return self.run()

    def cancel_run(self, *, reason: str = "cancelled-by-requester") -> LoopDecision:
        """Explicitly stop a waiting/reconciliation run without replaying it."""

        if self.state_store is not None:
            checkpoint = self.state_store.load(self.run_id)
            if checkpoint is not None:
                self._checkpoint = checkpoint
                self._tool_calls = checkpoint.tool_calls
                self._history = list(checkpoint.history)
                self._seen_failures = set(checkpoint.seen_failures)
                self._evidence = EvidenceAccumulator.from_snapshot(checkpoint.evidence)
        return self._terminal(LoopDecision.STOP, reason)

    def _safe_for_criterion_progress(
        self, observation: ExecutionObservation, report: VerificationReport
    ) -> bool:
        """Separate mandatory action safety from incomplete task criteria."""

        # Older development contracts did not contain a reviewed safety set.
        # Production contracts do, and only an all-PASS ActionSafetyReport may
        # contribute evidence there.
        if self.contract.action_safety_checks:
            return ActionSafetyReport.from_report(self.contract, report).accepted
        structural = next((check for check in report.checks if check.name == "structural"), None)
        return (
            observation.success
            and not any(check.status is CheckStatus.FAIL for check in report.checks)
            and (structural is None or structural.status is CheckStatus.PASS)
        )

    def _propose(self) -> ActionIntent:
        history = tuple(self._history)
        context: ContextPackage | None = None
        if self.context_provider is not None:
            context = self.context_provider.build(contract=self.contract, history=history)
            if context.contract_digest != self.contract.contract_digest:
                raise ValueError("context package belongs to a different task contract")
            self.ledger.append(
                "context.packaged",
                {
                    "run_id": self.run_id,
                    "contract_digest": context.contract_digest,
                    "environment_digest": context.environment_digest,
                    "trusted_sources": [item.source_id for item in context.trusted_items],
                    "untrusted_sources": [item.source_id for item in context.untrusted_items],
                    "truncated_sources": context.truncated_source_ids,
                },
            )
        contextual = getattr(self.planner, "propose_with_context", None)
        if context is not None and callable(contextual):
            intent = contextual(contract=self.contract, history=history, context=context)
        else:
            intent = self.planner.propose(contract=self.contract, history=history)
        if context is not None:
            # A controller cannot inspect a model's attention. Conservatively
            # mark every action argument proposed with a package containing
            # untrusted material as tainted, then preserve the exact package
            # inputs as nodes in each argument's provenance DAG.
            provenance = tuple(
                sorted(set(intent.provenance).union(context.provenance), key=lambda value: value.value)
            )
            argument_provenance = {
                name: tuple(
                    sorted(
                        set(intent.provenance_for_argument(name)).union(context.provenance),
                        key=lambda value: value.value,
                    )
                )
                for name in intent.arguments
            }
            intent = replace(intent, provenance=provenance, argument_provenance=argument_provenance)
        if self.contract.require_argument_provenance:
            return self._bind_argument_provenance(intent, context)
        return intent

    @staticmethod
    def _bind_argument_provenance(
        intent: ActionIntent, context: ContextPackage | None
    ) -> ActionIntent:
        """Bind each submitted value to a causal, non-content-bearing DAG.

        Planner-provided graphs are deliberately not trusted. The controller
        creates fresh source nodes from its own context package and a mandatory
        model-output derivation node for every value. This conservatively taints
        values when untrusted context was in scope without pretending to know
        the model's attention pattern.
        """

        context_nodes: list[ArgumentProvenanceNode] = []
        if context is not None:
            for index, item in enumerate((*context.trusted_items, *context.untrusted_items)):
                provenance = {
                    ContextTrust.USER: Provenance.USER,
                    ContextTrust.TRUSTED_REPOSITORY: Provenance.TRUSTED_REPOSITORY,
                    ContextTrust.VERIFIED_MEMORY: Provenance.VERIFIED_MEMORY,
                    ContextTrust.UNTRUSTED: Provenance.UNTRUSTED_RETRIEVAL,
                    ContextTrust.TRUSTED_SYSTEM: Provenance.TOOL_OUTPUT,
                }[item.trust]
                context_nodes.append(
                    ArgumentProvenanceNode(
                        node_id=f"context:{index}:{item.content_digest[:16]}",
                        provenance=provenance,
                        source_id=item.source_id,
                        content_digest=item.content_digest,
                    )
                )

        graphs: dict[str, ArgumentProvenance] = {}
        for name, value in intent.arguments.items():
            nodes = list(context_nodes)
            # A pre-existing untrusted label may only make the action more
            # restrictive. It cannot supply a trusted source node.
            if (
                Provenance.UNTRUSTED_RETRIEVAL in intent.provenance
                or Provenance.UNTRUSTED_RETRIEVAL in intent.argument_provenance.get(name, ())
            ):
                nodes.append(
                    ArgumentProvenanceNode(
                        node_id="declared-untrusted-input",
                        provenance=Provenance.UNTRUSTED_RETRIEVAL,
                        source_id="untrusted-planner-declaration",
                        content_digest=digest({"argument": name, "value": value}),
                    )
                )
            if not nodes:
                nodes.append(
                    ArgumentProvenanceNode(
                        node_id="planner-output-root",
                        provenance=Provenance.TOOL_OUTPUT,
                        source_id="planner-output",
                        content_digest=digest({"argument": name, "value": value}),
                    )
                )
            parents = {
                parent for node in nodes for parent in node.parent_ids
            }
            terminals = tuple(sorted(node.node_id for node in nodes if node.node_id not in parents))
            derived_id = f"controller-derived:{digest({'argument': name, 'value': value})[:16]}"
            nodes.append(
                ArgumentProvenanceNode(
                    node_id=derived_id,
                    provenance=Provenance.TOOL_OUTPUT,
                    source_id="controller-planner-boundary",
                    content_digest=digest(value),
                    parent_ids=terminals,
                )
            )
            graphs[name] = ArgumentProvenance(digest(value), tuple(nodes))
        return replace(intent, argument_provenance_graph=graphs)

    def _record_shadow_diagnostic(
        self, intent: ActionIntent, observation: ExecutionObservation, report: VerificationReport
    ) -> object | None:
        if self.shadow_verifier is None:
            return None
        try:
            diagnostic = self.shadow_verifier.diagnose(
                contract=self.contract,
                intent=intent,
                observation=observation,
                hard_report=report,
            )
            self.ledger.append(
                "neural.shadow.completed",
                {
                    "run_id": self.run_id,
                    "intent_digest": intent.intent_digest,
                    "diagnostic": diagnostic.ledger_payload(),
                },
            )
            return diagnostic
        except Exception as exc:
            self.ledger.append(
                "neural.shadow.unavailable",
                {
                    "run_id": self.run_id,
                    "intent_digest": intent.intent_digest,
                    "error_type": type(exc).__name__,
                },
            )
            return None

    def _verify_final_goal(
        self, report: VerificationReport
    ) -> tuple[CheckResult, str]:
        if self.final_verifier is None:
            result = CheckResult(
                "final-goal",
                CheckStatus.INCONCLUSIVE,
                {"error_type": "MissingFinalVerifier"},
                "an explicit final verifier is required before acceptance",
            )
        else:
            try:
                result = self.final_verifier.verify(
                    contract=self.contract,
                    action_report=report,
                    history=tuple(self._history),
                    evidence=self._evidence.snapshot(),
                )
            except Exception as exc:
                result = CheckResult(
                    "final-goal",
                    CheckStatus.INCONCLUSIVE,
                    {"error_type": type(exc).__name__},
                    "final verifier was unavailable",
                )
        if result.name != "final-goal":
            result = CheckResult(
                "final-goal",
                CheckStatus.INCONCLUSIVE,
                {"error_type": "InvalidFinalCheckName"},
                "final verifier returned an invalid check",
            )
        event_hash = self.ledger.append(
            "final-goal.completed",
            {
                "run_id": self.run_id,
                "status": result.status.value,
                "evidence": dict(result.evidence),
                "message": result.message[:500],
            },
        )
        return result, event_hash

    def _commit_verified_memory(
        self,
        *,
        report: VerificationReport,
        final_check: CheckResult,
        evidence_refs: tuple[str, ...],
    ) -> None:
        if self.memory_committer is None and self.memory_candidate_producer is None:
            return
        if self.memory_committer is None or self.memory_candidate_producer is None:
            self.ledger.append(
                "memory.commit.skipped",
                {"run_id": self.run_id, "reason": "memory-committer-not-fully-configured"},
            )
            return
        try:
            snapshot = self._evidence.snapshot()
            completion_report = TaskCompletionReport(
                action_reports=tuple(action.report for action in snapshot.actions),
                final_check=final_check,
                final_workspace_digest=snapshot.final_source_state_digest,
            )
            complete_evidence_refs = tuple(
                dict.fromkeys((*self.ledger.event_hashes_for_run(self.run_id), *evidence_refs))
            )
            candidate = self.memory_candidate_producer.propose(
                contract=self.contract,
                history=tuple(self._history),
                report=completion_report,
                final_check=final_check,
                available_evidence_refs=complete_evidence_refs,
            )
            if candidate is None:
                self.ledger.append(
                    "memory.commit.skipped",
                    {"run_id": self.run_id, "reason": "no-memory-candidate"},
                )
                return
            record = self.memory_committer.commit(
                candidate,
                report=completion_report,
                final_check=final_check,
                source_run_id=self.run_id,
                available_evidence_refs=complete_evidence_refs,
            )
            self.ledger.append(
                "memory.committed",
                {
                    "run_id": self.run_id,
                    "memory_id": record.memory_id,
                    "memory_ledger_event_hash": record.ledger_event_hash,
                },
            )
        except Exception as exc:
            # Memory is a post-success side effect.  It cannot make a verified
            # task fail, nor can an extractor error expose its raw content.
            self.ledger.append(
                "memory.commit.rejected",
                {"run_id": self.run_id, "error_type": type(exc).__name__},
            )

    def _run_probes(
        self,
        intent: ActionIntent,
        observation: ExecutionObservation,
        report: VerificationReport,
        stage: str,
        *,
        force: bool = False,
    ) -> ProbeReport | None:
        if stage not in {"probe", "pre-accept"} or self.probe_runner is None:
            return None
        probe_report = self.probe_runner.run(
            contract=self.contract,
            intent=intent,
            observation=observation,
            hard_report=report,
            force=force,
        )
        self.ledger.append(
            "probe.completed",
            {
                "run_id": self.run_id,
                "intent_digest": intent.intent_digest,
                "status": probe_report.status.value,
                "checks": [
                    {
                        "name": check.name,
                        "status": check.status.value,
                        "evidence": dict(check.evidence),
                    }
                    for check in probe_report.results
                ],
            },
        )
        return probe_report

    @staticmethod
    def _merge_probe_report(report: VerificationReport, probes: ProbeReport) -> VerificationReport:
        evidence = report.evidence
        if probes.status is CheckStatus.FAIL:
            evidence = CheckStatus.FAIL
        elif probes.status is CheckStatus.INCONCLUSIVE and evidence is CheckStatus.PASS:
            evidence = CheckStatus.INCONCLUSIVE
        return VerificationReport(
            report.correctness,
            report.policy,
            evidence,
            report.quality,
            (*report.checks, *probes.results),
        )

    @staticmethod
    def _failure_signature(intent: ActionIntent) -> str:
        """Stable retry key that deliberately excludes one-shot idempotency IDs."""

        return digest(
            {
                "tool": intent.tool,
                "effect": intent.effect.value,
                "target": intent.target,
                "arguments": dict(intent.arguments),
                "provenance": [value.value for value in intent.provenance],
                "contract_id": intent.contract_id,
                "contract_version": intent.contract_version,
            }
        )

    def _terminal(self, decision: LoopDecision, reason: str) -> LoopDecision:
        # Terminal state has no side effect, so persist it before its audit
        # event. A restart then returns the same decision rather than entering
        # the planner again if ledger publication is briefly unavailable.
        self._save_checkpoint(
            phase=RunPhase.TERMINAL,
            next_iteration=(self._checkpoint.next_iteration if self._checkpoint is not None else 1),
            terminal_decision=decision,
            terminal_reason=reason,
        )
        self.ledger.append(
            "run.terminal",
            {
                "run_id": self.run_id,
                "decision": decision.value,
                "reason": reason,
                "tool_calls": self._tool_calls,
                "occurred_at": datetime.now(UTC).isoformat(),
            },
        )
        return decision
