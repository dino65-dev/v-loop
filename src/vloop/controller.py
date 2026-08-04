"""Verified adaptive-loop state machine."""

from __future__ import annotations

from dataclasses import asdict, replace
from datetime import UTC, datetime
from typing import Iterable, Mapping, Protocol
from uuid import uuid4

from .canonical import digest
from .attestations import CompletionClient, CompletionResult, CompletionVerifier, DevelopmentCompletionFabric
from .completion import ActionSafetyReport, EvidenceAccumulator, FinalVerifier, TaskCompletionReport
from .context import ContextPackage, ContextTrust
from .evaluation import ProtectedEvaluationOrchestrator
from .executor import Executor
from .graph import EvidenceGraph, GraphManifest, build_evidence_graph, compile_control_graph
from .graph_events import CausalEvent, GraphEventStore, SemanticEvidenceGraph, build_semantic_evidence_graph
from .graph_runtime import DurableGraphScheduler, ScheduledReservation
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
    PreparedExecution,
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
        prepared_execution: PreparedExecution,
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
        graph_scheduler: DurableGraphScheduler | None = None,
        completion_client: CompletionClient | None = None,
        completion_verifier: CompletionVerifier | None = None,
        allow_unsafe_development_completions: bool = False,
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
        planned_receipts = (
            {plan.check_name: plan.receipt_type for plan in evaluation_orchestrator.plans}
            if evaluation_orchestrator is not None
            else {}
        )
        self.graph_manifest: GraphManifest = compile_control_graph(contract, evaluator_receipts=planned_receipts)
        self.graph_digest = self.graph_manifest.graph_digest
        self.run_id = run_id or str(uuid4())
        graph_database = getattr(state_store, "path", ":memory:")
        self.graph_event_store = GraphEventStore(graph_database)
        self._unsafe_development_completions = allow_unsafe_development_completions
        if allow_unsafe_development_completions:
            if completion_client is not None or completion_verifier is not None:
                raise ValueError("development completion mode cannot be combined with external completion trust")
            roles = {
                node.node_id: node.metadata.get("producer_role", "controller")
                for node in self.graph_manifest.nodes
            }
            fabric = DevelopmentCompletionFabric(graph_digest=self.graph_digest, template_roles=roles)
            completion_client, completion_verifier = fabric, fabric.verifier
        if graph_scheduler is None:
            if completion_client is None or completion_verifier is None:
                raise RuntimeError(
                    "evidence-native execution requires an external completion client and verifier; "
                    "local tests must explicitly opt in to unsafe development completions"
                )
            graph_scheduler = DurableGraphScheduler(
                self.graph_manifest,
                self.graph_event_store,
                graph_database,
                completion_verifier=completion_verifier,
            )
        if completion_client is None:
            raise RuntimeError("graph scheduler requires an external node-completion client")
        self.completion_client = completion_client
        self.graph_scheduler = graph_scheduler
        self._evidence = EvidenceAccumulator(self.run_id)
        self._history: list[dict] = []
        self._seen_failures: set[tuple[str, str]] = set()
        self._tool_calls = 0
        self._checkpoint: RunCheckpoint | None = None

    def evidence_graph(self) -> EvidenceGraph:
        """Return a hash-addressed graph projection of this run's ledger evidence."""

        return build_evidence_graph(self.ledger.events(), run_id=self.run_id)

    def causal_evidence_graph(self) -> SemanticEvidenceGraph:
        """Return the semantic causal graph, not a projection of ledger ordering."""

        return build_semantic_evidence_graph(self.graph_event_store.events(run_id=self.run_id), run_id=self.run_id)

    def run(self, approvals: Iterable[Approval | SignedApprovalReceipt] = ()) -> LoopDecision:
        resumed_terminal = self._restore_or_start()
        if resumed_terminal is not None:
            return resumed_terminal
        start_iteration = self._checkpoint.next_iteration if self._checkpoint is not None else 1
        for iteration in range(start_iteration, self.contract.maximum_iterations + 1):
            self._graph_begin_iteration(iteration)
            if self._tool_calls >= self.contract.maximum_tool_calls:
                return self._terminal(LoopDecision.STOP, "tool-call-budget-exhausted")
            reconciled = self._checkpoint is not None and self._checkpoint.phase is RunPhase.RECONCILED_EFFECT
            if reconciled:
                intent = self._checkpoint.pending_intent
                observation = self._checkpoint.reconciled_observation
                prepared_execution = self._checkpoint.prepared_execution
                assert intent is not None and observation is not None and prepared_execution is not None  # checkpoint invariant
                execution_event_hash = self.ledger.append(
                    "execution.reconciled",
                    {
                        "run_id": self.run_id,
                        "iteration": iteration,
                        "intent_digest": intent.intent_digest,
                        "executor_id": self._checkpoint.executor_id,
                        "operation_id": prepared_execution.operation_id,
                        "request_digest": prepared_execution.request_digest,
                        "graph_digest": prepared_execution.graph_digest,
                        "graph_node_id": prepared_execution.graph_node_id,
                        "success": observation.success,
                        "exit_code": observation.exit_code,
                        "artifact_digests": dict(observation.artifact_digests),
                    },
                )
                if observation.success:
                    self._graph_complete(
                        iteration,
                        "executor.result",
                        artifact_digest=digest(observation.artifact_digests),
                        facts={"success": "true", "reconciled": "true", "operation_id": prepared_execution.operation_id},
                        input_artifacts=observation.artifact_digests,
                    )
                    self._graph_complete(
                        iteration,
                        "artifact.manifest",
                        artifact_digest=digest(observation.artifact_digests),
                        facts={"success": "true", "reconciled": "true"},
                        input_artifacts=observation.artifact_digests,
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
                matched_rule_index = self._matching_action_rule_index(intent)
                self._graph_advance(
                    iteration,
                    "action.intent",
                    "intent.proposed",
                    {"intent_digest": intent.intent_digest, "rule_index": str(matched_rule_index)},
                )
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
                        graph_digest=self.graph_digest,
                        graph_node_id="capability.execute",
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

                self._graph_authorise_action_rule(iteration, intent)
                self._graph_complete(
                    iteration,
                    "capability.execute",
                    artifact_digest=digest({"capability_id": capability.capability_id, "intent_digest": intent.intent_digest}),
                    facts={"capability_id": capability.capability_id, "intent_digest": intent.intent_digest},
                    authority_refs=(capability.capability_id,),
                )

                self._tool_calls += 1
                # Persist before the effect starts. A process death after this
                # transition is never retried by the controller; the executor's
                # idempotency/supervisor record must be reconciled first.
                prepared_execution = self._prepare_execution(iteration, intent)
                self._graph_complete(
                    iteration,
                    "operation.prepared",
                    artifact_digest=prepared_execution.request_digest,
                    facts={"operation_id": prepared_execution.operation_id},
                    authority_refs=(capability.capability_id,),
                )
                self._checkpoint_pending_effect(iteration, intent, prepared_execution)
                self.ledger.append(
                    "execution.started",
                    {
                        "run_id": self.run_id,
                        "iteration": iteration,
                        "intent_digest": intent.intent_digest,
                        "operation_id": prepared_execution.operation_id,
                        "remote_job_id": prepared_execution.remote_job_id,
                        "request_digest": prepared_execution.request_digest,
                        "graph_digest": prepared_execution.graph_digest,
                        "graph_node_id": prepared_execution.graph_node_id,
                    },
                )
                binder = getattr(self.executor, "bind_run", None)
                if callable(binder):
                    binder(self.run_id, self.contract.contract_digest)
                self._graph_advance(
                    iteration,
                    "executor.dispatch",
                    "execution.dispatched",
                    {"operation_id": prepared_execution.operation_id},
                    authorization_ref=capability.capability_id,
                )
                execute_prepared = getattr(self.executor, "execute_prepared", None)
                observation = (
                    execute_prepared(intent, capability, prepared_execution)
                    if callable(execute_prepared)
                    else self.executor.execute(intent, capability)
                )
                self._graph_complete(
                    iteration,
                    "executor.result",
                    artifact_digest=digest(observation.artifact_digests),
                    result=CompletionResult.SUCCEEDED if observation.success else CompletionResult.FAILED,
                    facts={"success": str(observation.success).lower(), "operation_id": prepared_execution.operation_id},
                    input_artifacts=observation.artifact_digests,
                    authority_refs=(capability.capability_id,),
                )
                if observation.success:
                    self._graph_complete(
                        iteration,
                        "artifact.manifest",
                        artifact_digest=digest(observation.artifact_digests),
                        facts={"success": "true"},
                        input_artifacts=observation.artifact_digests,
                        authority_refs=(capability.capability_id,),
                    )
                execution_event_hash = self.ledger.append(
                    "execution.observed",
                    {
                        "run_id": self.run_id,
                        "intent_digest": intent.intent_digest,
                        "capability_id": capability.capability_id,
                        "executor_id": capability.executor_id,
                        "graph_digest": capability.graph_digest,
                        "graph_node_id": capability.graph_node_id,
                        "operation_id": prepared_execution.operation_id,
                        "request_digest": prepared_execution.request_digest,
                        "prepared_graph_digest": prepared_execution.graph_digest,
                        "prepared_graph_node_id": prepared_execution.graph_node_id,
                        "success": observation.success,
                        "exit_code": observation.exit_code,
                        "artifact_digests": dict(observation.artifact_digests),
                        "metadata": dict(observation.metadata),
                    },
                )
            if self.evaluation_orchestrator is not None:
                try:
                    if not observation.success:
                        raise ValueError("failed execution has no evaluator-admissible artifact")
                    snapshot = self.evaluation_orchestrator.materialize_snapshot(
                        contract=self.contract, intent=intent, observation=observation
                    )
                    self._graph_complete(
                        iteration,
                        "snapshot.materialized",
                        artifact_digest=snapshot.workspace_snapshot_digest,
                        facts={"success": "true", "workspace_generation": "0"},
                        input_artifacts={"snapshot": snapshot.workspace_snapshot_digest},
                    )
                    evaluator_receipts: dict[str, Mapping] = {}
                    graph_bindings: dict[str, Mapping[str, str]] = {}
                    for plan in self.evaluation_orchestrator.plans:
                        evaluator_node = f"evaluator.{self._graph_node_token(plan.check_name)}"
                        receipt_node = f"receipt.{self._graph_node_token(plan.receipt_type)}"
                        reservation = self._graph_reserve(iteration, evaluator_node)
                        raw_receipt = self.evaluation_orchestrator.evaluate_plan(
                            plan,
                            run_id=self.run_id,
                            contract=self.contract,
                            intent=intent,
                            observation=observation,
                            snapshot=snapshot,
                            graph_digest=self.graph_digest,
                            graph_node_id=evaluator_node,
                            graph_node_instance_id=reservation.node_instance_id,
                        )
                        evaluator_event = self._graph_complete(
                            iteration,
                            evaluator_node,
                            reservation=reservation,
                            artifact_digest=digest(raw_receipt),
                            facts={"receipt_type": plan.receipt_type},
                            input_artifacts=observation.artifact_digests,
                        )
                        receipt_event = self._graph_complete(
                            iteration,
                            receipt_node,
                            artifact_digest=digest(raw_receipt),
                            facts={"receipt_type": plan.receipt_type},
                            evidence_refs=(
                                str(evaluator_event.payload.get("completion_digest", ""))
                                if evaluator_event is not None
                                else "",
                            ),
                        )
                        del receipt_event
                        evaluator_receipts[plan.receipt_type] = raw_receipt
                        graph_bindings[plan.receipt_type] = {
                            "graph_digest": self.graph_digest,
                            "graph_node_id": evaluator_node,
                            "graph_node_instance_id": reservation.node_instance_id,
                        }
                    existing_receipts = observation.metadata.get("evaluator_receipts", {})
                    merged_receipts = (
                        dict(existing_receipts) if isinstance(existing_receipts, Mapping) else {}
                    )
                    merged_receipts.update(evaluator_receipts)
                    observation = replace(
                        observation,
                        metadata={
                            **observation.metadata,
                            "evaluator_receipts": merged_receipts,
                            "workspace_snapshot_digest": snapshot.workspace_snapshot_digest,
                            "workspace_snapshot_schema": snapshot.schema_version,
                            "workspace_exclusion_policy_digest": snapshot.exclusion_policy_digest,
                            "receipt_graph_bindings": graph_bindings,
                        },
                    )
                    self.ledger.append(
                        "evaluation.completed",
                        {
                            "run_id": self.run_id,
                            "intent_digest": intent.intent_digest,
                            "workspace_snapshot_digest": snapshot.workspace_snapshot_digest,
                            "receipt_types": sorted(evaluator_receipts),
                            "graph_digest": self.graph_digest,
                            "graph_node_ids": sorted(binding["graph_node_id"] for binding in graph_bindings.values()),
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
                observation.success
                and (report.accepted or self._safe_for_criterion_progress(observation, report))
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
            # Preserve the diagnostic report in the ledger, but never let a
            # failed side effect enter the completion evidence set.  Otherwise
            # a later final verifier could mistake its passing checks for
            # criterion evidence from a successful action.
            if observation.success:
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
                    self._graph_accept(iteration, final_check, observation)
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
            # A verifier report is evidence about an execution, never a
            # substitute for that execution actually succeeding.  This also
            # protects against an incorrectly implemented custom verifier.
            if report.accepted and observation.success:
                final_check, final_event_hash = self._verify_final_goal(report)
                if final_check.status is CheckStatus.PASS:
                    self._graph_accept(iteration, final_check, observation)
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
                {
                    "run_id": self.run_id,
                    "contract_digest": self.contract.contract_digest,
                    "graph_digest": self.graph_digest,
                },
            )
            return None
        checkpoint = self.state_store.load(self.run_id)
        if checkpoint is None:
            checkpoint = RunCheckpoint(
                run_id=self.run_id,
                contract_digest=self.contract.contract_digest,
                graph_digest=self.graph_digest,
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
                {
                    "run_id": self.run_id,
                    "contract_digest": self.contract.contract_digest,
                    "graph_digest": self.graph_digest,
                },
            )
            return None
        if checkpoint.contract_digest != self.contract.contract_digest:
            raise ValueError("persisted run state belongs to a different task contract")
        if checkpoint.graph_digest and checkpoint.graph_digest != self.graph_digest:
            raise ValueError("persisted run state belongs to a different control graph")
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
                prepared_execution=checkpoint.prepared_execution,
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

    def _graph_begin_iteration(self, iteration: int) -> None:
        """Start the immutable graph path before the controller can propose work."""

        state = self.graph_scheduler.state(run_id=self.run_id, iteration=iteration)
        if state.completed:
            return
        self._graph_advance(iteration, "task.contract", "task.bound", {"contract_digest": self.contract.contract_digest})
        self._graph_complete(
            iteration,
            "principal.contract",
            artifact_digest=self.contract.contract_digest,
            facts={"bound": "true"},
        )
        self._graph_advance(iteration, "snapshot.request", "snapshot.requested", {})

    def _graph_advance(
        self,
        iteration: int,
        template_node_id: str,
        event_type: str,
        payload: Mapping[str, object],
        *,
        output_artifacts: Mapping[str, str] = {},
        authorization_ref: str = "",
        receipt_refs: tuple[str, ...] = (),
    ) -> CausalEvent | None:
        """Advance only a graph-enabled node and retain an explicit causal link."""

        state = self.graph_scheduler.state(run_id=self.run_id, iteration=iteration)
        if template_node_id in state.completed:
            return next(
                (
                    event
                    for event in reversed(self.graph_event_store.events(run_id=self.run_id))
                    if event.template_node_id == template_node_id
                    and event.node_instance_id
                    and event.run_id == self.run_id
                ),
                None,
            )
        existing = self.graph_event_store.events(run_id=self.run_id)
        parent = (existing[-1].event_id,) if existing else ()
        return self.graph_scheduler.advance(
            run_id=self.run_id,
            iteration=iteration,
            template_node_id=template_node_id,
            event_type=event_type,
            payload=payload,
            causal_parents=parent,
            output_artifacts=output_artifacts,
            authorization_ref=authorization_ref,
            receipt_refs=receipt_refs,
        ).event

    def _graph_complete(
        self,
        iteration: int,
        template_node_id: str,
        *,
        artifact_digest: str,
        result: CompletionResult = CompletionResult.SUCCEEDED,
        facts: Mapping[str, str] = {},
        input_artifacts: Mapping[str, str] = {},
        authority_refs: tuple[str, ...] = (),
        evidence_refs: tuple[str, ...] = (),
        event_type: str = "node.completed",
        reservation: ScheduledReservation | None = None,
    ) -> CausalEvent | None:
        """Submit a producer-owned proof; the controller never advances it directly."""

        state = self.graph_scheduler.state(run_id=self.run_id, iteration=iteration)
        if template_node_id in state.completed:
            return next(
                (
                    event
                    for event in reversed(self.graph_event_store.events(run_id=self.run_id))
                    if event.template_node_id == template_node_id and event.iteration == iteration
                ),
                None,
            )
        reservation = reservation or self._graph_reserve(iteration, template_node_id)
        verifier = self.graph_scheduler.completion_verifier
        if verifier is None:  # pragma: no cover - constructor rejects this configuration
            raise RuntimeError("evidence-native scheduler has no completion verifier")
        completion = self.completion_client.complete(
            graph_digest=self.graph_digest,
            contract_digest=self.contract.contract_digest,
            run_id=self.run_id,
            template_node_id=template_node_id,
            node_instance_id=reservation.node_instance_id,
            artifact_digest=self._as_digest(artifact_digest),
            validator_policy_digest=verifier.ownership.policy_digest,
            result=result,
            facts=dict(facts),
            input_artifact_digests={key: self._as_digest(value) for key, value in input_artifacts.items()},
            authority_refs=authority_refs,
            evidence_refs=evidence_refs,
        )
        return self.graph_scheduler.complete(
            run_id=self.run_id,
            iteration=iteration,
            template_node_id=template_node_id,
            completion=completion,
            event_type=event_type,
            causal_parents=(reservation.event.event_id,),
        ).event

    def _graph_reserve(self, iteration: int, template_node_id: str) -> ScheduledReservation:
        existing = self.graph_event_store.events(run_id=self.run_id)
        return self.graph_scheduler.reserve(
            run_id=self.run_id,
            iteration=iteration,
            template_node_id=template_node_id,
            causal_parents=(existing[-1].event_id,) if existing else (),
        )

    @staticmethod
    def _graph_node_token(value: str) -> str:
        return "".join(character if character.isalnum() else "-" for character in value).strip("-") or "unnamed"

    @staticmethod
    def _as_digest(value: str) -> str:
        """Admit only canonical artifact digests into signed completion fields.

        Executors may expose opaque artifact handles to the controller.  They
        remain useful in the ledger, but the signed protocol binds a digest of
        that handle rather than treating arbitrary text as a digest.
        """

        if len(value) == 64 and all(character in "0123456789abcdef" for character in value):
            return value
        return digest({"artifact_reference": value})

    def _matching_action_rule_index(self, intent: ActionIntent) -> int:
        """Use the same closed rule shape as policy before the graph advances."""

        for index, rule in enumerate(self.contract.allowed_actions):
            prefix = rule.target_prefix.rstrip("/") or "/"
            if (
                rule.tool == intent.tool
                and rule.effect is intent.effect
                and (prefix == "/" or intent.target == prefix or intent.target.startswith(prefix + "/"))
            ):
                return index
        raise PolicyDenied("no action rule matches the proposed intent")

    def _graph_authorise_action_rule(self, iteration: int, intent: ActionIntent) -> None:
        """Make the selected contract rule (and its approval edge) executable."""

        index = self._matching_action_rule_index(intent)
        rule = self.contract.allowed_actions[index]
        self._graph_complete(
            iteration,
            f"action.rule.{index}",
            artifact_digest=intent.intent_digest,
            facts={"rule_index": str(index)},
        )
        self._graph_advance(iteration, "join.action.rule.any", "action.rule.joined", {})
        approval_required = self._dynamic_approval_required(intent, rule)
        self._graph_complete(
            iteration,
            "policy.decision",
            artifact_digest=digest({"intent": intent.intent_digest, "approval_required": approval_required}),
            facts={"approval_required": str(approval_required).lower()},
            authority_refs=(intent.intent_digest,),
        )
        if approval_required:
            self._graph_complete(
                iteration,
                f"approval.rule.{index}",
                artifact_digest=digest({"intent": intent.intent_digest, "approval": "consumed"}),
                facts={"approved": "true", "rule_index": str(index)},
                authority_refs=(intent.intent_digest,),
            )
        self._graph_advance(iteration, "join.action.authority.any", "action.authority.joined", {})

    @staticmethod
    def _dynamic_approval_required(intent: ActionIntent, rule) -> bool:
        high_impact = {"write", "execute", "network", "delete", "publish"}
        tainted = any(value in {Provenance.UNTRUSTED_RETRIEVAL, Provenance.TOOL_OUTPUT} for value in intent.provenance)
        return rule.approval_required or intent.effect.value in {"delete", "publish"} or (
            tainted and intent.effect.value in high_impact
        )

    def _graph_accept(
        self,
        iteration: int,
        final_check: CheckResult,
        observation: ExecutionObservation,
    ) -> None:
        """Accept only independently owned completions grounded in check evidence."""

        # Whole-task acceptance is intentionally based on the accumulated,
        # final-verifier-approved evidence set rather than only the most recent
        # action.  This is what permits a multi-step task to bind distinct
        # required checks to their own prior receipts.
        report_checks: dict[str, CheckResult] = {}
        for action in self._evidence.snapshot().actions:
            for check in action.report.checks:
                # A later action that does not exercise an earlier criterion
                # reports it as inconclusive; that is not contradictory
                # evidence and must not erase the already admitted passing
                # receipt.  A later explicit pass refreshes the binding.
                if check.status is CheckStatus.PASS or check.name not in report_checks:
                    report_checks[check.name] = check
        if self._unsafe_development_completions:
            self._admit_development_evidence(iteration, observation, report_checks)
        for node in self.graph_manifest.nodes:
            if node.node_type.value != "criterion":
                continue
            guard = node.metadata.get("guard", "")
            required_checks = tuple(filter(None, node.metadata.get("required_checks", guard).split(",")))
            checks = tuple(report_checks.get(name) for name in required_checks)
            if (
                self._unsafe_development_completions
                and guard not in self.contract.success_condition_bindings
                and report_checks.get("execution", CheckResult("execution", CheckStatus.FAIL, {})).status is CheckStatus.PASS
            ):
                # Legacy unit contracts without declared condition bindings
                # are admissible only in the explicitly unsafe test adapter.
                required_checks, checks = ("execution",), (report_checks["execution"],)
            if not checks or any(check is None or check.status is not CheckStatus.PASS for check in checks):
                raise PermissionError(f"no passing criterion evidence exists for guard {guard!r}")
            receipt_node = f"receipt.{self._graph_node_token(guard)}"
            receipt_event = next(
                (
                    event
                    for event in reversed(self.graph_event_store.events(run_id=self.run_id))
                    if event.template_node_id == receipt_node
                ),
                None,
            )
            if receipt_event is None:
                raise PermissionError(f"criterion {guard!r} has no admitted receipt evidence")
            self._graph_complete(
                iteration,
                node.node_id,
                artifact_digest=digest({
                    "guard": guard,
                    "checks": [(check.name, dict(check.evidence)) for check in checks if check is not None],
                }),
                facts={"passed": "true", "guard": guard},
                evidence_refs=(str(receipt_event.payload.get("completion_digest", "")),),
            )
        self._graph_advance(iteration, "join.guards.all", "guards.joined", {})
        self._graph_complete(
            iteration,
            "decision.accept",
            artifact_digest=digest({"final_check": final_check.name, "evidence": dict(final_check.evidence)}),
            facts={"decision": "accept"},
            evidence_refs=tuple(
                str(event.payload.get("completion_digest", ""))
                for event in self.graph_event_store.events(run_id=self.run_id)
                if event.template_node_id.startswith("criterion.") and event.iteration == iteration
            ),
        )

    def _admit_development_evidence(
        self,
        iteration: int,
        observation: ExecutionObservation,
        report_checks: Mapping[str, CheckResult],
    ) -> None:
        """Explicit test-only adapter; production must use protected services."""

        snapshot_state = self.graph_scheduler.state(run_id=self.run_id, iteration=iteration)
        if "snapshot.materialized" not in snapshot_state.completed:
            self._graph_complete(
                iteration,
                "snapshot.materialized",
                artifact_digest=digest({"development": "snapshot", "artifacts": observation.artifact_digests}),
                facts={"success": "true", "workspace_generation": "0"},
            )
        for node in self.graph_manifest.nodes:
            if node.node_type.value == "evaluator":
                self._graph_complete(
                    iteration,
                    node.node_id,
                    artifact_digest=digest({"development": "evaluator", "check": node.metadata.get("check_name", "")}),
                    facts={"check_name": node.metadata.get("check_name", "")},
                    input_artifacts=observation.artifact_digests,
                )
        for node in self.graph_manifest.nodes:
            if node.node_type.value != "receipt":
                continue
            receipt_type = node.metadata.get("receipt_type", "")
            criterion = next(
                (
                    candidate
                    for candidate in self.graph_manifest.nodes
                    if candidate.node_type.value == "criterion" and candidate.metadata.get("guard") == receipt_type
                ),
                None,
            )
            required = tuple(filter(None, (criterion.metadata.get("required_checks", receipt_type) if criterion else receipt_type).split(",")))
            checks = tuple(report_checks.get(name) for name in required)
            if (
                receipt_type not in self.contract.success_condition_bindings
                and report_checks.get("execution", CheckResult("execution", CheckStatus.FAIL, {})).status is CheckStatus.PASS
            ):
                required, checks = ("execution",), (report_checks["execution"],)
            # This path exists only behind the explicit development switch.
            # It synthesizes a condition receipt from the contract-declared
            # bound checks, preserving the same condition/receipt topology as
            # a protected evaluator deployment.
            if checks and all(check is not None and check.status is CheckStatus.PASS for check in checks):
                self._graph_complete(
                    iteration,
                    node.node_id,
                    artifact_digest=digest({
                        "development": "receipt",
                        "receipt_type": receipt_type,
                        "checks": [(check.name, dict(check.evidence)) for check in checks if check is not None],
                    }),
                    facts={"receipt_type": receipt_type, "passed": "true"},
                )

    def _save_checkpoint(
        self,
        *,
        phase: RunPhase,
        next_iteration: int,
        pending_intent: ActionIntent | None = None,
        executor_id: str = "",
        prepared_execution: PreparedExecution | None = None,
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
            graph_digest=self.graph_digest,
            phase=phase,
            next_iteration=next_iteration,
            tool_calls=self._tool_calls,
            history=tuple(self._history),
            seen_failures=tuple(sorted(self._seen_failures)),
            evidence=self._evidence.snapshot(),
            pending_intent=pending_intent,
            executor_id=executor_id,
            prepared_execution=prepared_execution,
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

    def _prepare_execution(self, iteration: int, intent: ActionIntent) -> PreparedExecution:
        """Mint the operation identity before durable effect dispatch."""

        operation_id = digest(
            {
                "run_id": self.run_id,
                "iteration": iteration,
                "intent_digest": intent.intent_digest,
                "idempotency_key": intent.idempotency_key,
                "executor_id": self.executor.executor_id,
            }
        )
        prepare = getattr(self.executor, "prepare_execution", None)
        if callable(prepare):
            prepared = prepare(
                intent,
                run_id=self.run_id,
                contract_digest=self.contract.contract_digest,
                iteration=iteration,
                operation_id=operation_id,
                graph_digest=self.graph_digest,
                graph_node_id="operation.prepared",
            )
            if not isinstance(prepared, PreparedExecution):
                raise TypeError("executor returned an invalid prepared execution")
        else:
            prepared = PreparedExecution(
                operation_id=operation_id,
                executor_id=self.executor.executor_id,
                intent_digest=intent.intent_digest,
                request_digest=digest(
                    {
                        "operation_id": operation_id,
                        "run_id": self.run_id,
                        "contract_digest": self.contract.contract_digest,
                        "iteration": iteration,
                        "intent_digest": intent.intent_digest,
                        "graph_digest": self.graph_digest,
                        "graph_node_id": "operation.prepared",
                    }
                ),
                remote_job_id=operation_id,
                graph_digest=self.graph_digest,
                graph_node_id="operation.prepared",
            )
        if (
            prepared.operation_id != operation_id
            or prepared.executor_id != self.executor.executor_id
            or prepared.intent_digest != intent.intent_digest
            or prepared.graph_digest != self.graph_digest
            or prepared.graph_node_id != "operation.prepared"
        ):
            raise ValueError("prepared execution does not bind this operation")
        return prepared

    def _checkpoint_pending_effect(
        self,
        iteration: int,
        intent: ActionIntent,
        prepared_execution: PreparedExecution,
    ) -> None:
        self._save_checkpoint(
            phase=RunPhase.PENDING_EFFECT,
            next_iteration=iteration,
            pending_intent=intent,
            executor_id=self.executor.executor_id,
            prepared_execution=prepared_execution,
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
        prepared_execution = checkpoint.prepared_execution
        assert intent is not None and prepared_execution is not None  # checkpoint invariant
        observation = self.effect_reconciler.reconcile(
            run_id=self.run_id,
            contract=self.contract,
            intent=intent,
            executor_id=checkpoint.executor_id,
            prepared_execution=prepared_execution,
        )
        if (
            observation.metadata.get("operation_id") != prepared_execution.operation_id
            or observation.metadata.get("request_digest") != prepared_execution.request_digest
            or observation.metadata.get("graph_digest") != prepared_execution.graph_digest
            or observation.metadata.get("graph_node_id") != prepared_execution.graph_node_id
        ):
            raise PermissionError("reconciliation observation is not bound to the prepared operation")
        self._checkpoint = checkpoint
        self._save_checkpoint(
            phase=RunPhase.RECONCILED_EFFECT,
            next_iteration=checkpoint.next_iteration,
            pending_intent=intent,
            executor_id=checkpoint.executor_id,
            prepared_execution=prepared_execution,
            reconciled_observation=observation,
        )
        self.ledger.append(
            "effect.reconciled",
            {
                "run_id": self.run_id,
                "intent_digest": intent.intent_digest,
                "executor_id": checkpoint.executor_id,
                "operation_id": prepared_execution.operation_id,
                "request_digest": prepared_execution.request_digest,
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

        if not observation.success:
            return False
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
