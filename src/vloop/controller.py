"""Verified adaptive-loop state machine."""

from __future__ import annotations

from dataclasses import asdict, replace
from datetime import UTC, datetime
from typing import Iterable, Protocol
from uuid import uuid4

from .canonical import digest
from .completion import EvidenceAccumulator, FinalVerifier
from .context import ContextPackage
from .executor import Executor
from .ledger import EvidenceLedger
from .memory import MemoryCandidateProducer, VerifiedMemoryCommitter
from .models import (
    ActionIntent,
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
        self.run_id = str(uuid4())
        self._evidence = EvidenceAccumulator(self.run_id)
        self._history: list[dict] = []
        self._seen_failures: set[tuple[str, str]] = set()
        self._tool_calls = 0

    def run(self, approvals: Iterable[Approval | SignedApprovalReceipt] = ()) -> LoopDecision:
        self.ledger.append(
            "run.started",
            {"run_id": self.run_id, "contract_digest": self.contract.contract_digest},
        )
        for iteration in range(1, self.contract.maximum_iterations + 1):
            if self._tool_calls >= self.contract.maximum_tool_calls:
                return self._terminal(LoopDecision.STOP, "tool-call-budget-exhausted")
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
                    return self._terminal(LoopDecision.WAITING, "approval-required")
                return self._terminal(LoopDecision.ESCALATE, "policy-denied")

            self._tool_calls += 1
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
            if report.accepted and self.probe_runner is not None:
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
        return self._terminal(LoopDecision.STOP, "iteration-budget-exhausted")

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
        if context is None:
            return intent
        # A controller cannot inspect a model's attention.  Conservatively mark
        # every action argument proposed with a package containing untrusted
        # material as tainted; PolicyGate can then reason at argument granularity
        # instead of relying on a single self-declared action label.
        provenance = tuple(sorted(set(intent.provenance).union(context.provenance), key=lambda value: value.value))
        argument_provenance = {
            name: tuple(
                sorted(
                    set(intent.provenance_for_argument(name)).union(context.provenance),
                    key=lambda value: value.value,
                )
            )
            for name in intent.arguments
        }
        return replace(intent, provenance=provenance, argument_provenance=argument_provenance)

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
            candidate = self.memory_candidate_producer.propose(
                contract=self.contract,
                history=tuple(self._history),
                report=report,
                final_check=final_check,
                available_evidence_refs=evidence_refs,
            )
            if candidate is None:
                self.ledger.append(
                    "memory.commit.skipped",
                    {"run_id": self.run_id, "reason": "no-memory-candidate"},
                )
                return
            record = self.memory_committer.commit(
                candidate,
                report=report,
                final_check=final_check,
                source_run_id=self.run_id,
                available_evidence_refs=evidence_refs,
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
