"""Verified adaptive-loop state machine."""

from __future__ import annotations

from dataclasses import asdict
from datetime import UTC, datetime
from typing import Iterable, Protocol
from uuid import uuid4

from .executor import Executor
from .ledger import EvidenceLedger
from .models import ActionIntent, CheckStatus, LoopDecision, TaskContract, VerificationReport
from .neural_verifier import ShadowNeuralVerifier
from .policy import Approval, PolicyDenied, PolicyGate
from .repair import RepairController
from .verifiers import HybridVerifier


class Planner(Protocol):
    def propose(self, *, contract: TaskContract, history: tuple[dict, ...]) -> ActionIntent: ...


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
    ) -> None:
        self.contract = contract
        self.planner = planner
        self.gate = gate
        self.executor = executor
        self.verifier = verifier
        self.ledger = ledger
        self.shadow_verifier = shadow_verifier
        self.repair_controller = repair_controller or RepairController()
        self.run_id = str(uuid4())
        self._history: list[dict] = []
        self._seen_failures: set[tuple[str, str]] = set()
        self._tool_calls = 0

    def run(self, approvals: Iterable[Approval] = ()) -> LoopDecision:
        self.ledger.append(
            "run.started",
            {"run_id": self.run_id, "contract_digest": self.contract.contract_digest},
        )
        for iteration in range(1, self.contract.maximum_iterations + 1):
            if self._tool_calls >= self.contract.maximum_tool_calls:
                return self._terminal(LoopDecision.STOP, "tool-call-budget-exhausted")
            intent = self.planner.propose(contract=self.contract, history=tuple(self._history))
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
                capability = self.gate.authorize(intent, approvals)
                self.gate.validate_and_consume(capability, intent)
            except PolicyDenied as exc:
                self._history.append({"intent": intent.intent_digest, "failure": str(exc)})
                self.ledger.append(
                    "intent.denied",
                    {"run_id": self.run_id, "intent_digest": intent.intent_digest, "reason": str(exc)},
                )
                return self._terminal(LoopDecision.ESCALATE, "policy-denied")

            self._tool_calls += 1
            observation = self.executor.execute(intent)
            self.ledger.append(
                "execution.observed",
                {
                    "run_id": self.run_id,
                    "intent_digest": intent.intent_digest,
                    "success": observation.success,
                    "exit_code": observation.exit_code,
                    "artifact_digests": dict(observation.artifact_digests),
                    "metadata": dict(observation.metadata),
                },
            )
            report = self.verifier.verify(self.contract, observation)
            self.ledger.append(
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
                return self._terminal(LoopDecision.ACCEPT, "verified-success")

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
            failure_key = (intent.intent_digest, directive.stage)
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
                }
            )
        return self._terminal(LoopDecision.STOP, "iteration-budget-exhausted")

    def _record_shadow_diagnostic(
        self, intent: ActionIntent, observation, report: VerificationReport
    ):
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

    def _failure_class(self, report: VerificationReport) -> str:
        if report.policy is CheckStatus.FAIL:
            return "policy"
        if report.correctness is CheckStatus.FAIL:
            return "correctness"
        if report.evidence is not CheckStatus.PASS:
            return "evidence"
        if report.quality is not CheckStatus.PASS:
            return "quality"
        return "unknown"

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
