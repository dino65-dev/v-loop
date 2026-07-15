"""Explicit final-goal verification for V-Loop runs.

An action-level verification report proves facts about one execution.  It does
not, by itself, prove that a multi-step user goal is complete.  This module
keeps that distinction explicit: a deployment must bind every contractual
success condition to a protected final check before the controller may treat a
run as complete.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Protocol

from .models import (
    ActionIntent,
    CheckResult,
    CheckStatus,
    ExecutionObservation,
    TaskContract,
    VerificationReport,
)


@dataclass(frozen=True, slots=True)
class ActionEvidence:
    """Immutable verification outcome for one completed action."""

    sequence: int
    intent_digest: str
    artifact_digests: Mapping[str, str]
    source_state_digest: str | None
    report: VerificationReport


@dataclass(frozen=True, slots=True)
class EvidenceSnapshot:
    run_id: str
    actions: tuple[ActionEvidence, ...]

    @property
    def final_source_state_digest(self) -> str | None:
        return self.actions[-1].source_state_digest if self.actions else None


class EvidenceAccumulator:
    """Run-local evidence index used by final-state verification."""

    def __init__(self, run_id: str) -> None:
        self._run_id = run_id
        self._actions: list[ActionEvidence] = []

    def append(
        self,
        *,
        intent: ActionIntent,
        observation: ExecutionObservation,
        report: VerificationReport,
    ) -> None:
        # Workspace state is authoritative only when it is contained in a
        # verified evaluator/supervisor receipt. Guest metadata is attacker
        # controlled and may never select which historical checks are fresh.
        signed_source_digests = {
            value
            for check in report.checks
            if check.evidence.get("signed_receipt") is True
            for value in (check.evidence.get("workspace_snapshot_digest"),)
            if isinstance(value, str) and value
        }
        source_digest = next(iter(signed_source_digests)) if len(signed_source_digests) == 1 else None
        self._actions.append(
            ActionEvidence(
                sequence=len(self._actions) + 1,
                intent_digest=intent.intent_digest,
                artifact_digests=dict(observation.artifact_digests),
                source_state_digest=source_digest,
                report=report,
            )
        )

    def snapshot(self) -> EvidenceSnapshot:
        return EvidenceSnapshot(self._run_id, tuple(self._actions))

    @classmethod
    def from_snapshot(cls, snapshot: EvidenceSnapshot) -> "EvidenceAccumulator":
        """Restore only already-verified action evidence from a checkpoint."""

        accumulator = cls(snapshot.run_id)
        accumulator._actions = list(snapshot.actions)
        return accumulator


@dataclass(frozen=True, slots=True)
class ActionSafetyReport:
    """Non-negotiable gates that permit an action to add task evidence."""

    required_checks: tuple[str, ...]
    statuses: Mapping[str, CheckStatus]

    @property
    def accepted(self) -> bool:
        return bool(self.required_checks) and all(
            self.statuses.get(name) is CheckStatus.PASS for name in self.required_checks
        )

    @classmethod
    def from_report(cls, contract: TaskContract, report: VerificationReport) -> "ActionSafetyReport":
        statuses: dict[str, CheckStatus] = {}
        for check in report.checks:
            prior = statuses.get(check.name)
            statuses[check.name] = (
                CheckStatus.INCONCLUSIVE
                if prior is not None and prior is not check.status
                else check.status
            )
        return cls(contract.action_safety_checks, statuses)


@dataclass(frozen=True, slots=True)
class TaskCompletionReport:
    """Aggregate, whole-task evidence used for post-acceptance side effects."""

    action_reports: tuple[VerificationReport, ...]
    final_check: CheckResult
    final_workspace_digest: str | None

    @property
    def accepted(self) -> bool:
        return self.final_check.status is CheckStatus.PASS


class FinalVerifier(Protocol):
    """Protected evaluator for whole-task completion."""

    def verify(
        self,
        *,
        contract: TaskContract,
        action_report: VerificationReport,
        history: tuple[dict, ...],
        evidence: EvidenceSnapshot,
    ) -> CheckResult: ...


class RequiredChecksFinalVerifier:
    """Binds immutable success conditions to named hard checks.

    This adapter is intentionally strict.  Every contractual condition must
    have an explicit non-empty set of check names, and every bound check must
    pass in the final action report.  Production deployments can replace this
    with a protected evaluator that performs a broader end-to-end check.
    """

    def __init__(
        self,
        required_checks: Mapping[str, tuple[str, ...]],
        global_completion_guards: tuple[str, ...] = (),
    ) -> None:
        self._required_checks = {condition: tuple(names) for condition, names in required_checks.items()}
        self._global_completion_guards = tuple(global_completion_guards)

    @property
    def required_checks(self) -> Mapping[str, tuple[str, ...]]:
        return dict(self._required_checks)

    @property
    def global_completion_guards(self) -> tuple[str, ...]:
        return self._global_completion_guards

    @classmethod
    def from_contract(cls, contract: TaskContract) -> "RequiredChecksFinalVerifier":
        if not contract.success_condition_bindings:
            raise ValueError("contract has no immutable success-condition bindings")
        return cls(contract.success_condition_bindings, contract.global_completion_guards)

    def verify(
        self,
        *,
        contract: TaskContract,
        action_report: VerificationReport,
        history: tuple[dict, ...],
        evidence: EvidenceSnapshot,
    ) -> CheckResult:
        del history, action_report  # Bound receipts, not planner text, are authoritative.
        check_statuses = self._fresh_check_statuses(evidence)
        missing_bindings = [
            condition
            for condition in contract.success_conditions
            if not self._required_checks.get(condition)
        ]
        unknown_conditions = sorted(set(self._required_checks).difference(contract.success_conditions))
        failed_conditions: dict[str, dict[str, str]] = {}
        for condition in contract.success_conditions:
            names = self._required_checks.get(condition, ())
            statuses = {name: check_statuses.get(name, CheckStatus.INCONCLUSIVE).value for name in names}
            if any(status != CheckStatus.PASS.value for status in statuses.values()):
                failed_conditions[condition] = statuses
        guard_statuses = {
            name: check_statuses.get(name, CheckStatus.INCONCLUSIVE).value
            for name in self._global_completion_guards
        }
        failed_guards = {
            name: status for name, status in guard_statuses.items() if status != CheckStatus.PASS.value
        }

        payload = {
            "contract_digest": contract.contract_digest,
            "required_checks": self._required_checks,
            "missing_bindings": tuple(missing_bindings),
            "unknown_conditions": tuple(unknown_conditions),
            "condition_statuses": failed_conditions,
            "global_guard_statuses": guard_statuses,
        }
        if missing_bindings or unknown_conditions:
            return CheckResult(
                "final-goal",
                CheckStatus.INCONCLUSIVE,
                payload,
                "each success condition needs an explicit protected check binding",
            )
        if not evidence.actions or failed_conditions or failed_guards:
            return CheckResult(
                "final-goal",
                CheckStatus.FAIL,
                payload,
                "a bound final check or global completion guard did not pass",
            )
        return CheckResult("final-goal", CheckStatus.PASS, payload)

    @staticmethod
    def _fresh_check_statuses(evidence: EvidenceSnapshot) -> dict[str, CheckStatus]:
        if not evidence.actions:
            return {}
        final_source = evidence.final_source_state_digest
        relevant = (
            [action for action in evidence.actions if action.source_state_digest == final_source]
            if final_source is not None
            else [evidence.actions[-1]]
        )
        candidates: dict[str, list[tuple[int, str, CheckStatus]]] = {}
        for action in relevant:
            for check in action.report.checks:
                issued_at = check.evidence.get("issued_at")
                receipt_time = issued_at if isinstance(issued_at, str) else ""
                candidates.setdefault(check.name, []).append((action.sequence, receipt_time, check.status))
        statuses: dict[str, CheckStatus] = {}
        for name, values in candidates.items():
            # Latest valid receipt wins. A tie on sequence/time with different
            # deterministic outcomes is a contradictory attestation, not a
            # reason to silently prefer failure or success.
            latest = max(values, key=lambda value: (value[1], value[0]))
            tied = [value for value in values if value[:2] == latest[:2]]
            statuses[name] = (
                CheckStatus.INCONCLUSIVE
                if len({value[2] for value in tied}) > 1
                else latest[2]
            )
        return statuses


@dataclass(frozen=True, slots=True)
class CallableFinalVerifier:
    """Adapter for a separately protected end-to-end evaluator.

    The callback belongs to deployment-owned verifier code.  It must return a
    ``CheckResult`` named ``final-goal``; planner or executor callbacks are not
    suitable inputs here.
    """

    check: Callable[[TaskContract, VerificationReport, tuple[dict, ...], EvidenceSnapshot], CheckResult]

    def verify(
        self,
        *,
        contract: TaskContract,
        action_report: VerificationReport,
        history: tuple[dict, ...],
        evidence: EvidenceSnapshot,
    ) -> CheckResult:
        result = self.check(contract, action_report, history, evidence)
        if result.name != "final-goal":
            raise ValueError("final verifier must return a final-goal check")
        return result
