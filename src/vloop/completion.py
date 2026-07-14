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

from .models import CheckResult, CheckStatus, TaskContract, VerificationReport


class FinalVerifier(Protocol):
    """Protected evaluator for whole-task completion."""

    def verify(
        self,
        *,
        contract: TaskContract,
        action_report: VerificationReport,
        history: tuple[dict, ...],
    ) -> CheckResult: ...


class RequiredChecksFinalVerifier:
    """Binds immutable success conditions to named hard checks.

    This adapter is intentionally strict.  Every contractual condition must
    have an explicit non-empty set of check names, and every bound check must
    pass in the final action report.  Production deployments can replace this
    with a protected evaluator that performs a broader end-to-end check.
    """

    def __init__(self, required_checks: Mapping[str, tuple[str, ...]]) -> None:
        self._required_checks = {condition: tuple(names) for condition, names in required_checks.items()}

    def verify(
        self,
        *,
        contract: TaskContract,
        action_report: VerificationReport,
        history: tuple[dict, ...],
    ) -> CheckResult:
        del history  # The condition/check binding, not planner text, is authoritative.
        check_statuses = {check.name: check.status for check in action_report.checks}
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

        evidence = {
            "contract_digest": contract.contract_digest,
            "required_checks": self._required_checks,
            "missing_bindings": tuple(missing_bindings),
            "unknown_conditions": tuple(unknown_conditions),
            "condition_statuses": failed_conditions,
        }
        if missing_bindings or unknown_conditions:
            return CheckResult(
                "final-goal",
                CheckStatus.INCONCLUSIVE,
                evidence,
                "each success condition needs an explicit protected check binding",
            )
        if not action_report.accepted or failed_conditions:
            return CheckResult(
                "final-goal",
                CheckStatus.FAIL,
                evidence,
                "a bound final check did not pass",
            )
        return CheckResult("final-goal", CheckStatus.PASS, evidence)


@dataclass(frozen=True, slots=True)
class CallableFinalVerifier:
    """Adapter for a separately protected end-to-end evaluator.

    The callback belongs to deployment-owned verifier code.  It must return a
    ``CheckResult`` named ``final-goal``; planner or executor callbacks are not
    suitable inputs here.
    """

    check: Callable[[TaskContract, VerificationReport, tuple[dict, ...]], CheckResult]

    def verify(
        self,
        *,
        contract: TaskContract,
        action_report: VerificationReport,
        history: tuple[dict, ...],
    ) -> CheckResult:
        result = self.check(contract, action_report, history)
        if result.name != "final-goal":
            raise ValueError("final verifier must return a final-goal check")
        return result
