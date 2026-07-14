"""Evidence-gated optional specialist delegation.

Specialists are not autonomous authority holders.  They receive bounded,
advisory tasks only after a same-budget experiment has demonstrated benefit;
their output is evidence-labelled data that still goes through the primary
controller and policy gate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Protocol

from .canonical import digest
from .ledger import EvidenceLedger


@dataclass(frozen=True, slots=True)
class DelegationEvidence:
    task_kind: str
    specialist_role: str
    verified: bool
    single_agent_success_rate: float
    specialist_success_rate: float
    measured_total_cost: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.single_agent_success_rate <= 1.0:
            raise ValueError("invalid single-agent success rate")
        if not 0.0 <= self.specialist_success_rate <= 1.0:
            raise ValueError("invalid specialist success rate")
        if self.measured_total_cost <= 0:
            raise ValueError("measured total cost must be positive")


@dataclass(frozen=True, slots=True)
class DelegationDecision:
    allowed: bool
    reason: str


class DelegationGate:
    """Allows a specialist only after verified same-budget improvement."""

    def __init__(self, evidence: tuple[DelegationEvidence, ...], minimum_gain: float = 0.05) -> None:
        if not 0.0 < minimum_gain <= 1.0:
            raise ValueError("minimum_gain must be in (0, 1]")
        self._evidence = evidence
        self._minimum_gain = minimum_gain

    def decide(self, *, task_kind: str, specialist_role: str, total_budget: float) -> DelegationDecision:
        matches = [
            item
            for item in self._evidence
            if item.task_kind == task_kind and item.specialist_role == specialist_role and item.verified
        ]
        if not matches:
            return DelegationDecision(False, "no verified delegation experiment for this task and role")
        best = max(matches, key=lambda item: item.specialist_success_rate - item.single_agent_success_rate)
        gain = best.specialist_success_rate - best.single_agent_success_rate
        if gain < self._minimum_gain:
            return DelegationDecision(False, "specialist did not show sufficient same-budget improvement")
        if total_budget < best.measured_total_cost:
            return DelegationDecision(False, "requested budget is below the verified specialist budget")
        return DelegationDecision(True, "verified same-budget specialist improvement")


@dataclass(frozen=True, slots=True)
class SpecialistTask:
    """A bounded request with no inherited capabilities or credentials."""

    task_kind: str
    specialist_role: str
    objective: str
    context: Mapping[str, str]
    total_budget: float

    def __post_init__(self) -> None:
        if not self.task_kind.strip() or not self.specialist_role.strip() or not self.objective.strip():
            raise ValueError("specialist task needs task kind, role, and objective")
        if self.total_budget <= 0:
            raise ValueError("specialist task budget must be positive")

    @property
    def task_digest(self) -> str:
        return digest(
            {
                "task_kind": self.task_kind,
                "specialist_role": self.specialist_role,
                "objective": self.objective,
                "context": dict(self.context),
                "total_budget": self.total_budget,
            }
        )


@dataclass(frozen=True, slots=True)
class SpecialistResult:
    """Advisory output.  Raw content is never written directly to the ledger."""

    summary: str
    evidence: Mapping[str, str]
    measured_cost: float

    def __post_init__(self) -> None:
        if not self.summary.strip():
            raise ValueError("specialist result needs a summary")
        if self.measured_cost <= 0:
            raise ValueError("specialist cost must be positive")


class Specialist(Protocol):
    def execute(self, task: SpecialistTask) -> SpecialistResult: ...


@dataclass(frozen=True, slots=True)
class SpecialistDispatch:
    allowed: bool
    invoked: bool
    reason: str
    result: SpecialistResult | None = None


class SpecialistDispatcher:
    """Dispatches only registered roles after the immutable delegation gate."""

    def __init__(
        self,
        gate: DelegationGate,
        specialists: Mapping[str, Specialist],
        ledger: EvidenceLedger | None = None,
    ) -> None:
        if not specialists:
            raise ValueError("at least one server-registered specialist is required")
        if any(not role.strip() for role in specialists):
            raise ValueError("specialist roles must not be blank")
        self._gate = gate
        self._specialists = dict(specialists)
        self._ledger = ledger

    def dispatch(self, task: SpecialistTask) -> SpecialistDispatch:
        decision = self._gate.decide(
            task_kind=task.task_kind,
            specialist_role=task.specialist_role,
            total_budget=task.total_budget,
        )
        specialist = self._specialists.get(task.specialist_role)
        if not decision.allowed or specialist is None:
            reason = decision.reason if specialist is not None else "specialist role is not server-registered"
            self._record("specialist.dispatch.denied", task, {"reason": reason})
            return SpecialistDispatch(False, False, reason)

        self._record("specialist.dispatch.started", task, {})
        try:
            result = specialist.execute(task)
        except Exception as exc:
            self._record("specialist.dispatch.failed", task, {"error_type": type(exc).__name__})
            return SpecialistDispatch(True, True, "specialist failed", None)
        if result.measured_cost > task.total_budget:
            self._record(
                "specialist.dispatch.rejected",
                task,
                {"reason": "specialist exceeded total budget", "measured_cost": result.measured_cost},
            )
            return SpecialistDispatch(True, True, "specialist exceeded total budget", None)
        self._record(
            "specialist.dispatch.completed",
            task,
            {
                "measured_cost": result.measured_cost,
                "summary_digest": digest(result.summary),
                "evidence_digest": digest(dict(result.evidence)),
                "advisory_only": True,
            },
        )
        return SpecialistDispatch(True, True, decision.reason, result)

    def _record(self, event_type: str, task: SpecialistTask, payload: Mapping[str, object]) -> None:
        if self._ledger is None:
            return
        self._ledger.append(
            event_type,
            {
                "task_digest": task.task_digest,
                "task_kind": task.task_kind,
                "specialist_role": task.specialist_role,
                **payload,
            },
        )
