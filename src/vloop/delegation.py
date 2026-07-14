"""Evidence-gated optional specialist delegation.

This does not create agents. It determines whether a caller may ask an
orchestrator for a specialist under a fixed total budget.
"""

from __future__ import annotations

from dataclasses import dataclass


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
