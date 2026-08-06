"""Matched-budget evidence gate for the experimental Prime-style plane."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Iterable


class PrimeAblationVariant(StrEnum):
    BASELINE = "baseline"
    SEQUENTIAL = "sequential-attempts"
    PARALLEL = "parallel-candidates"
    PROGRAMMABLE_CONTEXT = "programmable-context"
    RECURSIVE = "recursive-subagents"
    PERSISTENT = "persistent-sessions"


@dataclass(frozen=True, slots=True)
class PrimeAblationRun:
    task_id: str
    seed: int
    variant: PrimeAblationVariant
    token_budget: int
    wall_clock_seconds: float
    success: bool
    false_acceptance: bool
    policy_violations: int
    memory_contamination: int
    model_calls: int

    def __post_init__(self) -> None:
        if not self.task_id.strip() or min(self.token_budget, self.model_calls, self.policy_violations, self.memory_contamination) < 0 or self.wall_clock_seconds < 0:
            raise ValueError("ablation run fields are invalid")


@dataclass(frozen=True, slots=True)
class PrimeAblationResult:
    candidate: PrimeAblationVariant
    tasks: int
    success_delta: float
    false_acceptance_delta: float
    policy_violation_delta: int
    memory_contamination_delta: int
    cost_delta: float

    @property
    def promotable(self) -> bool:
        return (
            self.success_delta > 0 and self.false_acceptance_delta <= 0 and self.policy_violation_delta == 0
            and self.memory_contamination_delta <= 0 and (self.cost_delta < 0 or self.success_delta >= 0.05)
        )


def compare_matched_budget(
    baseline: Iterable[PrimeAblationRun], candidate: Iterable[PrimeAblationRun], *, variant: PrimeAblationVariant,
) -> PrimeAblationResult:
    left = {(run.task_id, run.seed): run for run in baseline}
    right = {(run.task_id, run.seed): run for run in candidate}
    if not left or set(left) != set(right):
        raise ValueError("ablation comparison requires identical non-empty task/seed pairs")
    if any(run.variant is not PrimeAblationVariant.BASELINE for run in left.values()) or any(run.variant is not variant for run in right.values()):
        raise ValueError("ablation variants do not match the requested comparison")
    for key in left:
        if (left[key].token_budget, left[key].wall_clock_seconds) != (right[key].token_budget, right[key].wall_clock_seconds):
            raise ValueError("ablation comparison requires matched token and wall-clock budgets")
    count = len(left)
    return PrimeAblationResult(
        candidate, count,
        sum(int(right[key].success) - int(left[key].success) for key in left) / count,
        sum(int(right[key].false_acceptance) - int(left[key].false_acceptance) for key in left) / count,
        sum(right[key].policy_violations - left[key].policy_violations for key in left),
        sum(right[key].memory_contamination - left[key].memory_contamination for key in left),
        sum(right[key].model_calls - left[key].model_calls for key in left) / count,
    )
