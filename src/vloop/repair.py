"""Deterministic repair directives with advisory neural context."""

from __future__ import annotations

from dataclasses import dataclass

from .models import CheckStatus, LoopDecision, VerificationReport
from .neural_verifier import NeuralDiagnostic


@dataclass(frozen=True, slots=True)
class RepairDirective:
    decision: LoopDecision
    stage: str
    reason: str
    advisory_stage: str | None = None
    evidence_gaps: tuple[str, ...] = ()


class RepairController:
    """Maps hard verifier outcomes to bounded next steps.

    Model output is retained as advice only. In particular, a neural pass never
    converts a failed deterministic check into acceptance.
    """

    _STAGES = frozenset({"local-repair", "replan", "probe", "escalate"})

    def direct(
        self, report: VerificationReport, diagnostic: NeuralDiagnostic | None = None
    ) -> RepairDirective:
        advisory = diagnostic.suggested_stage if diagnostic and diagnostic.suggested_stage in self._STAGES else None
        gaps = diagnostic.evidence_gaps if diagnostic else ()
        if report.policy is CheckStatus.FAIL:
            return RepairDirective(
                LoopDecision.ESCALATE,
                "escalate",
                "hard policy failure",
                advisory,
                gaps,
            )
        if report.correctness is CheckStatus.FAIL:
            return RepairDirective(
                LoopDecision.REPAIR,
                "local-repair",
                "hard correctness failure",
                advisory,
                gaps,
            )
        if report.evidence is not CheckStatus.PASS:
            return RepairDirective(
                LoopDecision.REPLAN,
                "probe",
                "evidence is missing or conflicting",
                advisory,
                gaps,
            )
        if report.quality is not CheckStatus.PASS:
            return RepairDirective(
                LoopDecision.REPAIR,
                "local-repair",
                "quality target was not met",
                advisory,
                gaps,
            )
        return RepairDirective(LoopDecision.REPLAN, "replan", "inconclusive verification", advisory, gaps)
