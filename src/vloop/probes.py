"""Bounded adversarial probes executed by protected verifier code.

The planner cannot supply executable probe code.  It can at most influence a
normal action that later causes the deterministic repair controller to request
probing.  Probe implementations are registered by the verifier deployment,
run outside the editable agent workspace, and return evidence only.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Callable, Iterable, Protocol

from .canonical import digest
from .models import ActionIntent, CheckResult, CheckStatus, ExecutionObservation, TaskContract, VerificationReport


class ProbeKind(StrEnum):
    EDGE_CASE = "edge-case"
    MUTATION = "mutation"
    COUNTEREXAMPLE = "counterexample"
    CONSISTENCY = "consistency"


def probe_policy_digest(policy_id: str, definitions: Iterable["ProbeDefinition"]) -> str:
    """Hash the complete reviewed probe manifest, not a mutable label alone."""

    return digest(
        {
            "probe_policy_id": policy_id,
            "probes": [definition.manifest for definition in sorted(definitions, key=lambda item: item.probe_id)],
        }
    )


@dataclass(frozen=True, slots=True)
class ProbeDefinition:
    probe_id: str
    kind: ProbeKind
    description: str
    trigger_categories: tuple[str, ...] = ("evidence",)
    implementation_image_digest: str = ""
    test_suite_digest: str = ""
    resource_profile_digest: str = ""

    def __post_init__(self) -> None:
        if not self.probe_id.strip() or not self.description.strip():
            raise ValueError("probe needs an id and description")
        if not self.trigger_categories or any(
            category not in {"correctness", "policy", "evidence", "quality"}
            for category in self.trigger_categories
        ):
            raise ValueError("probe has an invalid trigger category")

    @property
    def manifest(self) -> dict[str, object]:
        return {
            "probe_id": self.probe_id,
            "kind": self.kind.value,
            "trigger_categories": tuple(sorted(self.trigger_categories)),
            "implementation_image_digest": self.implementation_image_digest,
            "test_suite_digest": self.test_suite_digest,
            "resource_profile_digest": self.resource_profile_digest,
        }

    @property
    def production_ready(self) -> bool:
        return all(
            value.strip()
            and len(value) == 64
            and all(character in "0123456789abcdef" for character in value)
            for value in (
                self.implementation_image_digest,
                self.test_suite_digest,
                self.resource_profile_digest,
            )
        )


class ProtectedProbe(Protocol):
    definition: ProbeDefinition

    def run(
        self,
        *,
        contract: TaskContract,
        intent: ActionIntent,
        observation: ExecutionObservation,
        hard_report: VerificationReport,
    ) -> CheckResult: ...


@dataclass(frozen=True, slots=True)
class CallableProbe:
    """Test adapter for a probe hosted in the verifier environment."""

    definition: ProbeDefinition
    check: Callable[[TaskContract, ActionIntent, ExecutionObservation, VerificationReport], CheckResult]

    def run(
        self,
        *,
        contract: TaskContract,
        intent: ActionIntent,
        observation: ExecutionObservation,
        hard_report: VerificationReport,
    ) -> CheckResult:
        result = self.check(contract, intent, observation, hard_report)
        expected_name = f"probe:{self.definition.probe_id}"
        if result.name != expected_name:
            raise ValueError("probe result name does not match registered probe")
        return result


@dataclass(frozen=True, slots=True)
class ProbeReport:
    status: CheckStatus
    results: tuple[CheckResult, ...]


class ProtectedProbeRunner:
    """Runs pre-registered probes selected from hard verifier categories."""

    def __init__(self, probes: Iterable[ProtectedProbe], *, policy_id: str = "") -> None:
        registered = tuple(probes)
        ids = [probe.definition.probe_id for probe in registered]
        if len(ids) != len(set(ids)):
            raise ValueError("probe ids must be unique")
        if policy_id and not policy_id.strip():
            raise ValueError("probe policy id must be non-empty when supplied")
        self._probes = registered
        self._policy_id = policy_id

    @property
    def definitions(self) -> tuple[ProbeDefinition, ...]:
        return tuple(probe.definition for probe in self._probes)

    @property
    def policy_id(self) -> str:
        """Reviewed identity used by a production task profile."""

        return self._policy_id

    @property
    def policy_digest(self) -> str:
        return probe_policy_digest(self._policy_id, self.definitions) if self._policy_id else ""

    @property
    def production_ready(self) -> bool:
        return bool(self._policy_id) and bool(self._probes) and all(
            probe.definition.production_ready for probe in self._probes
        )

    def run(
        self,
        *,
        contract: TaskContract,
        intent: ActionIntent,
        observation: ExecutionObservation,
        hard_report: VerificationReport,
        force: bool = False,
    ) -> ProbeReport:
        categories = {
            "correctness": hard_report.correctness,
            "policy": hard_report.policy,
            "evidence": hard_report.evidence,
            "quality": hard_report.quality,
        }
        selected = [
            probe
            for probe in self._probes
            if force
            or any(categories[category] is not CheckStatus.PASS for category in probe.definition.trigger_categories)
        ]
        results: list[CheckResult] = []
        for probe in selected:
            try:
                results.append(
                    probe.run(
                        contract=contract,
                        intent=intent,
                        observation=observation,
                        hard_report=hard_report,
                    )
                )
            except Exception as exc:
                results.append(
                    CheckResult(
                        f"probe:{probe.definition.probe_id}",
                        CheckStatus.INCONCLUSIVE,
                        {"probe_kind": probe.definition.kind.value, "error_type": type(exc).__name__},
                        "protected probe did not complete",
                    )
                )
        if not results:
            return ProbeReport(CheckStatus.INCONCLUSIVE, ())
        if any(result.status is CheckStatus.FAIL for result in results):
            return ProbeReport(CheckStatus.FAIL, tuple(results))
        if any(result.status is CheckStatus.INCONCLUSIVE for result in results):
            return ProbeReport(CheckStatus.INCONCLUSIVE, tuple(results))
        return ProbeReport(CheckStatus.PASS, tuple(results))
