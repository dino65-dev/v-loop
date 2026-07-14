"""Hard verifier composition and deterministic acceptance."""

from __future__ import annotations

from typing import Callable, Iterable, Mapping, Protocol

from .models import ActionIntent, CheckResult, CheckStatus, ExecutionObservation, TaskContract, VerificationReport
from .receipts import EvaluationReceipt, ReceiptRejected, ReceiptVerifier


class Verifier(Protocol):
    def verify(self, contract: TaskContract, observation: ExecutionObservation) -> CheckResult: ...


class StructuralVerifier:
    """L0 schema validation before any semantic interpretation of evidence."""

    category = "evidence"
    phase = 0

    def verify(self, contract: TaskContract, observation: ExecutionObservation) -> CheckResult:
        del contract
        valid = (
            isinstance(observation.success, bool)
            and (observation.exit_code is None or isinstance(observation.exit_code, int))
            and isinstance(observation.stdout, str)
            and isinstance(observation.stderr, str)
            and isinstance(observation.artifact_digests, Mapping)
            and isinstance(observation.metadata, Mapping)
            and all(
                isinstance(name, str) and isinstance(value, str)
                for name, value in observation.artifact_digests.items()
            )
        )
        return CheckResult(
            "structural",
            CheckStatus.PASS if valid else CheckStatus.FAIL,
            {
                "has_mapping_artifacts": isinstance(observation.artifact_digests, Mapping),
                "has_mapping_metadata": isinstance(observation.metadata, Mapping),
            },
            "execution observation does not satisfy the protected schema" if not valid else "",
        )


class ExecutionVerifier:
    """Hard check that an external tool actually completed successfully."""

    phase = 2

    def verify(self, contract: TaskContract, observation: ExecutionObservation) -> CheckResult:
        return CheckResult(
            name="execution",
            status=CheckStatus.PASS if observation.success else CheckStatus.FAIL,
            evidence={"exit_code": observation.exit_code, "artifacts": dict(observation.artifact_digests)},
            message=observation.stderr[-500:],
        )


class IsolationEvidenceVerifier:
    """Hard policy check for the expected Firecracker isolation evidence."""

    category = "policy"
    phase = 1

    def verify(self, contract: TaskContract, observation: ExecutionObservation) -> CheckResult:
        metadata = observation.metadata
        required = {
            "executor": "firecracker",
            "isolation": "microvm",
            "rootfs_read_only": True,
            "network_enabled": False,
        }
        missing_or_wrong = {
            key: {"expected": expected, "actual": metadata.get(key)}
            for key, expected in required.items()
            if metadata.get(key) != expected
        }
        for digest_key in ("config_digest", "guest_manifest_digest", "guest_result_path"):
            if not metadata.get(digest_key):
                missing_or_wrong[digest_key] = {"expected": "present", "actual": metadata.get(digest_key)}
        return CheckResult(
            name="firecracker-isolation",
            status=CheckStatus.PASS if not missing_or_wrong else CheckStatus.FAIL,
            evidence={"metadata": dict(metadata), "violations": missing_or_wrong},
            message="microVM isolation evidence missing or inconsistent" if missing_or_wrong else "",
        )


class BenchmarkEvidenceVerifier:
    """Checks repeatable benchmark evidence without trusting a textual claim."""

    category = "quality"
    phase = 3

    def __init__(
        self,
        *,
        minimum_samples: int = 5,
        minimum_warmups: int = 1,
        maximum_regression_ratio: float = 0.0,
    ) -> None:
        self.minimum_samples = minimum_samples
        self.minimum_warmups = minimum_warmups
        self.maximum_regression_ratio = maximum_regression_ratio

    def verify(self, contract: TaskContract, observation: ExecutionObservation) -> CheckResult:
        benchmark = observation.metadata.get("benchmark")
        if not isinstance(benchmark, Mapping):
            return CheckResult("benchmark-evidence", CheckStatus.INCONCLUSIVE, {}, "missing benchmark data")
        required = ("config_digest", "synchronized", "samples", "warmups", "baseline_median_ms", "candidate_median_ms")
        if any(field not in benchmark for field in required):
            return CheckResult(
                "benchmark-evidence",
                CheckStatus.INCONCLUSIVE,
                {"benchmark": dict(benchmark)},
                "incomplete benchmark evidence",
            )
        samples = benchmark["samples"]
        warmups = benchmark["warmups"]
        baseline = benchmark["baseline_median_ms"]
        candidate = benchmark["candidate_median_ms"]
        valid = (
            isinstance(samples, int)
            and isinstance(warmups, int)
            and isinstance(baseline, (int, float))
            and isinstance(candidate, (int, float))
            and samples >= self.minimum_samples
            and warmups >= self.minimum_warmups
            and benchmark["synchronized"] is True
            and bool(benchmark["config_digest"])
            and baseline > 0
            and candidate > 0
        )
        if not valid:
            return CheckResult(
                "benchmark-evidence",
                CheckStatus.FAIL,
                {"benchmark": dict(benchmark)},
                "invalid or insufficient benchmark evidence",
            )
        regression = candidate / baseline - 1
        return CheckResult(
            "benchmark-evidence",
            CheckStatus.PASS if regression <= self.maximum_regression_ratio else CheckStatus.FAIL,
            {"benchmark": dict(benchmark), "regression_ratio": regression},
            "benchmark regression" if regression > self.maximum_regression_ratio else "",
        )


class DifferentialEvidenceVerifier:
    """L2 reference-comparison receipt verifier.

    A protected evaluator runs the reference and candidate.  The agent only
    receives its compact receipt; fabricated or incomplete receipts are not a
    successful differential test.
    """

    category = "correctness"
    phase = 2

    def verify(self, contract: TaskContract, observation: ExecutionObservation) -> CheckResult:
        del contract
        receipt = observation.metadata.get("differential")
        if not isinstance(receipt, Mapping):
            return CheckResult("differential", CheckStatus.INCONCLUSIVE, {}, "missing differential receipt")
        required = ("reference_digest", "candidate_digest", "passed")
        valid = (
            all(field in receipt for field in required)
            and isinstance(receipt.get("reference_digest"), str)
            and bool(receipt.get("reference_digest"))
            and isinstance(receipt.get("candidate_digest"), str)
            and bool(receipt.get("candidate_digest"))
            and isinstance(receipt.get("passed"), bool)
        )
        if not valid:
            return CheckResult(
                "differential",
                CheckStatus.FAIL,
                {"receipt": dict(receipt)},
                "invalid differential receipt",
            )
        return CheckResult(
            "differential",
            CheckStatus.PASS if receipt["passed"] else CheckStatus.FAIL,
            {"receipt": dict(receipt)},
            "reference comparison failed" if not receipt["passed"] else "",
        )


class MetamorphicEvidenceVerifier:
    """L2 transformation-invariant receipt verifier."""

    category = "correctness"
    phase = 2

    def verify(self, contract: TaskContract, observation: ExecutionObservation) -> CheckResult:
        del contract
        receipt = observation.metadata.get("metamorphic")
        if not isinstance(receipt, Mapping) or not isinstance(receipt.get("relations"), list):
            return CheckResult("metamorphic", CheckStatus.INCONCLUSIVE, {}, "missing metamorphic receipt")
        relations = receipt["relations"]
        valid = bool(relations) and all(
            isinstance(relation, Mapping)
            and isinstance(relation.get("name"), str)
            and bool(relation.get("name"))
            and isinstance(relation.get("evidence_digest"), str)
            and bool(relation.get("evidence_digest"))
            and isinstance(relation.get("passed"), bool)
            for relation in relations
        )
        if not valid:
            return CheckResult(
                "metamorphic",
                CheckStatus.FAIL,
                {"receipt": dict(receipt)},
                "invalid metamorphic receipt",
            )
        passed = all(relation["passed"] for relation in relations)
        return CheckResult(
            "metamorphic",
            CheckStatus.PASS if passed else CheckStatus.FAIL,
            {"receipt": dict(receipt)},
            "a transformation invariant failed" if not passed else "",
        )


class SignedReceiptVerifier:
    """Authenticates an evaluator or supervisor receipt before trusting claims."""

    phase = 1

    def __init__(
        self,
        *,
        name: str,
        category: str,
        receipt_type: str,
        receipt_verifier: ReceiptVerifier,
    ) -> None:
        if category not in {"correctness", "policy", "evidence", "quality"}:
            raise ValueError("invalid receipt verifier category")
        self.name = name
        self.category = category
        self.receipt_type = receipt_type
        self.receipt_verifier = receipt_verifier

    def verify_with_context(
        self,
        contract: TaskContract,
        observation: ExecutionObservation,
        *,
        run_id: str | None,
        intent: ActionIntent | None,
    ) -> CheckResult:
        del contract
        if run_id is None or intent is None:
            return CheckResult(self.name, CheckStatus.INCONCLUSIVE, {}, "missing verification binding")
        receipts = observation.metadata.get("evaluator_receipts")
        raw = receipts.get(self.receipt_type) if isinstance(receipts, Mapping) else None
        if not isinstance(raw, Mapping):
            return CheckResult(self.name, CheckStatus.INCONCLUSIVE, {}, "missing signed evaluator receipt")
        try:
            receipt = EvaluationReceipt.from_mapping(raw)
            self.receipt_verifier.validate(
                receipt,
                receipt_type=self.receipt_type,
                run_id=run_id,
                intent_digest=intent.intent_digest,
                artifact_digests=observation.artifact_digests,
            )
        except (KeyError, TypeError, ValueError, ReceiptRejected) as exc:
            return CheckResult(
                self.name,
                CheckStatus.FAIL,
                {"receipt_type": self.receipt_type, "error_type": type(exc).__name__},
                "signed evaluator receipt was rejected",
            )
        status = {
            "pass": CheckStatus.PASS,
            "fail": CheckStatus.FAIL,
            "inconclusive": CheckStatus.INCONCLUSIVE,
        }[receipt.result]
        return CheckResult(
            self.name,
            status,
            {
                "receipt_type": receipt.receipt_type,
                "candidate_artifact_digest": receipt.candidate_artifact_digest,
                "evaluator_image_digest": receipt.evaluator_image_digest,
                "test_suite_digest": receipt.test_suite_digest,
                "nonce": receipt.nonce,
            },
            "protected evaluator reported a failure" if status is CheckStatus.FAIL else "",
        )

class CallableVerifier:
    """Adapter for test, compiler, differential, policy, or quality checks."""

    def __init__(
        self,
        name: str,
        category: str,
        check: Callable[[TaskContract, ExecutionObservation], CheckResult],
    ) -> None:
        if category not in {"correctness", "policy", "evidence", "quality"}:
            raise ValueError("invalid verifier category")
        self.name, self.category, self._check = name, category, check
        self.phase = 2 if category in {"correctness", "policy", "evidence"} else 3

    def verify(self, contract: TaskContract, observation: ExecutionObservation) -> CheckResult:
        result = self._check(contract, observation)
        if result.name != self.name:
            raise ValueError("verifier result name does not match registered verifier")
        return result


class HybridVerifier:
    """Combines deterministic checks; learned scores may only add evidence."""

    def __init__(
        self,
        checks: Iterable[Verifier],
    ) -> None:
        self._checks = tuple(checks)

    def verify(
        self,
        contract: TaskContract,
        observation: ExecutionObservation,
        *,
        run_id: str | None = None,
        intent: ActionIntent | None = None,
    ) -> VerificationReport:
        results: list[CheckResult] = []
        categories: dict[str, list[CheckStatus]] = {
            "correctness": [],
            "policy": [],
            "evidence": [],
            "quality": [],
        }
        seen_names: set[str] = set()
        for check in sorted(self._checks, key=lambda item: getattr(item, "phase", 2)):
            try:
                contextual = getattr(check, "verify_with_context", None)
                result = (
                    contextual(contract, observation, run_id=run_id, intent=intent)
                    if callable(contextual)
                    else check.verify(contract, observation)
                )
            except Exception as exc:
                result = CheckResult(
                    getattr(check, "name", type(check).__name__),
                    CheckStatus.INCONCLUSIVE,
                    {"error_type": type(exc).__name__},
                    "verifier raised an exception",
                )
            if result.name in seen_names:
                raise ValueError(f"duplicate verifier result name: {result.name}")
            seen_names.add(result.name)
            results.append(result)
            categories[getattr(check, "category", "correctness")].append(result.status)
            if getattr(check, "phase", 2) <= 1 and result.status is not CheckStatus.PASS:
                break

        def reduce(statuses: list[CheckStatus]) -> CheckStatus:
            if not statuses:
                return CheckStatus.INCONCLUSIVE
            if CheckStatus.FAIL in statuses:
                return CheckStatus.FAIL
            if CheckStatus.INCONCLUSIVE in statuses:
                return CheckStatus.INCONCLUSIVE
            return CheckStatus.PASS

        return VerificationReport(
            correctness=reduce(categories["correctness"]),
            policy=reduce(categories["policy"]) if categories["policy"] else CheckStatus.PASS,
            evidence=reduce(categories["evidence"]) if categories["evidence"] else CheckStatus.PASS,
            quality=reduce(categories["quality"]) if categories["quality"] else CheckStatus.PASS,
            checks=tuple(results),
        )
