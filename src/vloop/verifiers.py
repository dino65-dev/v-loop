"""Hard verifier composition and deterministic acceptance."""

from __future__ import annotations

from typing import Callable, Iterable, Mapping, Protocol

from .models import CheckResult, CheckStatus, ExecutionObservation, TaskContract, VerificationReport


class Verifier(Protocol):
    def verify(self, contract: TaskContract, observation: ExecutionObservation) -> CheckResult: ...


class ExecutionVerifier:
    """Hard check that an external tool actually completed successfully."""

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

    def verify(self, contract: TaskContract, observation: ExecutionObservation) -> CheckResult:
        result = self._check(contract, observation)
        if result.name != self.name:
            raise ValueError("verifier result name does not match registered verifier")
        return result


class HybridVerifier:
    """Combines deterministic checks; learned scores may only add evidence."""

    def __init__(
        self,
        checks: Iterable[
            CallableVerifier | ExecutionVerifier | IsolationEvidenceVerifier | BenchmarkEvidenceVerifier
        ],
    ) -> None:
        self._checks = tuple(checks)

    def verify(self, contract: TaskContract, observation: ExecutionObservation) -> VerificationReport:
        results: list[CheckResult] = []
        categories: dict[str, list[CheckStatus]] = {
            "correctness": [],
            "policy": [],
            "evidence": [],
            "quality": [],
        }
        for check in self._checks:
            result = check.verify(contract, observation)
            results.append(result)
            categories[getattr(check, "category", "correctness")].append(result.status)

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
