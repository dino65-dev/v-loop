"""Signed paired evaluations for safety-constrained harness promotion."""

from __future__ import annotations

import base64
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from random import Random
from statistics import fmean, median
from typing import Iterable, Mapping

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey

from .canonical import canonical_json, digest


class MetricDirection(StrEnum):
    MAXIMIZE = "maximize"
    MINIMIZE = "minimize"


@dataclass(frozen=True, slots=True)
class GraphExperimentKey:
    graph_digest: str
    harness_bundle_digest: str
    model_id: str
    model_config_digest: str
    dataset_digest: str
    tool_environment_digest: str
    evaluator_policy_digest: str
    compute_budget: int

    def __post_init__(self) -> None:
        if not self.model_id.strip() or self.compute_budget < 1:
            raise ValueError("experiments need a model and positive compute budget")
        for value in (self.graph_digest, self.harness_bundle_digest, self.model_config_digest, self.dataset_digest, self.tool_environment_digest, self.evaluator_policy_digest):
            if len(value) != 64:
                raise ValueError("experiment identities must be SHA-256 digests")

    @property
    def experiment_digest(self) -> str:
        return digest(self)


@dataclass(frozen=True, slots=True)
class PairedRunEvidence:
    task_id: str
    seed: int
    baseline_run_id: str
    candidate_run_id: str
    baseline_trace_root: str
    candidate_trace_root: str
    baseline_metrics: Mapping[str, float]
    candidate_metrics: Mapping[str, float]

    def __post_init__(self) -> None:
        if not all((self.task_id.strip(), self.baseline_run_id.strip(), self.candidate_run_id.strip())) or self.seed < 0:
            raise ValueError("paired evidence needs task, seed, and run identities")
        if len(self.baseline_trace_root) != 64 or len(self.candidate_trace_root) != 64:
            raise ValueError("paired evidence needs trace roots")
        if set(self.baseline_metrics) != set(self.candidate_metrics):
            raise ValueError("paired runs must report the same metric set")


@dataclass(frozen=True, slots=True)
class TopologyRun:
    """One measured run before it is admitted to a paired topology comparison."""

    task_id: str
    seed: int
    run_id: str
    trace_root: str
    metrics: Mapping[str, float]
    experiment: GraphExperimentKey

    def __post_init__(self) -> None:
        if not self.task_id.strip() or not self.run_id.strip() or self.seed < 0 or len(self.trace_root) != 64:
            raise ValueError("topology runs need task, seed, run, and trace identities")


def pair_topology_runs(
    baseline: Iterable[TopologyRun], candidate: Iterable[TopologyRun]
) -> tuple[PairedRunEvidence, ...]:
    """Match topologies by exact task and seed before any score is compared."""

    baseline_runs, candidate_runs = tuple(baseline), tuple(candidate)
    baseline_by_case = {(run.task_id, run.seed): run for run in baseline_runs}
    candidate_by_case = {(run.task_id, run.seed): run for run in candidate_runs}
    if not baseline_by_case or set(baseline_by_case) != set(candidate_by_case):
        raise ValueError("topology comparisons require exactly matched task IDs and random seeds")
    if len(baseline_by_case) != len(baseline_runs) or len(candidate_by_case) != len(candidate_runs):
        raise ValueError("topology comparisons cannot contain duplicate task-and-seed cases")
    pairs: list[PairedRunEvidence] = []
    for case in sorted(baseline_by_case):
        left, right = baseline_by_case[case], candidate_by_case[case]
        if not _comparable_experiments(left.experiment, right.experiment):
            raise ValueError("topology comparisons must hold model, data, evaluator, environment, and budget constant")
        pairs.append(
            PairedRunEvidence(
                left.task_id,
                left.seed,
                left.run_id,
                right.run_id,
                left.trace_root,
                right.trace_root,
                left.metrics,
                right.metrics,
            )
        )
    return tuple(pairs)


def _comparable_experiments(left: GraphExperimentKey, right: GraphExperimentKey) -> bool:
    return (
        left.model_id == right.model_id
        and left.model_config_digest == right.model_config_digest
        and left.dataset_digest == right.dataset_digest
        and left.tool_environment_digest == right.tool_environment_digest
        and left.evaluator_policy_digest == right.evaluator_policy_digest
        and left.compute_budget == right.compute_budget
    )


@dataclass(frozen=True, slots=True)
class PairedMetricStatistics:
    metric: str
    direction: MetricDirection
    mean_delta: float
    median_delta: float
    confidence_interval_95: tuple[float, float]


def paired_metric_statistics(
    pairs: Iterable[PairedRunEvidence], *, metric: str, direction: MetricDirection, bootstrap_samples: int = 2_000
) -> PairedMetricStatistics:
    """Deterministic non-parametric CI for a fixed paired evaluation set."""

    values = tuple(pairs)
    if not values or bootstrap_samples < 100 or any(metric not in pair.baseline_metrics for pair in values):
        raise ValueError("statistics need a metric in every paired result and a meaningful bootstrap count")
    signed = tuple(
        (pair.candidate_metrics[metric] - pair.baseline_metrics[metric])
        * (1 if direction is MetricDirection.MAXIMIZE else -1)
        for pair in values
    )
    random = Random(digest({"metric": metric, "pairs": values, "samples": bootstrap_samples}))
    samples = sorted(fmean(random.choice(signed) for _ in signed) for _ in range(bootstrap_samples))
    lower = samples[int(0.025 * (bootstrap_samples - 1))]
    upper = samples[int(0.975 * (bootstrap_samples - 1))]
    return PairedMetricStatistics(metric, direction, fmean(signed), median(signed), (lower, upper))


@dataclass(frozen=True, slots=True)
class HarnessEvaluationBundle:
    change_id: str
    baseline: GraphExperimentKey
    candidate: GraphExperimentKey
    primary_metric: str
    direction: MetricDirection
    minimum_improvement: float
    minimum_samples: int
    pairs: tuple[PairedRunEvidence, ...]
    signer_id: str
    signature: str = ""

    def __post_init__(self) -> None:
        if not self.change_id.strip() or not self.primary_metric.strip() or not self.signer_id.strip() or self.minimum_samples < 1:
            raise ValueError("evaluation bundles need identity, metric, signer, and sample floor")
        if self.baseline.model_id != self.candidate.model_id or self.baseline.model_config_digest != self.candidate.model_config_digest:
            raise ValueError("paired evaluations must hold model configuration constant")
        if self.baseline.dataset_digest != self.candidate.dataset_digest or self.baseline.tool_environment_digest != self.candidate.tool_environment_digest:
            raise ValueError("paired evaluations must hold dataset and tool environment constant")
        if len(self.pairs) < self.minimum_samples:
            raise ValueError("evaluation bundle has too few paired samples")
        if len({(pair.task_id, pair.seed) for pair in self.pairs}) != len(self.pairs):
            raise ValueError("evaluation bundle contains duplicate paired task-and-seed cases")
        if any(self.primary_metric not in pair.baseline_metrics for pair in self.pairs):
            raise ValueError("primary metric is missing from a paired run")

    def payload(self) -> bytes:
        return canonical_json({**asdict(self), "signature": ""}).encode("utf-8")

    @property
    def improvement(self) -> float:
        deltas = [pair.candidate_metrics[self.primary_metric] - pair.baseline_metrics[self.primary_metric] for pair in self.pairs]
        raw = fmean(deltas)
        return raw if self.direction is MetricDirection.MAXIMIZE else -raw

    def safety_regression(self, metric: str) -> float:
        return fmean(pair.candidate_metrics[metric] - pair.baseline_metrics[metric] for pair in self.pairs)

    def primary_statistics(self) -> PairedMetricStatistics:
        return paired_metric_statistics(self.pairs, metric=self.primary_metric, direction=self.direction)


class HarnessEvaluationSigner:
    def __init__(self, private_key: bytes | Ed25519PrivateKey | None = None, *, signer_id: str) -> None:
        self.signer_id = signer_id
        self._key = Ed25519PrivateKey.generate() if private_key is None else (Ed25519PrivateKey.from_private_bytes(private_key) if isinstance(private_key, bytes) else private_key)

    @property
    def public_key_bytes(self) -> bytes:
        return self._key.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)

    def sign(self, bundle: HarnessEvaluationBundle) -> HarnessEvaluationBundle:
        if bundle.signer_id != self.signer_id:
            raise ValueError("evaluation bundle signer differs from signing key")
        return replace(bundle, signature=base64.urlsafe_b64encode(self._key.sign(bundle.payload())).decode("ascii"))


class HarnessEvaluationVerifier:
    def __init__(self, public_keys: Mapping[str, bytes | Ed25519PublicKey]) -> None:
        self._keys = {name: Ed25519PublicKey.from_public_bytes(value) if isinstance(value, bytes) else value for name, value in public_keys.items()}

    def validate(self, bundle: HarnessEvaluationBundle) -> None:
        key = self._keys.get(bundle.signer_id)
        if key is None:
            raise PermissionError("evaluation signer is not trusted")
        try:
            key.verify(base64.urlsafe_b64decode(bundle.signature.encode("ascii")), bundle.payload())
        except (InvalidSignature, ValueError) as exc:
            raise PermissionError("evaluation bundle signature is invalid") from exc
        if bundle.improvement < bundle.minimum_improvement:
            raise PermissionError("evaluation did not meet its declared improvement")
        if bundle.primary_statistics().confidence_interval_95[0] < bundle.minimum_improvement:
            raise PermissionError("evaluation confidence interval does not meet its declared improvement")
        for metric in ("false_acceptance", "policy_violation"):
            if all(metric in pair.baseline_metrics for pair in bundle.pairs) and bundle.safety_regression(metric) > 0:
                raise PermissionError(f"evaluation regresses {metric}")


@dataclass(frozen=True, slots=True)
class HarnessReviewReceipt:
    change_id: str
    reviewer_id: str
    role: str
    issued_at: datetime
    expires_at: datetime
    nonce: str
    signer_id: str
    signature: str = ""

    def __post_init__(self) -> None:
        if not all((self.change_id.strip(), self.reviewer_id.strip(), self.role.strip(), self.nonce.strip(), self.signer_id.strip())):
            raise ValueError("review receipts need change, reviewer, role, nonce, and signer")
        if self.issued_at.tzinfo is None or self.expires_at <= self.issued_at:
            raise ValueError("review receipt validity is invalid")

    def payload(self) -> bytes:
        return canonical_json({**asdict(self), "issued_at": self.issued_at.isoformat(), "expires_at": self.expires_at.isoformat(), "signature": ""}).encode("utf-8")


class HarnessReviewSigner(HarnessEvaluationSigner):
    def issue(self, *, change_id: str, reviewer_id: str, role: str, ttl: timedelta = timedelta(minutes=10)) -> HarnessReviewReceipt:
        issued = datetime.now(UTC)
        receipt = HarnessReviewReceipt(change_id, reviewer_id, role, issued, issued + ttl, digest({"change_id": change_id, "reviewer_id": reviewer_id, "issued_at": issued.isoformat()}), self.signer_id)
        return replace(receipt, signature=base64.urlsafe_b64encode(self._key.sign(receipt.payload())).decode("ascii"))


class HarnessReviewVerifier(HarnessEvaluationVerifier):
    def validate_review(self, receipt: HarnessReviewReceipt, *, role: str) -> None:
        key = self._keys.get(receipt.signer_id)
        if key is None or receipt.role != role or receipt.expires_at <= datetime.now(UTC):
            raise PermissionError("harness review receipt is not trusted or current")
        try:
            key.verify(base64.urlsafe_b64decode(receipt.signature.encode("ascii")), receipt.payload())
        except (InvalidSignature, ValueError) as exc:
            raise PermissionError("harness review receipt signature is invalid") from exc
