"""Explicit development and production V-Loop runtime configurations.

The prototype retains useful local metadata verifiers. They must not become a
production deployment by omission, so production construction validates the
whole enforcement graph before any controller runs.
"""

from __future__ import annotations

from dataclasses import dataclass

from .authorization import SQLiteNonceStore
from .completion import FinalVerifier
from .executor import CapabilityEnforcingExecutor, SQLiteIdempotencyStore
from .firecracker import FirecrackerExecutor
from .models import TaskContract
from .policy import PolicyGate, SQLitePolicyUseCounterStore
from .probes import ProtectedProbeRunner
from .verifiers import (
    DevelopmentBenchmarkVerifier,
    DevelopmentDifferentialVerifier,
    DevelopmentIsolationVerifier,
    DevelopmentMetamorphicVerifier,
    HybridVerifier,
    SignedReceiptVerifier,
    StructuralVerifier,
)


class ProductionConfigurationError(ValueError):
    """The configured components cannot enforce the production trust boundary."""


@dataclass(frozen=True, slots=True)
class DevelopmentRuntime:
    """Explicitly labelled prototype configuration that may use local adapters."""

    contract: TaskContract
    executor: object
    verifier: HybridVerifier
    final_verifier: FinalVerifier | None = None
    probe_runner: ProtectedProbeRunner | None = None


@dataclass(slots=True)
class ProductionRuntimeBuilder:
    """Fail startup unless every mandatory production trust control is present."""

    contract: TaskContract
    executor: object
    verifier: HybridVerifier
    final_verifier: FinalVerifier | None
    probe_runner: ProtectedProbeRunner | None
    gate: PolicyGate | None = None

    def validate(self) -> None:
        failures: list[str] = []
        if not isinstance(self.executor, CapabilityEnforcingExecutor):
            failures.append("CapabilityEnforcingExecutor is required")
        else:
            if not isinstance(self.executor.idempotency_store, SQLiteIdempotencyStore):
                failures.append("a durable SQLite idempotency store is required")
            if not isinstance(self.executor.capability_verifier.nonce_store, SQLiteNonceStore):
                failures.append("a durable SQLite capability nonce store is required")
            if not isinstance(self.executor.raw_executor, FirecrackerExecutor):
                failures.append("production untrusted-code execution requires FirecrackerExecutor")
            else:
                supervisor = self.executor.raw_executor.supervisor_receipt_verifier
                if supervisor is None or supervisor.policy is None:
                    failures.append("Firecracker requires a policy-bound supervisor receipt verifier")

        if self.final_verifier is None:
            failures.append("a final verifier is required")
        if self.gate is None or self.gate.signed_approval_verifier is None:
            failures.append("a policy gate with signed approval verification is required")
        elif not isinstance(self.gate.use_counter_store, SQLitePolicyUseCounterStore):
            failures.append("a durable SQLite policy use-counter store is required")
        elif isinstance(self.executor, CapabilityEnforcingExecutor) and (
            self.gate.capability_public_key != self.executor.capability_verifier.public_key_bytes
        ):
            failures.append("policy gate and executor capability verifier use different public keys")
        if self.probe_runner is None or not self.probe_runner.definitions:
            failures.append("a non-empty protected probe policy is required")

        required = self.contract.required_verifiers
        for category in ("correctness", "policy", "evidence", "quality"):
            if not required.get(category):
                failures.append(f"contract requires at least one {category} verifier")

        checks = self.verifier.checks
        if not any(isinstance(check, StructuralVerifier) for check in checks):
            failures.append("StructuralVerifier is required")
        development_types = (
            DevelopmentBenchmarkVerifier,
            DevelopmentDifferentialVerifier,
            DevelopmentIsolationVerifier,
            DevelopmentMetamorphicVerifier,
        )
        if any(isinstance(check, development_types) for check in checks):
            failures.append("development metadata verifiers are forbidden in production")

        registered = {getattr(check, "name", ""): check for check in checks}
        for category, names in required.items():
            for name in names:
                check = registered.get(name)
                if not isinstance(check, SignedReceiptVerifier):
                    failures.append(f"required {category} verifier {name!r} must be a signed receipt verifier")
                elif check.receipt_verifier.policy is None:
                    failures.append(f"required verifier {name!r} lacks an immutable receipt policy")

        if failures:
            raise ProductionConfigurationError("production runtime rejected: " + "; ".join(failures))

        for check in checks:
            if isinstance(check, SignedReceiptVerifier) and check.receipt_verifier.policy is not None:
                if check.receipt_verifier.policy.minimum_schema_version < 2:
                    raise ProductionConfigurationError("production receipts must require schema v2")

    def build(self) -> "ProductionRuntimeBuilder":
        """Validate eagerly; the returned object is the deployment recipe."""

        self.validate()
        return self
