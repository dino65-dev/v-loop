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
from .policy import PolicyGate, SQLiteApprovalConsumptionStore, SQLitePolicyUseCounterStore
from .probes import ProtectedProbeRunner
from .run_state import SQLiteRunStateStore
from .services import FirecrackerSupervisorHTTPClient, LedgerAnchorHTTPClient
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


@dataclass(frozen=True, slots=True)
class ProductionRuntime:
    """Frozen, validated construction recipe for a production V-Loop run."""

    contract: TaskContract
    executor: CapabilityEnforcingExecutor
    verifier: HybridVerifier
    final_verifier: FinalVerifier
    probe_runner: ProtectedProbeRunner
    gate: PolicyGate
    state_store: SQLiteRunStateStore
    ledger_anchor: LedgerAnchorHTTPClient

    def validate(self) -> None:
        """Revalidate mutable component internals immediately before startup."""

        ProductionRuntimeBuilder(
            contract=self.contract,
            executor=self.executor,
            verifier=self.verifier,
            final_verifier=self.final_verifier,
            probe_runner=self.probe_runner,
            gate=self.gate,
            state_store=self.state_store,
            ledger_anchor=self.ledger_anchor,
        ).validate()

    def create_loop(self, *, planner, ledger, **kwargs):
        """Construct a controller without exposing mutable runtime components."""

        from .controller import VerifiedLoop

        self.validate()

        memory_committer = kwargs.get("memory_committer")
        memory_candidate_producer = kwargs.get("memory_candidate_producer")
        if (memory_committer is None) != (memory_candidate_producer is None):
            raise ProductionConfigurationError("production memory requires both a committer and server-owned producer")
        if memory_committer is not None:
            write_authority = getattr(memory_committer.write_gate, "claim_authority", None)
            ledger_authority = getattr(memory_committer.memory_ledger, "claim_authority", None)
            if write_authority is None or ledger_authority is None or write_authority is not ledger_authority:
                raise ProductionConfigurationError(
                    "production memory requires one shared server-owned memory claim authority"
                )

        return VerifiedLoop(
            contract=self.contract,
            planner=planner,
            gate=self.gate,
            executor=self.executor,
            verifier=self.verifier,
            ledger=ledger,
            final_verifier=self.final_verifier,
            probe_runner=self.probe_runner,
            state_store=self.state_store,
            **kwargs,
        )

    def create_anchor_worker(self, *, ledger, worker_id: str):
        """Create the separately scheduled external ledger-anchor publisher."""

        from .ledger import LedgerAnchorWorker

        return LedgerAnchorWorker(ledger, self.ledger_anchor, worker_id=worker_id)


@dataclass(slots=True)
class ProductionRuntimeBuilder:
    """Fail startup unless every mandatory production trust control is present."""

    contract: TaskContract
    executor: object
    verifier: HybridVerifier
    final_verifier: FinalVerifier | None
    probe_runner: ProtectedProbeRunner | None
    gate: PolicyGate | None = None
    state_store: SQLiteRunStateStore | None = None
    ledger_anchor: LedgerAnchorHTTPClient | None = None

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
                if not isinstance(self.executor.raw_executor.supervisor, FirecrackerSupervisorHTTPClient):
                    failures.append("Firecracker requires an authenticated remote supervisor client")
                supervisor = self.executor.raw_executor.supervisor_receipt_verifier
                if supervisor is None or supervisor.policy is None:
                    failures.append("Firecracker requires a policy-bound supervisor receipt verifier")
                elif not supervisor.has_verifier_owned_trust:
                    failures.append("Firecracker supervisor receipt verifier needs key lifecycle trust")
                elif supervisor.policy.workspace_snapshot_schema is None:
                    failures.append("Firecracker supervisor receipts need a canonical workspace snapshot schema")

        if self.final_verifier is None:
            failures.append("a final verifier is required")
        elif not self.contract.success_condition_bindings:
            failures.append("contract needs immutable success-condition bindings")
        elif getattr(self.final_verifier, "required_checks", None) != dict(self.contract.success_condition_bindings):
            failures.append("final verifier does not match the contract success-condition bindings")
        if self.gate is None or self.gate.signed_approval_verifier is None:
            failures.append("a policy gate with signed approval verification is required")
        elif not self.gate.signed_approval_verifier.has_verifier_owned_trust:
            failures.append("signed approvals need verifier-owned key-to-subject trust entries")
        elif not isinstance(self.gate.signed_approval_verifier.consumption_store, SQLiteApprovalConsumptionStore):
            failures.append("signed approvals need durable one-time consumption")
        elif not isinstance(self.gate.use_counter_store, SQLitePolicyUseCounterStore):
            failures.append("a durable SQLite policy use-counter store is required")
        elif isinstance(self.executor, CapabilityEnforcingExecutor) and (
            self.gate.capability_public_key != self.executor.capability_verifier.public_key_bytes
        ):
            failures.append("policy gate and executor capability verifier use different public keys")
        elif self.gate.contract.contract_digest != self.contract.contract_digest:
            failures.append("policy gate and runtime use different contracts")
        if self.probe_runner is None or not self.probe_runner.definitions:
            failures.append("a non-empty protected probe policy is required")
        if not isinstance(self.state_store, SQLiteRunStateStore):
            failures.append("a durable SQLite controller run-state store is required")
        if not isinstance(self.ledger_anchor, LedgerAnchorHTTPClient):
            failures.append("an authenticated external ledger-anchor client is required")

        required = self.contract.required_verifiers
        if any(rule.allow_unlisted_arguments for rule in self.contract.allowed_actions):
            failures.append("production action rules must use closed argument schemas")
        if not self.contract.require_argument_provenance:
            failures.append("production contracts must require per-argument provenance DAGs")
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
                elif not check.receipt_verifier.has_verifier_owned_trust:
                    failures.append(f"required verifier {name!r} lacks verifier-owned key lifecycle trust")

        if failures:
            raise ProductionConfigurationError("production runtime rejected: " + "; ".join(failures))

        for check in checks:
            if isinstance(check, SignedReceiptVerifier) and check.receipt_verifier.policy is not None:
                if check.receipt_verifier.policy.minimum_schema_version < 2:
                    raise ProductionConfigurationError("production receipts must require schema v2")
                if check.receipt_verifier.policy.workspace_snapshot_schema is None:
                    raise ProductionConfigurationError("production receipts must require a canonical workspace snapshot schema")

    def build(self) -> ProductionRuntime:
        """Return a frozen deployment recipe only after all checks pass."""

        self.validate()
        assert isinstance(self.executor, CapabilityEnforcingExecutor)
        assert self.final_verifier is not None
        assert self.probe_runner is not None
        assert self.gate is not None
        assert self.state_store is not None
        assert self.ledger_anchor is not None
        return ProductionRuntime(
            contract=self.contract,
            executor=self.executor,
            verifier=self.verifier,
            final_verifier=self.final_verifier,
            probe_runner=self.probe_runner,
            gate=self.gate,
            state_store=self.state_store,
            ledger_anchor=self.ledger_anchor,
        )
