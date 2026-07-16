"""Explicit development and production V-Loop runtime configurations.

The prototype retains useful local metadata verifiers. They must not become a
production deployment by omission, so production construction validates the
whole enforcement graph before any controller runs.
"""

from __future__ import annotations

from dataclasses import dataclass

from .authorization import SQLiteNonceStore
from .completion import FinalVerifier
from .evaluation import ProtectedEvaluationOrchestrator
from .executor import CapabilityEnforcingExecutor, SQLiteIdempotencyStore
from .firecracker import FirecrackerEffectReconciler, FirecrackerExecutor
from .models import TaskContract
from .policy import PolicyGate, SQLiteApprovalConsumptionStore, SQLitePolicyUseCounterStore
from .probes import ProtectedProbeRunner
from .run_state import SQLiteRunStateStore
from .services import FirecrackerSupervisorHTTPClient, LedgerAnchorHTTPClient, ProtectedEvaluatorHTTPClient
from .verifiers import (
    DevelopmentBenchmarkVerifier,
    DevelopmentDifferentialVerifier,
    DevelopmentIsolationVerifier,
    DevelopmentMetamorphicVerifier,
    ExecutionVerifier,
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
    evaluation_orchestrator: ProtectedEvaluationOrchestrator
    effect_reconciler: FirecrackerEffectReconciler

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
            evaluation_orchestrator=self.evaluation_orchestrator,
            effect_reconciler=self.effect_reconciler,
        ).validate()

    def create_loop(self, *, planner, ledger, **kwargs):
        """Construct a controller without exposing mutable runtime components."""

        from .controller import VerifiedLoop

        self.validate()

        protected_dependencies = {
            "contract",
            "gate",
            "executor",
            "verifier",
            "final_verifier",
            "probe_runner",
            "state_store",
            "evaluation_orchestrator",
            "effect_reconciler",
        }
        overridden = protected_dependencies.intersection(kwargs)
        if overridden:
            raise ProductionConfigurationError(
                "production loop dependencies are fixed by the validated runtime: "
                + ", ".join(sorted(overridden))
            )

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
            evaluation_orchestrator=self.evaluation_orchestrator,
            effect_reconciler=self.effect_reconciler,
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
    evaluation_orchestrator: ProtectedEvaluationOrchestrator | None = None
    effect_reconciler: FirecrackerEffectReconciler | None = None

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
                elif not self.executor.raw_executor.remote_asset_request:
                    failures.append("remote Firecracker requires opaque registered asset identities")
                supervisor = self.executor.raw_executor.supervisor_receipt_verifier
                if supervisor is None or supervisor.policy is None:
                    failures.append("Firecracker requires a policy-bound supervisor receipt verifier")
                elif not supervisor.has_verifier_owned_trust:
                    failures.append("Firecracker supervisor receipt verifier needs key lifecycle trust")
                elif (
                    supervisor.policy.workspace_snapshot_schema is None
                    or not supervisor.policy.workspace_exclusion_policy_digests
                ):
                    failures.append("Firecracker supervisor receipts need a canonical snapshot schema and exclusion policy")

        if self.final_verifier is None:
            failures.append("a final verifier is required")
        elif not self.contract.success_condition_bindings:
            failures.append("contract needs immutable success-condition bindings")
        elif getattr(self.final_verifier, "required_checks", None) != dict(self.contract.success_condition_bindings):
            failures.append("final verifier does not match the contract success-condition bindings")
        elif getattr(self.final_verifier, "global_completion_guards", None) != self.contract.global_completion_guards:
            failures.append("final verifier does not match the contract global completion guards")
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
        elif self.probe_runner.policy_digest != self.contract.probe_policy_digest:
            failures.append("protected probe policy does not match the task-profile digest")
        elif not self.probe_runner.production_ready:
            failures.append("production probes need immutable image, suite, and resource identities")
        if not isinstance(self.state_store, SQLiteRunStateStore):
            failures.append("a durable SQLite controller run-state store is required")
        if not isinstance(self.ledger_anchor, LedgerAnchorHTTPClient):
            failures.append("an authenticated external ledger-anchor client is required")
        if not isinstance(self.evaluation_orchestrator, ProtectedEvaluationOrchestrator):
            failures.append("a protected evaluator orchestrator is required")
        if not isinstance(self.effect_reconciler, FirecrackerEffectReconciler):
            failures.append("a signed Firecracker effect reconciler is required")
        elif not isinstance(self.executor, CapabilityEnforcingExecutor) or (
            self.effect_reconciler.executor is not self.executor.raw_executor
            or self.effect_reconciler.executor_id != self.executor.executor_id
        ):
            failures.append("effect reconciler is not bound to the configured Firecracker executor")

        required = self.contract.required_verifiers
        if any(rule.allow_unlisted_arguments for rule in self.contract.allowed_actions):
            failures.append("production action rules must use closed argument schemas")
        if not self.contract.require_argument_provenance:
            failures.append("production contracts must require per-argument provenance DAGs")
        if not all(
            (
                self.contract.task_kind,
                self.contract.risk_class,
                self.contract.probe_policy_digest,
                self.contract.profile_version,
                self.contract.profile_digest,
            )
        ):
            failures.append("production contracts need immutable task-profile metadata")
        if not self.contract.action_safety_checks:
            failures.append("production contracts need mandatory action-safety checks")
        elif "execution" not in self.contract.action_safety_checks:
            failures.append("production action-safety checks must require execution success")
        if not self.contract.global_completion_guards:
            failures.append("production contracts need global completion guards")
        elif not set(self.contract.action_safety_checks).issubset(self.contract.global_completion_guards):
            failures.append("mandatory action-safety checks must also be global completion guards")
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

        # Most verifier implementations expose their stable result name as
        # ``name``.  StructuralVerifier is intentionally stateless, so make
        # its protocol name explicit rather than relying on an implementation
        # attribute that it does not need at execution time.
        registered = {
            (
                "structural"
                if isinstance(check, StructuralVerifier)
                else "execution"
                if isinstance(check, ExecutionVerifier)
                else getattr(check, "name", "")
            ): check
            for check in checks
        }
        for category, names in required.items():
            for name in names:
                check = registered.get(name)
                if not isinstance(check, SignedReceiptVerifier):
                    failures.append(f"required {category} verifier {name!r} must be a signed receipt verifier")
                elif check.receipt_verifier.policy is None:
                    failures.append(f"required verifier {name!r} lacks an immutable receipt policy")
                elif not check.receipt_verifier.has_verifier_owned_trust:
                    failures.append(f"required verifier {name!r} lacks verifier-owned key lifecycle trust")
                elif not check.receipt_verifier.policy.workspace_exclusion_policy_digests:
                    failures.append(f"required verifier {name!r} lacks an approved snapshot exclusion policy")
        if isinstance(self.evaluation_orchestrator, ProtectedEvaluationOrchestrator):
            plans = {plan.check_name: plan for plan in self.evaluation_orchestrator.plans}
            for name in (
                name
                for names in required.values()
                for name in names
                if isinstance(registered.get(name), SignedReceiptVerifier)
            ):
                verifier = registered[name]
                assert isinstance(verifier, SignedReceiptVerifier)  # narrowed by comprehension
                plan = plans.get(name)
                policy = verifier.receipt_verifier.policy
                if plan is None:
                    failures.append(f"protected evaluator orchestration does not cover required receipt check {name!r}")
                elif not isinstance(plan.client, ProtectedEvaluatorHTTPClient):
                    failures.append(f"required evaluator plan {name!r} needs the protected evaluator service client")
                elif policy is None or (
                    plan.receipt_type != verifier.receipt_type
                    or plan.evaluator_image_digest not in policy.allowed_evaluator_images
                    or plan.test_suite_digest not in policy.allowed_test_suites
                ):
                    failures.append(f"required evaluator plan {name!r} does not match its receipt verifier policy")

        bound_checks = {
            name for names in self.contract.success_condition_bindings.values() for name in names
        }.union(self.contract.global_completion_guards)
        required_names = {name for names in required.values() for name in names}
        if not required_names.issubset(bound_checks):
            failures.append("all contract-required verifiers must bind to final criteria or global guards")
        if not set(self.contract.action_safety_checks).issubset(registered):
            failures.append("mandatory action-safety checks are not registered")
        available_completion_guards = set(registered).union(
            {f"probe:{definition.probe_id}" for definition in self.probe_runner.definitions}
            if self.probe_runner is not None
            else set()
        )
        if not set(self.contract.global_completion_guards).issubset(available_completion_guards):
            failures.append("global completion guards are not registered checks or protected probes")
        if self.probe_runner is not None:
            required_probe_guards = {f"probe:{definition.probe_id}" for definition in self.probe_runner.definitions}
            if not required_probe_guards.issubset(self.contract.global_completion_guards):
                failures.append("every production protected probe must be a global completion guard")

        if failures:
            raise ProductionConfigurationError("production runtime rejected: " + "; ".join(failures))

        for check in checks:
            if isinstance(check, SignedReceiptVerifier) and check.receipt_verifier.policy is not None:
                if check.receipt_verifier.policy.minimum_schema_version < 2:
                    raise ProductionConfigurationError("production receipts must require schema v2")
                if (
                    check.receipt_verifier.policy.workspace_snapshot_schema is None
                    or not check.receipt_verifier.policy.workspace_exclusion_policy_digests
                ):
                    raise ProductionConfigurationError(
                        "production receipts must require a canonical snapshot schema and exclusion policy"
                    )

    def build(self) -> ProductionRuntime:
        """Return a frozen deployment recipe only after all checks pass."""

        self.validate()
        assert isinstance(self.executor, CapabilityEnforcingExecutor)
        assert self.final_verifier is not None
        assert self.probe_runner is not None
        assert self.gate is not None
        assert self.state_store is not None
        assert self.ledger_anchor is not None
        assert self.evaluation_orchestrator is not None
        assert self.effect_reconciler is not None
        return ProductionRuntime(
            contract=self.contract,
            executor=self.executor,
            verifier=self.verifier,
            final_verifier=self.final_verifier,
            probe_runner=self.probe_runner,
            gate=self.gate,
            state_store=self.state_store,
            ledger_anchor=self.ledger_anchor,
            evaluation_orchestrator=self.evaluation_orchestrator,
            effect_reconciler=self.effect_reconciler,
        )
