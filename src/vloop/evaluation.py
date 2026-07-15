"""Protected evaluator orchestration for signed task evidence."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Protocol

from .models import ActionIntent, ExecutionObservation, TaskContract
from .services import ProtectedEvaluatorHTTPClient
from .snapshot import WorkspaceSnapshot


class WorkspaceSnapshotProvider(Protocol):
    """Deployment-owned provider over an immutable evaluator workspace."""

    def snapshot(
        self,
        *,
        contract: TaskContract,
        intent: ActionIntent,
        observation: ExecutionObservation,
    ) -> WorkspaceSnapshot: ...


@dataclass(frozen=True, slots=True)
class ProtectedEvaluatorPlan:
    check_name: str
    receipt_type: str
    client: ProtectedEvaluatorHTTPClient
    evaluator_image_digest: str
    test_suite_digest: str

    def __post_init__(self) -> None:
        if not all(
            (
                self.check_name.strip(),
                self.receipt_type.strip(),
                self.evaluator_image_digest.strip(),
                self.test_suite_digest.strip(),
            )
        ):
            raise ValueError("protected evaluator plans need complete immutable identities")


@dataclass(frozen=True, slots=True)
class EvidenceBundle:
    workspace_snapshot: WorkspaceSnapshot
    evaluator_receipts: Mapping[str, Mapping]


class ProtectedEvaluationOrchestrator:
    """Calls only deployment-registered evaluators for an immutable snapshot."""

    def __init__(
        self,
        snapshot_provider: WorkspaceSnapshotProvider,
        plans: tuple[ProtectedEvaluatorPlan, ...],
    ) -> None:
        if not plans:
            raise ValueError("protected evaluation needs at least one evaluator plan")
        names = [plan.check_name for plan in plans]
        receipt_types = [plan.receipt_type for plan in plans]
        if len(names) != len(set(names)) or len(receipt_types) != len(set(receipt_types)):
            raise ValueError("protected evaluator plans need unique check and receipt names")
        self.snapshot_provider = snapshot_provider
        self.plans = plans

    @property
    def check_names(self) -> tuple[str, ...]:
        return tuple(plan.check_name for plan in self.plans)

    def evaluate(
        self,
        *,
        run_id: str,
        contract: TaskContract,
        intent: ActionIntent,
        observation: ExecutionObservation,
    ) -> EvidenceBundle:
        snapshot = self.snapshot_provider.snapshot(
            contract=contract,
            intent=intent,
            observation=observation,
        )
        receipts: dict[str, Mapping] = {}
        for plan in self.plans:
            receipts[plan.receipt_type] = plan.client.evaluate(
                run_id=run_id,
                contract_digest=contract.contract_digest,
                intent_digest=intent.intent_digest,
                receipt_type=plan.receipt_type,
                artifact_digests=observation.artifact_digests,
                workspace_snapshot_digest=snapshot.workspace_snapshot_digest,
                evaluator_image_digest=plan.evaluator_image_digest,
                test_suite_digest=plan.test_suite_digest,
            )
        return EvidenceBundle(snapshot, receipts)
