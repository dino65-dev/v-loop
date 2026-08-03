"""Durable, fail-closed checkpoints for the verified controller.

The state store records controller progress separately from the evidence
ledger.  It never attempts to make an uncertain external effect retryable: a
checkpoint written immediately before execution is restored as
``pending-effect`` and requires supervisor/operator reconciliation.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Mapping, Protocol

from .canonical import canonical_json
from .completion import ActionEvidence, EvidenceSnapshot
from .models import (
    ActionIntent,
    ArgumentProvenance,
    ArgumentProvenanceNode,
    CheckResult,
    CheckStatus,
    Effect,
    ExecutionObservation,
    PreparedExecution,
    Provenance,
    VerificationReport,
)


class RunPhase(StrEnum):
    READY = "ready"
    PENDING_AUTHORIZATION = "pending-authorization"
    PENDING_EFFECT = "pending-effect"
    AWAITING_APPROVAL = "awaiting-approval"
    RECONCILIATION_REQUIRED = "reconciliation-required"
    RECONCILED_EFFECT = "reconciled-effect"
    TERMINAL = "terminal"


class RunStateConflict(RuntimeError):
    """Another controller instance advanced the same run checkpoint."""


@dataclass(frozen=True, slots=True)
class RunCheckpoint:
    run_id: str
    contract_digest: str
    phase: RunPhase
    next_iteration: int
    tool_calls: int
    history: tuple[dict[str, Any], ...]
    seen_failures: tuple[tuple[str, str], ...]
    evidence: EvidenceSnapshot
    graph_digest: str = ""
    pending_intent: ActionIntent | None = None
    executor_id: str = ""
    prepared_execution: PreparedExecution | None = None
    reconciled_observation: ExecutionObservation | None = None
    terminal_decision: str | None = None
    terminal_reason: str | None = None
    revision: int = 0
    updated_at: datetime = datetime.min.replace(tzinfo=UTC)

    def __post_init__(self) -> None:
        if not self.run_id.strip() or not self.contract_digest.strip():
            raise ValueError("run checkpoints need a run and contract identity")
        if self.graph_digest and (len(self.graph_digest) != 64 or any(c not in "0123456789abcdef" for c in self.graph_digest)):
            raise ValueError("run checkpoint graph digest must be SHA-256 hex")
        if self.next_iteration < 1 or self.tool_calls < 0 or self.revision < 0:
            raise ValueError("run checkpoint counters are invalid")
        if self.evidence.run_id != self.run_id:
            raise ValueError("run checkpoint evidence belongs to another run")
        pending_phases = {
            RunPhase.PENDING_AUTHORIZATION,
            RunPhase.PENDING_EFFECT,
            RunPhase.AWAITING_APPROVAL,
            RunPhase.RECONCILIATION_REQUIRED,
            RunPhase.RECONCILED_EFFECT,
        }
        if self.phase in pending_phases:
            if self.pending_intent is None:
                raise ValueError("pending run checkpoints need their exact intent")
            if not self.executor_id.strip():
                raise ValueError("pending run checkpoints need their intended executor")
        if self.phase in {
            RunPhase.PENDING_EFFECT,
            RunPhase.RECONCILIATION_REQUIRED,
            RunPhase.RECONCILED_EFFECT,
        }:
            if self.prepared_execution is None:
                raise ValueError("effect checkpoints need a prepared operation")
            if self.prepared_execution.executor_id != self.executor_id:
                raise ValueError("prepared execution belongs to another executor")
            if self.pending_intent is not None and self.prepared_execution.intent_digest != self.pending_intent.intent_digest:
                raise ValueError("prepared execution belongs to another intent")
            if self.graph_digest and (
                self.prepared_execution.graph_digest != self.graph_digest
                or self.prepared_execution.graph_node_id != "operation.prepared"
            ):
                raise ValueError("prepared execution is not bound to this checkpoint graph")
        elif self.prepared_execution is not None:
            raise ValueError("only effect checkpoints may retain a prepared operation")
        if self.phase not in pending_phases and (self.pending_intent is not None or self.executor_id):
            raise ValueError("only a pending checkpoint may retain an intent")
        if self.phase is RunPhase.RECONCILED_EFFECT and self.reconciled_observation is None:
            raise ValueError("reconciled effects need an attested observation")
        if self.phase is not RunPhase.RECONCILED_EFFECT and self.reconciled_observation is not None:
            raise ValueError("only reconciled effects may retain an observation")
        if self.phase is RunPhase.TERMINAL:
            if not self.terminal_decision or not self.terminal_reason:
                raise ValueError("terminal checkpoints need a decision and reason")
        elif self.terminal_decision is not None or self.terminal_reason is not None:
            raise ValueError("non-terminal checkpoint cannot have a terminal result")


class RunStateStore(Protocol):
    def load(self, run_id: str) -> RunCheckpoint | None: ...

    def save(self, checkpoint: RunCheckpoint) -> RunCheckpoint: ...


class SQLiteRunStateStore:
    """Optimistically versioned controller checkpoints backed by SQLite."""

    def __init__(self, database: str | Path) -> None:
        path = Path(database)
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(path, isolation_level=None, timeout=5.0)
        self._connection.execute("PRAGMA busy_timeout=5000")
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS run_checkpoints (
                run_id TEXT PRIMARY KEY,
                contract_digest TEXT NOT NULL,
                graph_digest TEXT,
                phase TEXT NOT NULL,
                next_iteration INTEGER NOT NULL,
                tool_calls INTEGER NOT NULL,
                history_json TEXT NOT NULL,
                seen_failures_json TEXT NOT NULL,
                evidence_json TEXT NOT NULL,
                pending_intent_json TEXT,
                executor_id TEXT,
                prepared_execution_json TEXT,
                reconciled_observation_json TEXT,
                terminal_decision TEXT,
                terminal_reason TEXT,
                revision INTEGER NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        existing = {row[1] for row in self._connection.execute("PRAGMA table_info(run_checkpoints)").fetchall()}
        for column, declaration in (
            ("graph_digest", "TEXT"),
            ("executor_id", "TEXT"),
            ("prepared_execution_json", "TEXT"),
            ("reconciled_observation_json", "TEXT"),
        ):
            if column not in existing:
                self._connection.execute(f"ALTER TABLE run_checkpoints ADD COLUMN {column} {declaration}")

    def load(self, run_id: str) -> RunCheckpoint | None:
        row = self._connection.execute(
            """
            SELECT contract_digest, graph_digest, phase, next_iteration, tool_calls, history_json,
                   seen_failures_json, evidence_json, pending_intent_json, executor_id,
                   prepared_execution_json,
                   reconciled_observation_json,
                   terminal_decision, terminal_reason, revision, updated_at
            FROM run_checkpoints WHERE run_id = ?
            """,
            (run_id,),
        ).fetchone()
        if row is None:
            return None
        return RunCheckpoint(
            run_id=run_id,
            contract_digest=row[0],
            graph_digest=row[1] or "",
            phase=RunPhase(row[2]),
            next_iteration=int(row[3]),
            tool_calls=int(row[4]),
            history=tuple(json.loads(row[5])),
            seen_failures=tuple(tuple(item) for item in json.loads(row[6])),
            evidence=_decode_evidence(run_id, json.loads(row[7])),
            pending_intent=_decode_intent(json.loads(row[8])) if row[8] else None,
            executor_id=row[9] or "",
            prepared_execution=_decode_prepared_execution(json.loads(row[10])) if row[10] else None,
            reconciled_observation=_decode_observation(json.loads(row[11])) if row[11] else None,
            terminal_decision=row[12],
            terminal_reason=row[13],
            revision=int(row[14]),
            updated_at=datetime.fromisoformat(row[15]),
        )

    def save(self, checkpoint: RunCheckpoint) -> RunCheckpoint:
        encoded = (
            checkpoint.contract_digest,
            checkpoint.graph_digest or None,
            checkpoint.phase.value,
            checkpoint.next_iteration,
            checkpoint.tool_calls,
            canonical_json(checkpoint.history),
            canonical_json(checkpoint.seen_failures),
            canonical_json(_encode_evidence(checkpoint.evidence)),
            canonical_json(_encode_intent(checkpoint.pending_intent)) if checkpoint.pending_intent else None,
            checkpoint.executor_id or None,
            canonical_json(_encode_prepared_execution(checkpoint.prepared_execution))
            if checkpoint.prepared_execution
            else None,
            canonical_json(_encode_observation(checkpoint.reconciled_observation))
            if checkpoint.reconciled_observation
            else None,
            checkpoint.terminal_decision,
            checkpoint.terminal_reason,
        )
        now = datetime.now(UTC)
        self._connection.execute("BEGIN IMMEDIATE")
        try:
            row = self._connection.execute(
                "SELECT revision FROM run_checkpoints WHERE run_id = ?", (checkpoint.run_id,)
            ).fetchone()
            if row is None:
                if checkpoint.revision != 0:
                    raise RunStateConflict("run checkpoint was deleted or replaced")
                revision = 1
                self._connection.execute(
                    """
                    INSERT INTO run_checkpoints (
                        run_id, contract_digest, graph_digest, phase, next_iteration, tool_calls,
                        history_json, seen_failures_json, evidence_json, pending_intent_json,
                        executor_id, prepared_execution_json, reconciled_observation_json, terminal_decision, terminal_reason,
                        revision, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (checkpoint.run_id, *encoded, revision, now.isoformat()),
                )
            else:
                if int(row[0]) != checkpoint.revision:
                    raise RunStateConflict("run checkpoint has a newer revision")
                revision = checkpoint.revision + 1
                cursor = self._connection.execute(
                    """
                    UPDATE run_checkpoints
                    SET contract_digest = ?, graph_digest = ?, phase = ?, next_iteration = ?, tool_calls = ?,
                        history_json = ?, seen_failures_json = ?, evidence_json = ?,
                        pending_intent_json = ?, executor_id = ?, prepared_execution_json = ?, reconciled_observation_json = ?,
                        terminal_decision = ?, terminal_reason = ?,
                        revision = ?, updated_at = ?
                    WHERE run_id = ? AND revision = ?
                    """,
                    (*encoded, revision, now.isoformat(), checkpoint.run_id, checkpoint.revision),
                )
                if cursor.rowcount != 1:  # pragma: no cover - BEGIN IMMEDIATE serializes writers
                    raise RunStateConflict("run checkpoint update lost its compare-and-swap")
            self._connection.execute("COMMIT")
        except Exception:
            self._connection.execute("ROLLBACK")
            raise
        return RunCheckpoint(
            run_id=checkpoint.run_id,
            contract_digest=checkpoint.contract_digest,
            graph_digest=checkpoint.graph_digest,
            phase=checkpoint.phase,
            next_iteration=checkpoint.next_iteration,
            tool_calls=checkpoint.tool_calls,
            history=checkpoint.history,
            seen_failures=checkpoint.seen_failures,
            evidence=checkpoint.evidence,
            pending_intent=checkpoint.pending_intent,
            executor_id=checkpoint.executor_id,
            prepared_execution=checkpoint.prepared_execution,
            reconciled_observation=checkpoint.reconciled_observation,
            terminal_decision=checkpoint.terminal_decision,
            terminal_reason=checkpoint.terminal_reason,
            revision=revision,
            updated_at=now,
        )

    def close(self) -> None:
        self._connection.close()


def _encode_intent(intent: ActionIntent) -> dict[str, Any]:
    return {
        "tool": intent.tool,
        "effect": intent.effect.value,
        "target": intent.target,
        "arguments": dict(intent.arguments),
        "provenance": [value.value for value in intent.provenance],
        "explanation": intent.explanation,
        "contract_id": intent.contract_id,
        "contract_version": intent.contract_version,
        "idempotency_key": intent.idempotency_key,
        "argument_provenance": {
            name: [value.value for value in values]
            for name, values in intent.argument_provenance.items()
        },
        "argument_provenance_graph": {
            name: {
                "value_digest": graph.value_digest,
                "nodes": [
                    {
                        "node_id": node.node_id,
                        "provenance": node.provenance.value,
                        "source_id": node.source_id,
                        "content_digest": node.content_digest,
                        "parent_ids": list(node.parent_ids),
                    }
                    for node in graph.nodes
                ],
            }
            for name, graph in intent.argument_provenance_graph.items()
        },
    }


def _decode_intent(value: Mapping[str, Any]) -> ActionIntent:
    graphs = {
        name: ArgumentProvenance(
            value_digest=graph["value_digest"],
            nodes=tuple(
                ArgumentProvenanceNode(
                    node_id=node["node_id"],
                    provenance=Provenance(node["provenance"]),
                    source_id=node["source_id"],
                    content_digest=node["content_digest"],
                    parent_ids=tuple(node.get("parent_ids", ())),
                )
                for node in graph["nodes"]
            ),
        )
        for name, graph in value.get("argument_provenance_graph", {}).items()
    }
    return ActionIntent(
        tool=value["tool"],
        effect=Effect(value["effect"]),
        target=value["target"],
        arguments=dict(value["arguments"]),
        provenance=tuple(Provenance(item) for item in value["provenance"]),
        explanation=value["explanation"],
        contract_id=value["contract_id"],
        contract_version=int(value["contract_version"]),
        idempotency_key=value["idempotency_key"],
        argument_provenance={
            name: tuple(Provenance(item) for item in values)
            for name, values in value.get("argument_provenance", {}).items()
        },
        argument_provenance_graph=graphs,
    )


def _encode_report(report: VerificationReport) -> dict[str, Any]:
    return {
        "correctness": report.correctness.value,
        "policy": report.policy.value,
        "evidence": report.evidence.value,
        "quality": report.quality.value,
        "checks": [
            {"name": check.name, "status": check.status.value, "evidence": dict(check.evidence), "message": check.message}
            for check in report.checks
        ],
    }


def _decode_report(value: Mapping[str, Any]) -> VerificationReport:
    return VerificationReport(
        CheckStatus(value["correctness"]),
        CheckStatus(value["policy"]),
        CheckStatus(value["evidence"]),
        CheckStatus(value["quality"]),
        tuple(
            CheckResult(
                str(check["name"]),
                CheckStatus(check["status"]),
                dict(check["evidence"]),
                str(check.get("message", "")),
            )
            for check in value["checks"]
        ),
    )


def _encode_observation(observation: ExecutionObservation) -> dict[str, Any]:
    return {
        "success": observation.success,
        "exit_code": observation.exit_code,
        "stdout": observation.stdout,
        "stderr": observation.stderr,
        "artifact_digests": dict(observation.artifact_digests),
        "metadata": dict(observation.metadata),
    }


def _decode_observation(value: Mapping[str, Any]) -> ExecutionObservation:
    return ExecutionObservation(
        success=bool(value["success"]),
        exit_code=value["exit_code"],
        stdout=str(value["stdout"]),
        stderr=str(value["stderr"]),
        artifact_digests=dict(value["artifact_digests"]),
        metadata=dict(value["metadata"]),
    )


def _encode_prepared_execution(prepared: PreparedExecution) -> dict[str, str]:
    return {
        "operation_id": prepared.operation_id,
        "executor_id": prepared.executor_id,
        "intent_digest": prepared.intent_digest,
        "request_digest": prepared.request_digest,
        "remote_job_id": prepared.remote_job_id,
        "graph_digest": prepared.graph_digest,
        "graph_node_id": prepared.graph_node_id,
    }


def _decode_prepared_execution(value: Mapping[str, Any]) -> PreparedExecution:
    return PreparedExecution(
        operation_id=str(value["operation_id"]),
        executor_id=str(value["executor_id"]),
        intent_digest=str(value["intent_digest"]),
        request_digest=str(value["request_digest"]),
        remote_job_id=str(value["remote_job_id"]),
        graph_digest=str(value.get("graph_digest", "")),
        graph_node_id=str(value.get("graph_node_id", "")),
    )


def _encode_evidence(snapshot: EvidenceSnapshot) -> list[dict[str, Any]]:
    return [
        {
            "sequence": action.sequence,
            "intent_digest": action.intent_digest,
            "artifact_digests": dict(action.artifact_digests),
            "source_state_digest": action.source_state_digest,
            "report": _encode_report(action.report),
        }
        for action in snapshot.actions
    ]


def _decode_evidence(run_id: str, value: list[Mapping[str, Any]]) -> EvidenceSnapshot:
    return EvidenceSnapshot(
        run_id,
        tuple(
            ActionEvidence(
                sequence=int(action["sequence"]),
                intent_digest=str(action["intent_digest"]),
                artifact_digests=dict(action["artifact_digests"]),
                source_state_digest=action.get("source_state_digest"),
                report=_decode_report(action["report"]),
            )
            for action in value
        ),
    )
