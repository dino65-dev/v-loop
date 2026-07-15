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
    Provenance,
    VerificationReport,
)


class RunPhase(StrEnum):
    READY = "ready"
    PENDING_AUTHORIZATION = "pending-authorization"
    PENDING_EFFECT = "pending-effect"
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
    pending_intent: ActionIntent | None = None
    terminal_decision: str | None = None
    terminal_reason: str | None = None
    revision: int = 0
    updated_at: datetime = datetime.min.replace(tzinfo=UTC)

    def __post_init__(self) -> None:
        if not self.run_id.strip() or not self.contract_digest.strip():
            raise ValueError("run checkpoints need a run and contract identity")
        if self.next_iteration < 1 or self.tool_calls < 0 or self.revision < 0:
            raise ValueError("run checkpoint counters are invalid")
        if self.evidence.run_id != self.run_id:
            raise ValueError("run checkpoint evidence belongs to another run")
        if self.phase in {RunPhase.PENDING_AUTHORIZATION, RunPhase.PENDING_EFFECT}:
            if self.pending_intent is None:
                raise ValueError("pending run checkpoints need their exact intent")
        elif self.pending_intent is not None:
            raise ValueError("only a pending checkpoint may retain an intent")
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
        path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(path, isolation_level=None, timeout=5.0)
        self._connection.execute("PRAGMA busy_timeout=5000")
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS run_checkpoints (
                run_id TEXT PRIMARY KEY,
                contract_digest TEXT NOT NULL,
                phase TEXT NOT NULL,
                next_iteration INTEGER NOT NULL,
                tool_calls INTEGER NOT NULL,
                history_json TEXT NOT NULL,
                seen_failures_json TEXT NOT NULL,
                evidence_json TEXT NOT NULL,
                pending_intent_json TEXT,
                terminal_decision TEXT,
                terminal_reason TEXT,
                revision INTEGER NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )

    def load(self, run_id: str) -> RunCheckpoint | None:
        row = self._connection.execute(
            """
            SELECT contract_digest, phase, next_iteration, tool_calls, history_json,
                   seen_failures_json, evidence_json, pending_intent_json,
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
            phase=RunPhase(row[1]),
            next_iteration=int(row[2]),
            tool_calls=int(row[3]),
            history=tuple(json.loads(row[4])),
            seen_failures=tuple(tuple(item) for item in json.loads(row[5])),
            evidence=_decode_evidence(run_id, json.loads(row[6])),
            pending_intent=_decode_intent(json.loads(row[7])) if row[7] else None,
            terminal_decision=row[8],
            terminal_reason=row[9],
            revision=int(row[10]),
            updated_at=datetime.fromisoformat(row[11]),
        )

    def save(self, checkpoint: RunCheckpoint) -> RunCheckpoint:
        encoded = (
            checkpoint.contract_digest,
            checkpoint.phase.value,
            checkpoint.next_iteration,
            checkpoint.tool_calls,
            canonical_json(checkpoint.history),
            canonical_json(checkpoint.seen_failures),
            canonical_json(_encode_evidence(checkpoint.evidence)),
            canonical_json(_encode_intent(checkpoint.pending_intent)) if checkpoint.pending_intent else None,
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
                        run_id, contract_digest, phase, next_iteration, tool_calls,
                        history_json, seen_failures_json, evidence_json, pending_intent_json,
                        terminal_decision, terminal_reason, revision, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                    SET contract_digest = ?, phase = ?, next_iteration = ?, tool_calls = ?,
                        history_json = ?, seen_failures_json = ?, evidence_json = ?,
                        pending_intent_json = ?, terminal_decision = ?, terminal_reason = ?,
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
            phase=checkpoint.phase,
            next_iteration=checkpoint.next_iteration,
            tool_calls=checkpoint.tool_calls,
            history=checkpoint.history,
            seen_failures=checkpoint.seen_failures,
            evidence=checkpoint.evidence,
            pending_intent=checkpoint.pending_intent,
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
