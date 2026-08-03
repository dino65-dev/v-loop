"""Governed, shadow-evaluated evolution for versioned V-Loop harness components."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Mapping
from uuid import uuid4

from .canonical import canonical_json, digest


class HarnessChangeStatus(StrEnum):
    PROPOSED = "proposed"
    SHADOW_PASSED = "shadow-passed"
    REJECTED = "rejected"
    PROMOTED = "promoted"
    ROLLED_BACK = "rolled-back"


@dataclass(frozen=True, slots=True)
class HarnessComponent:
    component_id: str
    kind: str
    version: str
    configuration: Mapping[str, str]

    def __post_init__(self) -> None:
        if not self.component_id.strip() or not self.kind.strip() or not self.version.strip():
            raise ValueError("harness components need stable id, kind, and version")

    @property
    def component_digest(self) -> str:
        return digest(
            {
                "component_id": self.component_id,
                "kind": self.kind,
                "version": self.version,
                "configuration": dict(self.configuration),
            }
        )


@dataclass(frozen=True, slots=True)
class HarnessChangeProposal:
    change_id: str
    component: HarnessComponent
    base_component_digest: str
    proposer_id: str
    predicted_metric: str
    minimum_improvement: float
    expected_failure_mode: str
    affected_task_classes: tuple[str, ...]

    def __post_init__(self) -> None:
        if not all(
            (
                self.change_id.strip(),
                self.base_component_digest,
                self.proposer_id.strip(),
                self.predicted_metric.strip(),
                self.expected_failure_mode.strip(),
            )
        ):
            raise ValueError("harness proposals need stable identity and a predicted metric")
        if len(self.base_component_digest) != 64:
            raise ValueError("harness proposal base digest must be SHA-256 hex")


@dataclass(frozen=True, slots=True)
class ShadowEvaluation:
    change_id: str
    baseline_score: float
    observed_score: float
    held_out_passed: bool
    rollback_threshold: float

    @property
    def improvement(self) -> float:
        return self.observed_score - self.baseline_score


class HarnessRegistry:
    """Durable promotion state; a proposer can never self-promote a change."""

    def __init__(self, database: str | Path) -> None:
        path = Path(database)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(path, isolation_level=None, timeout=5.0)
        self._connection.execute("PRAGMA busy_timeout=5000")
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS harness_components (
                component_id TEXT PRIMARY KEY,
                component_json TEXT NOT NULL,
                component_digest TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS harness_changes (
                change_id TEXT PRIMARY KEY,
                component_id TEXT NOT NULL,
                proposal_json TEXT NOT NULL,
                status TEXT NOT NULL,
                shadow_json TEXT,
                reviewer_id TEXT,
                previous_component_json TEXT,
                updated_at TEXT NOT NULL
            );
            """
        )
        columns = {row[1] for row in self._connection.execute("PRAGMA table_info(harness_changes)")}
        if "previous_component_json" not in columns:
            self._connection.execute("ALTER TABLE harness_changes ADD COLUMN previous_component_json TEXT")

    def register(self, component: HarnessComponent) -> None:
        self._connection.execute(
            """INSERT INTO harness_components VALUES (?, ?, ?, ?)
               ON CONFLICT(component_id) DO UPDATE SET component_json = excluded.component_json,
               component_digest = excluded.component_digest, updated_at = excluded.updated_at""",
            (component.component_id, _encode_component(component), component.component_digest, _now()),
        )

    def propose(self, proposal: HarnessChangeProposal) -> None:
        current = self._connection.execute(
            "SELECT component_digest FROM harness_components WHERE component_id = ?", (proposal.component.component_id,)
        ).fetchone()
        if current is None or current[0] != proposal.base_component_digest:
            raise ValueError("harness proposal is not based on the current component")
        self._connection.execute(
            """INSERT INTO harness_changes
               (change_id, component_id, proposal_json, status, shadow_json, reviewer_id, previous_component_json, updated_at)
               VALUES (?, ?, ?, ?, NULL, NULL, NULL, ?)""",
            (proposal.change_id, proposal.component.component_id, _encode_proposal(proposal), HarnessChangeStatus.PROPOSED.value, _now()),
        )

    def record_shadow(self, evaluation: ShadowEvaluation) -> HarnessChangeStatus:
        row = self._connection.execute(
            "SELECT proposal_json, status FROM harness_changes WHERE change_id = ?", (evaluation.change_id,)
        ).fetchone()
        if row is None or row[1] != HarnessChangeStatus.PROPOSED.value:
            raise ValueError("only proposed harness changes may receive shadow evidence")
        proposal = _decode_proposal(row[0])
        passed = (
            evaluation.held_out_passed
            and evaluation.improvement >= proposal.minimum_improvement
            and evaluation.observed_score >= evaluation.rollback_threshold
        )
        status = HarnessChangeStatus.SHADOW_PASSED if passed else HarnessChangeStatus.REJECTED
        self._connection.execute(
            "UPDATE harness_changes SET status = ?, shadow_json = ?, updated_at = ? WHERE change_id = ?",
            (status.value, _encode_shadow(evaluation), _now(), evaluation.change_id),
        )
        return status

    def promote(self, change_id: str, *, reviewer_id: str) -> HarnessComponent:
        self._connection.execute("BEGIN IMMEDIATE")
        try:
            row = self._connection.execute(
                "SELECT proposal_json, status FROM harness_changes WHERE change_id = ?", (change_id,)
            ).fetchone()
            if row is None:
                raise ValueError("unknown harness change")
            proposal = _decode_proposal(row[0])
            if row[1] != HarnessChangeStatus.SHADOW_PASSED.value:
                raise PermissionError("only held-out shadow-passed changes may be promoted")
            if reviewer_id == proposal.proposer_id:
                raise PermissionError("a harness proposer cannot promote its own change")
            current = self._connection.execute(
                "SELECT component_json, component_digest FROM harness_components WHERE component_id = ?",
                (proposal.component.component_id,),
            ).fetchone()
            if current is None or current[1] != proposal.base_component_digest:
                raise PermissionError("harness component changed since the shadow experiment")
            self.register(proposal.component)
            self._connection.execute(
                "UPDATE harness_changes SET status = ?, reviewer_id = ?, previous_component_json = ?, updated_at = ? WHERE change_id = ?",
                (HarnessChangeStatus.PROMOTED.value, reviewer_id, current[0], _now(), change_id),
            )
            self._connection.execute("COMMIT")
            return proposal.component
        except Exception:
            self._connection.execute("ROLLBACK")
            raise

    def rollback(self, change_id: str, *, reviewer_id: str) -> HarnessComponent:
        """Restore the retained baseline without overwriting a newer component version."""

        self._connection.execute("BEGIN IMMEDIATE")
        try:
            row = self._connection.execute(
                "SELECT proposal_json, status, reviewer_id, previous_component_json FROM harness_changes WHERE change_id = ?",
                (change_id,),
            ).fetchone()
            if row is None:
                raise ValueError("unknown harness change")
            proposal = _decode_proposal(row[0])
            if row[1] != HarnessChangeStatus.PROMOTED.value or not row[3]:
                raise PermissionError("only a promoted change with a retained baseline may be rolled back")
            if reviewer_id in {proposal.proposer_id, row[2]}:
                raise PermissionError("rollback needs an independent reviewer")
            current = self._connection.execute(
                "SELECT component_digest FROM harness_components WHERE component_id = ?", (proposal.component.component_id,)
            ).fetchone()
            if current is None or current[0] != proposal.component.component_digest:
                raise PermissionError("harness component changed after promotion; rollback would overwrite newer work")
            previous = _decode_component(row[3])
            self.register(previous)
            self._connection.execute(
                "UPDATE harness_changes SET status = ?, reviewer_id = ?, updated_at = ? WHERE change_id = ?",
                (HarnessChangeStatus.ROLLED_BACK.value, reviewer_id, _now(), change_id),
            )
            self._connection.execute("COMMIT")
            return previous
        except Exception:
            self._connection.execute("ROLLBACK")
            raise

    def status(self, change_id: str) -> HarnessChangeStatus:
        row = self._connection.execute("SELECT status FROM harness_changes WHERE change_id = ?", (change_id,)).fetchone()
        if row is None:
            raise ValueError("unknown harness change")
        return HarnessChangeStatus(row[0])


def new_change_id() -> str:
    return str(uuid4())


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _encode_component(component: HarnessComponent) -> str:
    return canonical_json({"component_id": component.component_id, "kind": component.kind, "version": component.version, "configuration": dict(component.configuration)})


def _decode_component(value: str) -> HarnessComponent:
    data = json.loads(value)
    return HarnessComponent(str(data["component_id"]), str(data["kind"]), str(data["version"]), dict(data["configuration"]))


def _encode_proposal(proposal: HarnessChangeProposal) -> str:
    return canonical_json({"change_id": proposal.change_id, "component": json.loads(_encode_component(proposal.component)), "base_component_digest": proposal.base_component_digest, "proposer_id": proposal.proposer_id, "predicted_metric": proposal.predicted_metric, "minimum_improvement": proposal.minimum_improvement, "expected_failure_mode": proposal.expected_failure_mode, "affected_task_classes": proposal.affected_task_classes})


def _decode_proposal(value: str) -> HarnessChangeProposal:
    data = json.loads(value)
    return HarnessChangeProposal(str(data["change_id"]), HarnessComponent(**data["component"]), str(data["base_component_digest"]), str(data["proposer_id"]), str(data["predicted_metric"]), float(data["minimum_improvement"]), str(data["expected_failure_mode"]), tuple(data["affected_task_classes"]))


def _encode_shadow(evaluation: ShadowEvaluation) -> str:
    return canonical_json({"change_id": evaluation.change_id, "baseline_score": evaluation.baseline_score, "observed_score": evaluation.observed_score, "held_out_passed": evaluation.held_out_passed, "rollback_threshold": evaluation.rollback_threshold})
