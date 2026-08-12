"""Durable, graph-bound, capability-free sessions for advisory subagents."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

from .canonical import digest


def _require_digest(value: str, label: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{label} must be a SHA-256 digest")


class SessionRejected(PermissionError):
    """A session crossed a graph, contract, budget, or expiry boundary."""


@dataclass(frozen=True, slots=True)
class ReasoningSession:
    session_id: str
    run_id: str
    contract_digest: str
    graph_digest: str
    parent_node_instance_id: str
    node_instance_id: str
    model_digest: str
    context_root_digest: str
    state_snapshot_digest: str
    created_at: datetime
    expires_at: datetime
    remaining_token_budget: int
    remaining_call_budget: int
    parent_session_id: str = ""
    status: str = "active"
    root_graph_digest: str = ""
    objective_digest: str = ""
    allowed_context_handles_digest: str = ""
    parent_artifact_digest: str = ""
    spawn_event_id: str = ""
    state_blob_ref: str = ""
    last_processed_message_id: str = ""
    continuation_status: str = "ready"

    def __post_init__(self) -> None:
        required = (self.session_id, self.run_id, self.parent_node_instance_id, self.node_instance_id)
        if not all(required) or self.status not in {"active", "archived"} or self.continuation_status not in {"ready", "waiting", "completed", "failed"}:
            raise ValueError("reasoning sessions need identity and a closed status")
        for value in (self.contract_digest, self.graph_digest, self.model_digest, self.context_root_digest, self.state_snapshot_digest):
            if len(value) != 64:
                raise ValueError("reasoning session bindings must be SHA-256 digests")
        if self.root_graph_digest:
            _require_digest(self.root_graph_digest, "reasoning session root graph")
        if self.created_at.tzinfo is None or self.expires_at.tzinfo is None or self.expires_at <= self.created_at:
            raise ValueError("reasoning session lifetime is invalid")
        if min(self.remaining_token_budget, self.remaining_call_budget) < 0:
            raise ValueError("reasoning session budget is invalid")


@dataclass(frozen=True, slots=True)
class SessionSnapshot:
    snapshot_digest: str
    session_id: str
    state: dict[str, object]
    previous_snapshot_digest: str
    last_processed_message_id: str
    continuation_status: str
    created_at: datetime

    def __post_init__(self) -> None:
        if not self.session_id or self.continuation_status not in {"ready", "waiting", "completed", "failed"}:
            raise ValueError("session snapshot is invalid")
        if self.created_at.tzinfo is None:
            raise ValueError("session snapshots require timezone-aware timestamps")
        if self.previous_snapshot_digest:
            _require_digest(self.previous_snapshot_digest, "previous snapshot")
        expected = digest({
            "session_id": self.session_id, "state": self.state,
            "previous_snapshot_digest": self.previous_snapshot_digest,
            "last_processed_message_id": self.last_processed_message_id,
            "continuation_status": self.continuation_status,
        })
        if self.snapshot_digest != expected:
            raise ValueError("session snapshot digest does not match its recoverable state")


@dataclass(frozen=True, slots=True)
class ChildSessionAdmission:
    """Trusted admission record supplied after a child GraphIR node was reserved."""

    child_node_instance_id: str
    child_graph_digest: str
    objective: str
    allowed_context_handles: tuple[str, ...]
    context_manifest_digest: str
    parent_artifact_digest: str
    spawn_event_id: str
    token_budget: int
    call_budget: int

    def __post_init__(self) -> None:
        if (
            not self.child_node_instance_id or not self.objective.strip() or not self.allowed_context_handles
            or not self.spawn_event_id or min(self.token_budget, self.call_budget) < 1
        ):
            raise ValueError("child session admission needs realised graph identity, context, and budget")
        if len(self.allowed_context_handles) != len(set(self.allowed_context_handles)) or any(not value.startswith("context://") for value in self.allowed_context_handles):
            raise ValueError("child session admission handles are invalid")
        for value, label in (
            (self.child_graph_digest, "child graph"), (self.context_manifest_digest, "child context manifest"),
            (self.parent_artifact_digest, "parent artifact"),
        ):
            _require_digest(value, label)

    @property
    def objective_digest(self) -> str:
        return digest(self.objective)

    @property
    def allowed_context_handles_digest(self) -> str:
        return digest(self.allowed_context_handles)


class ReasoningSessionStore:
    """Server-owned persistent state; workers receive only a session reference."""

    def __init__(self, database: str | Path, *, connection: sqlite3.Connection | None = None) -> None:
        path = Path(database)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = connection or sqlite3.connect(path, isolation_level=None, timeout=5.0)
        self._connection.execute("PRAGMA busy_timeout=5000")
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.execute(
            """CREATE TABLE IF NOT EXISTS reasoning_sessions (
                session_id TEXT PRIMARY KEY, value_json TEXT NOT NULL,
                parent_session_id TEXT NOT NULL, spawn_sequence INTEGER NOT NULL
            )"""
        )
        self._connection.execute(
            """CREATE TABLE IF NOT EXISTS reasoning_session_snapshots (
                snapshot_digest TEXT PRIMARY KEY, session_id TEXT NOT NULL,
                state_json TEXT NOT NULL, previous_snapshot_digest TEXT NOT NULL,
                last_processed_message_id TEXT NOT NULL, continuation_status TEXT NOT NULL,
                created_at TEXT NOT NULL
            )"""
        )
        self._connection.execute(
            """CREATE TABLE IF NOT EXISTS reasoning_session_nodes (
                node_instance_id TEXT PRIMARY KEY, session_id TEXT NOT NULL UNIQUE
            )"""
        )

    def create_root(
        self, *, run_id: str, contract_digest: str, graph_digest: str, node_instance_id: str,
        model_digest: str, context_root_digest: str, token_budget: int, call_budget: int,
        ttl: timedelta = timedelta(hours=1), now: datetime | None = None,
    ) -> ReasoningSession:
        if min(token_budget, call_budget) < 1 or ttl <= timedelta(0):
            raise ValueError("root session needs positive budgets and lifetime")
        created = now or datetime.now(UTC)
        session_id = digest({"run": run_id, "contract": contract_digest, "graph": graph_digest, "node": node_instance_id})
        session = ReasoningSession(
            session_id, run_id, contract_digest, graph_digest, node_instance_id, node_instance_id,
            model_digest, context_root_digest, digest({"state": "initial", "session": session_id}),
            created, created + ttl, token_budget, call_budget, root_graph_digest=graph_digest,
        )
        self._insert(session, parent_session_id="", sequence=0)
        return self._persist_snapshot(session, {"status": "ready"}, previous_snapshot_digest="", last_processed_message_id="", continuation_status="ready", now=created)

    def admit_reasoning_step(
        self,
        parent_session_id: str,
        *,
        token_usage: int,
        call_usage: int,
        children: tuple[ChildSessionAdmission, ...] = (),
        state: dict[str, object] | None = None,
        last_processed_message_id: str = "",
        continuation_status: str = "ready",
        ttl: timedelta = timedelta(minutes=30),
        now: datetime | None = None,
        transaction_open: bool = False,
    ) -> tuple[ReasoningSession, tuple[ReasoningSession, ...]]:
        """Atomically charge a reasoning result, persist recovery state, and create children.

        The caller must provide GraphIR-reserved node/event identities in each
        admission; this store never manufactures node identities itself.
        """
        if token_usage < 0 or call_usage < 0 or ttl <= timedelta(0):
            raise ValueError("reasoning admission usage or lifetime is invalid")
        if continuation_status not in {"ready", "waiting", "completed", "failed"}:
            raise ValueError("reasoning admission continuation status is invalid")
        if not transaction_open:
            self._connection.execute("BEGIN IMMEDIATE")
        try:
            parent = self._get(parent_session_id, now=now, archive_expired=True)
            reserved_tokens = token_usage + sum(item.token_budget for item in children)
            reserved_calls = call_usage + sum(item.call_budget for item in children)
            if parent.remaining_token_budget < reserved_tokens or parent.remaining_call_budget < reserved_calls:
                raise SessionRejected("parent recursive budget is exhausted")
            created = now or datetime.now(UTC)
            updated_parent = self._update(parent, tokens=reserved_tokens, calls=reserved_calls)
            updated_parent = self._persist_snapshot(
                updated_parent, state or {"status": continuation_status}, previous_snapshot_digest=parent.state_snapshot_digest,
                last_processed_message_id=last_processed_message_id, continuation_status=continuation_status, now=created,
            )
            realised_children: list[ReasoningSession] = []
            for admission in children:
                if self._connection.execute("SELECT 1 FROM reasoning_session_nodes WHERE node_instance_id = ?", (admission.child_node_instance_id,)).fetchone():
                    raise SessionRejected("child graph node instance was already admitted")
                sequence = self._connection.execute(
                    "SELECT COALESCE(MAX(spawn_sequence), 0) + 1 FROM reasoning_sessions WHERE parent_session_id = ?", (parent_session_id,)
                ).fetchone()[0]
                session_id = digest({"parent": parent.session_id, "node": admission.child_node_instance_id, "spawn_sequence": sequence})
                child = ReasoningSession(
                    session_id, parent.run_id, parent.contract_digest, admission.child_graph_digest,
                    parent.node_instance_id, admission.child_node_instance_id, parent.model_digest,
                    admission.context_manifest_digest, digest({"state": "initial", "session": session_id}),
                    created, min(parent.expires_at, created + ttl), admission.token_budget, admission.call_budget,
                    parent.session_id, objective_digest=admission.objective_digest,
                    allowed_context_handles_digest=admission.allowed_context_handles_digest,
                    parent_artifact_digest=admission.parent_artifact_digest, spawn_event_id=admission.spawn_event_id,
                    root_graph_digest=parent.root_graph_digest or parent.graph_digest,
                )
                self._insert(child, parent_session_id=parent.session_id, sequence=sequence)
                child = self._persist_snapshot(
                    child,
                    {
                        "objective": admission.objective,
                        "allowed_context_handles": admission.allowed_context_handles,
                        "parent_artifact_digest": admission.parent_artifact_digest,
                    },
                    previous_snapshot_digest="",
                    last_processed_message_id="",
                    continuation_status="ready",
                    now=created,
                )
                realised_children.append(child)
            if not transaction_open:
                self._connection.execute("COMMIT")
            return updated_parent, tuple(realised_children)
        except Exception:
            if not transaction_open:
                self._connection.execute("ROLLBACK")
            raise

    def spawn_child(
        self, parent_session_id: str, *, child_node_instance_id: str, token_budget: int,
        call_budget: int, ttl: timedelta = timedelta(minutes=30), now: datetime | None = None,
    ) -> ReasoningSession:
        """Deprecated test helper; real RLM paths must call ``admit_reasoning_step``."""
        parent = self.get(parent_session_id, now=now)
        admission = ChildSessionAdmission(
            child_node_instance_id, parent.graph_digest, "legacy child", ("context://legacy/handle",), parent.context_root_digest,
            digest({"legacy-parent": parent.session_id}), f"legacy:{child_node_instance_id}", token_budget, call_budget,
        )
        _parent, children = self.admit_reasoning_step(parent_session_id, token_usage=0, call_usage=0, children=(admission,), ttl=ttl, now=now)
        return children[0]

    def get(self, session_id: str, *, now: datetime | None = None) -> ReasoningSession:
        return self._get(session_id, now=now, archive_expired=True)

    def require_binding(
        self, session_id: str, *, run_id: str, contract_digest: str, graph_digest: str, node_instance_id: str,
        context_root_digest: str, now: datetime | None = None,
    ) -> ReasoningSession:
        value = self.get(session_id, now=now)
        if (value.run_id, value.contract_digest, value.graph_digest, value.node_instance_id, value.context_root_digest) != (
            run_id, contract_digest, graph_digest, node_instance_id, context_root_digest,
        ):
            raise SessionRejected("reasoning session is bound to another graph, contract, node, or context")
        return value

    def consume(self, session_id: str, *, tokens: int, calls: int, now: datetime | None = None) -> ReasoningSession:
        if tokens < 0 or calls < 0:
            raise ValueError("usage cannot be negative")
        self._connection.execute("BEGIN IMMEDIATE")
        try:
            value = self._get(session_id, now=now, archive_expired=True)
            if value.remaining_token_budget < tokens or value.remaining_call_budget < calls:
                raise SessionRejected("reasoning session budget is exhausted")
            updated = self._update(value, tokens=tokens, calls=calls)
            self._connection.execute("COMMIT")
            return updated
        except Exception:
            self._connection.execute("ROLLBACK")
            raise

    def snapshot(self, session_id: str, state: dict[str, object], *, now: datetime | None = None) -> ReasoningSession:
        self._connection.execute("BEGIN IMMEDIATE")
        try:
            value = self._get(session_id, now=now, archive_expired=True)
            updated = self._persist_snapshot(
                value, state, previous_snapshot_digest=value.state_snapshot_digest,
                last_processed_message_id=value.last_processed_message_id,
                continuation_status=value.continuation_status, now=now or datetime.now(UTC),
            )
            self._connection.execute("COMMIT")
            return updated
        except Exception:
            self._connection.execute("ROLLBACK")
            raise

    def resolve_child(
        self,
        child_session_id: str,
        *,
        completion_event_id: str,
        artifact_digest: str,
        transaction_open: bool = False,
        now: datetime | None = None,
    ) -> ReasoningSession:
        """Persist a signed child result and make its parent resumable when joined.

        This never interprets the child artifact.  The caller must first have
        verified and recorded the graph-node completion in the same transaction.
        """

        _require_digest(artifact_digest, "child completion artifact")
        if not completion_event_id.strip():
            raise ValueError("child completion needs its causal event identity")
        if not transaction_open:
            self._connection.execute("BEGIN IMMEDIATE")
        try:
            child = self._get(child_session_id, now=now, archive_expired=True)
            if not child.parent_session_id:
                raise SessionRejected("root reasoning sessions cannot be resolved as children")
            created = now or datetime.now(UTC)
            child_state = dict(self.load_snapshot(child_session_id).state)
            child_state.update({"completion_event_id": completion_event_id, "artifact_digest": artifact_digest})
            self._persist_snapshot(
                child, child_state, previous_snapshot_digest=child.state_snapshot_digest,
                last_processed_message_id=child.last_processed_message_id,
                continuation_status="completed", now=created,
            )
            parent = self._get(child.parent_session_id, now=now, archive_expired=True)
            sibling_rows = self._connection.execute(
                "SELECT value_json FROM reasoning_sessions WHERE parent_session_id = ?", (parent.session_id,)
            ).fetchall()
            siblings = tuple(_decode(row[0]) for row in sibling_rows)
            all_completed = all(
                item.session_id == child.session_id or item.continuation_status == "completed" for item in siblings
            )
            if all_completed:
                parent_state = dict(self.load_snapshot(parent.session_id).state)
                parent_state["child_join"] = {"completed": len(siblings), "last_event_id": completion_event_id}
                parent = self._persist_snapshot(
                    parent, parent_state, previous_snapshot_digest=parent.state_snapshot_digest,
                    last_processed_message_id=parent.last_processed_message_id,
                    continuation_status="ready", now=created,
                )
            if not transaction_open:
                self._connection.execute("COMMIT")
            return parent
        except Exception:
            if not transaction_open:
                self._connection.execute("ROLLBACK")
            raise

    def load_snapshot(self, session_id: str) -> SessionSnapshot:
        session = self.get(session_id)
        row = self._connection.execute(
            "SELECT snapshot_digest, session_id, state_json, previous_snapshot_digest, last_processed_message_id, continuation_status, created_at "
            "FROM reasoning_session_snapshots WHERE snapshot_digest = ?", (session.state_snapshot_digest,)
        ).fetchone()
        if row is None:
            raise SessionRejected("reasoning session snapshot is unavailable for recovery")
        return SessionSnapshot(
            row[0], row[1], json.loads(row[2]), row[3], row[4], row[5], datetime.fromisoformat(row[6]),
        )

    def _get(self, session_id: str, *, now: datetime | None, archive_expired: bool) -> ReasoningSession:
        row = self._connection.execute("SELECT value_json FROM reasoning_sessions WHERE session_id = ?", (session_id,)).fetchone()
        if row is None:
            raise SessionRejected("unknown reasoning session")
        value = _decode(row[0])
        current = now or datetime.now(UTC)
        if value.status != "active":
            raise SessionRejected("reasoning session is archived")
        if value.expires_at <= current:
            if archive_expired:
                self._replace(replace(value, status="archived"))
            raise SessionRejected("reasoning session expired and was archived")
        return value

    def _insert(self, value: ReasoningSession, *, parent_session_id: str, sequence: int) -> None:
        self._connection.execute(
            "INSERT INTO reasoning_sessions VALUES (?, ?, ?, ?)",
            (value.session_id, json.dumps(_encode(value), sort_keys=True), parent_session_id, sequence),
        )
        self._connection.execute(
            "INSERT INTO reasoning_session_nodes VALUES (?, ?)", (value.node_instance_id, value.session_id)
        )

    def _replace(self, value: ReasoningSession) -> None:
        self._connection.execute("UPDATE reasoning_sessions SET value_json = ? WHERE session_id = ?", (json.dumps(_encode(value), sort_keys=True), value.session_id))

    def _update(self, value: ReasoningSession, *, tokens: int, calls: int) -> ReasoningSession:
        updated = replace(
            value,
            remaining_token_budget=value.remaining_token_budget - tokens,
            remaining_call_budget=value.remaining_call_budget - calls,
        )
        self._replace(updated)
        return updated

    def _persist_snapshot(
        self,
        value: ReasoningSession,
        state: dict[str, object],
        *,
        previous_snapshot_digest: str,
        last_processed_message_id: str,
        continuation_status: str,
        now: datetime,
    ) -> ReasoningSession:
        snapshot_digest = digest({
            "session_id": value.session_id, "state": state,
            "previous_snapshot_digest": previous_snapshot_digest,
            "last_processed_message_id": last_processed_message_id,
            "continuation_status": continuation_status,
        })
        snapshot = SessionSnapshot(
            snapshot_digest, value.session_id, state, previous_snapshot_digest,
            last_processed_message_id, continuation_status, now,
        )
        self._connection.execute(
            "INSERT INTO reasoning_session_snapshots VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                snapshot.snapshot_digest, snapshot.session_id, json.dumps(snapshot.state, sort_keys=True),
                snapshot.previous_snapshot_digest, snapshot.last_processed_message_id,
                snapshot.continuation_status, snapshot.created_at.isoformat(),
            ),
        )
        updated = replace(
            value, state_snapshot_digest=snapshot.snapshot_digest, state_blob_ref=snapshot.snapshot_digest,
            last_processed_message_id=last_processed_message_id, continuation_status=continuation_status,
        )
        self._replace(updated)
        return updated


def _encode(value: ReasoningSession) -> dict[str, object]:
    result = asdict(value)
    result["created_at"] = value.created_at.isoformat()
    result["expires_at"] = value.expires_at.isoformat()
    return result


def _decode(value: str) -> ReasoningSession:
    data = json.loads(value)
    data["created_at"] = datetime.fromisoformat(data["created_at"])
    data["expires_at"] = datetime.fromisoformat(data["expires_at"])
    return ReasoningSession(**data)
