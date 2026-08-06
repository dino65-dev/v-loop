"""Durable, graph-bound, capability-free sessions for advisory subagents."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

from .canonical import digest


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

    def __post_init__(self) -> None:
        required = (self.session_id, self.run_id, self.parent_node_instance_id, self.node_instance_id)
        if not all(required) or self.status not in {"active", "archived"}:
            raise ValueError("reasoning sessions need identity and a closed status")
        for value in (self.contract_digest, self.graph_digest, self.model_digest, self.context_root_digest, self.state_snapshot_digest):
            if len(value) != 64:
                raise ValueError("reasoning session bindings must be SHA-256 digests")
        if self.created_at.tzinfo is None or self.expires_at.tzinfo is None or self.expires_at <= self.created_at:
            raise ValueError("reasoning session lifetime is invalid")
        if min(self.remaining_token_budget, self.remaining_call_budget) < 0:
            raise ValueError("reasoning session budget is invalid")


class ReasoningSessionStore:
    """Server-owned persistent state; workers receive only a session reference."""

    def __init__(self, database: str | Path) -> None:
        path = Path(database)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(path, isolation_level=None, timeout=5.0)
        self._connection.execute("PRAGMA busy_timeout=5000")
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.execute(
            """CREATE TABLE IF NOT EXISTS reasoning_sessions (
                session_id TEXT PRIMARY KEY, value_json TEXT NOT NULL,
                parent_session_id TEXT NOT NULL, spawn_sequence INTEGER NOT NULL
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
            created, created + ttl, token_budget, call_budget,
        )
        self._insert(session, parent_session_id="", sequence=0)
        return session

    def spawn_child(
        self, parent_session_id: str, *, child_node_instance_id: str, token_budget: int,
        call_budget: int, ttl: timedelta = timedelta(minutes=30), now: datetime | None = None,
    ) -> ReasoningSession:
        if min(token_budget, call_budget) < 1 or ttl <= timedelta(0):
            raise ValueError("child session needs positive budgets and lifetime")
        self._connection.execute("BEGIN IMMEDIATE")
        try:
            parent = self._get(parent_session_id, now=now, archive_expired=True)
            if parent.remaining_token_budget < token_budget or parent.remaining_call_budget < call_budget:
                raise SessionRejected("parent recursive budget is exhausted")
            sequence = self._connection.execute(
                "SELECT COALESCE(MAX(spawn_sequence), 0) + 1 FROM reasoning_sessions WHERE parent_session_id = ?", (parent_session_id,)
            ).fetchone()[0]
            created = now or datetime.now(UTC)
            session_id = digest({"parent": parent.session_id, "node": child_node_instance_id, "spawn_sequence": sequence})
            child = ReasoningSession(
                session_id, parent.run_id, parent.contract_digest, parent.graph_digest,
                parent.node_instance_id, child_node_instance_id, parent.model_digest,
                parent.context_root_digest, digest({"state": "initial", "session": session_id}),
                created, min(parent.expires_at, created + ttl), token_budget, call_budget, parent.session_id,
            )
            self._update(parent, tokens=token_budget, calls=call_budget)
            self._insert(child, parent_session_id=parent.session_id, sequence=sequence)
            self._connection.execute("COMMIT")
            return child
        except Exception:
            self._connection.execute("ROLLBACK")
            raise

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
            updated = replace(value, state_snapshot_digest=digest(state))
            self._replace(updated)
            self._connection.execute("COMMIT")
            return updated
        except Exception:
            self._connection.execute("ROLLBACK")
            raise

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
