"""Causal runtime events and semantic evidence projections."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Mapping
from uuid import uuid4

from .canonical import canonical_json, digest


def node_instance_id(*, run_id: str, graph_digest: str, template_node_id: str, iteration: int, attempt: int) -> str:
    if not run_id.strip() or not template_node_id.strip() or iteration < 1 or attempt < 1:
        raise ValueError("node instances need a run, template, iteration, and attempt")
    return digest(
        {
            "run_id": run_id,
            "graph_digest": graph_digest,
            "template_node_id": template_node_id,
            "iteration": iteration,
            "attempt": attempt,
        }
    )


@dataclass(frozen=True, slots=True)
class CausalEvent:
    event_id: str
    run_id: str
    graph_digest: str
    node_instance_id: str
    template_node_id: str
    event_type: str
    iteration: int = 1
    attempt: int = 1
    causal_parents: tuple[str, ...] = ()
    input_artifacts: Mapping[str, str] = field(default_factory=dict)
    output_artifacts: Mapping[str, str] = field(default_factory=dict)
    authorization_ref: str = ""
    receipt_refs: tuple[str, ...] = ()
    payload: Mapping[str, Any] = field(default_factory=dict)
    occurred_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def __post_init__(self) -> None:
        if not all((self.event_id.strip(), self.run_id.strip(), self.node_instance_id, self.template_node_id, self.event_type)):
            raise ValueError("causal events need stable identities")
        if self.iteration < 1 or self.attempt < 1:
            raise ValueError("causal events need positive iteration and attempt numbers")
        if len(self.graph_digest) != 64:
            raise ValueError("causal events need a graph digest")
        if self.occurred_at.tzinfo is None:
            raise ValueError("causal event time must be timezone-aware")
        if len(self.causal_parents) != len(set(self.causal_parents)):
            raise ValueError("causal parents must be unique")

    @property
    def event_digest(self) -> str:
        return digest(
            {
                **asdict(self),
                "occurred_at": self.occurred_at.isoformat(),
                "input_artifacts": dict(self.input_artifacts),
                "output_artifacts": dict(self.output_artifacts),
                "payload": dict(self.payload),
            }
        )


class GraphEventStore:
    """Durable per-run causal store, independent of the global ledger chain."""

    def __init__(self, database: str | Path) -> None:
        self.path = Path(database)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(self.path, isolation_level=None, timeout=5.0)
        self._connection.execute("PRAGMA busy_timeout=5000")
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.execute(
            """CREATE TABLE IF NOT EXISTS graph_events (
                event_id TEXT PRIMARY KEY, run_id TEXT NOT NULL, graph_digest TEXT NOT NULL,
                event_json TEXT NOT NULL, event_digest TEXT NOT NULL UNIQUE
            )"""
        )

    def append(self, event: CausalEvent, *, commit: bool = True) -> CausalEvent:
        encoded = _encode_event(event)
        if commit:
            with self._connection:
                self._connection.execute(
                    "INSERT INTO graph_events VALUES (?, ?, ?, ?, ?)",
                    (event.event_id, event.run_id, event.graph_digest, encoded, event.event_digest),
                )
        else:
            self._connection.execute(
                "INSERT INTO graph_events VALUES (?, ?, ?, ?, ?)",
                (event.event_id, event.run_id, event.graph_digest, encoded, event.event_digest),
            )
        return event

    def events(self, *, run_id: str) -> tuple[CausalEvent, ...]:
        rows = self._connection.execute(
            "SELECT event_json FROM graph_events WHERE run_id = ? ORDER BY rowid", (run_id,)
        ).fetchall()
        return tuple(_decode_event(row[0]) for row in rows)

    def causal_root(self, *, run_id: str) -> str:
        return digest([event.event_digest for event in self.events(run_id=run_id)])


@dataclass(frozen=True, slots=True)
class SemanticEvidenceGraph:
    run_id: str
    graph_digest: str
    nodes: Mapping[str, str]
    edges: tuple[tuple[str, str, str], ...]

    def to_dot(self) -> str:
        lines = ["digraph vloop_causal {"]
        for node_id, label in sorted(self.nodes.items()):
            lines.append(f'  "{node_id}" [label="{label.replace(chr(34), chr(39))}"];')
        for source, target, label in self.edges:
            lines.append(f'  "{source}" -> "{target}" [label="{label}"];')
        lines.append("}")
        return "\n".join(lines)


def build_semantic_evidence_graph(events: Iterable[CausalEvent], *, run_id: str) -> SemanticEvidenceGraph:
    selected = tuple(event for event in events if event.run_id == run_id)
    if not selected:
        raise ValueError("semantic graph needs events for the selected run")
    graph_digests = {event.graph_digest for event in selected}
    if len(graph_digests) != 1:
        raise ValueError("semantic graph cannot mix graph versions")
    nodes: dict[str, str] = {}
    edges: list[tuple[str, str, str]] = []
    for event in selected:
        nodes[event.event_id] = event.template_node_id
        for parent in event.causal_parents:
            edges.append((parent, event.event_id, "caused"))
        for artifact in event.input_artifacts.values():
            nodes.setdefault(artifact, "artifact")
            edges.append((artifact, event.event_id, "input"))
        for artifact in event.output_artifacts.values():
            nodes.setdefault(artifact, "artifact")
            edges.append((event.event_id, artifact, "output"))
        if event.authorization_ref:
            nodes.setdefault(event.authorization_ref, "authorization")
            edges.append((event.authorization_ref, event.event_id, "authorised"))
        for receipt in event.receipt_refs:
            nodes.setdefault(receipt, "receipt")
            edges.append((receipt, event.event_id, "evidence"))
    return SemanticEvidenceGraph(run_id, next(iter(graph_digests)), nodes, tuple(edges))


def new_event_id() -> str:
    return str(uuid4())


def _encode_event(event: CausalEvent) -> str:
    data = asdict(event)
    data["occurred_at"] = event.occurred_at.isoformat()
    return canonical_json(data)


def _decode_event(value: str) -> CausalEvent:
    data = json.loads(value)
    return CausalEvent(
        event_id=str(data["event_id"]), run_id=str(data["run_id"]), graph_digest=str(data["graph_digest"]),
        node_instance_id=str(data["node_instance_id"]), template_node_id=str(data["template_node_id"]),
        event_type=str(data["event_type"]), iteration=int(data.get("iteration", 1)), attempt=int(data.get("attempt", 1)),
        causal_parents=tuple(data.get("causal_parents", ())),
        input_artifacts=dict(data.get("input_artifacts", {})), output_artifacts=dict(data.get("output_artifacts", {})),
        authorization_ref=str(data.get("authorization_ref", "")), receipt_refs=tuple(data.get("receipt_refs", ())),
        payload=dict(data.get("payload", {})), occurred_at=datetime.fromisoformat(data["occurred_at"]),
    )
