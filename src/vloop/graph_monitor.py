"""Runtime transition monitor for compiled V-Loop manifests."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from .graph import GraphEdge, GraphEdgeType, GraphManifest
from .graph_schema import GraphJoin


class GraphTransitionRejected(PermissionError):
    """A caller attempted a transition absent from the compiled graph."""


@dataclass(frozen=True, slots=True)
class TransitionState:
    iteration: int
    completed: frozenset[str] = frozenset()
    events: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)


class TransitionMonitor:
    """Derives enabled nodes from manifest edges and explicit join policies."""

    def __init__(self, manifest: GraphManifest, *, joins: tuple[GraphJoin, ...] = ()) -> None:
        manifest.validate().require_valid()
        self.manifest = manifest
        self._nodes = {node.node_id: node for node in manifest.nodes}
        self._incoming: dict[str, list[GraphEdge]] = {node_id: [] for node_id in self._nodes}
        for edge in manifest.edges:
            self._incoming[edge.target].append(edge)
        self._joins = {join.node_id: join for join in joins}
        if set(self._joins).difference(self._nodes):
            raise ValueError("joins refer to nodes absent from the manifest")

    def initial_state(self, *, iteration: int = 1) -> TransitionState:
        if iteration < 1:
            raise ValueError("graph iteration must be positive")
        return TransitionState(iteration)

    def enabled(self, state: TransitionState, node_id: str, payload: Mapping[str, Any]) -> bool:
        if node_id not in self._nodes or node_id in state.completed:
            return False
        join = self._joins.get(node_id)
        if join is not None:
            return join.satisfied_by(set(state.completed))
        incoming = self._incoming[node_id]
        if not incoming:
            return not state.completed
        return all(edge.source in state.completed and _condition_matches(edge, state.events, payload) for edge in incoming)

    def advance(self, state: TransitionState, node_id: str, payload: Mapping[str, Any]) -> TransitionState:
        if not self.enabled(state, node_id, payload):
            raise GraphTransitionRejected(f"transition to {node_id!r} is not enabled by graph {self.manifest.graph_digest}")
        return TransitionState(
            state.iteration,
            state.completed | {node_id},
            {**state.events, node_id: dict(payload)},
        )


def _condition_matches(edge: GraphEdge, events: Mapping[str, Mapping[str, Any]], payload: Mapping[str, Any]) -> bool:
    """Evaluate only compiler-owned predicates and the closed legacy vocabulary."""

    if not edge.condition:
        legacy_matches = True
    else:
        source_payload = events.get(edge.source, payload)
        if edge.condition == "execution.success":
            legacy_matches = source_payload.get("success") is True or (
                source_payload.get("success") is None and payload.get("success") is True
            )
        elif edge.condition == "outcome-unknown-or-failed":
            legacy_matches = source_payload.get("success") is False or (
                source_payload.get("success") is None and payload.get("success") is not True
            )
        elif edge.condition == "guard-pass":
            legacy_matches = source_payload.get("passed") is True
        else:  # GraphEdge rejects this at construction; preserve fail-closed semantics.
            legacy_matches = False
    if not legacy_matches:
        return False
    source_payload = events.get(edge.source, payload)
    return edge.predicate.evaluate(source_payload)
