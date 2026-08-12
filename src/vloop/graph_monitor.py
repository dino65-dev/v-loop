"""Runtime transition monitor for compiled V-Loop manifests."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from .graph import GraphEdge, GraphManifest
from .graph_schema import GraphJoin


class GraphTransitionRejected(PermissionError):
    """A caller attempted a transition absent from the compiled graph."""


@dataclass(frozen=True, slots=True)
class TransitionState:
    iteration: int
    completed: frozenset[str] = frozenset()
    events: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    started: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)


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
        if node_id not in self._nodes or node_id in state.completed or node_id in state.started:
            return False
        join = self._joins.get(node_id)
        if join is not None:
            completed_predecessors = {
                edge.source
                for edge in self._incoming[node_id]
                if edge.source in state.completed and _condition_matches(edge, state.events, payload)
            }
            return join.satisfied_by(completed_predecessors)
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
            {key: value for key, value in state.started.items() if key != node_id},
        )

    def reserve(self, state: TransitionState, node_id: str, payload: Mapping[str, Any]) -> TransitionState:
        if not self.enabled(state, node_id, payload):
            raise GraphTransitionRejected(f"reservation for {node_id!r} is not enabled by graph {self.manifest.graph_digest}")
        return TransitionState(
            state.iteration,
            state.completed,
            state.events,
            {**state.started, node_id: dict(payload)},
        )

    def complete(self, state: TransitionState, node_id: str, payload: Mapping[str, Any]) -> TransitionState:
        if node_id not in state.started:
            raise GraphTransitionRejected(f"completion for {node_id!r} has no enabled reservation")
        # Prerequisites were checked at reservation time; completion may carry
        # only producer-authenticated result data and cannot rewrite them.
        return TransitionState(
            state.iteration,
            state.completed | {node_id},
            {**state.events, node_id: dict(payload)},
            {key: value for key, value in state.started.items() if key != node_id},
        )


def _condition_matches(edge: GraphEdge, events: Mapping[str, Mapping[str, Any]], payload: Mapping[str, Any]) -> bool:
    """Evaluate only compiler-owned predicates and the closed legacy vocabulary."""

    if not edge.condition:
        legacy_matches = True
    else:
        source_payload = events.get(edge.source, payload)
        if edge.condition == "execution.success":
            # A target may never manufacture the fact required to traverse an
            # incoming edge.  Only an authenticated predecessor result can
            # establish successful execution.
            legacy_matches = source_payload.get("success") is True
        elif edge.condition == "outcome-unknown-or-failed":
            legacy_matches = source_payload.get("success") in {False, "indeterminate"}
        elif edge.condition == "guard-pass":
            legacy_matches = source_payload.get("passed") is True
        else:  # GraphEdge rejects this at construction; preserve fail-closed semantics.
            legacy_matches = False
    if not legacy_matches:
        return False
    source_payload = events.get(edge.source, payload)
    return edge.predicate.evaluate(source_payload)
