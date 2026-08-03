"""Durable, graph-native scheduler for V-Loop state advancement."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol

from .canonical import canonical_json
from .graph import DynamicSubgraphPolicy, GraphManifest, GraphNode
from .graph_schema import NodeImplementation
from .graph_events import CausalEvent, GraphEventStore, new_event_id, node_instance_id
from .graph_monitor import TransitionMonitor, TransitionState
from .graph_schema import GraphJoin
from .models import TaskContract


@dataclass(frozen=True, slots=True)
class ScheduledTransition:
    event: CausalEvent
    state: TransitionState


class GraphBudgetExceeded(RuntimeError):
    """A server-owned dynamic implementation exceeded its admitted budget."""


@dataclass(frozen=True, slots=True)
class DynamicNodeResult:
    output_artifacts: Mapping[str, str]
    tokens_used: int
    calls_used: int


class DynamicNodeRunner(Protocol):
    def run(self, implementation: NodeImplementation, node: GraphNode, inputs: Mapping[str, str]) -> DynamicNodeResult: ...


class ReadOnlyDynamicExecutor:
    """Execute admitted dynamic nodes exclusively through registered templates."""

    def __init__(self, policy: DynamicSubgraphPolicy, runners: Mapping[str, DynamicNodeRunner]) -> None:
        self.policy = policy
        self._runners = dict(runners)

    def execute_node(
        self,
        graph: GraphManifest,
        node_id: str,
        *,
        contract: TaskContract,
        inputs: Mapping[str, str],
    ) -> DynamicNodeResult:
        admitted = self.policy.admit(graph, contract=contract)
        node = next((candidate for candidate in admitted.nodes if candidate.node_id == node_id), None)
        if node is None or node.node_type.value == "task":
            raise PermissionError("dynamic execution needs an admitted non-task node")
        implementation = self.policy.implementations[node.metadata["implementation_id"]]
        runner = self._runners.get(implementation.implementation_id)
        if runner is None:
            raise PermissionError("no deployment runner is registered for this dynamic implementation")
        result = runner.run(implementation, node, inputs)
        if result.tokens_used < 0 or result.calls_used < 0:
            raise GraphBudgetExceeded("dynamic runner reported an invalid resource count")
        declared_tokens = int(node.metadata["maximum_tokens"])
        if result.tokens_used > declared_tokens or result.calls_used > (node.budget or 0):
            raise GraphBudgetExceeded("dynamic node exceeded its admitted token or call budget")
        return result


class DurableGraphScheduler:
    """The only state-transition writer for an executable graph run.

    It persists enabled/completed template state per iteration and emits a
    causal event in the same caller operation. A restart can therefore resume
    monitoring without trusting imperative controller history.
    """

    def __init__(
        self,
        manifest: GraphManifest,
        event_store: GraphEventStore,
        database: str | Path = ":memory:",
        *,
        joins: tuple[GraphJoin, ...] = (),
    ) -> None:
        self.monitor = TransitionMonitor(manifest, joins=joins or manifest.joins)
        self.event_store = event_store
        if Path(database) != event_store.path:
            raise ValueError("scheduler state and causal events must share one SQLite database")
        self._connection = event_store._connection
        self._connection.execute(
            """CREATE TABLE IF NOT EXISTS graph_scheduler_state (
                run_id TEXT NOT NULL, graph_digest TEXT NOT NULL, iteration INTEGER NOT NULL,
                state_json TEXT NOT NULL, attempts_json TEXT NOT NULL,
                PRIMARY KEY (run_id, graph_digest, iteration)
            )"""
        )

    @property
    def manifest(self) -> GraphManifest:
        return self.monitor.manifest

    def state(self, *, run_id: str, iteration: int) -> TransitionState:
        row = self._connection.execute(
            "SELECT state_json FROM graph_scheduler_state WHERE run_id = ? AND graph_digest = ? AND iteration = ?",
            (run_id, self.manifest.graph_digest, iteration),
        ).fetchone()
        if row is None:
            return self.monitor.initial_state(iteration=iteration)
        value = json.loads(row[0])
        return TransitionState(iteration, frozenset(value["completed"]), dict(value["events"]))

    def advance(
        self,
        *,
        run_id: str,
        iteration: int,
        template_node_id: str,
        event_type: str,
        payload: Mapping[str, Any] = {},
        causal_parents: tuple[str, ...] = (),
        input_artifacts: Mapping[str, str] = {},
        output_artifacts: Mapping[str, str] = {},
        authorization_ref: str = "",
        receipt_refs: tuple[str, ...] = (),
    ) -> ScheduledTransition:
        if not run_id.strip() or iteration < 1:
            raise ValueError("scheduled transitions need a run and positive iteration")
        self._connection.execute("BEGIN IMMEDIATE")
        try:
            state = self.state(run_id=run_id, iteration=iteration)
            next_state = self.monitor.advance(state, template_node_id, payload)
            attempts = self._attempts(run_id, iteration)
            attempt = attempts.get(template_node_id, 0) + 1
            attempts[template_node_id] = attempt
            event = CausalEvent(
                event_id=new_event_id(),
                run_id=run_id,
                graph_digest=self.manifest.graph_digest,
                node_instance_id=node_instance_id(
                    run_id=run_id,
                    graph_digest=self.manifest.graph_digest,
                    template_node_id=template_node_id,
                    iteration=iteration,
                    attempt=attempt,
                ),
                template_node_id=template_node_id,
                event_type=event_type,
                iteration=iteration,
                attempt=attempt,
                causal_parents=causal_parents,
                input_artifacts=input_artifacts,
                output_artifacts=output_artifacts,
                authorization_ref=authorization_ref,
                receipt_refs=receipt_refs,
                payload=dict(payload),
            )
            self.event_store.append(event, commit=False)
            self._save(run_id, next_state, attempts)
            self._connection.execute("COMMIT")
        except Exception:
            self._connection.execute("ROLLBACK")
            raise
        return ScheduledTransition(event, next_state)

    def _attempts(self, run_id: str, iteration: int) -> dict[str, int]:
        row = self._connection.execute(
            "SELECT attempts_json FROM graph_scheduler_state WHERE run_id = ? AND graph_digest = ? AND iteration = ?",
            (run_id, self.manifest.graph_digest, iteration),
        ).fetchone()
        return dict(json.loads(row[0])) if row else {}

    def _save(self, run_id: str, state: TransitionState, attempts: Mapping[str, int]) -> None:
        payload = canonical_json({"completed": sorted(state.completed), "events": dict(state.events)})
        self._connection.execute(
            """INSERT INTO graph_scheduler_state VALUES (?, ?, ?, ?, ?)
               ON CONFLICT(run_id, graph_digest, iteration) DO UPDATE SET
               state_json = excluded.state_json, attempts_json = excluded.attempts_json""",
            (run_id, self.manifest.graph_digest, state.iteration, payload, canonical_json(dict(attempts))),
        )


# The scheduler owns durable state and validation, so it is the graph executor
# rather than a separate imperative dispatch loop.  Keep this public name for
# deployments that depend on the architecture terminology.
GraphExecutor = DurableGraphScheduler
