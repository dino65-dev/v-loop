"""Durable, graph-native scheduler for V-Loop state advancement."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol

from .canonical import canonical_json
from .attestations import CompletionResult, CompletionVerifier, ValidatedNodeCompletion
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


@dataclass(frozen=True, slots=True)
class ScheduledReservation:
    event: CausalEvent
    node_instance_id: str
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
        completion_verifier: CompletionVerifier | None = None,
    ) -> None:
        self.monitor = TransitionMonitor(manifest, joins=joins or manifest.joins)
        self.event_store = event_store
        self.completion_verifier = completion_verifier
        self._roles = {
            node.node_id: node.metadata.get("producer_role", "controller")
            for node in manifest.nodes
        }
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
        return TransitionState(
            iteration,
            frozenset(value["completed"]),
            dict(value["events"]),
            dict(value.get("started", {})),
        )

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
        completion: ValidatedNodeCompletion | None = None,
        transaction_open: bool = False,
    ) -> ScheduledTransition:
        if not run_id.strip() or iteration < 1:
            raise ValueError("scheduled transitions need a run and positive iteration")
        role = self._roles.get(template_node_id)
        if role is None:
            raise PermissionError("graph node has no producer ownership assignment")
        if self.completion_verifier is not None and role not in {"controller", "scheduler"}:
            if completion is None:
                raise PermissionError("externally owned graph nodes require a validated completion")
            return self._complete_direct(
                run_id=run_id,
                iteration=iteration,
                template_node_id=template_node_id,
                event_type=event_type,
                completion=completion,
                payload=payload,
                causal_parents=causal_parents,
                input_artifacts=input_artifacts,
                output_artifacts=output_artifacts,
                authorization_ref=authorization_ref,
                receipt_refs=receipt_refs,
                transaction_open=transaction_open,
            )
        if not transaction_open:
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
            if not transaction_open:
                self._connection.execute("COMMIT")
        except Exception:
            if not transaction_open:
                self._connection.execute("ROLLBACK")
            raise
        return ScheduledTransition(event, next_state)

    def reserve(
        self,
        *,
        run_id: str,
        iteration: int,
        template_node_id: str,
        event_type: str = "node.started",
        payload: Mapping[str, Any] = {},
        causal_parents: tuple[str, ...] = (),
        transaction_open: bool = False,
    ) -> ScheduledReservation:
        """Reserve an enabled external node before asking its producer to act."""

        if not run_id.strip() or iteration < 1:
            raise ValueError("scheduled reservations need a run and positive iteration")
        if not transaction_open:
            self._connection.execute("BEGIN IMMEDIATE")
        try:
            state = self.state(run_id=run_id, iteration=iteration)
            next_state = self.monitor.reserve(state, template_node_id, payload)
            attempts = self._attempts(run_id, iteration)
            attempt = attempts.get(template_node_id, 0) + 1
            attempts[template_node_id] = attempt
            instance = node_instance_id(
                run_id=run_id, graph_digest=self.manifest.graph_digest,
                template_node_id=template_node_id, iteration=iteration, attempt=attempt,
            )
            started = {**next_state.started, template_node_id: {**next_state.started[template_node_id], "node_instance_id": instance, "attempt": attempt}}
            next_state = TransitionState(next_state.iteration, next_state.completed, next_state.events, started)
            event = CausalEvent(
                new_event_id(), run_id, self.manifest.graph_digest, instance, template_node_id,
                event_type, iteration, attempt, causal_parents=causal_parents, payload={"lifecycle": "started", **dict(payload)},
            )
            self.event_store.append(event, commit=False)
            self._save(run_id, next_state, attempts)
            if not transaction_open:
                self._connection.execute("COMMIT")
        except Exception:
            if not transaction_open:
                self._connection.execute("ROLLBACK")
            raise
        return ScheduledReservation(event, instance, next_state)

    def complete(
        self,
        *,
        run_id: str,
        iteration: int,
        template_node_id: str,
        completion: ValidatedNodeCompletion,
        event_type: str = "node.completed",
        causal_parents: tuple[str, ...] = (),
        transaction_open: bool = False,
    ) -> ScheduledTransition:
        """Accept only a producer-authenticated completion for a reservation."""

        return self._complete_direct(
            run_id=run_id, iteration=iteration, template_node_id=template_node_id,
            event_type=event_type, completion=completion, payload={}, causal_parents=causal_parents,
            input_artifacts=completion.input_artifact_digests,
            output_artifacts={"primary": completion.artifact_digest},
            authorization_ref=completion.authority_refs[0] if completion.authority_refs else "",
            receipt_refs=completion.evidence_refs,
            transaction_open=transaction_open,
        )

    def _complete_direct(
        self,
        *,
        run_id: str,
        iteration: int,
        template_node_id: str,
        event_type: str,
        completion: ValidatedNodeCompletion,
        payload: Mapping[str, Any],
        causal_parents: tuple[str, ...],
        input_artifacts: Mapping[str, str],
        output_artifacts: Mapping[str, str],
        authorization_ref: str,
        receipt_refs: tuple[str, ...],
        transaction_open: bool = False,
    ) -> ScheduledTransition:
        if self.completion_verifier is None:
            raise PermissionError("validated completions require a configured ownership verifier")
        if not transaction_open:
            self._connection.execute("BEGIN IMMEDIATE")
        try:
            state = self.state(run_id=run_id, iteration=iteration)
            started = state.started.get(template_node_id)
            if started is None:
                raise PermissionError("external node completion has no prior reservation")
            instance = str(started.get("node_instance_id", ""))
            self.completion_verifier.validate(
                completion,
                expected_graph_digest=self.manifest.graph_digest,
                expected_contract_digest=self.manifest.contract_digest,
                expected_run_id=run_id,
                expected_template_node_id=template_node_id,
                expected_node_instance_id=instance,
            )
            # Completion facts are deliberately serialized as strings for the
            # signed wire format.  The graph's closed condition vocabulary,
            # however, models pass/fail as booleans.  Decode only that closed
            # field; leave policy predicates (for example ``rule_index``) as
            # their authenticated textual values.
            facts = dict(completion.facts)
            if facts.get("passed") in {"true", "false"}:
                facts["passed"] = facts["passed"] == "true"
            completion_payload = {
                **facts,
                "result": completion.result.value,
                "success": completion.result is CompletionResult.SUCCEEDED,
                "completion_digest": completion.completion_digest,
                "producer_identity": completion.producer_identity,
                "producer_role": completion.producer_role,
                "artifact_digest": completion.artifact_digest,
                **dict(payload),
            }
            next_state = self.monitor.complete(state, template_node_id, completion_payload)
            event = CausalEvent(
                new_event_id(), run_id, self.manifest.graph_digest, instance, template_node_id,
                event_type, iteration, int(started.get("attempt", 1)), causal_parents=causal_parents,
                input_artifacts=input_artifacts, output_artifacts=output_artifacts,
                authorization_ref=authorization_ref, receipt_refs=receipt_refs,
                payload={"lifecycle": completion.result.value, **completion_payload},
            )
            self.event_store.append(event, commit=False)
            self._save(run_id, next_state, self._attempts(run_id, iteration))
            if not transaction_open:
                self._connection.execute("COMMIT")
        except Exception:
            if not transaction_open:
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
        payload = canonical_json({"completed": sorted(state.completed), "events": dict(state.events), "started": dict(state.started)})
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
