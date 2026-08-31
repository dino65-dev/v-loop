"""Graph-native admission for durable, read-only recursive RLM children.

This service is the only bridge from an untrusted ``ChildSessionProposal`` to
a durable child.  It uses the existing dynamic graph policy and scheduler,
shares their SQLite transaction with the session store, and never grants an
effect capability.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from .attestations import CompletionVerifier, ValidatedNodeCompletion
from .canonical import canonical_json, digest
from .graph import DynamicSubgraphPolicy, GraphEdge, GraphEdgeType, GraphManifest, GraphNode, GraphNodeType
from .graph_events import GraphEventStore
from .graph_runtime import DurableGraphScheduler
from .graph_schema import NodeImplementation, NodePort, PortDirection, schema_digest
from .models import TaskContract
from .programmable_context import ProgrammableContextStore
from .reasoning_sessions import ChildSessionAdmission, ReasoningSession, ReasoningSessionStore
from .rlm_protocol import ChildSessionProposal, RLMReasoningRequest, RLMWorkerOutput


@dataclass(frozen=True, slots=True)
class RecursiveChildPolicy:
    """Server-owned template and resource ceiling for a recursive child."""

    implementation: NodeImplementation
    maximum_children_per_step: int = 4

    def __post_init__(self) -> None:
        if self.maximum_children_per_step < 1:
            raise ValueError("recursive child limit must be positive")
        if self.implementation.network_allowed:
            raise ValueError("recursive RLM child implementations must be networkless")


class GraphNativeChildAdmissionProvider:
    """Atomically reserve GraphIR children and persist their bound sessions."""

    def __init__(
        self,
        *,
        contract: TaskContract,
        event_store: GraphEventStore,
        sessions: ReasoningSessionStore,
        dynamic_policy: DynamicSubgraphPolicy,
        child_policy: RecursiveChildPolicy,
    ) -> None:
        if event_store._connection is not sessions._connection:
            raise ValueError("recursive graph events and session state must share one SQLite connection")
        if child_policy.implementation.implementation_id not in dynamic_policy.implementations:
            raise ValueError("recursive child implementation is not registered in dynamic graph policy")
        self.contract = contract
        self.event_store = event_store
        self.sessions = sessions
        self.dynamic_policy = dynamic_policy
        self.child_policy = child_policy
        self._connection = event_store._connection
        self._connection.execute(
            """CREATE TABLE IF NOT EXISTS recursive_subgraphs (
                graph_digest TEXT PRIMARY KEY, parent_session_id TEXT NOT NULL,
                parent_artifact_digest TEXT NOT NULL, manifest_json TEXT NOT NULL
            )"""
        )

    def admit_step(
        self,
        *,
        request: RLMReasoningRequest,
        parent_session: ReasoningSession,
        parent_artifact_digest: str,
        output: RLMWorkerOutput,
        proposals: tuple[ChildSessionProposal, ...],
        context: ProgrammableContextStore,
    ) -> tuple[ReasoningSession, ...]:
        if request.contract_digest != self.contract.contract_digest or request.graph_digest != parent_session.graph_digest:
            raise PermissionError("recursive admission belongs to another task contract or parent graph")
        if not request.causal_parent_event_id:
            raise PermissionError("recursive admission requires a causal parent event")
        if len(proposals) > self.child_policy.maximum_children_per_step:
            raise PermissionError("recursive admission exceeds the server-owned child limit")
        self._connection.execute("BEGIN IMMEDIATE")
        try:
            admissions: list[ChildSessionAdmission] = []
            for ordinal, proposal in enumerate(proposals, 1):
                manifest = self._build_child_graph(request, proposal, ordinal)
                self.dynamic_policy.admit(manifest, contract=self.contract)
                self._connection.execute(
                    "INSERT INTO recursive_subgraphs VALUES (?, ?, ?, ?)",
                    (manifest.graph_digest, parent_session.session_id, parent_artifact_digest, canonical_json(_manifest_record(manifest))),
                )
                scheduler = DurableGraphScheduler(manifest, self.event_store, self.event_store.path)
                task = scheduler.advance(
                    run_id=request.run_id, iteration=1, template_node_id="child.task", event_type="rlm.child.task",
                    payload={"objective_digest": digest(proposal.objective)}, causal_parents=(request.causal_parent_event_id,),
                    transaction_open=True,
                )
                reservation = scheduler.reserve(
                    run_id=request.run_id, iteration=1, template_node_id="child.reasoning",
                    event_type="rlm.child.started", causal_parents=(task.event.event_id,),
                    payload={"parent_artifact_digest": parent_artifact_digest}, transaction_open=True,
                )
                child_manifest = context.manifest(allowed_handles=proposal.context_handles)
                admissions.append(
                    ChildSessionAdmission(
                        reservation.node_instance_id, manifest.graph_digest, proposal.objective,
                        proposal.context_handles, child_manifest.manifest_digest, parent_artifact_digest,
                        reservation.event.event_id, proposal.token_budget, proposal.call_budget,
                    )
                )
            _parent, children = self.sessions.admit_reasoning_step(
                parent_session.session_id, token_usage=output.token_usage, call_usage=output.model_calls,
                children=tuple(admissions),
                state={"program_digest": digest(dict(output.program)), "summary": output.final_summary},
                continuation_status="waiting" if admissions else "ready", transaction_open=True,
            )
            self._connection.execute("COMMIT")
            return children
        except Exception:
            self._connection.execute("ROLLBACK")
            raise

    def _build_child_graph(
        self, request: RLMReasoningRequest, proposal: ChildSessionProposal, ordinal: int,
    ) -> GraphManifest:
        implementation = self.child_policy.implementation
        output_schema = implementation.output_schema_digests[0]
        graph = GraphManifest(
            graph_id=f"rlm-child-{digest({'parent': request.node_instance_id, 'ordinal': ordinal, 'objective': proposal.objective})[:24]}",
            schema_version=1,
            contract_digest=request.contract_digest,
            profile_digest=self.contract.profile_digest or "0" * 64,
            nodes=(
                GraphNode("child.task", GraphNodeType.TASK, metadata={"child": "true"}),
                GraphNode(
                    "child.reasoning", GraphNodeType.ACTION, budget=proposal.call_budget,
                    timeout_seconds=implementation.timeout_seconds,
                    output_ports=(NodePort("reasoning-artifact", PortDirection.OUTPUT, output_schema),),
                    metadata={
                        "mode": "read-only-reasoning", "terminal": "true",
                        "implementation_id": implementation.implementation_id,
                        "implementation_digest": implementation.implementation_digest,
                        "maximum_tokens": str(proposal.token_budget),
                        "producer_role": "rlm-worker",
                    },
                ),
            ),
            edges=(GraphEdge("child.task", "child.reasoning", GraphEdgeType.DEPENDS_ON),),
            metadata={"mode": "read-only-reasoning", "maximum_parallelism": "1", "parent_graph_digest": request.graph_digest},
        )
        return graph

    def load_manifest(self, graph_digest: str) -> GraphManifest:
        """Rebuild a persisted child GraphIR manifest after a controller restart."""

        row = self._connection.execute(
            "SELECT manifest_json FROM recursive_subgraphs WHERE graph_digest = ?", (graph_digest,)
        ).fetchone()
        if row is None:
            raise PermissionError("recursive child graph was not durably admitted")
        record = json.loads(row[0])
        ports = tuple(
            NodePort(item["name"], PortDirection(item["direction"]), item["schema_digest"])
            for item in record["reasoning_output_ports"]
        )
        manifest = GraphManifest(
            graph_id=record["graph_id"], schema_version=1,
            contract_digest=record["contract_digest"], profile_digest=record["profile_digest"],
            nodes=(
                GraphNode("child.task", GraphNodeType.TASK, metadata={"child": "true"}),
                GraphNode(
                    "child.reasoning", GraphNodeType.ACTION, budget=record["call_budget"],
                    timeout_seconds=record["timeout_seconds"], output_ports=ports,
                    metadata=record["reasoning_metadata"],
                ),
            ),
            edges=(GraphEdge("child.task", "child.reasoning", GraphEdgeType.DEPENDS_ON),),
            metadata=record["graph_metadata"],
        )
        if manifest.graph_digest != graph_digest:
            raise PermissionError("persisted recursive child GraphIR digest mismatch")
        self.dynamic_policy.admit(manifest, contract=self.contract)
        return manifest

    def request_for_child(
        self,
        *,
        child_session_id: str,
        context: ProgrammableContextStore,
        harness_digest: str,
        timeout_seconds: int = 60,
    ) -> RLMReasoningRequest:
        """Reconstruct the only permitted worker request for an admitted child."""

        child = self.sessions.get(child_session_id)
        if not child.parent_session_id or not child.spawn_event_id:
            raise PermissionError("only durably admitted child sessions may run recursively")
        snapshot = self.sessions.load_snapshot(child_session_id)
        objective = snapshot.state.get("objective")
        handles = snapshot.state.get("allowed_context_handles")
        if not isinstance(objective, str) or not isinstance(handles, list | tuple) or not all(isinstance(item, str) for item in handles):
            raise PermissionError("recursive child recovery state is malformed")
        allowed_handles = tuple(handles)
        manifest = context.manifest(allowed_handles=allowed_handles)
        if manifest.manifest_digest != child.context_root_digest:
            raise PermissionError("recursive child context no longer matches its admitted manifest")
        return RLMReasoningRequest(
            run_id=child.run_id, contract_digest=child.contract_digest, graph_digest=child.graph_digest,
            node_instance_id=child.node_instance_id, context_manifest_digest=child.context_root_digest,
            allowed_context_handles=allowed_handles, maximum_recursive_calls=child.remaining_call_budget,
            maximum_tokens=child.remaining_token_budget, timeout_seconds=timeout_seconds,
            model_digest=child.model_digest, harness_digest=harness_digest, session_id=child.session_id,
            objective=objective, causal_parent_event_id=child.spawn_event_id,
        )

    def complete_child(
        self,
        *,
        child_session_id: str,
        completion: ValidatedNodeCompletion,
        completion_verifier: CompletionVerifier,
    ) -> ReasoningSession:
        """Atomically accept a signed child completion, join it, and resume its parent."""

        child = self.sessions.get(child_session_id)
        manifest = self.load_manifest(child.graph_digest)
        scheduler = DurableGraphScheduler(
            manifest, self.event_store, self.event_store.path, completion_verifier=completion_verifier,
        )
        self._connection.execute("BEGIN IMMEDIATE")
        try:
            transition = scheduler.complete(
                run_id=child.run_id, iteration=1, template_node_id="child.reasoning", completion=completion,
                event_type="rlm.child.completed", causal_parents=(child.spawn_event_id,), transaction_open=True,
            )
            parent = self.sessions.resolve_child(
                child_session_id, completion_event_id=transition.event.event_id,
                artifact_digest=completion.artifact_digest, transaction_open=True,
            )
            self._connection.execute("COMMIT")
            return parent
        except Exception:
            self._connection.execute("ROLLBACK")
            raise


def default_recursive_implementation(*, maximum_tokens: int = 4_000, maximum_calls: int = 4, timeout_seconds: int = 60) -> NodeImplementation:
    """Return the narrow server-owned implementation identity for RLM children."""

    return NodeImplementation(
        "rlm-reasoning-v1", digest({"image": "rlm-reasoning-v1"}),
        (), (schema_digest("vloop.rlm.reasoning-artifact.v1"),), maximum_tokens, maximum_calls,
        timeout_seconds, network_allowed=False,
    )


def _manifest_record(manifest: GraphManifest) -> dict[str, object]:
    reasoning = next(node for node in manifest.nodes if node.node_id == "child.reasoning")
    return {
        "graph_id": manifest.graph_id,
        "graph_digest": manifest.graph_digest,
        "contract_digest": manifest.contract_digest,
        "profile_digest": manifest.profile_digest,
        "call_budget": reasoning.budget,
        "timeout_seconds": reasoning.timeout_seconds,
        "reasoning_metadata": dict(reasoning.metadata),
        "reasoning_output_ports": [
            {"name": port.name, "direction": port.direction.value, "schema_digest": port.schema_digest}
            for port in reasoning.output_ports
        ],
        "graph_metadata": dict(manifest.metadata),
    }
