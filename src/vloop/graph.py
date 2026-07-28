"""Typed, statically checked graph manifests for V-Loop control and evidence."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Iterable, Mapping

from .canonical import digest
from .models import TaskContract


class GraphNodeType(StrEnum):
    TASK = "task"
    ACTION = "action"
    CAPABILITY = "capability"
    OPERATION = "operation"
    EXECUTOR = "executor"
    EVALUATOR = "evaluator"
    RECEIPT = "receipt"
    CRITERION = "criterion"
    DECISION = "decision"


class GraphEdgeType(StrEnum):
    DEPENDS_ON = "depends-on"
    AUTHORISED_BY = "authorised-by"
    DISPATCHED_TO = "dispatched-to"
    PRODUCED = "produced"
    VERIFIED_BY = "verified-by"
    SATISFIES = "satisfies"
    ESCALATES_TO = "escalates-to"


@dataclass(frozen=True, slots=True)
class GraphNode:
    node_id: str
    node_type: GraphNodeType
    effect: str = "read-only"
    authority: str = ""
    budget: int | None = None
    required_guards: tuple[str, ...] = ()
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.node_id.strip():
            raise ValueError("graph nodes need an identity")
        if self.effect not in {"read-only", "side-effect", "terminal"}:
            raise ValueError("graph node effect is invalid")
        if self.effect == "side-effect" and not self.authority.strip():
            raise ValueError("side-effect graph nodes need an authority")
        if self.budget is not None and self.budget < 1:
            raise ValueError("graph node budgets must be positive")


@dataclass(frozen=True, slots=True)
class GraphEdge:
    source: str
    target: str
    edge_type: GraphEdgeType
    condition: str = ""

    def __post_init__(self) -> None:
        if not self.source.strip() or not self.target.strip():
            raise ValueError("graph edges need source and target")


@dataclass(frozen=True, slots=True)
class GraphValidationReport:
    errors: tuple[str, ...] = ()

    @property
    def accepted(self) -> bool:
        return not self.errors

    def require_valid(self) -> None:
        if self.errors:
            raise ValueError("invalid V-Loop graph: " + "; ".join(self.errors))


@dataclass(frozen=True, slots=True)
class GraphManifest:
    graph_id: str
    schema_version: int
    contract_digest: str
    profile_digest: str
    nodes: tuple[GraphNode, ...]
    edges: tuple[GraphEdge, ...]
    compiler_version: str = "vloop.graph.v1"
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.graph_id.strip() or self.schema_version < 1:
            raise ValueError("graph manifest needs an id and schema version")
        if len(self.contract_digest) != 64 or len(self.profile_digest) != 64:
            raise ValueError("graph manifest needs bound contract and profile digests")

    @property
    def graph_digest(self) -> str:
        return digest(
            {
                "graph_id": self.graph_id,
                "schema_version": self.schema_version,
                "contract_digest": self.contract_digest,
                "profile_digest": self.profile_digest,
                "compiler_version": self.compiler_version,
                "metadata": dict(self.metadata),
                "nodes": [
                    {
                        "node_id": node.node_id,
                        "node_type": node.node_type.value,
                        "effect": node.effect,
                        "authority": node.authority,
                        "budget": node.budget,
                        "required_guards": node.required_guards,
                        "metadata": dict(node.metadata),
                    }
                    for node in self.nodes
                ],
                "edges": [
                    {"source": edge.source, "target": edge.target, "edge_type": edge.edge_type.value,
                     "condition": edge.condition}
                    for edge in self.edges
                ],
            }
        )

    def validate(self) -> GraphValidationReport:
        errors: list[str] = []
        node_ids = {node.node_id for node in self.nodes}
        if len(node_ids) != len(self.nodes):
            errors.append("node identities are not unique")
        for edge in self.edges:
            if edge.source not in node_ids or edge.target not in node_ids:
                errors.append(f"edge {edge.source}->{edge.target} refers to an unknown node")
        incoming: dict[str, list[GraphEdge]] = {node_id: [] for node_id in node_ids}
        for edge in self.edges:
            if edge.target in incoming:
                incoming[edge.target].append(edge)
        for node in self.nodes:
            if node.effect == "side-effect" and not any(
                edge.edge_type is GraphEdgeType.AUTHORISED_BY for edge in incoming[node.node_id]
            ):
                errors.append(f"side-effect node {node.node_id!r} has no authority edge")
            if node.node_type is GraphNodeType.DECISION and node.node_id == "decision.accept":
                if not node.required_guards:
                    errors.append("accept decision has no completion requirements")
        return GraphValidationReport(tuple(errors))


def compile_control_graph(contract: TaskContract) -> GraphManifest:
    """Compile V-Loop's fixed authority/control skeleton into a typed graph."""

    graph = GraphManifest(
        graph_id="vloop-control",
        schema_version=1,
        contract_digest=contract.contract_digest,
        profile_digest=contract.profile_digest or "0" * 64,
        nodes=(
            GraphNode("task.contract", GraphNodeType.TASK),
            GraphNode("action.intent", GraphNodeType.ACTION),
            GraphNode("capability.execute", GraphNodeType.CAPABILITY, authority="policy-gate"),
            GraphNode("operation.prepared", GraphNodeType.OPERATION),
            GraphNode("executor.effect", GraphNodeType.EXECUTOR, effect="side-effect", authority="capability.execute"),
            GraphNode("evaluator.protected", GraphNodeType.EVALUATOR),
            GraphNode("receipt.evidence", GraphNodeType.RECEIPT),
            GraphNode("criterion.global", GraphNodeType.CRITERION),
            GraphNode(
                "decision.accept",
                GraphNodeType.DECISION,
                effect="terminal",
                required_guards=contract.global_completion_guards or contract.success_conditions,
            ),
            GraphNode("decision.escalate", GraphNodeType.DECISION, effect="terminal"),
        ),
        edges=(
            GraphEdge("task.contract", "action.intent", GraphEdgeType.DEPENDS_ON),
            GraphEdge("action.intent", "capability.execute", GraphEdgeType.DEPENDS_ON),
            GraphEdge("capability.execute", "operation.prepared", GraphEdgeType.AUTHORISED_BY),
            GraphEdge("operation.prepared", "executor.effect", GraphEdgeType.AUTHORISED_BY),
            GraphEdge("executor.effect", "evaluator.protected", GraphEdgeType.PRODUCED, "execution.success"),
            GraphEdge("evaluator.protected", "receipt.evidence", GraphEdgeType.PRODUCED),
            GraphEdge("receipt.evidence", "criterion.global", GraphEdgeType.VERIFIED_BY),
            GraphEdge("criterion.global", "decision.accept", GraphEdgeType.SATISFIES, "all-global-guards-pass"),
            GraphEdge("executor.effect", "decision.escalate", GraphEdgeType.ESCALATES_TO, "reconciliation-or-failure"),
        ),
    )
    graph.validate().require_valid()
    return graph


@dataclass(frozen=True, slots=True)
class EvidenceGraph:
    """Read-only, hash-addressed graph view of immutable ledger events."""

    run_id: str
    graph_digest: str
    nodes: tuple[GraphNode, ...]
    edges: tuple[GraphEdge, ...]

    def to_dot(self) -> str:
        """Render a deterministic Graphviz DOT projection for inspection."""

        lines = ["digraph vloop_evidence {"]
        for node in self.nodes:
            label = node.metadata.get("event_type", node.node_type.value).replace('"', "'")
            lines.append(f'  "{node.node_id}" [label="{label}"];')
        for edge in self.edges:
            lines.append(f'  "{edge.source}" -> "{edge.target}" [label="{edge.edge_type.value}"];')
        lines.append("}")
        return "\n".join(lines)


def build_evidence_graph(events: Iterable[Mapping[str, Any]], *, run_id: str) -> EvidenceGraph:
    """Project one run's ledger chain into an inspectable evidence graph."""

    selected = [event for event in events if event.get("payload", {}).get("run_id") == run_id]
    if not run_id.strip() or not selected:
        raise ValueError("evidence graph needs a run with ledger events")
    nodes = tuple(
        GraphNode(
            node_id=str(event["event_hash"]),
            node_type=GraphNodeType.RECEIPT,
            metadata={
                "event_type": str(event["event_type"]),
                "sequence": str(event["sequence"]),
                "occurred_at": str(event["occurred_at"]),
            },
        )
        for event in selected
    )
    hashes = {node.node_id for node in nodes}
    edges = tuple(
        GraphEdge(str(event["parent_hash"]), str(event["event_hash"]), GraphEdgeType.DEPENDS_ON)
        for event in selected
        if str(event["parent_hash"]) in hashes
    )
    return EvidenceGraph(
        run_id=run_id,
        graph_digest=digest(
            {
                "run_id": run_id,
                "events": [str(event["event_hash"]) for event in selected],
                "edges": [(edge.source, edge.target) for edge in edges],
            }
        ),
        nodes=nodes,
        edges=edges,
    )
