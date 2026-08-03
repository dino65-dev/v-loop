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
    PRINCIPAL = "principal"
    APPROVAL = "approval"
    CAPABILITY = "capability"
    OPERATION = "operation"
    EXECUTOR = "executor"
    SNAPSHOT = "snapshot"
    ARTIFACT = "artifact"
    EVALUATOR = "evaluator"
    RECEIPT = "receipt"
    CRITERION = "criterion"
    DECISION = "decision"
    MEMORY = "memory"
    CONTEXT = "context"


class GraphEdgeType(StrEnum):
    DEPENDS_ON = "depends-on"
    AUTHORISED_BY = "authorised-by"
    DERIVED_FROM = "derived-from"
    REQUIRES = "requires"
    DISPATCHED_TO = "dispatched-to"
    PRODUCED = "produced"
    VERIFIED_BY = "verified-by"
    SATISFIES = "satisfies"
    RECONCILES = "reconciles"
    ESCALATES_TO = "escalates-to"


@dataclass(frozen=True, slots=True)
class GraphNode:
    node_id: str
    node_type: GraphNodeType
    effect: str = "read-only"
    authority: str = ""
    budget: int | None = None
    timeout_seconds: int | None = None
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
        if self.timeout_seconds is not None and self.timeout_seconds < 1:
            raise ValueError("graph node timeouts must be positive")


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
                        "timeout_seconds": node.timeout_seconds,
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
        by_id = {node.node_id: node for node in self.nodes}
        if len(node_ids) != len(self.nodes):
            errors.append("node identities are not unique")
        for edge in self.edges:
            if edge.source not in node_ids or edge.target not in node_ids:
                errors.append(f"edge {edge.source}->{edge.target} refers to an unknown node")
        incoming: dict[str, list[GraphEdge]] = {node_id: [] for node_id in node_ids}
        outgoing: dict[str, list[GraphEdge]] = {node_id: [] for node_id in node_ids}
        for edge in self.edges:
            if edge.source in outgoing and edge.target in incoming:
                incoming[edge.target].append(edge)
                outgoing[edge.source].append(edge)
        entries = {node_id for node_id, edges in incoming.items() if not edges}
        if not entries:
            errors.append("graph has no entry node")
        reachable = self._reachable_from(entries, outgoing)
        for node in self.nodes:
            if node.node_id not in reachable:
                errors.append(f"node {node.node_id!r} is unreachable")
            if (
                node.node_id in reachable
                and not outgoing[node.node_id]
                and node.effect != "terminal"
                and node.effect != "side-effect"
                and node.metadata.get("terminal") != "true"
            ):
                errors.append(f"reachable node {node.node_id!r} has no successor or terminal declaration")
        for node in self.nodes:
            if node.effect == "side-effect" and not any(
                edge.edge_type is GraphEdgeType.AUTHORISED_BY for edge in incoming[node.node_id]
            ):
                errors.append(f"side-effect node {node.node_id!r} has no authority edge")
            if node.effect == "side-effect" and not self._has_capability_path(node.node_id, incoming, by_id):
                errors.append(f"side-effect node {node.node_id!r} has no capability authority path")
            if node.effect == "side-effect" and not self._reaches_recovery(node.node_id, outgoing, by_id):
                errors.append(f"side-effect node {node.node_id!r} has no reconciliation path")
            if node.metadata.get("approval_required") == "true" and not self._has_approval_path(
                node.node_id, incoming, by_id
            ):
                errors.append(f"approval-required node {node.node_id!r} has no approval authority path")
            if node.metadata.get("human_control") == "required" and not self._reaches_human_exit(
                node.node_id, outgoing, by_id
            ):
                errors.append(f"human-controlled node {node.node_id!r} has no escalation or cancellation path")
            if node.node_type is GraphNodeType.DECISION and node.node_id == "decision.accept":
                if not node.required_guards:
                    errors.append("accept decision has no completion requirements")
                else:
                    guard_nodes = {
                        edge.source
                        for edge in incoming[node.node_id]
                        if edge.edge_type is GraphEdgeType.SATISFIES
                        and by_id[edge.source].node_type is GraphNodeType.CRITERION
                    }
                    actual_guards = {by_id[node_id].metadata.get("guard", "") for node_id in guard_nodes}
                    missing = set(node.required_guards).difference(actual_guards)
                    if missing:
                        errors.append(f"accept decision lacks criterion edges for: {sorted(missing)}")
                    if self._reachable_without(entries, node.node_id, guard_nodes, outgoing):
                        errors.append("accept decision is reachable without completion criteria")
        for component in self._cyclic_components(outgoing):
            component_nodes = [by_id[node_id] for node_id in component]
            has_budget = all(node.budget is not None for node in component_nodes)
            has_timeout = all(node.timeout_seconds is not None for node in component_nodes)
            has_escalation = any(
                edge.edge_type is GraphEdgeType.ESCALATES_TO and edge.target not in component
                for node_id in component
                for edge in outgoing[node_id]
            )
            if not has_budget or not has_timeout or not has_escalation:
                errors.append(
                    "cycle requires a budget, timeout, and escalation edge: " + ", ".join(sorted(component))
                )
        for node in self.nodes:
            if node.metadata.get("trust") != "untrusted":
                continue
            for protected_type in (GraphNodeType.APPROVAL, GraphNodeType.CAPABILITY):
                if self._reaches_type(node.node_id, protected_type, outgoing, by_id):
                    errors.append(f"untrusted node {node.node_id!r} can influence an authority node")
        for node in self.nodes:
            if node.node_type is not GraphNodeType.MEMORY or not node.authority:
                continue
            origins = [
                by_id[edge.source]
                for edge in incoming[node.node_id]
                if edge.edge_type is GraphEdgeType.DERIVED_FROM
            ]
            if not origins:
                errors.append(f"memory node {node.node_id!r} with authority has no provenance origin")
                continue
            try:
                authority = int(node.authority)
                origin_authorities = [int(origin.authority) for origin in origins]
            except ValueError:
                errors.append(f"memory authority for {node.node_id!r} and its origins must be numeric")
                continue
            if any(authority > origin_authority for origin_authority in origin_authorities):
                errors.append(f"memory node {node.node_id!r} exceeds its origin authority")
        return GraphValidationReport(tuple(errors))

    @staticmethod
    def _reachable_from(starts: set[str], outgoing: Mapping[str, list[GraphEdge]]) -> set[str]:
        pending, seen = list(starts), set()
        while pending:
            node_id = pending.pop()
            if node_id in seen:
                continue
            seen.add(node_id)
            pending.extend(edge.target for edge in outgoing[node_id])
        return seen

    @staticmethod
    def _has_capability_path(
        start: str, incoming: Mapping[str, list[GraphEdge]], by_id: Mapping[str, GraphNode]
    ) -> bool:
        pending, seen = [start], set()
        while pending:
            node_id = pending.pop()
            if node_id in seen:
                continue
            seen.add(node_id)
            if node_id != start and by_id[node_id].node_type is GraphNodeType.CAPABILITY:
                return True
            pending.extend(edge.source for edge in incoming[node_id] if edge.edge_type is GraphEdgeType.AUTHORISED_BY)
        return False

    @staticmethod
    def _has_approval_path(
        start: str, incoming: Mapping[str, list[GraphEdge]], by_id: Mapping[str, GraphNode]
    ) -> bool:
        pending, seen = [start], set()
        while pending:
            node_id = pending.pop()
            if node_id in seen:
                continue
            seen.add(node_id)
            if node_id != start and by_id[node_id].node_type is GraphNodeType.APPROVAL:
                return True
            pending.extend(edge.source for edge in incoming[node_id] if edge.edge_type is GraphEdgeType.AUTHORISED_BY)
        return False

    @staticmethod
    def _reaches_human_exit(
        start: str, outgoing: Mapping[str, list[GraphEdge]], by_id: Mapping[str, GraphNode]
    ) -> bool:
        pending, seen = [start], set()
        while pending:
            node_id = pending.pop()
            if node_id in seen:
                continue
            seen.add(node_id)
            node = by_id[node_id]
            if node_id != start and node.node_type is GraphNodeType.DECISION and (
                node.node_id.endswith("escalate") or node.node_id.endswith("cancel")
            ):
                return True
            pending.extend(edge.target for edge in outgoing[node_id])
        return False

    @staticmethod
    def _reaches_recovery(
        start: str, outgoing: Mapping[str, list[GraphEdge]], by_id: Mapping[str, GraphNode]
    ) -> bool:
        pending, seen = [start], set()
        while pending:
            node_id = pending.pop()
            if node_id in seen:
                continue
            seen.add(node_id)
            node = by_id[node_id]
            if node.node_type is GraphNodeType.OPERATION and node.metadata.get("recovery") == "true":
                return True
            pending.extend(edge.target for edge in outgoing[node_id])
        return False

    @staticmethod
    def _reaches_type(
        start: str,
        node_type: GraphNodeType,
        outgoing: Mapping[str, list[GraphEdge]],
        by_id: Mapping[str, GraphNode],
    ) -> bool:
        pending, seen = [start], set()
        while pending:
            node_id = pending.pop()
            if node_id in seen:
                continue
            seen.add(node_id)
            if node_id != start and by_id[node_id].node_type is node_type:
                return True
            pending.extend(edge.target for edge in outgoing[node_id])
        return False

    @staticmethod
    def _reachable_without(
        starts: set[str], target: str, excluded: set[str], outgoing: Mapping[str, list[GraphEdge]]
    ) -> bool:
        pending, seen = list(starts.difference(excluded)), set()
        while pending:
            node_id = pending.pop()
            if node_id in seen or node_id in excluded:
                continue
            if node_id == target:
                return True
            seen.add(node_id)
            pending.extend(edge.target for edge in outgoing[node_id])
        return False

    @staticmethod
    def _cyclic_components(outgoing: Mapping[str, list[GraphEdge]]) -> tuple[frozenset[str], ...]:
        """Return strongly connected components that form a directed cycle."""

        index, stack, indices, lowlinks, on_stack, components = 0, [], {}, {}, set(), []

        def visit(node_id: str) -> None:
            nonlocal index
            indices[node_id] = lowlinks[node_id] = index
            index += 1
            stack.append(node_id)
            on_stack.add(node_id)
            for edge in outgoing[node_id]:
                target = edge.target
                if target not in indices:
                    visit(target)
                    lowlinks[node_id] = min(lowlinks[node_id], lowlinks[target])
                elif target in on_stack:
                    lowlinks[node_id] = min(lowlinks[node_id], indices[target])
            if lowlinks[node_id] == indices[node_id]:
                component: set[str] = set()
                while True:
                    target = stack.pop()
                    on_stack.remove(target)
                    component.add(target)
                    if target == node_id:
                        break
                if len(component) > 1 or any(edge.target == node_id for edge in outgoing[node_id]):
                    components.append(frozenset(component))

        for node_id in outgoing:
            if node_id not in indices:
                visit(node_id)
        return tuple(components)


def compile_control_graph(contract: TaskContract) -> GraphManifest:
    """Compile V-Loop's fixed authority/control skeleton into a typed graph."""

    guards = contract.global_completion_guards or contract.success_conditions
    criteria = tuple(
        GraphNode(f"criterion.{index}", GraphNodeType.CRITERION, metadata={"guard": guard})
        for index, guard in enumerate(guards)
    )
    graph = GraphManifest(
        graph_id="vloop-control",
        schema_version=1,
        contract_digest=contract.contract_digest,
        profile_digest=contract.profile_digest or "0" * 64,
        nodes=(
            GraphNode("task.contract", GraphNodeType.TASK),
            GraphNode("principal.contract", GraphNodeType.PRINCIPAL, authority="1"),
            GraphNode("action.intent", GraphNodeType.ACTION),
            GraphNode("capability.execute", GraphNodeType.CAPABILITY, authority="policy-gate"),
            GraphNode("operation.prepared", GraphNodeType.OPERATION),
            GraphNode("executor.effect", GraphNodeType.EXECUTOR, effect="side-effect", authority="capability.execute"),
            GraphNode("operation.reconcile", GraphNodeType.OPERATION, metadata={"recovery": "true"}),
            GraphNode("snapshot.workspace", GraphNodeType.SNAPSHOT),
            GraphNode("artifact.manifest", GraphNodeType.ARTIFACT),
            GraphNode("evaluator.protected", GraphNodeType.EVALUATOR),
            GraphNode("receipt.evidence", GraphNodeType.RECEIPT),
            GraphNode(
                "decision.accept",
                GraphNodeType.DECISION,
                effect="terminal",
                required_guards=guards,
            ),
            GraphNode("decision.escalate", GraphNodeType.DECISION, effect="terminal"),
            *criteria,
        ),
        edges=(
            GraphEdge("task.contract", "principal.contract", GraphEdgeType.DEPENDS_ON),
            GraphEdge("task.contract", "action.intent", GraphEdgeType.DEPENDS_ON),
            GraphEdge("principal.contract", "capability.execute", GraphEdgeType.AUTHORISED_BY),
            GraphEdge("action.intent", "capability.execute", GraphEdgeType.DEPENDS_ON),
            GraphEdge("capability.execute", "operation.prepared", GraphEdgeType.AUTHORISED_BY),
            GraphEdge("operation.prepared", "executor.effect", GraphEdgeType.AUTHORISED_BY),
            GraphEdge("task.contract", "snapshot.workspace", GraphEdgeType.DEPENDS_ON),
            GraphEdge("executor.effect", "artifact.manifest", GraphEdgeType.PRODUCED, "execution.success"),
            GraphEdge("artifact.manifest", "evaluator.protected", GraphEdgeType.VERIFIED_BY, "execution.success"),
            GraphEdge("snapshot.workspace", "evaluator.protected", GraphEdgeType.REQUIRES),
            GraphEdge("executor.effect", "operation.reconcile", GraphEdgeType.RECONCILES, "outcome-unknown-or-failed"),
            GraphEdge("operation.reconcile", "decision.escalate", GraphEdgeType.ESCALATES_TO),
            GraphEdge("evaluator.protected", "receipt.evidence", GraphEdgeType.PRODUCED),
            *tuple(
                GraphEdge("receipt.evidence", criterion.node_id, GraphEdgeType.VERIFIED_BY)
                for criterion in criteria
            ),
            *tuple(
                GraphEdge(criterion.node_id, "decision.accept", GraphEdgeType.SATISFIES, "guard-pass")
                for criterion in criteria
            ),
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


@dataclass(frozen=True, slots=True)
class DynamicSubgraphPolicy:
    """Server-owned ceiling for planner-proposed read-only reasoning graphs."""

    allowed_node_types: frozenset[GraphNodeType] = frozenset(
        {
            GraphNodeType.TASK,
            GraphNodeType.CONTEXT,
            GraphNodeType.ACTION,
            GraphNodeType.ARTIFACT,
            GraphNodeType.CRITERION,
        }
    )
    allowed_edge_types: frozenset[GraphEdgeType] = frozenset(
        {
            GraphEdgeType.DEPENDS_ON,
            GraphEdgeType.DERIVED_FROM,
            GraphEdgeType.REQUIRES,
            GraphEdgeType.PRODUCED,
            GraphEdgeType.VERIFIED_BY,
        }
    )
    maximum_nodes: int = 24
    maximum_edges: int = 48

    def __post_init__(self) -> None:
        if self.maximum_nodes < 1 or self.maximum_edges < 0:
            raise ValueError("dynamic graph limits are invalid")

    def admit(self, graph: GraphManifest, *, contract: TaskContract) -> GraphManifest:
        """Validate a proposal against the maximum-authority graph policy."""

        expected_profile = contract.profile_digest or "0" * 64
        if graph.contract_digest != contract.contract_digest or graph.profile_digest != expected_profile:
            raise PermissionError("dynamic graph belongs to another task contract or profile")
        if len(graph.nodes) > self.maximum_nodes or len(graph.edges) > self.maximum_edges:
            raise PermissionError("dynamic graph exceeds server-owned size limits")
        if graph.metadata.get("mode") != "read-only-reasoning":
            raise PermissionError("dynamic graph is not labelled read-only reasoning")
        for node in graph.nodes:
            if node.node_type not in self.allowed_node_types or node.effect != "read-only" or node.authority:
                raise PermissionError(f"dynamic graph node {node.node_id!r} exceeds read-only authority")
        if any(edge.edge_type not in self.allowed_edge_types for edge in graph.edges):
            raise PermissionError("dynamic graph contains a non-approved edge type")
        if GraphManifest._cyclic_components(
            {node.node_id: [edge for edge in graph.edges if edge.source == node.node_id] for node in graph.nodes}
        ):
            raise PermissionError("dynamic reasoning graphs must be acyclic")
        graph.validate().require_valid()
        return graph
