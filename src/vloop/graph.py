"""Typed, statically checked graph manifests for V-Loop control and evidence."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import StrEnum
from typing import Any, Iterable, Mapping

from .canonical import digest
from .graph_schema import (
    GraphJoin,
    GraphPredicate,
    JoinPolicy,
    NodeImplementation,
    NodePort,
    PortDirection,
    PredicateKind,
    schema_digest,
)
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
    JOIN = "join"


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
    input_ports: tuple[NodePort, ...] = ()
    output_ports: tuple[NodePort, ...] = ()
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
        ports = self.input_ports + self.output_ports
        if len({port.name for port in ports}) != len(ports):
            raise ValueError("node port names must be unique across inputs and outputs")
        if any(port.direction is not PortDirection.INPUT for port in self.input_ports) or any(
            port.direction is not PortDirection.OUTPUT for port in self.output_ports
        ):
            raise ValueError("node ports must match their declared direction")


@dataclass(frozen=True, slots=True)
class GraphEdge:
    source: str
    target: str
    edge_type: GraphEdgeType
    condition: str = ""
    source_port: str = ""
    target_port: str = ""
    predicate: GraphPredicate = GraphPredicate()

    def __post_init__(self) -> None:
        if not self.source.strip() or not self.target.strip():
            raise ValueError("graph edges need source and target")
        if bool(self.source_port) != bool(self.target_port):
            raise ValueError("typed edges need both source and target ports")
        if self.condition not in {"", "execution.success", "outcome-unknown-or-failed", "guard-pass"}:
            raise ValueError("graph edges must use a closed predicate vocabulary")


@dataclass(frozen=True, slots=True)
class GraphValidationReport:
    errors: tuple[str, ...] = ()

    @property
    def accepted(self) -> bool:
        return not self.errors

    def require_valid(self) -> None:
        if self.errors:
            raise ValueError("invalid V-Loop graph: " + "; ".join(self.errors))


_EDGE_COMPATIBILITY: Mapping[GraphEdgeType, frozenset[tuple[GraphNodeType, GraphNodeType]]] = {
    GraphEdgeType.AUTHORISED_BY: frozenset(
        {
            (GraphNodeType.ACTION, GraphNodeType.APPROVAL),
            (GraphNodeType.CAPABILITY, GraphNodeType.APPROVAL),
            (GraphNodeType.PRINCIPAL, GraphNodeType.CAPABILITY),
            (GraphNodeType.APPROVAL, GraphNodeType.CAPABILITY),
            (GraphNodeType.CAPABILITY, GraphNodeType.OPERATION),
        }
    ),
    GraphEdgeType.DISPATCHED_TO: frozenset({(GraphNodeType.OPERATION, GraphNodeType.EXECUTOR)}),
    GraphEdgeType.PRODUCED: frozenset(
        {
            (GraphNodeType.EXECUTOR, GraphNodeType.EXECUTOR),
            (GraphNodeType.EXECUTOR, GraphNodeType.ARTIFACT),
            (GraphNodeType.EVALUATOR, GraphNodeType.RECEIPT),
        }
    ),
    GraphEdgeType.VERIFIED_BY: frozenset(
        {(GraphNodeType.ARTIFACT, GraphNodeType.EVALUATOR), (GraphNodeType.RECEIPT, GraphNodeType.CRITERION)}
    ),
    GraphEdgeType.SATISFIES: frozenset(
        {
            (GraphNodeType.ACTION, GraphNodeType.JOIN),
            (GraphNodeType.APPROVAL, GraphNodeType.JOIN),
            (GraphNodeType.CAPABILITY, GraphNodeType.JOIN),
            (GraphNodeType.CRITERION, GraphNodeType.JOIN),
            (GraphNodeType.JOIN, GraphNodeType.CAPABILITY),
            (GraphNodeType.JOIN, GraphNodeType.DECISION),
        }
    ),
    GraphEdgeType.RECONCILES: frozenset({(GraphNodeType.EXECUTOR, GraphNodeType.OPERATION)}),
    GraphEdgeType.REQUIRES: frozenset({(GraphNodeType.SNAPSHOT, GraphNodeType.EVALUATOR)}),
}


def _edge_pair_is_valid(source: GraphNodeType, edge_type: GraphEdgeType, target: GraphNodeType) -> bool:
    allowed = _EDGE_COMPATIBILITY.get(edge_type)
    return allowed is None or (source, target) in allowed


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
    joins: tuple[GraphJoin, ...] = ()

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
                        "input_ports": [
                            {"name": port.name, "direction": port.direction.value, "schema_digest": port.schema_digest,
                             "required": port.required}
                            for port in node.input_ports
                        ],
                        "output_ports": [
                            {"name": port.name, "direction": port.direction.value, "schema_digest": port.schema_digest,
                             "required": port.required}
                            for port in node.output_ports
                        ],
                        "metadata": dict(node.metadata),
                    }
                    for node in self.nodes
                ],
                "edges": [
                    {"source": edge.source, "target": edge.target, "edge_type": edge.edge_type.value,
                     "condition": edge.condition, "source_port": edge.source_port, "target_port": edge.target_port,
                     "predicate": {"kind": edge.predicate.kind.value, "field": edge.predicate.field,
                                   "value": edge.predicate.value, "values": edge.predicate.values}}
                    for edge in self.edges
                ],
                "joins": [
                    {"node_id": join.node_id, "predecessors": join.predecessors, "policy": join.policy.value,
                     "threshold": join.threshold}
                    for join in self.joins
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
                continue
            source = by_id[edge.source]
            target = by_id[edge.target]
            if not _edge_pair_is_valid(source.node_type, edge.edge_type, target.node_type):
                errors.append(
                    f"edge {edge.edge_type.value} is invalid for {source.node_type.value}->{target.node_type.value}"
                )
            if edge.source_port:
                source_ports = {port.name: port for port in source.output_ports}
                target_ports = {port.name: port for port in target.input_ports}
                source_port = source_ports.get(edge.source_port)
                target_port = target_ports.get(edge.target_port)
                if source_port is None or target_port is None:
                    errors.append(f"edge {edge.source}->{edge.target} names an unknown port")
                elif source_port.schema_digest != target_port.schema_digest:
                    errors.append(f"edge {edge.source}->{edge.target} joins incompatible port schemas")
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
                edge.edge_type in {GraphEdgeType.AUTHORISED_BY, GraphEdgeType.DISPATCHED_TO}
                for edge in incoming[node.node_id]
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
                    barrier_edges = [
                        edge for edge in incoming[node.node_id]
                        if edge.edge_type is GraphEdgeType.SATISFIES and by_id[edge.source].node_type is GraphNodeType.JOIN
                    ]
                    if len(barrier_edges) != 1:
                        errors.append("accept decision needs exactly one explicit guard join")
                        continue
                    join = next((candidate for candidate in self.joins if candidate.node_id == barrier_edges[0].source), None)
                    if join is None or join.policy is not JoinPolicy.ALL:
                        errors.append("accept decision requires an ALL guard join")
                        continue
                    guard_nodes = set(join.predecessors)
                    actual_guards = {by_id[node_id].metadata.get("guard", "") for node_id in guard_nodes}
                    missing = set(node.required_guards).difference(actual_guards)
                    if missing:
                        errors.append(f"accept decision lacks criterion edges for: {sorted(missing)}")
                    if self._reachable_without(entries, node.node_id, guard_nodes, outgoing):
                        errors.append("accept decision is reachable without completion criteria")
        for join in self.joins:
            if join.node_id not in by_id or by_id[join.node_id].node_type is not GraphNodeType.JOIN:
                errors.append(f"join {join.node_id!r} does not name a join node")
                continue
            join_inputs = {edge.source for edge in incoming[join.node_id] if edge.edge_type is GraphEdgeType.SATISFIES}
            if join_inputs != set(join.predecessors):
                errors.append(f"join {join.node_id!r} does not have its declared predecessor edges")
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
            pending.extend(
                edge.source
                for edge in incoming[node_id]
                if edge.edge_type in {GraphEdgeType.AUTHORISED_BY, GraphEdgeType.DISPATCHED_TO}
            )
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


@dataclass(frozen=True, slots=True)
class GraphCompiler:
    """Compiler-owned authority boundary from a task contract to executable law.

    Planners never construct the production control graph.  They may later
    select an admitted read-only subgraph, but this compiler owns every
    authority, effect, reconciliation, evaluator, and acceptance transition.
    """

    compiler_version: str = "vloop.graph.v2"

    def compile(
        self,
        contract: TaskContract,
        *,
        evaluator_receipts: Mapping[str, str] = {},
    ) -> GraphManifest:
        return replace(
            _compile_control_graph(contract, evaluator_receipts=evaluator_receipts),
            compiler_version=self.compiler_version,
        )


def compile_control_graph(
    contract: TaskContract,
    *,
    evaluator_receipts: Mapping[str, str] = {},
) -> GraphManifest:
    """Compatibility entrypoint for the deployment-owned graph compiler."""

    return GraphCompiler().compile(contract, evaluator_receipts=evaluator_receipts)


def _compile_control_graph(
    contract: TaskContract,
    *,
    evaluator_receipts: Mapping[str, str],
) -> GraphManifest:
    """Compile an evidence-native control graph with exclusive node owners."""

    guards = contract.global_completion_guards or contract.success_conditions
    contract_schema, principal_schema = schema_digest("TaskContract.v1"), schema_digest("PrincipalAuthority.v1")
    intent_schema, policy_schema = schema_digest("ActionIntent.v1"), schema_digest("PolicyDecision.v1")
    capability_schema, operation_schema = schema_digest("Capability.v1"), schema_digest("PreparedOperation.v1")
    result_schema, artifact_schema = schema_digest("SupervisorResult.v1"), schema_digest("ArtifactManifest.v2")
    snapshot_schema, receipt_schema = schema_digest("WorkspaceSnapshot.v1"), schema_digest("SignedEvaluationReceipt.v2")
    guard_schema, barrier_schema = schema_digest("VerifiedGuard.v1"), schema_digest("AllGuardsBarrier.v1")

    def inp(name: str, schema: str) -> NodePort:
        return NodePort(name, PortDirection.INPUT, schema)

    def out(name: str, schema: str) -> NodePort:
        return NodePort(name, PortDirection.OUTPUT, schema)

    def token(value: str) -> str:
        return "".join(character if character.isalnum() else "-" for character in value).strip("-") or "unnamed"

    def owned(role: str, **metadata: str) -> Mapping[str, str]:
        return {"producer_role": role, **metadata}

    verifier_names = tuple(dict.fromkeys(name for names in contract.required_verifiers.values() for name in names))
    evaluator_specs = list((name, name) for name in (*verifier_names, *guards))
    for check_name, receipt_type in evaluator_receipts.items():
        if not check_name.strip() or not receipt_type.strip():
            raise ValueError("evaluator receipt mappings need non-empty check and receipt identities")
        if check_name not in {name for name, _receipt in evaluator_specs}:
            evaluator_specs.append((check_name, receipt_type))
    receipt_names = tuple(dict.fromkeys((*(_receipt for _name, _receipt in evaluator_specs), *guards)))
    action_rules = tuple(
        GraphNode(
            f"action.rule.{index}", GraphNodeType.ACTION,
            input_ports=(inp("intent", intent_schema),), output_ports=(out("matched_intent", intent_schema),),
            metadata=owned("policy", rule_index=str(index), tool=rule.tool, effect=rule.effect.value,
                           target_prefix=rule.target_prefix),
        )
        for index, rule in enumerate(contract.allowed_actions)
    )
    approval_nodes = tuple(
        GraphNode(
            f"approval.rule.{index}", GraphNodeType.APPROVAL, authority="human-review",
            input_ports=(inp("decision", policy_schema),), output_ports=(out("approved_intent", intent_schema),),
            metadata=owned("approval-verifier", rule_index=str(index), dynamic="true"),
        )
        for index, _rule in enumerate(contract.allowed_actions)
    )
    rule_join = GraphNode(
        "join.action.rule.any", GraphNodeType.JOIN,
        input_ports=tuple(inp(f"intent.{index}", intent_schema) for index, _ in enumerate(action_rules)),
        output_ports=(out("matched_intent", intent_schema),), metadata=owned("scheduler"),
    )
    authority_predecessors = ("policy.decision", *(node.node_id for node in approval_nodes))
    authority_join = GraphNode(
        "join.action.authority.any", GraphNodeType.JOIN,
        input_ports=tuple(inp(f"authority.{index}", intent_schema) for index, _ in enumerate(authority_predecessors)),
        output_ports=(out("authorised_intent", intent_schema),), metadata=owned("scheduler"),
    )
    evaluator_nodes = tuple(
        GraphNode(
            f"evaluator.{token(name)}", GraphNodeType.EVALUATOR,
            input_ports=(inp("artifact", artifact_schema), inp("snapshot", snapshot_schema)),
            output_ports=(out("receipt", receipt_schema),), metadata=owned(f"evaluator/{token(name)}", check_name=name),
        )
        for name, _receipt in evaluator_specs
    )
    receipt_nodes = tuple(
        GraphNode(
            f"receipt.{token(name)}", GraphNodeType.RECEIPT,
            input_ports=(inp("receipt", receipt_schema),), output_ports=(out("verified", receipt_schema),),
            metadata=owned("receipt-verifier", receipt_type=name, terminal="true"),
        )
        for name in receipt_names
    )
    criteria = tuple(
        GraphNode(
            f"criterion.{index}", GraphNodeType.CRITERION,
            input_ports=(inp("receipt", receipt_schema),), output_ports=(out("guard", guard_schema),),
            # The criterion verifier receives a receipt for the named task
            # condition, whose required check set is part of the immutable
            # graph contract.  A missing binding deliberately falls back to
            # the condition name itself, never to whatever check happens to
            # be present in a report.
            metadata=owned(
                "criterion-verifier",
                guard=guard,
                required_checks=",".join(contract.success_condition_bindings.get(guard, (guard,))),
            ),
        )
        for index, guard in enumerate(guards)
    )
    guard_join = GraphNode(
        "join.guards.all", GraphNodeType.JOIN,
        input_ports=tuple(inp(f"guard.{index}", guard_schema) for index in range(len(criteria))),
        output_ports=(out("barrier", barrier_schema),), metadata=owned("scheduler"),
    )
    nodes = (
        GraphNode("task.contract", GraphNodeType.TASK, output_ports=(out("contract", contract_schema),), metadata=owned("controller")),
        GraphNode("principal.contract", GraphNodeType.PRINCIPAL, authority="1", input_ports=(inp("contract", contract_schema),), output_ports=(out("authority", principal_schema),), metadata=owned("policy")),
        GraphNode("action.intent", GraphNodeType.ACTION, input_ports=(inp("contract", contract_schema),), output_ports=(out("intent", intent_schema),), metadata=owned("controller")),
        *action_rules,
        rule_join,
        GraphNode("policy.decision", GraphNodeType.CAPABILITY, authority="policy-gate", input_ports=(inp("intent", intent_schema),), output_ports=(out("decision", policy_schema), out("direct_intent", intent_schema)), metadata=owned("policy")),
        *approval_nodes,
        authority_join,
        GraphNode("capability.execute", GraphNodeType.CAPABILITY, authority="policy-gate", input_ports=(inp("intent", intent_schema), inp("authority", principal_schema)), output_ports=(out("capability", capability_schema),), metadata=owned("policy")),
        GraphNode("operation.prepared", GraphNodeType.OPERATION, input_ports=(inp("capability", capability_schema),), output_ports=(out("operation", operation_schema),), metadata=owned("executor-supervisor")),
        GraphNode("executor.dispatch", GraphNodeType.EXECUTOR, effect="side-effect", authority="capability.execute", input_ports=(inp("operation", operation_schema),), output_ports=(out("dispatch", operation_schema),), metadata=owned("controller")),
        GraphNode("executor.result", GraphNodeType.EXECUTOR, input_ports=(inp("dispatch", operation_schema),), output_ports=(out("result", result_schema),), metadata=owned("executor-supervisor")),
        GraphNode("operation.reconcile", GraphNodeType.OPERATION, metadata=owned("executor-supervisor", recovery="true")),
        GraphNode("snapshot.request", GraphNodeType.SNAPSHOT, input_ports=(inp("contract", contract_schema),), output_ports=(out("request", contract_schema),), metadata=owned("controller")),
        GraphNode("snapshot.materialized", GraphNodeType.SNAPSHOT, input_ports=(inp("request", contract_schema),), output_ports=(out("snapshot", snapshot_schema),), metadata=owned("snapshot-service")),
        GraphNode("artifact.manifest", GraphNodeType.ARTIFACT, input_ports=(inp("result", result_schema),), output_ports=(out("manifest", artifact_schema),), metadata=owned("artifact-validator")),
        *evaluator_nodes,
        *receipt_nodes,
        *criteria,
        guard_join,
        GraphNode("decision.accept", GraphNodeType.DECISION, effect="terminal", required_guards=guards, input_ports=(inp("barrier", barrier_schema),), metadata=owned("execution-validator")),
        GraphNode("decision.escalate", GraphNodeType.DECISION, effect="terminal", metadata=owned("controller")),
    )
    edges = (
        GraphEdge("task.contract", "principal.contract", GraphEdgeType.DEPENDS_ON, source_port="contract", target_port="contract"),
        GraphEdge("task.contract", "action.intent", GraphEdgeType.DEPENDS_ON, source_port="contract", target_port="contract"),
        GraphEdge("task.contract", "snapshot.request", GraphEdgeType.DEPENDS_ON, source_port="contract", target_port="contract"),
        *tuple(GraphEdge("action.intent", rule.node_id, GraphEdgeType.DEPENDS_ON, source_port="intent", target_port="intent", predicate=GraphPredicate(PredicateKind.FIELD_EQUALS, "rule_index", rule.metadata["rule_index"])) for rule in action_rules),
        *tuple(GraphEdge(rule.node_id, rule_join.node_id, GraphEdgeType.SATISFIES, source_port="matched_intent", target_port=f"intent.{index}") for index, rule in enumerate(action_rules)),
        GraphEdge(rule_join.node_id, "policy.decision", GraphEdgeType.DEPENDS_ON, source_port="matched_intent", target_port="intent"),
        *tuple(GraphEdge("policy.decision", approval.node_id, GraphEdgeType.AUTHORISED_BY, source_port="decision", target_port="decision", predicate=GraphPredicate(PredicateKind.FIELD_EQUALS, "approval_required", "true")) for approval in approval_nodes),
        GraphEdge("policy.decision", authority_join.node_id, GraphEdgeType.SATISFIES, source_port="direct_intent", target_port="authority.0", predicate=GraphPredicate(PredicateKind.FIELD_EQUALS, "approval_required", "false")),
        *tuple(GraphEdge(approval.node_id, authority_join.node_id, GraphEdgeType.SATISFIES, source_port="approved_intent", target_port=f"authority.{index + 1}") for index, approval in enumerate(approval_nodes)),
        GraphEdge("principal.contract", "capability.execute", GraphEdgeType.AUTHORISED_BY, source_port="authority", target_port="authority"),
        GraphEdge(authority_join.node_id, "capability.execute", GraphEdgeType.SATISFIES, source_port="authorised_intent", target_port="intent"),
        GraphEdge("capability.execute", "operation.prepared", GraphEdgeType.AUTHORISED_BY, source_port="capability", target_port="capability"),
        GraphEdge("operation.prepared", "executor.dispatch", GraphEdgeType.DISPATCHED_TO, source_port="operation", target_port="operation"),
        GraphEdge("executor.dispatch", "executor.result", GraphEdgeType.PRODUCED, source_port="dispatch", target_port="dispatch"),
        GraphEdge("executor.result", "artifact.manifest", GraphEdgeType.PRODUCED, "execution.success", "result", "result"),
        GraphEdge("executor.result", "operation.reconcile", GraphEdgeType.RECONCILES, "outcome-unknown-or-failed"),
        GraphEdge("operation.reconcile", "decision.escalate", GraphEdgeType.ESCALATES_TO),
        GraphEdge("snapshot.request", "snapshot.materialized", GraphEdgeType.DEPENDS_ON, source_port="request", target_port="request"),
        *tuple(GraphEdge("artifact.manifest", evaluator.node_id, GraphEdgeType.VERIFIED_BY, "execution.success", "manifest", "artifact") for evaluator in evaluator_nodes),
        *tuple(GraphEdge("snapshot.materialized", evaluator.node_id, GraphEdgeType.REQUIRES, source_port="snapshot", target_port="snapshot") for evaluator in evaluator_nodes),
        *tuple(
            GraphEdge(
                evaluator.node_id,
                f"receipt.{token(receipt_type)}",
                GraphEdgeType.PRODUCED,
                source_port="receipt",
                target_port="receipt",
            )
            for evaluator, (_check_name, receipt_type) in zip(evaluator_nodes, evaluator_specs, strict=True)
        ),
        *tuple(GraphEdge(f"receipt.{token(criterion.metadata['guard'])}", criterion.node_id, GraphEdgeType.VERIFIED_BY, source_port="verified", target_port="receipt") for criterion in criteria),
        *tuple(GraphEdge(criterion.node_id, guard_join.node_id, GraphEdgeType.SATISFIES, "guard-pass", "guard", f"guard.{index}") for index, criterion in enumerate(criteria)),
        GraphEdge(guard_join.node_id, "decision.accept", GraphEdgeType.SATISFIES, source_port="barrier", target_port="barrier"),
    )
    graph = GraphManifest(
        "vloop-control", 2, contract.contract_digest, contract.profile_digest or "0" * 64, nodes, edges,
        metadata={"completion_protocol": "vloop.attestation.v1"},
        joins=(
            GraphJoin(rule_join.node_id, tuple(node.node_id for node in action_rules), JoinPolicy.ANY),
            GraphJoin(authority_join.node_id, authority_predecessors, JoinPolicy.ANY),
            GraphJoin(guard_join.node_id, tuple(node.node_id for node in criteria), JoinPolicy.ALL),
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
    maximum_total_tokens: int = 32_000
    maximum_parallelism: int = 4
    implementations: Mapping[str, NodeImplementation] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if min(self.maximum_nodes, self.maximum_total_tokens, self.maximum_parallelism) < 1 or self.maximum_edges < 0:
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
        try:
            requested_parallelism = int(graph.metadata.get("maximum_parallelism", "1"))
        except ValueError as exc:
            raise PermissionError("dynamic graph has an invalid parallelism limit") from exc
        if requested_parallelism < 1 or requested_parallelism > self.maximum_parallelism:
            raise PermissionError("dynamic graph exceeds server-owned parallelism limit")
        total_tokens = 0
        for node in graph.nodes:
            if node.node_type not in self.allowed_node_types or node.effect != "read-only" or node.authority:
                raise PermissionError(f"dynamic graph node {node.node_id!r} exceeds read-only authority")
            if node.node_type is GraphNodeType.TASK:
                continue
            implementation_id = node.metadata.get("implementation_id", "")
            implementation = self.implementations.get(implementation_id)
            if implementation is None or node.metadata.get("implementation_digest") != implementation.implementation_digest:
                raise PermissionError(f"dynamic node {node.node_id!r} lacks an approved implementation digest")
            if node.budget is None or node.timeout_seconds is None:
                raise PermissionError(f"dynamic node {node.node_id!r} lacks call or timeout budget")
            try:
                token_budget = int(node.metadata.get("maximum_tokens", "0"))
            except ValueError as exc:
                raise PermissionError(f"dynamic node {node.node_id!r} has an invalid token budget") from exc
            if token_budget < 1 or token_budget > implementation.maximum_tokens or node.budget > implementation.maximum_calls or node.timeout_seconds > implementation.timeout_seconds:
                raise PermissionError(f"dynamic node {node.node_id!r} exceeds its implementation budget")
            if any(key in node.metadata for key in ("credential", "secret", "network")) or implementation.network_allowed:
                raise PermissionError(f"dynamic node {node.node_id!r} requests credentials or network access")
            if not node.input_ports and not node.output_ports:
                raise PermissionError(f"dynamic node {node.node_id!r} lacks typed ports")
            schemas = {port.schema_digest for port in node.input_ports + node.output_ports}
            if not schemas.issubset(set(implementation.input_schema_digests + implementation.output_schema_digests)):
                raise PermissionError(f"dynamic node {node.node_id!r} uses schemas outside its implementation")
            total_tokens += token_budget
        if total_tokens > self.maximum_total_tokens:
            raise PermissionError("dynamic graph exceeds server-owned total token budget")
        if any(edge.edge_type not in self.allowed_edge_types for edge in graph.edges):
            raise PermissionError("dynamic graph contains a non-approved edge type")
        if GraphManifest._cyclic_components(
            {node.node_id: [edge for edge in graph.edges if edge.source == node.node_id] for node in graph.nodes}
        ):
            raise PermissionError("dynamic reasoning graphs must be acyclic")
        graph.validate().require_valid()
        return graph
