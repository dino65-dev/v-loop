from __future__ import annotations

import pytest

from vloop.canonical import digest
from vloop.execution_certificate import (
    ExecutionCertificateSigner,
    ExecutionCertificateValidator,
    certificate_from_trace,
)
from vloop.attestations import CompletionResult, DevelopmentCompletionFabric
from vloop.graph import compile_control_graph
from vloop.graph import DynamicSubgraphPolicy, GraphEdge, GraphEdgeType, GraphManifest, GraphNode, GraphNodeType
from vloop.graph_events import GraphEventStore, build_semantic_evidence_graph
from vloop.graph_formal import export_tla_plus, import_tla_counterexample, replay_counterexample
from vloop.graph_monitor import GraphTransitionRejected
from vloop.graph_runtime import DynamicNodeResult, DurableGraphScheduler, GraphBudgetExceeded, ReadOnlyDynamicExecutor
from vloop.graph_schema import NodeImplementation, NodePort, PortDirection, schema_digest
from vloop.harness_experiments import GraphExperimentKey, TopologyRun, pair_topology_runs
from vloop.models import ActionRule, Effect, TaskContract
from vloop.proof_artifacts import ArtifactSigner, ArtifactType, ArtifactVerifier, ProofCarryingArtifact, WorkspaceTransition
from vloop.receipts import ReceiptRejected, ReceiptSigner, ReceiptVerifier


def _contract() -> TaskContract:
    return TaskContract(
        "execute a bounded graph task",
        ("complete",),
        (ActionRule("command.run", Effect.EXECUTE, "/workspace"),),
        global_completion_guards=("guard-a", "guard-b", "guard-c", "guard-d"),
    )


def _advance_to_receipt(
    scheduler: DurableGraphScheduler,
    fabric: DevelopmentCompletionFabric,
    *,
    run_id: str,
    iteration: int,
) -> None:
    def advance(node: str, **payload: object) -> None:
        scheduler.advance(run_id=run_id, iteration=iteration, template_node_id=node, event_type=node, payload=payload)

    def complete(node: str, *, facts: dict[str, str] | None = None, result: CompletionResult = CompletionResult.SUCCEEDED) -> None:
        _complete_node(scheduler, fabric, run_id=run_id, iteration=iteration, node=node, facts=facts, result=result)

    assert scheduler.completion_verifier is fabric.verifier
    advance("task.contract")
    complete("principal.contract")
    advance("snapshot.request")
    complete("snapshot.materialized")
    advance("action.intent", rule_index="0")
    complete("action.rule.0", facts={"rule_index": "0"})
    advance("join.action.rule.any")
    complete("policy.decision", facts={"approval_required": "false"})
    advance("join.action.authority.any")
    complete("capability.execute")
    complete("operation.prepared")
    advance("executor.dispatch")
    complete("executor.result", facts={"success": "true"})
    complete("artifact.manifest", facts={"success": "true"})
    for node in scheduler.manifest.nodes:
        if node.node_type is GraphNodeType.EVALUATOR:
            complete(node.node_id)
    for node in scheduler.manifest.nodes:
        if node.node_type is GraphNodeType.RECEIPT:
            complete(node.node_id)


def _complete_node(
    scheduler: DurableGraphScheduler,
    fabric: DevelopmentCompletionFabric,
    *,
    run_id: str,
    iteration: int,
    node: str,
    facts: dict[str, str] | None = None,
    result: CompletionResult = CompletionResult.SUCCEEDED,
) -> None:
    reservation = scheduler.reserve(run_id=run_id, iteration=iteration, template_node_id=node)
    verifier = scheduler.completion_verifier
    assert verifier is not None
    completion = fabric.complete(
        graph_digest=scheduler.manifest.graph_digest,
        contract_digest=scheduler.manifest.contract_digest,
        run_id=run_id,
        template_node_id=node,
        node_instance_id=reservation.node_instance_id,
        artifact_digest=schema_digest(f"artifact:{run_id}:{node}"),
        validator_policy_digest=verifier.ownership.policy_digest,
        facts=facts or {},
        result=result,
    )
    scheduler.complete(run_id=run_id, iteration=iteration, template_node_id=node, completion=completion)


def test_executable_graph_enforces_all_guard_join_and_rejects_unknown_transitions(tmp_path) -> None:
    manifest = compile_control_graph(_contract())
    store = GraphEventStore(tmp_path / "graph.db")
    fabric = DevelopmentCompletionFabric(
        graph_digest=manifest.graph_digest,
        template_roles={node.node_id: node.metadata.get("producer_role", "controller") for node in manifest.nodes},
    )
    scheduler = DurableGraphScheduler(manifest, store, tmp_path / "graph.db", completion_verifier=fabric.verifier)

    with pytest.raises(PermissionError, match="validated completion"):
        scheduler.advance(run_id="run", iteration=1, template_node_id="decision.accept", event_type="bypass")

    _advance_to_receipt(scheduler, fabric, run_id="run", iteration=1)
    _complete_node(scheduler, fabric, run_id="run", iteration=1, node="criterion.0", facts={"passed": "true"})
    with pytest.raises(GraphTransitionRejected):
        scheduler.advance(run_id="run", iteration=1, template_node_id="join.guards.all", event_type="premature")
    for index in (1, 2, 3):
        _complete_node(scheduler, fabric, run_id="run", iteration=1, node=f"criterion.{index}", facts={"passed": "true"})
    scheduler.advance(run_id="run", iteration=1, template_node_id="join.guards.all", event_type="all-guards")
    _complete_node(scheduler, fabric, run_id="run", iteration=1, node="decision.accept", facts={"decision": "accept"})

    scheduler.advance(run_id="run", iteration=2, template_node_id="task.contract", event_type="task")
    events = store.events(run_id="run")
    task_instances = [event.node_instance_id for event in events if event.template_node_id == "task.contract"]
    assert len(task_instances) == 2 and task_instances[0] != task_instances[1]


def test_causal_graph_and_signed_certificate_survive_interleaving(tmp_path) -> None:
    manifest = compile_control_graph(_contract())
    store = GraphEventStore(tmp_path / "events.db")
    fabric = DevelopmentCompletionFabric(
        graph_digest=manifest.graph_digest,
        template_roles={node.node_id: node.metadata.get("producer_role", "controller") for node in manifest.nodes},
    )
    scheduler = DurableGraphScheduler(manifest, store, tmp_path / "events.db", completion_verifier=fabric.verifier)
    _advance_to_receipt(scheduler, fabric, run_id="run-a", iteration=1)
    _advance_to_receipt(scheduler, fabric, run_id="run-b", iteration=1)
    for run_id in ("run-a", "run-b"):
        for index in range(4):
            _complete_node(scheduler, fabric, run_id=run_id, iteration=1, node=f"criterion.{index}", facts={"passed": "true"})
        scheduler.advance(run_id=run_id, iteration=1, template_node_id="join.guards.all", event_type="joined")
        _complete_node(scheduler, fabric, run_id=run_id, iteration=1, node="decision.accept", facts={"decision": "accept"})

    trace = store.events(run_id="run-a")
    semantic = build_semantic_evidence_graph(trace, run_id="run-a")
    assert semantic.nodes and all("run-b" not in node for node in semantic.nodes)
    signer = ExecutionCertificateSigner(b"c" * 32)
    certificate = signer.issue(
        certificate_from_trace(
            run_id="run-a",
            contract_digest=_contract().contract_digest,
            graph_digest=manifest.graph_digest,
            harness_bundle_digest="a" * 64,
            initial_workspace_digest="b" * 64,
            final_workspace_digest="c" * 64,
            final_decision="accept",
            events=trace,
        )
    )
    ExecutionCertificateValidator(signer.public_key_bytes).validate(certificate, trace, manifest=manifest)


def test_formal_export_is_bound_to_the_compiled_graph() -> None:
    manifest = compile_control_graph(_contract())
    model = export_tla_plus(manifest)
    assert model.graph_digest == manifest.graph_digest
    assert "NoAcceptWithoutPredecessors" in model.tla_plus


def test_compiler_materialises_action_rules_and_approval_nodes() -> None:
    contract = TaskContract(
        "publish only with review",
        ("complete",),
        (
            ActionRule("command.run", Effect.EXECUTE, "/workspace"),
            ActionRule("file.write", Effect.WRITE, "/workspace", approval_required=True),
        ),
    )
    manifest = compile_control_graph(contract)
    assert {"action.rule.0", "action.rule.1", "approval.rule.1"}.issubset(
        {node.node_id for node in manifest.nodes}
    )


def test_stale_receipt_from_same_template_instance_is_rejected() -> None:
    signer = ReceiptSigner(b"s" * 32)
    receipt = signer.issue(
        receipt_type="protected",
        run_id="run",
        intent_digest="a" * 64,
        candidate_artifact_digest="b" * 64,
        evaluator_image_digest="image",
        test_suite_digest="suite",
        result="pass",
        artifact_digests={"result": "b" * 64},
        primary_artifact_name="result",
        graph_digest="c" * 64,
        graph_node_id="evaluator.differential",
        graph_node_instance_id="d" * 64,
        schema_version=1,
    )
    verifier = ReceiptVerifier(signer.public_key_bytes)
    with pytest.raises(ReceiptRejected, match="node instance"):
        verifier.validate(
            receipt,
            receipt_type="protected",
            run_id="run",
            intent_digest="a" * 64,
            artifact_digests={"result": "b" * 64},
            graph_digest="c" * 64,
            graph_node_id="evaluator.differential",
            graph_node_instance_id="e" * 64,
        )


def test_unmatched_topology_tasks_are_rejected_before_comparison() -> None:
    key = GraphExperimentKey(
        graph_digest="a" * 64,
        harness_bundle_digest="b" * 64,
        model_id="model",
        model_config_digest="c" * 64,
        dataset_digest="d" * 64,
        tool_environment_digest="e" * 64,
        evaluator_policy_digest="f" * 64,
        compute_budget=100,
    )
    baseline = (TopologyRun("task-a", 0, "base", "1" * 64, {"success": 1.0}, key),)
    candidate = (TopologyRun("task-b", 0, "candidate", "2" * 64, {"success": 1.0}, key),)
    with pytest.raises(ValueError, match="matched task IDs"):
        pair_topology_runs(baseline, candidate)


def test_tla_counterexample_replays_as_a_runtime_fault(tmp_path) -> None:
    manifest = compile_control_graph(_contract())
    store = GraphEventStore(tmp_path / "fault.db")
    scheduler = DurableGraphScheduler(manifest, store, tmp_path / "fault.db")
    steps = import_tla_counterexample({"steps": [{"template_node_id": "decision.accept"}]})
    with pytest.raises(GraphTransitionRejected):
        replay_counterexample(scheduler, run_id="fault", iteration=1, steps=steps)


def test_workspace_transition_is_a_signed_proof_carrying_artifact() -> None:
    transition_payload = digest(
        {
            "parent_snapshot_digest": "e" * 64,
            "output_snapshot_digest": "1" * 64,
            "operation_id": "operation",
            "changed_paths": ("/workspace/a.py",),
            "artifact_manifest_digest": "2" * 64,
            "supervisor_receipt_digest": "3" * 64,
        }
    )
    artifact = ProofCarryingArtifact(
        ArtifactType.WORKSPACE_TRANSITION,
        "a" * 64,
        "b" * 64,
        "c" * 64,
        "d" * 64,
        2,
        "supervisor",
        {"parent_snapshot": "e" * 64},
        {"supervisor": "pass", "schema": "pass"},
        transition_payload,
        "supervisor-key",
    )
    signer = ArtifactSigner(b"p" * 32, signer_id="supervisor-key")
    signed = signer.sign(artifact)
    ArtifactVerifier({"supervisor-key": signer.public_key_bytes}).validate(signed)
    transition = WorkspaceTransition(signed, "e" * 64, "1" * 64, "operation", ("/workspace/a.py",), "2" * 64, "3" * 64)
    assert transition.artifact.workspace_generation == 2
    with pytest.raises(ValueError, match="covered"):
        WorkspaceTransition(signed, "e" * 64, "4" * 64, "operation", ("/workspace/a.py",), "2" * 64, "3" * 64)


def test_dynamic_executor_uses_registered_templates_and_stops_over_budget() -> None:
    contract = _contract()
    value_schema = schema_digest("read-only-result.v1")
    implementation = NodeImplementation("bounded-reader", "a" * 64, (), (value_schema,), 10, 1, 10)
    graph = GraphManifest(
        "dynamic",
        1,
        contract.contract_digest,
        "0" * 64,
        (
            GraphNode("task", GraphNodeType.TASK),
            GraphNode(
                "analysis",
                GraphNodeType.CONTEXT,
                budget=1,
                timeout_seconds=10,
                metadata={
                    "mode": "read-only-reasoning",
                    "terminal": "true",
                    "implementation_id": implementation.implementation_id,
                    "implementation_digest": implementation.implementation_digest,
                    "maximum_tokens": "10",
                },
                output_ports=(NodePort("result", PortDirection.OUTPUT, value_schema),),
            ),
        ),
        (GraphEdge("task", "analysis", GraphEdgeType.DEPENDS_ON),),
        metadata={"mode": "read-only-reasoning"},
    )

    class OverBudgetRunner:
        def run(self, _implementation, _node, _inputs):
            return DynamicNodeResult({"result": "artifact"}, 11, 1)

    policy = DynamicSubgraphPolicy(implementations={implementation.implementation_id: implementation})
    executor = ReadOnlyDynamicExecutor(policy, {implementation.implementation_id: OverBudgetRunner()})
    with pytest.raises(GraphBudgetExceeded):
        executor.execute_node(graph, "analysis", contract=contract, inputs={})
