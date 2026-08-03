from __future__ import annotations

import pytest

from vloop.execution_certificate import (
    ExecutionCertificateSigner,
    ExecutionCertificateValidator,
    certificate_from_trace,
)
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


def _advance_to_receipt(scheduler: DurableGraphScheduler, *, run_id: str, iteration: int) -> None:
    def advance(node: str, **payload: object) -> None:
        scheduler.advance(run_id=run_id, iteration=iteration, template_node_id=node, event_type=node, payload=payload)

    advance("task.contract")
    advance("principal.contract")
    advance("snapshot.workspace")
    advance("action.intent", rule_index="0")
    advance("action.rule.0")
    advance("join.action.authority.any")
    advance("capability.execute")
    advance("operation.prepared")
    advance("executor.effect", success=None)
    advance("artifact.manifest", success=True)
    advance("evaluator.protected")
    for node in scheduler.manifest.nodes:
        if node.node_type is GraphNodeType.EVALUATOR and node.node_id != "evaluator.protected":
            advance(node.node_id, passed=True)
    for node in scheduler.manifest.nodes:
        if node.node_type is GraphNodeType.RECEIPT:
            advance(node.node_id, passed=True)


def test_executable_graph_enforces_all_guard_join_and_rejects_unknown_transitions(tmp_path) -> None:
    manifest = compile_control_graph(_contract())
    store = GraphEventStore(tmp_path / "graph.db")
    scheduler = DurableGraphScheduler(manifest, store, tmp_path / "graph.db")

    with pytest.raises(GraphTransitionRejected):
        scheduler.advance(run_id="run", iteration=1, template_node_id="decision.accept", event_type="bypass")

    _advance_to_receipt(scheduler, run_id="run", iteration=1)
    scheduler.advance(run_id="run", iteration=1, template_node_id="criterion.0", event_type="guard", payload={"passed": True})
    with pytest.raises(GraphTransitionRejected):
        scheduler.advance(run_id="run", iteration=1, template_node_id="join.guards.all", event_type="premature")
    for index in (1, 2, 3):
        scheduler.advance(run_id="run", iteration=1, template_node_id=f"criterion.{index}", event_type="guard", payload={"passed": True})
    scheduler.advance(run_id="run", iteration=1, template_node_id="join.guards.all", event_type="all-guards")
    scheduler.advance(run_id="run", iteration=1, template_node_id="decision.accept", event_type="accept")

    scheduler.advance(run_id="run", iteration=2, template_node_id="task.contract", event_type="task")
    events = store.events(run_id="run")
    task_instances = [event.node_instance_id for event in events if event.template_node_id == "task.contract"]
    assert len(task_instances) == 2 and task_instances[0] != task_instances[1]


def test_causal_graph_and_signed_certificate_survive_interleaving(tmp_path) -> None:
    manifest = compile_control_graph(_contract())
    store = GraphEventStore(tmp_path / "events.db")
    scheduler = DurableGraphScheduler(manifest, store, tmp_path / "events.db")
    _advance_to_receipt(scheduler, run_id="run-a", iteration=1)
    _advance_to_receipt(scheduler, run_id="run-b", iteration=1)
    for run_id in ("run-a", "run-b"):
        for index in range(4):
            scheduler.advance(run_id=run_id, iteration=1, template_node_id=f"criterion.{index}", event_type="guard", payload={"passed": True})
        scheduler.advance(run_id=run_id, iteration=1, template_node_id="join.guards.all", event_type="joined")
        scheduler.advance(run_id=run_id, iteration=1, template_node_id="decision.accept", event_type="accepted")

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
        "f" * 64,
        "supervisor-key",
    )
    signer = ArtifactSigner(b"p" * 32, signer_id="supervisor-key")
    signed = signer.sign(artifact)
    ArtifactVerifier({"supervisor-key": signer.public_key_bytes}).validate(signed)
    transition = WorkspaceTransition(signed, "e" * 64, "1" * 64, "operation", ("/workspace/a.py",), "2" * 64, "3" * 64)
    assert transition.artifact.workspace_generation == 2


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
