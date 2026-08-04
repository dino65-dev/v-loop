"""Bounded TLA+ export and counterexample replay for graph-runtime protocols."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from .graph import GraphManifest, GraphNodeType
from .graph_runtime import DurableGraphScheduler


@dataclass(frozen=True, slots=True)
class FormalModel:
    module_name: str
    tla_plus: str
    graph_digest: str


@dataclass(frozen=True, slots=True)
class CounterexampleStep:
    template_node_id: str
    payload: Mapping[str, Any]


def export_tla_plus(manifest: GraphManifest) -> FormalModel:
    """Emit a bounded, graph-digest-bound state model for external TLC review."""

    manifest.validate().require_valid()
    module = "VLoop_" + manifest.graph_digest[:12]
    nodes = ", ".join(f'"{node.node_id}"' for node in manifest.nodes)
    predecessors = ",\n    ".join(
        f'"{node.node_id}" |-> {{{", ".join(f"{edge.source!r}" for edge in manifest.edges if edge.target == node.node_id)}}}'
        for node in manifest.nodes
    )
    guards = tuple(
        node.node_id for node in manifest.nodes if node.node_type is GraphNodeType.CRITERION
    )
    guard_set = ", ".join(f'"{guard}"' for guard in guards)
    accept = "decision.accept"
    text = rf'''---- MODULE {module} ----
EXTENDS Sequences, FiniteSets, TLC
CONSTANT Nodes, Predecessors, GuardNodes, Accept
VARIABLES completed, cancelled, effectDispatched, effectResolved

Init == /\ completed = {{}} /\ cancelled = FALSE
        /\ effectDispatched = FALSE /\ effectResolved = FALSE
Enabled(node) == /\ node \in Nodes
                 /\ node \notin completed
                 /\ Predecessors[node] \subseteq completed
Advance(node) == /\ Enabled(node)
                 /\ completed' = completed \cup {{node}}
                 /\ effectDispatched' = effectDispatched \/ (node = "executor.dispatch")
                 /\ effectResolved' = effectResolved \/ (node \in {{"executor.result", "operation.reconcile"}})
                 /\ UNCHANGED cancelled
Cancel == /\ cancelled' = TRUE
          /\ UNCHANGED <<completed, effectDispatched, effectResolved>>
Next == (\E node \in Nodes : Advance(node)) \/ Cancel
NoEffectWithoutCapability == "executor.dispatch" \in completed => "capability.execute" \in completed
NoAcceptWithoutAllGuards == Accept \in completed => GuardNodes \subseteq completed
NoAcceptWithoutPredecessors == Accept \in completed => Predecessors[Accept] \subseteq completed
NoEffectReplayAfterIndeterminate == effectDispatched /\ ~effectResolved => "executor.dispatch" \notin (completed \setminus {{"executor.dispatch"}})
EventualReconciliation == effectDispatched => <>(effectResolved \/ cancelled)
Spec == Init /\ [][Next]_<<completed, cancelled, effectDispatched, effectResolved>>
TypeOK == completed \subseteq Nodes /\ GuardNodes \subseteq Nodes
====
\* graph_digest = "{manifest.graph_digest}"
\* Nodes = {{{nodes}}}
\* Predecessors = [{predecessors}]
\* GuardNodes = {{{guard_set}}}
\* Accept = "{accept}"
'''
    return FormalModel(module, text, manifest.graph_digest)


def import_tla_counterexample(value: str | Mapping[str, Any]) -> tuple[CounterexampleStep, ...]:
    """Read the intentionally tiny JSON interchange used by CI fault tests.

    TLC text stays authoritative for model checking; a translated trace uses
    this closed format so a counterexample can become a deterministic runtime
    regression test rather than a document-only finding.
    """

    data = json.loads(value) if isinstance(value, str) else dict(value)
    raw_steps = data.get("steps")
    if not isinstance(raw_steps, list) or not raw_steps:
        raise ValueError("TLA+ counterexample needs a non-empty steps array")
    steps: list[CounterexampleStep] = []
    for raw in raw_steps:
        if not isinstance(raw, Mapping) or not isinstance(raw.get("template_node_id"), str):
            raise ValueError("TLA+ counterexample step has no template node id")
        payload = raw.get("payload", {})
        if not isinstance(payload, Mapping):
            raise ValueError("TLA+ counterexample payload must be an object")
        steps.append(CounterexampleStep(raw["template_node_id"], dict(payload)))
    return tuple(steps)


def replay_counterexample(
    scheduler: DurableGraphScheduler,
    *,
    run_id: str,
    iteration: int,
    steps: Iterable[CounterexampleStep],
) -> None:
    """Replay an imported trace through the real monitor; invalid paths raise."""

    for step in steps:
        scheduler.advance(
            run_id=run_id,
            iteration=iteration,
            template_node_id=step.template_node_id,
            event_type="formal-counterexample-replay",
            payload=step.payload,
        )
