"""Comparable, topology-aware metrics for graph-runtime evaluations."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import fmean
from typing import Iterable, Mapping


@dataclass(frozen=True, slots=True)
class GraphRunMetric:
    run_id: str
    topology: str
    task_succeeded: bool
    false_acceptance: bool
    false_blocking: bool
    token_cost: int
    latency_seconds: float
    model_calls: int
    recoveries: int
    human_interventions: int
    graph_nodes: int
    critical_path_length: int
    model_id: str = ""
    parallel_efficiency: float = 0.0
    communication_redundancy: float = 0.0

    def __post_init__(self) -> None:
        if not self.run_id.strip() or not self.topology.strip():
            raise ValueError("graph metrics need a run and topology")
        if any(value < 0 for value in (self.token_cost, self.latency_seconds, self.model_calls, self.recoveries, self.human_interventions, self.graph_nodes, self.critical_path_length, self.parallel_efficiency, self.communication_redundancy)):
            raise ValueError("graph metrics cannot be negative")
        if self.parallel_efficiency > 1:
            raise ValueError("parallel efficiency must be in the range [0, 1]")


@dataclass(frozen=True, slots=True)
class GraphBenchmarkSummary:
    topology: str
    model_id: str
    run_count: int
    task_success_rate: float
    false_acceptance_rate: float
    false_blocking_rate: float
    mean_token_cost: float
    mean_latency_seconds: float
    mean_model_calls: float
    mean_recoveries: float
    mean_human_interventions: float
    mean_critical_path_length: float
    mean_parallel_efficiency: float
    mean_communication_redundancy: float


def summarize_graph_benchmark(metrics: Iterable[GraphRunMetric]) -> Mapping[str, GraphBenchmarkSummary]:
    """Aggregate measured topology outcomes while preventing cross-model comparisons."""

    groups: dict[str, list[GraphRunMetric]] = {}
    for metric in metrics:
        groups.setdefault(metric.topology, []).append(metric)
    benchmark_models = {metric.model_id for group in groups.values() for metric in group}
    if len(benchmark_models) > 1:
        raise ValueError("benchmark comparisons must hold model identity constant")
    summaries: dict[str, GraphBenchmarkSummary] = {}
    for topology, group in groups.items():
        model_ids = {metric.model_id for metric in group}
        if len(model_ids) != 1:
            raise ValueError(f"topology {topology!r} mixes model identities; benchmark comparisons must hold model constant")
        summaries[topology] = GraphBenchmarkSummary(
            topology=topology,
            model_id=next(iter(model_ids)),
            run_count=len(group),
            task_success_rate=fmean(metric.task_succeeded for metric in group),
            false_acceptance_rate=fmean(metric.false_acceptance for metric in group),
            false_blocking_rate=fmean(metric.false_blocking for metric in group),
            mean_token_cost=fmean(metric.token_cost for metric in group),
            mean_latency_seconds=fmean(metric.latency_seconds for metric in group),
            mean_model_calls=fmean(metric.model_calls for metric in group),
            mean_recoveries=fmean(metric.recoveries for metric in group),
            mean_human_interventions=fmean(metric.human_interventions for metric in group),
            mean_critical_path_length=fmean(metric.critical_path_length for metric in group),
            mean_parallel_efficiency=fmean(metric.parallel_efficiency for metric in group),
            mean_communication_redundancy=fmean(metric.communication_redundancy for metric in group),
        )
    return summaries
