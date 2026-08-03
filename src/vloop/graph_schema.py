"""Data-level schemas, ports, predicates, and joins for executable graphs.

The graph manifest remains intentionally small.  This module supplies the
parts that turn a structural graph into an executable protocol: values move
through typed ports, predicates have server-owned semantics, and joins make
AND/ANY behaviour explicit rather than implicit in graph reachability.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Mapping

from .canonical import digest


class PortDirection(StrEnum):
    INPUT = "input"
    OUTPUT = "output"


class JoinPolicy(StrEnum):
    ALL = "all"
    ANY = "any"
    QUORUM = "quorum"
    THRESHOLD = "threshold"


class PredicateKind(StrEnum):
    ALWAYS = "always"
    FIELD_EQUALS = "field-equals"
    FIELD_TRUTHY = "field-truthy"
    FIELD_IN = "field-in"


def schema_digest(schema_id: str) -> str:
    """Return the canonical identity of a server-owned artifact schema."""

    if not schema_id.strip():
        raise ValueError("artifact schema id is required")
    return digest({"artifact_schema": schema_id})


@dataclass(frozen=True, slots=True)
class NodePort:
    name: str
    direction: PortDirection
    schema_digest: str
    required: bool = True

    def __post_init__(self) -> None:
        if not self.name.strip() or len(self.schema_digest) != 64:
            raise ValueError("node ports need a name and SHA-256 schema digest")


@dataclass(frozen=True, slots=True)
class GraphPredicate:
    """Closed predicate AST; model-provided Python expressions are forbidden."""

    kind: PredicateKind = PredicateKind.ALWAYS
    field: str = ""
    value: str = ""
    values: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.kind is PredicateKind.ALWAYS and (self.field or self.value or self.values):
            raise ValueError("always predicates cannot carry operands")
        if self.kind in {PredicateKind.FIELD_EQUALS, PredicateKind.FIELD_TRUTHY, PredicateKind.FIELD_IN} and not self.field:
            raise ValueError("field predicates need a field name")
        if self.kind is PredicateKind.FIELD_EQUALS and not self.value:
            raise ValueError("equals predicates need a value")
        if self.kind is PredicateKind.FIELD_IN and not self.values:
            raise ValueError("membership predicates need values")

    def evaluate(self, event: Mapping[str, Any]) -> bool:
        if self.kind is PredicateKind.ALWAYS:
            return True
        observed = event.get(self.field)
        if self.kind is PredicateKind.FIELD_TRUTHY:
            return bool(observed)
        if self.kind is PredicateKind.FIELD_EQUALS:
            return str(observed) == self.value
        return str(observed) in self.values


@dataclass(frozen=True, slots=True)
class GraphJoin:
    """A compiled predecessor barrier with explicit execution semantics."""

    node_id: str
    predecessors: tuple[str, ...]
    policy: JoinPolicy
    threshold: int | None = None

    def __post_init__(self) -> None:
        if not self.node_id.strip() or not self.predecessors:
            raise ValueError("joins need an identity and predecessors")
        if len(self.predecessors) != len(set(self.predecessors)):
            raise ValueError("join predecessors must be unique")
        if self.policy in {JoinPolicy.QUORUM, JoinPolicy.THRESHOLD}:
            if self.threshold is None or not 1 <= self.threshold <= len(self.predecessors):
                raise ValueError("threshold joins need a threshold inside their predecessor count")
        elif self.threshold is not None:
            raise ValueError("only threshold joins may carry a threshold")

    def satisfied_by(self, completed: set[str]) -> bool:
        count = len(set(self.predecessors).intersection(completed))
        if self.policy is JoinPolicy.ALL:
            return count == len(self.predecessors)
        if self.policy is JoinPolicy.ANY:
            return count >= 1
        return count >= (self.threshold or 0)


@dataclass(frozen=True, slots=True)
class NodeImplementation:
    """Deployment-owned template a dynamic graph node may select, never define."""

    implementation_id: str
    image_digest: str
    input_schema_digests: tuple[str, ...]
    output_schema_digests: tuple[str, ...]
    maximum_tokens: int
    maximum_calls: int
    timeout_seconds: int
    network_allowed: bool = False

    def __post_init__(self) -> None:
        if not self.implementation_id.strip() or min(self.maximum_tokens, self.maximum_calls, self.timeout_seconds) < 1:
            raise ValueError("node implementations need identity and positive resource limits")
        for value in (self.image_digest, *self.input_schema_digests, *self.output_schema_digests):
            if len(value) != 64:
                raise ValueError("node implementation identities must be SHA-256 digests")

    @property
    def implementation_digest(self) -> str:
        return digest(self)
