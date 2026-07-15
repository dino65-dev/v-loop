"""Authority-bounded compilation of user intent into immutable task contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from .models import ActionRule, ArgumentRule, Effect, TaskContract


@dataclass(frozen=True, slots=True)
class RequestedAction:
    tool: str
    effect: Effect
    target_prefix: str


@dataclass(frozen=True, slots=True)
class ContractRequest:
    goal: str
    success_conditions: tuple[str, ...]
    requested_actions: tuple[RequestedAction, ...]
    maximum_iterations: int = 8
    maximum_tool_calls: int = 32


@dataclass(frozen=True, slots=True)
class ToolAuthority:
    """Server-owned maximum authority for one tool."""

    tool: str
    effect: Effect
    permitted_prefixes: tuple[str, ...]
    approval_required: bool = False
    max_uses: int | None = None
    argument_rules: tuple[ArgumentRule, ...] = ()
    allow_unlisted_arguments: bool = True


class ContractCompilationError(ValueError):
    pass


class TaskContractCompiler:
    """Intersects requested actions with server-owned authority.

    A model may help produce ContractRequest, but cannot alter this catalog,
    remove an approval requirement, or grant a broader target prefix.
    """

    def __init__(self, catalog: tuple[ToolAuthority, ...]) -> None:
        self._catalog = catalog

    def compile(self, request: ContractRequest) -> TaskContract:
        if not request.goal.strip():
            raise ContractCompilationError("goal is required")
        if not request.success_conditions:
            raise ContractCompilationError("at least one success condition is required")
        rules: list[ActionRule] = []
        for action in request.requested_actions:
            authorities = self._resolve_all(action)
            rules.append(
                ActionRule(
                    tool=action.tool,
                    effect=action.effect,
                    target_prefix=action.target_prefix,
                    approval_required=any(authority.approval_required for authority in authorities),
                    max_uses=self._minimum_max_uses(authorities),
                    argument_rules=self._intersect_argument_rules(authorities),
                    allow_unlisted_arguments=all(authority.allow_unlisted_arguments for authority in authorities),
                )
            )
        if not rules:
            raise ContractCompilationError("at least one permitted action is required")
        return TaskContract(
            goal=request.goal,
            success_conditions=request.success_conditions,
            allowed_actions=tuple(rules),
            forbidden_actions=("policy.update", "evaluator.modify", "memory.policy.write"),
            maximum_iterations=request.maximum_iterations,
            maximum_tool_calls=request.maximum_tool_calls,
        )

    def _resolve_all(self, action: RequestedAction) -> tuple[ToolAuthority, ...]:
        matches = [
            authority
            for authority in self._catalog
            if authority.tool == action.tool
            and authority.effect is action.effect
            and any(self._is_within(action.target_prefix, prefix) for prefix in authority.permitted_prefixes)
        ]
        if not matches:
            raise ContractCompilationError(
                f"requested action is outside server authority: {action.tool} {action.effect.value}"
            )
        return tuple(matches)

    @staticmethod
    def _minimum_max_uses(authorities: tuple[ToolAuthority, ...]) -> int | None:
        limits = [authority.max_uses for authority in authorities if authority.max_uses is not None]
        return min(limits) if limits else None

    @staticmethod
    def _intersect_argument_rules(authorities: tuple[ToolAuthority, ...]) -> tuple[ArgumentRule, ...]:
        """Keep every catalog constraint; PolicyGate applies all overlaps."""

        by_name: dict[str, list[ArgumentRule]] = {}
        for authority in authorities:
            for rule in authority.argument_rules:
                by_name.setdefault(rule.name, []).append(rule)
        combined: list[ArgumentRule] = []
        for name, rules in sorted(by_name.items()):
            first = rules[0]
            if any(rule.kind is not first.kind for rule in rules[1:]):
                raise ContractCompilationError(f"conflicting argument kinds for {name!r}")
            allowed_values = ()
            if first.kind.value == "enum":
                allowed = set(first.allowed_values)
                for rule in rules[1:]:
                    allowed.intersection_update(rule.allowed_values)
                if not allowed:
                    raise ContractCompilationError(f"overlapping authority has no common values for {name!r}")
                allowed_values = tuple(sorted(allowed))
            maximum_lengths = [rule.maximum_length for rule in rules if rule.maximum_length is not None]
            minimums = [rule.minimum for rule in rules if rule.minimum is not None]
            maximums = [rule.maximum for rule in rules if rule.maximum is not None]
            combined.append(
                ArgumentRule(
                    name=name,
                    kind=first.kind,
                    required=any(rule.required for rule in rules),
                    allowed_values=allowed_values,
                    maximum_length=min(maximum_lengths) if maximum_lengths else None,
                    minimum=max(minimums) if minimums else None,
                    maximum=min(maximums) if maximums else None,
                )
            )
        return tuple(combined)

    @staticmethod
    def _is_within(requested_prefix: str, permitted_prefix: str) -> bool:
        permitted = permitted_prefix.rstrip("/") or "/"
        requested = requested_prefix.rstrip("/") or "/"
        return permitted == "/" or requested == permitted or requested.startswith(permitted + "/")
