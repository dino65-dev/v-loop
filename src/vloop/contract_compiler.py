"""Authority-bounded compilation of user intent into immutable task contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from .models import ActionRule, Effect, TaskContract


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
            authority = self._resolve(action)
            rules.append(
                ActionRule(
                    tool=action.tool,
                    effect=action.effect,
                    target_prefix=action.target_prefix,
                    approval_required=authority.approval_required,
                    max_uses=authority.max_uses,
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

    def _resolve(self, action: RequestedAction) -> ToolAuthority:
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
        return matches[0]

    @staticmethod
    def _is_within(requested_prefix: str, permitted_prefix: str) -> bool:
        permitted = permitted_prefix.rstrip("/") or "/"
        requested = requested_prefix.rstrip("/") or "/"
        return permitted == "/" or requested == permitted or requested.startswith(permitted + "/")
