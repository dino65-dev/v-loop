"""Provenance-aware context and state packaging for a V-Loop iteration.

The model receives external material as labelled data. This engine keeps task
authority separate from retrieved content and derives conservative provenance
labels for every proposed action.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Mapping

from .canonical import digest
from .memory import RetrievalResult, WorkingState
from .models import Provenance, TaskContract


class ContextTrust(StrEnum):
    USER = "user"
    TRUSTED_SYSTEM = "trusted-system"
    TRUSTED_REPOSITORY = "trusted-repository"
    VERIFIED_MEMORY = "verified-memory"
    UNTRUSTED = "untrusted"


@dataclass(frozen=True, slots=True)
class ContextItem:
    source_id: str
    kind: str
    content: str
    trust: ContextTrust
    metadata: Mapping[str, str] = field(default_factory=dict)
    captured_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    @property
    def content_digest(self) -> str:
        return digest({"source_id": self.source_id, "kind": self.kind, "content": self.content})


@dataclass(frozen=True, slots=True)
class EnvironmentFingerprint:
    """Stable, non-secret facts needed to reproduce an action decision."""

    values: Mapping[str, str]

    @property
    def digest(self) -> str:
        return digest(self.values)


@dataclass(frozen=True, slots=True)
class ContextPackage:
    contract_digest: str
    environment_digest: str
    trusted_items: tuple[ContextItem, ...]
    untrusted_items: tuple[ContextItem, ...]
    working_state: WorkingState | None
    truncated_source_ids: tuple[str, ...]

    @property
    def provenance(self) -> tuple[Provenance, ...]:
        values: set[Provenance] = set()
        for item in (*self.trusted_items, *self.untrusted_items):
            values.add(
                {
                    ContextTrust.USER: Provenance.USER,
                    ContextTrust.TRUSTED_REPOSITORY: Provenance.TRUSTED_REPOSITORY,
                    ContextTrust.VERIFIED_MEMORY: Provenance.VERIFIED_MEMORY,
                    ContextTrust.UNTRUSTED: Provenance.UNTRUSTED_RETRIEVAL,
                    ContextTrust.TRUSTED_SYSTEM: Provenance.TOOL_OUTPUT,
                }[item.trust]
            )
        return tuple(sorted(values, key=lambda value: value.value))


class ContextEngine:
    """Builds bounded, provenance-labelled context without granting authority."""

    def __init__(self, *, maximum_chars: int = 24_000) -> None:
        if maximum_chars < 1:
            raise ValueError("maximum_chars must be positive")
        self.maximum_chars = maximum_chars
        self._items: list[ContextItem] = []

    def add(self, item: ContextItem) -> None:
        if not item.source_id.strip() or not item.kind.strip():
            raise ValueError("context item needs source_id and kind")
        self._items.append(item)

    def add_memory(self, result: RetrievalResult) -> None:
        record = result.record
        self.add(
            ContextItem(
                source_id=record.memory_id,
                kind="verified-memory",
                content=record.claim,
                trust=ContextTrust.VERIFIED_MEMORY,
                metadata={
                    "ledger_event_hash": record.ledger_event_hash,
                    "source": result.source,
                    "score": f"{result.score:.6f}",
                },
            )
        )

    def package(
        self,
        *,
        contract: TaskContract,
        environment: EnvironmentFingerprint,
        working_state: WorkingState | None = None,
    ) -> ContextPackage:
        trusted: list[ContextItem] = []
        untrusted: list[ContextItem] = []
        truncated: list[str] = []
        remaining = self.maximum_chars
        ordered = sorted(
            self._items,
            key=lambda item: (
                item.trust is ContextTrust.UNTRUSTED,
                item.captured_at,
                item.source_id,
            ),
        )
        for item in ordered:
            cost = len(item.content)
            if cost > remaining:
                truncated.append(item.source_id)
                continue
            remaining -= cost
            if item.trust is ContextTrust.UNTRUSTED:
                untrusted.append(item)
            else:
                trusted.append(item)
        return ContextPackage(
            contract_digest=contract.contract_digest,
            environment_digest=environment.digest,
            trusted_items=tuple(trusted),
            untrusted_items=tuple(untrusted),
            working_state=working_state,
            truncated_source_ids=tuple(truncated),
        )
