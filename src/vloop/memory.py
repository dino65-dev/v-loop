"""Verified memory ledger, working state, and selective retrieval.

Memory is evidence-backed data, not an instruction channel. Retrieval results
must be treated as hypotheses and never expand a task contract or capability.
"""

from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterable, Mapping, Protocol
from uuid import uuid4

from .canonical import canonical_json, digest
from .ledger import EvidenceLedger
from .models import VerificationReport


_TOKEN = re.compile(r"[a-z0-9_./-]+")


def _tokens(text: str) -> frozenset[str]:
    return frozenset(_TOKEN.findall(text.lower()))


@dataclass(frozen=True, slots=True)
class MemoryCandidate:
    claim: str
    scope: str
    conditions: Mapping[str, str]
    evidence_refs: tuple[str, ...]
    confidence: float
    sensitivity: str
    expires_at: datetime | None = None
    memory_id: str = ""


@dataclass(frozen=True, slots=True)
class VerifiedMemory:
    candidate: MemoryCandidate
    source_run_id: str
    promoted_at: datetime
    status: str = "verified"


class MemoryWriteGate:
    """Only independently accepted runs can promote reusable experience."""

    def promote(
        self,
        candidate: MemoryCandidate,
        report: VerificationReport,
        *,
        source_run_id: str,
        now: datetime | None = None,
    ) -> VerifiedMemory:
        if not candidate.claim.strip():
            raise ValueError("memory claim is required")
        if not candidate.scope.strip():
            raise ValueError("memory scope is required")
        if not candidate.evidence_refs:
            raise ValueError("memory needs evidence references")
        if not 0.0 <= candidate.confidence <= 1.0:
            raise ValueError("memory confidence must be in [0, 1]")
        if candidate.sensitivity not in {"public", "internal", "restricted"}:
            raise ValueError("unknown sensitivity label")
        if candidate.expires_at is not None and candidate.expires_at.tzinfo is None:
            raise ValueError("memory expiry must be timezone-aware")
        if not report.accepted:
            raise PermissionError("unverified runs cannot enter reusable memory")
        if not candidate.memory_id:
            candidate = replace(candidate, memory_id=str(uuid4()))
        return VerifiedMemory(candidate, source_run_id, now or datetime.now(UTC))


@dataclass(frozen=True, slots=True)
class WorkingState:
    """L0 session state. It is never considered reusable memory by itself."""

    task_id: str
    project_scope: str
    current_step: str
    hypotheses: tuple[str, ...] = ()
    observations: tuple[str, ...] = ()
    updated_at: datetime = datetime.min.replace(tzinfo=UTC)


class WorkingStateStore:
    def __init__(self) -> None:
        self._states: dict[str, WorkingState] = {}

    def put(self, state: WorkingState) -> None:
        self._states[state.task_id] = replace(state, updated_at=datetime.now(UTC))

    def get(self, task_id: str) -> WorkingState | None:
        return self._states.get(task_id)

    def clear(self, task_id: str) -> None:
        self._states.pop(task_id, None)


@dataclass(frozen=True, slots=True)
class MemoryRecord:
    memory_id: str
    scope: str
    claim: str
    conditions: Mapping[str, str]
    evidence_refs: tuple[str, ...]
    confidence: float
    sensitivity: str
    source_run_id: str
    promoted_at: datetime
    ledger_event_hash: str
    expires_at: datetime | None = None
    status: str = "verified"
    supersedes: str | None = None

    def is_live(self, now: datetime | None = None) -> bool:
        current = now or datetime.now(UTC)
        return self.status == "verified" and (self.expires_at is None or self.expires_at > current)


@dataclass(frozen=True, slots=True)
class MemoryQuery:
    text: str
    scope: str
    limit: int = 5
    include_scopes: tuple[str, ...] = ()
    allowed_sensitivities: tuple[str, ...] = ("public", "internal")
    historical: bool = False
    associative: bool = False
    latency_sensitive: bool = True

    def __post_init__(self) -> None:
        if not self.text.strip():
            raise ValueError("memory query text is required")
        if self.limit < 1 or self.limit > 50:
            raise ValueError("memory query limit must be in [1, 50]")
        if not self.allowed_sensitivities:
            raise ValueError("memory query needs at least one allowed sensitivity")


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    record: MemoryRecord
    score: float
    source: str


class MemoryIndex(Protocol):
    name: str

    def search(self, query: MemoryQuery, records: Iterable[MemoryRecord]) -> list[RetrievalResult]: ...


class HotLexicalIndex:
    """Dependency-free scoped index used until an external hot RAG is justified."""

    name = "hot-lexical"

    def search(self, query: MemoryQuery, records: Iterable[MemoryRecord]) -> list[RetrievalResult]:
        query_tokens = _tokens(query.text)
        results: list[RetrievalResult] = []
        for record in records:
            material = " ".join(
                [record.claim, *record.conditions.keys(), *record.conditions.values()]
            )
            overlap = len(query_tokens.intersection(_tokens(material)))
            if not overlap:
                continue
            score = overlap / max(1, len(query_tokens))
            score += record.confidence * 0.05
            results.append(RetrievalResult(record, score, self.name))
        return sorted(results, key=lambda result: (-result.score, result.record.promoted_at), reverse=False)


class ExternalMemoryIndex:
    """Adapter for LightRAG or HippoRAG results without giving them write authority.

    The adapter must return V-Loop memory IDs and a numeric score. The ledger
    rehydrates and re-authorizes every returned record before it reaches a
    planner.
    """

    def __init__(self, name: str, search_ids) -> None:
        self.name = name
        self._search_ids = search_ids

    def search(self, query: MemoryQuery, records: Iterable[MemoryRecord]) -> list[RetrievalResult]:
        by_id = {record.memory_id: record for record in records}
        raw = self._search_ids(query)
        results: list[RetrievalResult] = []
        for memory_id, score in raw:
            record = by_id.get(memory_id)
            if record is not None and isinstance(score, (float, int)):
                results.append(RetrievalResult(record, float(score), self.name))
        return results


class MemoryRouter:
    """Selects hot retrieval first and broad associative retrieval only when needed."""

    def select(self, query: MemoryQuery, *, associative_available: bool) -> tuple[str, ...]:
        if associative_available and (query.historical or query.associative) and not query.latency_sensitive:
            return ("hot", "associative")
        if associative_available and (query.historical or query.associative):
            return ("hot", "associative")
        return ("hot",)


class MemoryLedger:
    """Canonical verified-memory store; indexes are rebuildable projections."""

    def __init__(self, database: str | Path, evidence_ledger: EvidenceLedger) -> None:
        self.path = Path(database)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(self.path)
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS memory_records (
                memory_id TEXT PRIMARY KEY,
                scope TEXT NOT NULL,
                claim TEXT NOT NULL,
                conditions_json TEXT NOT NULL,
                evidence_refs_json TEXT NOT NULL,
                confidence REAL NOT NULL,
                sensitivity TEXT NOT NULL,
                source_run_id TEXT NOT NULL,
                promoted_at TEXT NOT NULL,
                ledger_event_hash TEXT NOT NULL,
                expires_at TEXT,
                status TEXT NOT NULL,
                supersedes TEXT
            )
            """
        )
        self._connection.commit()
        self._evidence_ledger = evidence_ledger

    def insert(self, verified: VerifiedMemory, *, supersedes: str | None = None) -> MemoryRecord:
        candidate = verified.candidate
        event_hash = self._evidence_ledger.append(
            "memory.promoted",
            {
                "memory_id": candidate.memory_id,
                "scope": candidate.scope,
                "claim_digest": digest(candidate.claim),
                "evidence_refs": candidate.evidence_refs,
                "source_run_id": verified.source_run_id,
                "confidence": candidate.confidence,
                "sensitivity": candidate.sensitivity,
                "supersedes": supersedes,
            },
        )
        record = MemoryRecord(
            memory_id=candidate.memory_id,
            scope=candidate.scope,
            claim=candidate.claim,
            conditions=dict(candidate.conditions),
            evidence_refs=candidate.evidence_refs,
            confidence=candidate.confidence,
            sensitivity=candidate.sensitivity,
            source_run_id=verified.source_run_id,
            promoted_at=verified.promoted_at,
            ledger_event_hash=event_hash,
            expires_at=candidate.expires_at,
            status=verified.status,
            supersedes=supersedes,
        )
        with self._connection:
            self._connection.execute(
                """
                INSERT INTO memory_records (
                    memory_id, scope, claim, conditions_json, evidence_refs_json,
                    confidence, sensitivity, source_run_id, promoted_at,
                    ledger_event_hash, expires_at, status, supersedes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.memory_id,
                    record.scope,
                    record.claim,
                    canonical_json(record.conditions),
                    canonical_json(record.evidence_refs),
                    record.confidence,
                    record.sensitivity,
                    record.source_run_id,
                    record.promoted_at.isoformat(),
                    record.ledger_event_hash,
                    record.expires_at.isoformat() if record.expires_at else None,
                    record.status,
                    record.supersedes,
                ),
            )
            if supersedes:
                self._connection.execute(
                    "UPDATE memory_records SET status = 'superseded' WHERE memory_id = ?",
                    (supersedes,),
                )
        return record

    def records(self, query: MemoryQuery) -> list[MemoryRecord]:
        allowed_scopes = {query.scope, *query.include_scopes}
        placeholders = ",".join("?" for _ in allowed_scopes)
        rows = self._connection.execute(
            f"""
            SELECT memory_id, scope, claim, conditions_json, evidence_refs_json,
                   confidence, sensitivity, source_run_id, promoted_at,
                   ledger_event_hash, expires_at, status, supersedes
            FROM memory_records
            WHERE scope IN ({placeholders}) AND sensitivity IN ({",".join("?" for _ in query.allowed_sensitivities)})
            """,
            (*sorted(allowed_scopes), *query.allowed_sensitivities),
        ).fetchall()
        records = [
            MemoryRecord(
                memory_id=row[0],
                scope=row[1],
                claim=row[2],
                conditions=json.loads(row[3]),
                evidence_refs=tuple(json.loads(row[4])),
                confidence=float(row[5]),
                sensitivity=row[6],
                source_run_id=row[7],
                promoted_at=datetime.fromisoformat(row[8]),
                ledger_event_hash=row[9],
                expires_at=datetime.fromisoformat(row[10]) if row[10] else None,
                status=row[11],
                supersedes=row[12],
            )
            for row in rows
        ]
        return [record for record in records if record.is_live()]

    def close(self) -> None:
        self._connection.close()


class MemoryService:
    """Retrieval facade with scope, sensitivity, expiry, and conflict controls."""

    def __init__(
        self,
        ledger: MemoryLedger,
        *,
        authorized_scopes: frozenset[str],
        allowed_sensitivities: frozenset[str] = frozenset({"public", "internal"}),
        hot_index: MemoryIndex | None = None,
        associative_index: MemoryIndex | None = None,
        router: MemoryRouter | None = None,
    ) -> None:
        if not authorized_scopes:
            raise ValueError("at least one authorized memory scope is required")
        if not allowed_sensitivities:
            raise ValueError("at least one allowed memory sensitivity is required")
        self.ledger = ledger
        self.authorized_scopes = authorized_scopes
        self.allowed_sensitivities = allowed_sensitivities
        self.hot_index = hot_index or HotLexicalIndex()
        self.associative_index = associative_index
        self.router = router or MemoryRouter()

    def retrieve(self, query: MemoryQuery) -> list[RetrievalResult]:
        if query.scope not in self.authorized_scopes or not set(query.include_scopes).issubset(
            self.authorized_scopes
        ):
            raise PermissionError("memory query requests an unauthorized scope")
        effective_sensitivities = tuple(
            sensitivity
            for sensitivity in query.allowed_sensitivities
            if sensitivity in self.allowed_sensitivities
        )
        if not effective_sensitivities:
            return []
        query = replace(query, allowed_sensitivities=effective_sensitivities)
        records = self.ledger.records(query)
        selected = self.router.select(query, associative_available=self.associative_index is not None)
        results: list[RetrievalResult] = []
        if "hot" in selected:
            results.extend(self.hot_index.search(query, records))
        if "associative" in selected and self.associative_index is not None:
            results.extend(self.associative_index.search(query, records))
        unique: dict[str, RetrievalResult] = {}
        for result in results:
            existing = unique.get(result.record.memory_id)
            if existing is None or result.score > existing.score:
                unique[result.record.memory_id] = result
        return sorted(
            unique.values(),
            key=lambda result: (-result.score, -result.record.promoted_at.timestamp()),
        )[: query.limit]
