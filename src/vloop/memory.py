"""Verified memory ledger, working state, and selective retrieval.

Memory is evidence-backed data, not an instruction channel. Retrieval results
must be treated as hypotheses and never expand a task contract or capability.
"""

from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Iterable, Mapping, Protocol
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen
from uuid import uuid4

from .canonical import canonical_json, digest
from .ledger import EvidenceLedger
from .models import CheckResult, CheckStatus, TaskContract, VerificationReport


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
    claim_kind: str = "operational-procedure"


@dataclass(frozen=True, slots=True)
class VerifiedMemory:
    candidate: MemoryCandidate
    source_run_id: str
    promoted_at: datetime
    status: str = "verified"


class MemoryWriteGate:
    """Only independently accepted runs can promote reusable experience."""

    def __init__(self, claim_authority: "MemoryClaimAuthority | None" = None) -> None:
        self.claim_authority = claim_authority

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
        if not candidate.claim_kind.strip():
            raise ValueError("memory claim kind is required")
        if candidate.expires_at is not None and candidate.expires_at.tzinfo is None:
            raise ValueError("memory expiry must be timezone-aware")
        if not report.accepted:
            raise PermissionError("unverified runs cannot enter reusable memory")
        if self.claim_authority is not None:
            self.claim_authority.validate(candidate)
        if not candidate.memory_id:
            candidate = replace(candidate, memory_id=str(uuid4()))
        return VerifiedMemory(candidate, source_run_id, now or datetime.now(UTC))


class DiagnosedFailureMemoryGate:
    """Admits reusable failure knowledge only for hard-diagnosed failures.

    A generic model reflection is not sufficient.  The failure must be visible
    in a deterministic verification report and retain references to the
    immutable evidence that established it.
    """

    def promote(
        self,
        candidate: MemoryCandidate,
        report: VerificationReport,
        *,
        source_run_id: str,
        now: datetime | None = None,
    ) -> VerifiedMemory:
        if report.correctness is not CheckStatus.FAIL and report.policy is not CheckStatus.FAIL:
            raise PermissionError("only hard-diagnosed failures can enter failure memory")
        # Reuse the structural validation in the success gate without allowing
        # a failed report to masquerade as an accepted one.
        accepted = VerificationReport(
            CheckStatus.PASS,
            CheckStatus.PASS,
            CheckStatus.PASS,
            CheckStatus.PASS,
            report.checks,
        )
        verified = MemoryWriteGate().promote(
            candidate,
            accepted,
            source_run_id=source_run_id,
            now=now,
        )
        return replace(verified, status="diagnosed-failure")


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
    claim_kind: str = "operational-procedure"

    def is_live(self, now: datetime | None = None) -> bool:
        current = now or datetime.now(UTC)
        return self.status in {"verified", "diagnosed-failure"} and (
            self.expires_at is None or self.expires_at > current
        )


class MemoryIndexProjection(Protocol):
    """A rebuildable external memory projection with idempotent upserts."""

    name: str

    def upsert(self, record: MemoryRecord) -> None: ...


@dataclass(frozen=True, slots=True)
class IndexOutboxItem:
    memory_id: str
    record: MemoryRecord
    attempts: int


@dataclass(frozen=True, slots=True)
class MemoryClaimRule:
    """Server-owned schema for a reusable memory claim category."""

    kind: str
    permitted_scopes: frozenset[str]
    required_conditions: frozenset[str] = frozenset()
    maximum_claim_characters: int = 2_000

    def __post_init__(self) -> None:
        if not self.kind.strip() or not self.permitted_scopes or self.maximum_claim_characters < 1:
            raise ValueError("memory claim rules need a kind, scopes, and positive size limit")


class MemoryClaimAuthority:
    """Validates memory semantics from a reviewed schema, not model preference."""

    def __init__(self, rules: Iterable[MemoryClaimRule]) -> None:
        self._rules = {rule.kind: rule for rule in rules}
        if not self._rules:
            raise ValueError("memory claim authority needs at least one rule")

    def validate(self, candidate: MemoryCandidate) -> None:
        rule = self._rules.get(candidate.claim_kind)
        if rule is None:
            raise PermissionError("memory claim kind is not server-authorized")
        if candidate.scope not in rule.permitted_scopes:
            raise PermissionError("memory claim scope is not authorized for this kind")
        if len(candidate.claim) > rule.maximum_claim_characters:
            raise PermissionError("memory claim exceeds its server-owned size limit")
        if not rule.required_conditions.issubset(candidate.conditions):
            raise PermissionError("memory claim omits required applicability conditions")


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


class HTTPJSONClient:
    """Small authenticated JSON client for deployment-owned index services."""

    def __init__(
        self,
        base_url: str,
        *,
        bearer_token: str | None = None,
        timeout_seconds: float = 10.0,
        allow_insecure_loopback: bool = False,
    ) -> None:
        parsed = urlparse(base_url)
        loopback = parsed.hostname in {"127.0.0.1", "::1", "localhost"}
        if parsed.scheme not in {"https", "http"} or not parsed.netloc:
            raise ValueError("index service URL must be absolute HTTP(S)")
        if parsed.scheme != "https" and not (allow_insecure_loopback and loopback):
            raise ValueError("index service must use HTTPS outside an explicit loopback test")
        if timeout_seconds <= 0:
            raise ValueError("index service timeout must be positive")
        self.base_url = base_url.rstrip("/")
        self.bearer_token = bearer_token
        self.timeout_seconds = timeout_seconds

    def post(self, path: str, payload: Mapping[str, Any]) -> Mapping[str, Any] | list[Any] | str:
        if not path.startswith("/"):
            raise ValueError("index service path must be absolute")
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if self.bearer_token:
            headers["Authorization"] = f"Bearer {self.bearer_token}"
        request = Request(
            self.base_url + path,
            data=canonical_json(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:  # nosec B310: validated scheme
                body = response.read().decode("utf-8")
        except (HTTPError, URLError, TimeoutError) as exc:
            raise RuntimeError("memory index service request failed") from exc
        try:
            return json.loads(body) if body else {}
        except json.JSONDecodeError:
            return body


_VLOOP_MEMORY_ID = re.compile(r"\[vloop-memory-id:([0-9a-f-]{8,64})\]")


def _projection_document(record: MemoryRecord) -> str:
    """Projection payload; external systems receive no authority-bearing data."""

    conditions = "\n".join(f"{key}={value}" for key, value in sorted(record.conditions.items()))
    return (
        f"[vloop-memory-id:{record.memory_id}]\n"
        f"[scope:{record.scope}]\n"
        f"[ledger-event:{record.ledger_event_hash}]\n"
        f"{record.claim}\n"
        f"Applicability:\n{conditions}"
    )


def _extract_memory_ids(value: Any) -> list[str]:
    """Treat an external result as opaque text and recover only V-Loop IDs."""

    material: list[str] = []
    if isinstance(value, str):
        material.append(value)
    elif isinstance(value, Mapping):
        for item in value.values():
            material.extend(_extract_memory_ids(item))
        return material
    elif isinstance(value, (list, tuple)):
        for item in value:
            material.extend(_extract_memory_ids(item))
        return material
    return [match.group(1) for text in material for match in _VLOOP_MEMORY_ID.finditer(text)]


class LightRAGIndex(ExternalMemoryIndex):
    """Concrete LightRAG REST projection and ID-only retrieval adapter.

    LightRAG's supported server interface accepts ``/insert`` and ``/query``.
    The response is never trusted as a claim: it is parsed only for the V-Loop
    memory identifier embedded in an inserted projection document, then the
    canonical memory ledger rehydrates and reauthorizes it.
    """

    def __init__(self, client: HTTPJSONClient, *, name: str = "lightrag") -> None:
        self.client = client
        super().__init__(name, self._search_ids)

    def upsert(self, record: MemoryRecord) -> None:
        self.client.post("/insert", {"text": _projection_document(record)})

    def _search_ids(self, query: MemoryQuery) -> list[tuple[str, float]]:
        response = self.client.post(
            "/query",
            {"query": query.text, "mode": "hybrid", "only_need_context": True},
        )
        return [
            (memory_id, 1.0 / rank)
            for rank, memory_id in enumerate(dict.fromkeys(_extract_memory_ids(response)), start=1)
        ]


class HippoRAGIndex(ExternalMemoryIndex):
    """Concrete adapter for the HippoRAG Python API supplied by deployment.

    HippoRAG is embedded rather than given V-Loop database authority. The
    deployment passes its document-ingest and retrieval callables (typically
    wrappers around ``index`` and ``retrieve``); this adapter stores only
    projection documents and returns only V-Loop memory IDs.
    """

    def __init__(self, index_documents, retrieve_documents, *, name: str = "hipporag") -> None:
        if not callable(index_documents) or not callable(retrieve_documents):
            raise ValueError("HippoRAG adapter needs callable index and retrieval operations")
        self._index_documents = index_documents
        self._retrieve_documents = retrieve_documents
        super().__init__(name, self._search_ids)

    def upsert(self, record: MemoryRecord) -> None:
        self._index_documents([_projection_document(record)])

    def _search_ids(self, query: MemoryQuery) -> list[tuple[str, float]]:
        raw = self._retrieve_documents(queries=[query.text], num_to_retrieve=query.limit)
        return [
            (memory_id, 1.0 / rank)
            for rank, memory_id in enumerate(dict.fromkeys(_extract_memory_ids(raw)), start=1)
        ]


class MemoryRouter:
    """Selects hot retrieval first and broad associative retrieval only when needed."""

    def select(self, query: MemoryQuery, *, associative_available: bool) -> tuple[str, ...]:
        if associative_available and (query.historical or query.associative) and not query.latency_sensitive:
            return ("hot", "associative")
        return ("hot",)


class MemoryLedger:
    """Canonical verified-memory store; indexes are rebuildable projections."""

    def __init__(
        self,
        database: str | Path,
        evidence_ledger: EvidenceLedger,
        *,
        projection_sensitivities: frozenset[str] = frozenset({"public", "internal"}),
        claim_authority: MemoryClaimAuthority | None = None,
    ) -> None:
        if not projection_sensitivities.issubset({"public", "internal", "restricted"}):
            raise ValueError("unknown projection sensitivity")
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
                supersedes TEXT,
                claim_kind TEXT NOT NULL DEFAULT 'operational-procedure'
            )
            """
        )
        columns = {
            row[1] for row in self._connection.execute("PRAGMA table_info(memory_records)").fetchall()
        }
        if "claim_kind" not in columns:
            self._connection.execute(
                "ALTER TABLE memory_records ADD COLUMN claim_kind TEXT NOT NULL DEFAULT 'operational-procedure'"
            )
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS memory_index_outbox (
                memory_id TEXT PRIMARY KEY,
                record_json TEXT NOT NULL,
                state TEXT NOT NULL,
                attempts INTEGER NOT NULL DEFAULT 0,
                lease_owner TEXT,
                lease_expires_at TEXT,
                last_error TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS memory_index_deliveries (
                memory_id TEXT NOT NULL,
                projection_name TEXT NOT NULL,
                state TEXT NOT NULL,
                attempts INTEGER NOT NULL DEFAULT 0,
                lease_owner TEXT,
                lease_expires_at TEXT,
                last_error TEXT,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (memory_id, projection_name)
            )
            """
        )
        self._connection.commit()
        self._evidence_ledger = evidence_ledger
        self._projection_sensitivities = projection_sensitivities
        self._claim_authority = claim_authority

    @property
    def claim_authority(self) -> MemoryClaimAuthority | None:
        return self._claim_authority

    def insert(self, verified: VerifiedMemory, *, supersedes: str | None = None) -> MemoryRecord:
        self._validate_attested_promotion(verified)
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
                "claim_kind": candidate.claim_kind,
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
            claim_kind=candidate.claim_kind,
        )
        with self._connection:
            self._connection.execute(
                """
                INSERT INTO memory_records (
                    memory_id, scope, claim, conditions_json, evidence_refs_json,
                    confidence, sensitivity, source_run_id, promoted_at,
                    ledger_event_hash, expires_at, status, supersedes, claim_kind
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                    record.claim_kind,
                ),
            )
            if supersedes:
                self._connection.execute(
                    "UPDATE memory_records SET status = 'superseded' WHERE memory_id = ?",
                    (supersedes,),
                )
            # The canonical record and its projection request commit together.
            # Delivery is at-least-once; adapters must upsert by memory_id.
            if record.sensitivity in self._projection_sensitivities:
                now = datetime.now(UTC).isoformat()
                self._connection.execute(
                    """
                    INSERT INTO memory_index_outbox
                    (memory_id, record_json, state, attempts, lease_owner,
                     lease_expires_at, last_error, created_at, updated_at)
                    VALUES (?, ?, 'pending', 0, NULL, NULL, NULL, ?, ?)
                    ON CONFLICT(memory_id) DO UPDATE SET
                        record_json = excluded.record_json, state = 'pending',
                        lease_owner = NULL, lease_expires_at = NULL,
                        updated_at = excluded.updated_at
                    """,
                    (record.memory_id, canonical_json(_encode_memory_record(record)), now, now),
                )
        return record

    def _validate_attested_promotion(self, verified: VerifiedMemory) -> None:
        """Defend the public insert API against forged ``VerifiedMemory`` values."""

        candidate = verified.candidate
        if self._claim_authority is not None:
            self._claim_authority.validate(candidate)
        if verified.status not in {"verified", "diagnosed-failure"}:
            raise PermissionError("memory status is not eligible for canonical storage")
        if not candidate.memory_id or not candidate.evidence_refs:
            raise PermissionError("memory must have an assigned id and evidence references")
        if not self._evidence_ledger.verify_chain():
            raise PermissionError("evidence ledger integrity check failed")
        events = self._evidence_ledger.events_for_hashes(set(candidate.evidence_refs))
        if len(events) != len(set(candidate.evidence_refs)):
            raise PermissionError("memory cites unknown evidence")
        if any(event["payload"].get("run_id") != verified.source_run_id for event in events.values()):
            raise PermissionError("memory evidence belongs to another run")
        if verified.status == "verified":
            accepted_final = any(
                event["event_type"] == "final-goal.completed" and event["payload"].get("status") == "pass"
                for event in events.values()
            )
            if not accepted_final:
                raise PermissionError("verified memory requires a cited passing final-goal event")
        else:
            diagnosed = any(
                event["event_type"] == "verification.completed"
                and (
                    event["payload"].get("correctness") == "fail"
                    or event["payload"].get("policy") == "fail"
                )
                for event in events.values()
            )
            if not diagnosed:
                raise PermissionError("failure memory requires a cited hard diagnosis")

    def records(self, query: MemoryQuery) -> list[MemoryRecord]:
        allowed_scopes = {query.scope, *query.include_scopes}
        placeholders = ",".join("?" for _ in allowed_scopes)
        rows = self._connection.execute(
            f"""
            SELECT memory_id, scope, claim, conditions_json, evidence_refs_json,
                   confidence, sensitivity, source_run_id, promoted_at,
                   ledger_event_hash, expires_at, status, supersedes, claim_kind
            FROM memory_records
            WHERE scope IN ({placeholders}) AND sensitivity IN ({",".join("?" for _ in query.allowed_sensitivities)})
            """,
            (*sorted(allowed_scopes), *query.allowed_sensitivities),
        ).fetchall()
        records = [self._record_from_row(row) for row in rows]
        return [record for record in records if record.is_live()]

    @staticmethod
    def _record_from_row(row) -> MemoryRecord:
        return MemoryRecord(
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
            claim_kind=row[13],
        )

    def claim_index_operations(
        self,
        *,
        projection_name: str,
        worker_id: str,
        limit: int = 20,
        lease_duration: timedelta = timedelta(minutes=2),
    ) -> list[IndexOutboxItem]:
        """Lease projection work without giving an index write authority."""

        if not projection_name.strip() or not worker_id.strip() or limit < 1 or lease_duration <= timedelta(0):
            raise ValueError("invalid projection outbox lease request")
        now = datetime.now(UTC)
        self._connection.execute("BEGIN IMMEDIATE")
        try:
            self._connection.execute(
                """
                INSERT OR IGNORE INTO memory_index_deliveries
                (memory_id, projection_name, state, attempts, lease_owner,
                 lease_expires_at, last_error, updated_at)
                SELECT memory_id, ?, 'pending', 0, NULL, NULL, NULL, ?
                FROM memory_index_outbox
                """,
                (projection_name, now.isoformat()),
            )
            rows = self._connection.execute(
                """
                SELECT outbox.memory_id, outbox.record_json, deliveries.attempts
                FROM memory_index_outbox AS outbox
                JOIN memory_index_deliveries AS deliveries
                  ON deliveries.memory_id = outbox.memory_id
                WHERE deliveries.projection_name = ?
                  AND (deliveries.state = 'pending'
                    OR (deliveries.state = 'leased' AND deliveries.lease_expires_at <= ?))
                ORDER BY outbox.created_at, outbox.memory_id LIMIT ?
                """,
                (projection_name, now.isoformat(), limit),
            ).fetchall()
            expires = (now + lease_duration).isoformat()
            for memory_id, _record, _attempts in rows:
                self._connection.execute(
                    """
                    UPDATE memory_index_deliveries
                    SET state = 'leased', attempts = attempts + 1, lease_owner = ?,
                        lease_expires_at = ?, updated_at = ?
                    WHERE memory_id = ? AND projection_name = ?
                    """,
                    (worker_id, expires, now.isoformat(), memory_id, projection_name),
                )
            self._connection.execute("COMMIT")
        except Exception:
            self._connection.execute("ROLLBACK")
            raise
        return [
            IndexOutboxItem(memory_id, _decode_memory_record(json.loads(record)), int(attempts) + 1)
            for memory_id, record, attempts in rows
        ]

    def complete_index_operation(self, memory_id: str, *, projection_name: str, worker_id: str) -> None:
        with self._connection:
            cursor = self._connection.execute(
                """
                UPDATE memory_index_deliveries
                SET state = 'delivered', lease_owner = NULL, lease_expires_at = NULL,
                    last_error = NULL, updated_at = ?
                WHERE memory_id = ? AND projection_name = ? AND state = 'leased' AND lease_owner = ?
                """,
                (datetime.now(UTC).isoformat(), memory_id, projection_name, worker_id),
            )
        if cursor.rowcount != 1:
            raise RuntimeError("memory index outbox lease was lost")

    def release_index_operation(
        self, memory_id: str, *, projection_name: str, worker_id: str, error: Exception
    ) -> None:
        with self._connection:
            self._connection.execute(
                """
                UPDATE memory_index_deliveries
                SET state = 'pending', lease_owner = NULL, lease_expires_at = NULL,
                    last_error = ?, updated_at = ?
                WHERE memory_id = ? AND projection_name = ? AND state = 'leased' AND lease_owner = ?
                """,
                (
                    type(error).__name__,
                    datetime.now(UTC).isoformat(),
                    memory_id,
                    projection_name,
                    worker_id,
                ),
            )

    def close(self) -> None:
        self._connection.close()


def _encode_memory_record(record: MemoryRecord) -> dict[str, Any]:
    return {
        "memory_id": record.memory_id,
        "scope": record.scope,
        "claim": record.claim,
        "conditions": dict(record.conditions),
        "evidence_refs": list(record.evidence_refs),
        "confidence": record.confidence,
        "sensitivity": record.sensitivity,
        "source_run_id": record.source_run_id,
        "promoted_at": record.promoted_at.isoformat(),
        "ledger_event_hash": record.ledger_event_hash,
        "expires_at": record.expires_at.isoformat() if record.expires_at else None,
        "status": record.status,
        "supersedes": record.supersedes,
        "claim_kind": record.claim_kind,
    }


def _decode_memory_record(value: Mapping[str, Any]) -> MemoryRecord:
    return MemoryRecord(
        memory_id=str(value["memory_id"]),
        scope=str(value["scope"]),
        claim=str(value["claim"]),
        conditions=dict(value["conditions"]),
        evidence_refs=tuple(value["evidence_refs"]),
        confidence=float(value["confidence"]),
        sensitivity=str(value["sensitivity"]),
        source_run_id=str(value["source_run_id"]),
        promoted_at=datetime.fromisoformat(value["promoted_at"]),
        ledger_event_hash=str(value["ledger_event_hash"]),
        expires_at=datetime.fromisoformat(value["expires_at"]) if value.get("expires_at") else None,
        status=str(value["status"]),
        supersedes=value.get("supersedes"),
        claim_kind=str(value.get("claim_kind", "operational-procedure")),
    )


class MemoryProjectionWorker:
    """At-least-once, durable delivery from the memory ledger to one index."""

    def __init__(self, ledger: MemoryLedger, projection: MemoryIndexProjection, *, worker_id: str) -> None:
        if not worker_id.strip():
            raise ValueError("projection worker needs a stable worker id")
        self.ledger = ledger
        self.projection = projection
        self.worker_id = worker_id

    def drain(self, *, limit: int = 20) -> int:
        delivered = 0
        for item in self.ledger.claim_index_operations(
            projection_name=self.projection.name, worker_id=self.worker_id, limit=limit
        ):
            try:
                self.projection.upsert(item.record)
                self.ledger.complete_index_operation(
                    item.memory_id, projection_name=self.projection.name, worker_id=self.worker_id
                )
                delivered += 1
            except Exception as exc:
                self.ledger.release_index_operation(
                    item.memory_id,
                    projection_name=self.projection.name,
                    worker_id=self.worker_id,
                    error=exc,
                )
        return delivered


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
        rrf_k: int = 60,
    ) -> None:
        if not authorized_scopes:
            raise ValueError("at least one authorized memory scope is required")
        if not allowed_sensitivities:
            raise ValueError("at least one allowed memory sensitivity is required")
        if rrf_k < 1:
            raise ValueError("rrf_k must be positive")
        self.ledger = ledger
        self.authorized_scopes = authorized_scopes
        self.allowed_sensitivities = allowed_sensitivities
        self.hot_index = hot_index or HotLexicalIndex()
        self.associative_index = associative_index
        self.router = router or MemoryRouter()
        self.rrf_k = rrf_k

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
        # External retrieval scores are not calibrated against the local hot
        # index. Reciprocal-rank fusion combines rank evidence without treating
        # one backend's arbitrary score scale as more authoritative.
        per_source: dict[str, dict[str, RetrievalResult]] = {}
        for result in results:
            source_results = per_source.setdefault(result.source, {})
            existing = source_results.get(result.record.memory_id)
            if existing is None or result.score > existing.score:
                source_results[result.record.memory_id] = result
        fused: dict[str, tuple[MemoryRecord, float, list[str]]] = {}
        for source, source_results in per_source.items():
            ranked = sorted(
                source_results.values(),
                key=lambda result: (-result.score, -result.record.promoted_at.timestamp()),
            )
            for rank, result in enumerate(ranked, start=1):
                record, score, sources = fused.get(result.record.memory_id, (result.record, 0.0, []))
                fused[result.record.memory_id] = (record, score + 1.0 / (self.rrf_k + rank), [*sources, source])
        unique = {
            memory_id: RetrievalResult(record, score, "rrf:" + "+".join(sorted(sources)))
            for memory_id, (record, score, sources) in fused.items()
        }
        return sorted(
            unique.values(),
            key=lambda result: (-result.score, -result.record.promoted_at.timestamp()),
        )[: query.limit]


class MemoryCandidateProducer(Protocol):
    """Server-owned extractor for a bounded, evidence-referenced lesson."""

    def propose(
        self,
        *,
        contract: TaskContract,
        history: tuple[dict, ...],
        report: VerificationReport,
        final_check: CheckResult,
        available_evidence_refs: tuple[str, ...],
    ) -> MemoryCandidate | None: ...


class VerifiedMemoryCommitter:
    """Commits a candidate only after final-goal verification and attestation.

    This component is intentionally separate from the planner.  It verifies
    that every cited event hash exists in the evidence ledger before the
    canonical memory ledger receives the record.
    """

    def __init__(
        self,
        memory_ledger: MemoryLedger,
        evidence_ledger: EvidenceLedger,
        write_gate: MemoryWriteGate | None = None,
    ) -> None:
        self.memory_ledger = memory_ledger
        self.evidence_ledger = evidence_ledger
        self.write_gate = write_gate or MemoryWriteGate()

    def commit(
        self,
        candidate: MemoryCandidate,
        *,
        report: VerificationReport,
        final_check: CheckResult,
        source_run_id: str,
        available_evidence_refs: tuple[str, ...],
    ) -> MemoryRecord:
        if final_check.status is not CheckStatus.PASS:
            raise PermissionError("final-goal verification is required for reusable memory")
        allowed = set(available_evidence_refs)
        cited = set(candidate.evidence_refs)
        if not cited.issubset(allowed):
            raise PermissionError("memory cites evidence outside this completed run")
        if not self.evidence_ledger.contains_event_hashes(cited):
            raise PermissionError("memory cites unknown evidence")
        verified = self.write_gate.promote(candidate, report, source_run_id=source_run_id)
        return self.memory_ledger.insert(verified)
