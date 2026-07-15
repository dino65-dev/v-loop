"""Tamper-evident evidence ledger."""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, Protocol

from .canonical import canonical_json, digest


@dataclass(frozen=True, slots=True)
class LedgerAnchorRecord:
    """One immutable evidence-ledger head eligible for external anchoring."""

    event_hash: str
    sequence: int
    occurred_at: datetime
    attempts: int


class LedgerAnchor(Protocol):
    """Externally operated append-only anchor service."""

    name: str

    def anchor(self, record: LedgerAnchorRecord) -> None: ...


class LedgerAnchorWorker:
    """Durably publishes evidence heads; replay must be idempotent by hash."""

    def __init__(self, ledger: "EvidenceLedger", anchor: LedgerAnchor, *, worker_id: str) -> None:
        if not worker_id.strip():
            raise ValueError("ledger anchor worker needs a stable worker id")
        self.ledger = ledger
        self.anchor = anchor
        self.worker_id = worker_id

    def drain(self, *, limit: int = 20) -> int:
        delivered = 0
        for record in self.ledger.claim_anchor_operations(worker_id=self.worker_id, limit=limit):
            try:
                self.anchor.anchor(record)
                self.ledger.complete_anchor_operation(record.event_hash, worker_id=self.worker_id)
                delivered += 1
            except Exception as exc:
                self.ledger.release_anchor_operation(record.event_hash, worker_id=self.worker_id, error=exc)
        return delivered


class EvidenceLedger:
    """Append-only SQLite table with a per-row hash chain.

    Production deployments must keep this database/object store inaccessible to
    executors and periodically anchor the head hash outside the service.
    """

    def __init__(self, database: str | Path) -> None:
        self.path = Path(database)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(self.path, isolation_level=None, timeout=5.0)
        self._connection.execute("PRAGMA busy_timeout=5000")
        # Setting journal mode itself needs an exclusive lock. Concurrent
        # first-openers retry that one-time transition rather than failing a
        # valid writer before the ledger schema is even initialized.
        for attempt in range(50):
            try:
                self._connection.execute("PRAGMA journal_mode=WAL")
                break
            except sqlite3.OperationalError as exc:
                if "locked" not in str(exc).lower() or attempt == 49:
                    self._connection.close()
                    raise
                time.sleep(0.01 * (attempt + 1))
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS ledger_events (
                sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                event_type TEXT NOT NULL,
                occurred_at TEXT NOT NULL,
                payload TEXT NOT NULL,
                parent_hash TEXT NOT NULL,
                event_hash TEXT NOT NULL UNIQUE
            )
            """
        )
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS ledger_anchor_outbox (
                event_hash TEXT PRIMARY KEY,
                sequence INTEGER NOT NULL,
                occurred_at TEXT NOT NULL,
                state TEXT NOT NULL,
                attempts INTEGER NOT NULL DEFAULT 0,
                lease_owner TEXT,
                lease_expires_at TEXT,
                last_error TEXT,
                updated_at TEXT NOT NULL
            )
            """
        )
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS ledger_head (
                singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
                event_hash TEXT NOT NULL
            )
            """
        )
        latest = self._connection.execute(
            "SELECT event_hash FROM ledger_events ORDER BY sequence DESC LIMIT 1"
        ).fetchone()
        self._connection.execute(
            "INSERT OR IGNORE INTO ledger_head (singleton, event_hash) VALUES (1, ?)",
            (latest[0] if latest else "0" * 64,),
        )

    def append(self, event_type: str, payload: Mapping[str, Any]) -> str:
        occurred_at = datetime.now(UTC).isoformat()
        serialized = canonical_json(payload)
        self._connection.execute("BEGIN IMMEDIATE")
        try:
            head = self._connection.execute(
                "SELECT event_hash FROM ledger_head WHERE singleton = 1"
            ).fetchone()
            if head is None:  # pragma: no cover - schema invariant
                raise RuntimeError("ledger head is missing")
            parent_hash = head[0]
            event_hash = digest(
                {
                    "event_type": event_type,
                    "occurred_at": occurred_at,
                    "payload": json.loads(serialized),
                    "parent_hash": parent_hash,
                }
            )
            cursor = self._connection.execute(
                """
                INSERT INTO ledger_events
                    (event_type, occurred_at, payload, parent_hash, event_hash)
                VALUES (?, ?, ?, ?, ?)
                """,
                (event_type, occurred_at, serialized, parent_hash, event_hash),
            )
            sequence = cursor.lastrowid
            cursor = self._connection.execute(
                "UPDATE ledger_head SET event_hash = ? WHERE singleton = 1 AND event_hash = ?",
                (event_hash, parent_hash),
            )
            if cursor.rowcount != 1:  # pragma: no cover - BEGIN IMMEDIATE serializes writers
                raise RuntimeError("ledger head changed during append")
            self._connection.execute(
                """
                INSERT OR IGNORE INTO ledger_anchor_outbox
                (event_hash, sequence, occurred_at, state, attempts, lease_owner,
                 lease_expires_at, last_error, updated_at)
                VALUES (?, ?, ?, 'pending', 0, NULL, NULL, NULL, ?)
                """,
                (event_hash, sequence, occurred_at, occurred_at),
            )
            self._connection.execute("COMMIT")
        except Exception:
            self._connection.execute("ROLLBACK")
            raise
        return event_hash

    def claim_anchor_operations(
        self,
        *,
        worker_id: str,
        limit: int = 20,
        lease_seconds: int = 120,
    ) -> list["LedgerAnchorRecord"]:
        if not worker_id.strip() or limit < 1 or lease_seconds < 1:
            raise ValueError("invalid ledger anchor lease request")
        now = datetime.now(UTC)
        self._connection.execute("BEGIN IMMEDIATE")
        try:
            rows = self._connection.execute(
                """
                SELECT event_hash, sequence, occurred_at, attempts
                FROM ledger_anchor_outbox
                WHERE state = 'pending' OR (state = 'leased' AND lease_expires_at <= ?)
                ORDER BY sequence LIMIT ?
                """,
                (now.isoformat(), limit),
            ).fetchall()
            expires_at = datetime.fromtimestamp(now.timestamp() + lease_seconds, UTC).isoformat()
            for event_hash, _sequence, _occurred_at, _attempts in rows:
                self._connection.execute(
                    """
                    UPDATE ledger_anchor_outbox
                    SET state = 'leased', attempts = attempts + 1, lease_owner = ?,
                        lease_expires_at = ?, updated_at = ?
                    WHERE event_hash = ?
                    """,
                    (worker_id, expires_at, now.isoformat(), event_hash),
                )
            self._connection.execute("COMMIT")
        except Exception:
            self._connection.execute("ROLLBACK")
            raise
        return [
            LedgerAnchorRecord(event_hash, int(sequence), datetime.fromisoformat(occurred_at), int(attempts) + 1)
            for event_hash, sequence, occurred_at, attempts in rows
        ]

    def complete_anchor_operation(self, event_hash: str, *, worker_id: str) -> None:
        with self._connection:
            cursor = self._connection.execute(
                """
                UPDATE ledger_anchor_outbox
                SET state = 'delivered', lease_owner = NULL, lease_expires_at = NULL,
                    last_error = NULL, updated_at = ?
                WHERE event_hash = ? AND state = 'leased' AND lease_owner = ?
                """,
                (datetime.now(UTC).isoformat(), event_hash, worker_id),
            )
        if cursor.rowcount != 1:
            raise RuntimeError("ledger anchor outbox lease was lost")

    def release_anchor_operation(self, event_hash: str, *, worker_id: str, error: Exception) -> None:
        with self._connection:
            self._connection.execute(
                """
                UPDATE ledger_anchor_outbox
                SET state = 'pending', lease_owner = NULL, lease_expires_at = NULL,
                    last_error = ?, updated_at = ?
                WHERE event_hash = ? AND state = 'leased' AND lease_owner = ?
                """,
                (type(error).__name__, datetime.now(UTC).isoformat(), event_hash, worker_id),
            )

    def events(self) -> list[dict[str, Any]]:
        rows = self._connection.execute(
            """
            SELECT sequence, event_type, occurred_at, payload, parent_hash, event_hash
            FROM ledger_events ORDER BY sequence
            """
        ).fetchall()
        return [
            {
                "sequence": row[0],
                "event_type": row[1],
                "occurred_at": row[2],
                "payload": json.loads(row[3]),
                "parent_hash": row[4],
                "event_hash": row[5],
            }
            for row in rows
        ]

    def contains_event_hashes(self, event_hashes: set[str] | frozenset[str]) -> bool:
        """Return whether every supplied hash is present in this ledger.

        This is deliberately a membership check rather than a lookup that
        returns event payloads.  Memory promotion needs to attest references
        without exposing all historic evidence to a caller.
        """

        if not event_hashes:
            return False
        placeholders = ",".join("?" for _ in event_hashes)
        row = self._connection.execute(
            f"SELECT COUNT(*) FROM ledger_events WHERE event_hash IN ({placeholders})",
            tuple(event_hashes),
        ).fetchone()
        return bool(row and row[0] == len(event_hashes))

    def events_for_hashes(self, event_hashes: set[str] | frozenset[str]) -> dict[str, dict[str, Any]]:
        """Return exact immutable events for a small attestation set.

        This is intentionally for trusted committers such as ``MemoryLedger``;
        callers still receive only events they already cite by hash.
        """

        if not event_hashes:
            return {}
        placeholders = ",".join("?" for _ in event_hashes)
        rows = self._connection.execute(
            f"""
            SELECT sequence, event_type, occurred_at, payload, parent_hash, event_hash
            FROM ledger_events WHERE event_hash IN ({placeholders})
            """,
            tuple(event_hashes),
        ).fetchall()
        return {
            row[5]: {
                "sequence": row[0],
                "event_type": row[1],
                "occurred_at": row[2],
                "payload": json.loads(row[3]),
                "parent_hash": row[4],
                "event_hash": row[5],
            }
            for row in rows
        }

    def event_hashes_for_run(self, run_id: str) -> tuple[str, ...]:
        """Return every immutable event explicitly bound to one controller run."""

        if not run_id.strip():
            raise ValueError("run id is required")
        return tuple(
            event["event_hash"]
            for event in self.events()
            if event["payload"].get("run_id") == run_id
        )

    def verify_chain(self) -> bool:
        parent_hash = "0" * 64
        for event in self.events():
            if event["parent_hash"] != parent_hash:
                return False
            expected = digest(
                {
                    "event_type": event["event_type"],
                    "occurred_at": event["occurred_at"],
                    "payload": event["payload"],
                    "parent_hash": parent_hash,
                }
            )
            if event["event_hash"] != expected:
                return False
            parent_hash = event["event_hash"]
        head = self._connection.execute(
            "SELECT event_hash FROM ledger_head WHERE singleton = 1"
        ).fetchone()
        return bool(head and head[0] == parent_hash)

    def close(self) -> None:
        self._connection.close()
