"""Tamper-evident evidence ledger."""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping

from .canonical import canonical_json, digest


class EvidenceLedger:
    """Append-only SQLite table with a per-row hash chain.

    Production deployments must keep this database/object store inaccessible to
    executors and periodically anchor the head hash outside the service.
    """

    def __init__(self, database: str | Path) -> None:
        self.path = Path(database)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(self.path)
        self._connection.execute("PRAGMA journal_mode=WAL")
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
        self._connection.commit()

    def append(self, event_type: str, payload: Mapping[str, Any]) -> str:
        occurred_at = datetime.now(UTC).isoformat()
        previous = self._connection.execute(
            "SELECT event_hash FROM ledger_events ORDER BY sequence DESC LIMIT 1"
        ).fetchone()
        parent_hash = previous[0] if previous else "0" * 64
        serialized = canonical_json(payload)
        event_hash = digest(
            {
                "event_type": event_type,
                "occurred_at": occurred_at,
                "payload": json.loads(serialized),
                "parent_hash": parent_hash,
            }
        )
        with self._connection:
            self._connection.execute(
                """
                INSERT INTO ledger_events
                    (event_type, occurred_at, payload, parent_hash, event_hash)
                VALUES (?, ?, ?, ?, ?)
                """,
                (event_type, occurred_at, serialized, parent_hash, event_hash),
            )
        return event_hash

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
        return True

    def close(self) -> None:
        self._connection.close()
