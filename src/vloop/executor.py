"""Executor interfaces and the capability-enforcement point.

Raw executors implement the side effect.  Production controllers invoke them
only through :class:`CapabilityEnforcingExecutor`, which verifies the policy
signature, audience, expiry, exact intent digest, nonce, and idempotency key
at the enforcement point immediately before execution.
"""

from __future__ import annotations

import os
import json
import sqlite3
import subprocess
import threading
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Protocol, Sequence

from .authorization import CapabilityRejected, CapabilityVerifier
from .canonical import digest
from .models import ActionIntent, Capability, ExecutionObservation, PreparedExecution


class Executor(Protocol):
    executor_id: str

    def execute(self, intent: ActionIntent, capability: Capability) -> ExecutionObservation: ...


class RawExecutor(Protocol):
    def execute(self, intent: ActionIntent) -> ExecutionObservation: ...


class IdempotencyStore(Protocol):
    def reserve(self, key: tuple[str, str, str, str]) -> tuple[str, ExecutionObservation | None]: ...

    def complete(self, key: tuple[str, str, str, str], observation: ExecutionObservation) -> None: ...

    def mark_indeterminate(self, key: tuple[str, str, str, str], observation: ExecutionObservation) -> None: ...


class InMemoryIdempotencyStore:
    """Thread-safe test/development idempotency store."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._records: dict[tuple[str, str, str, str], ExecutionObservation | None] = {}
        self._intent_by_operation: dict[tuple[str, str, str], str] = {}
        self._indeterminate: set[tuple[str, str, str, str]] = set()

    def reserve(self, key: tuple[str, str, str, str]) -> tuple[str, ExecutionObservation | None]:
        with self._lock:
            operation_key, intent_digest = key[:3], key[3]
            existing_intent = self._intent_by_operation.get(operation_key)
            if existing_intent is not None and existing_intent != intent_digest:
                return "conflict", None
            if key not in self._records:
                self._intent_by_operation[operation_key] = intent_digest
                self._records[key] = None
                return "reserved", None
            if key in self._indeterminate:
                return "indeterminate", self._records[key]
            record = self._records[key]
            return ("pending", None) if record is None else ("completed", record)

    def complete(self, key: tuple[str, str, str, str], observation: ExecutionObservation) -> None:
        with self._lock:
            self._records[key] = observation

    def mark_indeterminate(self, key: tuple[str, str, str, str], observation: ExecutionObservation) -> None:
        with self._lock:
            self._records[key] = observation
            self._indeterminate.add(key)


class SQLiteIdempotencyStore:
    """Durable, lease-aware idempotency state with fail-closed crash recovery.

    A process dying after a side effect cannot safely retry the operation.  An
    expired pending lease is therefore recorded as ``indeterminate`` and must
    be reconciled by an operator/supervisor, never silently replayed.
    """

    def __init__(self, database: str | Path, *, lease_duration: timedelta = timedelta(minutes=5)) -> None:
        if lease_duration <= timedelta(0):
            raise ValueError("idempotency lease duration must be positive")
        path = Path(database)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(path, isolation_level=None)
        self._connection.execute("PRAGMA busy_timeout=5000")
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._lease_duration = lease_duration
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS executor_idempotency_v3 (
                executor_id TEXT NOT NULL,
                contract_digest TEXT NOT NULL,
                idempotency_key TEXT NOT NULL,
                intent_digest TEXT NOT NULL,
                state TEXT NOT NULL,
                observation_json TEXT,
                owner_id TEXT,
                lease_expires_at TEXT,
                attempts INTEGER NOT NULL DEFAULT 0,
                created_at TEXT,
                updated_at TEXT,
                PRIMARY KEY (executor_id, contract_digest, idempotency_key)
            )
            """
        )

    def reserve(self, key: tuple[str, str, str, str]) -> tuple[str, ExecutionObservation | None]:
        now = datetime.now(UTC)
        owner_id = f"pid-{threading.get_ident()}-{now.timestamp()}"
        lease_expires_at = (now + self._lease_duration).isoformat()
        self._connection.execute("BEGIN IMMEDIATE")
        try:
            row = self._connection.execute(
                """
                SELECT state, observation_json, lease_expires_at, intent_digest
                FROM executor_idempotency_v3
                WHERE executor_id = ? AND contract_digest = ? AND idempotency_key = ?
                """,
                key[:3],
            ).fetchone()
            if row is None:
                self._connection.execute(
                    """
                    INSERT INTO executor_idempotency_v3
                    (executor_id, contract_digest, idempotency_key, intent_digest, state, observation_json,
                     owner_id, lease_expires_at, attempts, created_at, updated_at)
                    VALUES (?, ?, ?, ?, 'pending', NULL, ?, ?, 1, ?, ?)
                    """,
                    (*key, owner_id, lease_expires_at, now.isoformat(), now.isoformat()),
                )
                self._connection.execute("COMMIT")
                return "reserved", None
            state, encoded, lease, stored_intent_digest = row
            if stored_intent_digest != key[3]:
                self._connection.execute("COMMIT")
                return "conflict", None
            if state == "completed":
                self._connection.execute("COMMIT")
                return "completed", self._decode_observation(encoded)
            if state == "indeterminate":
                self._connection.execute("COMMIT")
                return "indeterminate", self._decode_observation(encoded) if encoded else None
            if state != "pending":
                raise RuntimeError(f"unknown idempotency state: {state}")
            expired = not lease or datetime.fromisoformat(lease) <= now
            if expired:
                observation = self._indeterminate_observation("prior executor lease expired; side effect outcome is unknown")
                self._connection.execute(
                    """
                    UPDATE executor_idempotency_v3
                    SET state = 'indeterminate', observation_json = ?, updated_at = ?
                    WHERE executor_id = ? AND contract_digest = ? AND idempotency_key = ? AND intent_digest = ?
                    """,
                    (self._encode_observation(observation), now.isoformat(), *key),
                )
                self._connection.execute("COMMIT")
                return "indeterminate", observation
            self._connection.execute("COMMIT")
            return "pending", None
        except Exception:
            self._connection.execute("ROLLBACK")
            raise

    def complete(self, key: tuple[str, str, str, str], observation: ExecutionObservation) -> None:
        self._connection.execute("BEGIN IMMEDIATE")
        try:
            cursor = self._connection.execute(
                """
                UPDATE executor_idempotency_v3
                SET state = 'completed', observation_json = ?, owner_id = NULL,
                    lease_expires_at = NULL, updated_at = ?
                WHERE executor_id = ? AND contract_digest = ? AND idempotency_key = ? AND intent_digest = ? AND state = 'pending'
                """,
                (self._encode_observation(observation), datetime.now(UTC).isoformat(), *key),
            )
            if cursor.rowcount != 1:
                raise RuntimeError("idempotency reservation was lost before completion")
            self._connection.execute("COMMIT")
        except Exception:
            self._connection.execute("ROLLBACK")
            raise

    def mark_indeterminate(self, key: tuple[str, str, str, str], observation: ExecutionObservation) -> None:
        self._connection.execute("BEGIN IMMEDIATE")
        try:
            cursor = self._connection.execute(
                """
                UPDATE executor_idempotency_v3
                SET state = 'indeterminate', observation_json = ?, owner_id = NULL,
                    lease_expires_at = NULL, updated_at = ?
                WHERE executor_id = ? AND contract_digest = ? AND idempotency_key = ? AND intent_digest = ? AND state = 'pending'
                """,
                (self._encode_observation(observation), datetime.now(UTC).isoformat(), *key),
            )
            if cursor.rowcount != 1:
                raise RuntimeError("idempotency reservation was lost before indeterminate recovery")
            self._connection.execute("COMMIT")
        except Exception:
            self._connection.execute("ROLLBACK")
            raise

    @staticmethod
    def _encode_observation(observation: ExecutionObservation) -> str:
        return json.dumps(
            {
                "success": observation.success,
                "exit_code": observation.exit_code,
                "stdout": observation.stdout,
                "stderr": observation.stderr,
                "artifact_digests": dict(observation.artifact_digests),
                "metadata": dict(observation.metadata),
            },
            sort_keys=True,
        )

    @staticmethod
    def _indeterminate_observation(reason: str) -> ExecutionObservation:
        return ExecutionObservation(
            False,
            None,
            "",
            reason,
            metadata={"idempotency_state": "indeterminate"},
        )

    @staticmethod
    def _decode_observation(encoded: str | None) -> ExecutionObservation:
        if not encoded:
            raise RuntimeError("completed idempotency record has no observation")
        value = json.loads(encoded)
        return ExecutionObservation(
            bool(value["success"]),
            value["exit_code"],
            str(value["stdout"]),
            str(value["stderr"]),
            dict(value["artifact_digests"]),
            dict(value["metadata"]),
        )

    def close(self) -> None:
        self._connection.close()


class CapabilityEnforcingExecutor:
    """Policy enforcement point placed directly in front of a raw executor."""

    def __init__(
        self,
        *,
        executor_id: str,
        raw_executor: RawExecutor,
        capability_verifier: CapabilityVerifier,
        idempotency_store: IdempotencyStore,
    ) -> None:
        if not executor_id.strip():
            raise ValueError("executor_id is required")
        self.executor_id = executor_id
        self._raw_executor = raw_executor
        self._capability_verifier = capability_verifier
        self._idempotency_store = idempotency_store

    @property
    def raw_executor(self) -> RawExecutor:
        return self._raw_executor

    @property
    def idempotency_store(self) -> IdempotencyStore:
        return self._idempotency_store

    @property
    def capability_verifier(self) -> CapabilityVerifier:
        return self._capability_verifier

    def prepare_execution(
        self,
        intent: ActionIntent,
        *,
        run_id: str,
        contract_digest: str,
        iteration: int,
        operation_id: str,
        graph_digest: str = "",
        graph_node_id: str = "",
    ) -> PreparedExecution:
        """Prepare a durable operation identity without starting its effect."""

        prepare = getattr(self._raw_executor, "prepare_execution", None)
        if callable(prepare):
            prepared = prepare(
                intent,
                run_id=run_id,
                contract_digest=contract_digest,
                iteration=iteration,
                operation_id=operation_id,
                executor_id=self.executor_id,
                graph_digest=graph_digest,
                graph_node_id=graph_node_id,
            )
            if not isinstance(prepared, PreparedExecution):
                raise TypeError("raw executor returned an invalid prepared execution")
            if (
                prepared.operation_id != operation_id
                or prepared.executor_id != self.executor_id
                or prepared.intent_digest != intent.intent_digest
                or prepared.graph_digest != graph_digest
                or prepared.graph_node_id != graph_node_id
            ):
                raise ValueError("raw executor prepared another operation")
            return prepared
        return PreparedExecution(
            operation_id=operation_id,
            executor_id=self.executor_id,
            intent_digest=intent.intent_digest,
            request_digest=digest(
                {
                    "operation_id": operation_id,
                    "run_id": run_id,
                    "contract_digest": contract_digest,
                    "iteration": iteration,
                    "intent_digest": intent.intent_digest,
                    "idempotency_key": intent.idempotency_key,
                }
            ),
            remote_job_id=operation_id,
            graph_digest=graph_digest,
            graph_node_id=graph_node_id,
        )

    def execute(self, intent: ActionIntent, capability: Capability) -> ExecutionObservation:
        return self._execute(intent, capability, prepared=None)

    def execute_prepared(
        self,
        intent: ActionIntent,
        capability: Capability,
        prepared: PreparedExecution,
    ) -> ExecutionObservation:
        if prepared.executor_id != self.executor_id or prepared.intent_digest != intent.intent_digest:
            return self._authorization_failure("prepared execution is not bound to this executor and intent")
        return self._execute(intent, capability, prepared=prepared)

    def _execute(
        self,
        intent: ActionIntent,
        capability: Capability,
        *,
        prepared: PreparedExecution | None,
    ) -> ExecutionObservation:
        try:
            self._capability_verifier.validate(capability, intent, executor_id=self.executor_id)
        except CapabilityRejected as exc:
            return self._authorization_failure(str(exc))
        key = (
            self.executor_id,
            capability.contract_digest,
            intent.idempotency_key,
            intent.intent_digest,
        )
        state, cached = self._idempotency_store.reserve(key)
        if state == "completed" and cached is not None:
            return replace(
                cached,
                metadata={**cached.metadata, "idempotency_replay": True, "executor_id": self.executor_id},
            )
        if state == "pending":
            return self._authorization_failure("idempotency key is already executing")
        if state == "conflict":
            return self._authorization_failure("idempotency key was already bound to a different intent")
        if state == "indeterminate":
            return replace(
                cached or SQLiteIdempotencyStore._indeterminate_observation(
                    "previous execution outcome is unknown"
                ),
                metadata={
                    **(cached.metadata if cached else {}),
                    "executor_id": self.executor_id,
                    "capability_verified": True,
                    "idempotency_state": "indeterminate",
                },
            )
        try:
            self._capability_verifier.validate_and_consume(
                capability, intent, executor_id=self.executor_id
            )
        except CapabilityRejected as exc:
            observation = self._authorization_failure(str(exc))
            self._idempotency_store.complete(key, observation)
            return observation
        try:
            execute_prepared = getattr(self._raw_executor, "execute_prepared", None)
            observation = (
                execute_prepared(intent, prepared)
                if prepared is not None and callable(execute_prepared)
                else self._raw_executor.execute(intent)
            )
        except Exception as exc:
            observation = SQLiteIdempotencyStore._indeterminate_observation(
                f"raw executor raised {type(exc).__name__}; side effect outcome is unknown"
            )
            self._idempotency_store.mark_indeterminate(key, observation)
            return replace(
                observation,
                metadata={**observation.metadata, "executor_id": self.executor_id, "capability_verified": True},
            )
        observation = replace(
            observation,
            metadata={**observation.metadata, "executor_id": self.executor_id, "capability_verified": True},
        )
        try:
            self._idempotency_store.complete(key, observation)
        except Exception as exc:
            indeterminate = SQLiteIdempotencyStore._indeterminate_observation(
                f"execution completed but durable receipt failed: {type(exc).__name__}"
            )
            try:
                self._idempotency_store.mark_indeterminate(key, indeterminate)
            except Exception:
                pass
            return replace(
                indeterminate,
                metadata={**indeterminate.metadata, "executor_id": self.executor_id, "capability_verified": True},
            )
        return observation

    def bind_run(self, run_id: str, contract_digest: str | None = None) -> None:
        """Forward the controller run binding to raw executors that need it."""

        binder = getattr(self._raw_executor, "bind_run", None)
        if callable(binder):
            binder(run_id, contract_digest)

    def _authorization_failure(self, reason: str) -> ExecutionObservation:
        return ExecutionObservation(
            False,
            None,
            "",
            reason,
            metadata={"executor_id": self.executor_id, "capability_verified": False},
        )


@dataclass(frozen=True, slots=True)
class LocalCommandExecutor:
    """Development-only executor: no shell and explicit binary allowlist."""

    allowed_binaries: frozenset[str]
    workspace_root: Path

    def execute(self, intent: ActionIntent) -> ExecutionObservation:
        if intent.tool != "command.run":
            return ExecutionObservation(False, None, "", "unsupported local tool")
        command = intent.arguments.get("command")
        if not isinstance(command, list) or not command or not all(isinstance(x, str) for x in command):
            return ExecutionObservation(False, None, "", "command must be a non-empty argv array")
        if command[0] not in self.allowed_binaries:
            return ExecutionObservation(False, None, "", "binary is not allowlisted")
        requested_cwd = Path(str(intent.arguments.get("cwd", self.workspace_root))).resolve()
        root = self.workspace_root.resolve()
        if requested_cwd != root and root not in requested_cwd.parents:
            return ExecutionObservation(False, None, "", "working directory escapes workspace")
        timeout_seconds = int(intent.arguments.get("timeout_seconds", 60))
        if timeout_seconds < 1 or timeout_seconds > 600:
            return ExecutionObservation(False, None, "", "timeout is outside policy bounds")
        environment = {"PATH": os.environ.get("PATH", ""), "LANG": "C.UTF-8"}
        try:
            result = subprocess.run(
                command,
                cwd=requested_cwd,
                env=environment,
                capture_output=True,
                text=True,
                shell=False,
                timeout=timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return ExecutionObservation(False, None, "", "executor timeout")
        return ExecutionObservation(
            success=result.returncode == 0,
            exit_code=result.returncode,
            stdout=result.stdout[-16_384:],
            stderr=result.stderr[-16_384:],
            artifact_digests={"stdout": digest(result.stdout), "stderr": digest(result.stderr)},
            metadata={"executor": "local-development-only"},
        )


@dataclass(frozen=True, slots=True)
class BubblewrapExecutor:
    """Fail-closed Bubblewrap command builder for a separate sandbox worker."""

    bwrap_binary: str = "bwrap"

    def build_argv(
        self,
        command: Sequence[str],
        *,
        workspace: Path,
        readonly_paths: Sequence[Path] = (),
        network: bool = False,
    ) -> list[str]:
        if not command:
            raise ValueError("command is required")
        if network:
            raise ValueError(
                "network is disabled in the V0 sandbox; use a separately deployed network executor"
            )
        argv = [
            self.bwrap_binary,
            "--die-with-parent",
            "--new-session",
            "--unshare-all",
            "--ro-bind",
            "/usr",
            "/usr",
            "--ro-bind",
            "/bin",
            "/bin",
            "--proc",
            "/proc",
            "--dev",
            "/dev",
            "--bind",
            str(workspace),
            "/workspace",
            "--chdir",
            "/workspace",
        ]
        for path in readonly_paths:
            argv.extend(["--ro-bind", str(path), str(path)])
        argv.extend(["--", *command])
        return argv

    def execute(self, intent: ActionIntent) -> ExecutionObservation:
        return ExecutionObservation(
            False,
            None,
            "",
            "BubblewrapExecutor requires deployment-specific mounts and is not directly invoked",
        )
