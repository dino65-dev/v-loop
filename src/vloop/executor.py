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
from pathlib import Path
from typing import Protocol, Sequence

from .authorization import CapabilityRejected, CapabilityVerifier
from .canonical import digest
from .models import ActionIntent, Capability, ExecutionObservation


class Executor(Protocol):
    executor_id: str

    def execute(self, intent: ActionIntent, capability: Capability) -> ExecutionObservation: ...


class RawExecutor(Protocol):
    def execute(self, intent: ActionIntent) -> ExecutionObservation: ...


class IdempotencyStore(Protocol):
    def reserve(self, key: tuple[str, str, str]) -> tuple[str, ExecutionObservation | None]: ...

    def complete(self, key: tuple[str, str, str], observation: ExecutionObservation) -> None: ...


class InMemoryIdempotencyStore:
    """Thread-safe test/development idempotency store."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._records: dict[tuple[str, str, str], ExecutionObservation | None] = {}

    def reserve(self, key: tuple[str, str, str]) -> tuple[str, ExecutionObservation | None]:
        with self._lock:
            if key not in self._records:
                self._records[key] = None
                return "reserved", None
            record = self._records[key]
            return ("pending", None) if record is None else ("completed", record)

    def complete(self, key: tuple[str, str, str], observation: ExecutionObservation) -> None:
        with self._lock:
            self._records[key] = observation


class SQLiteIdempotencyStore:
    """Durable executor-side idempotency table with transactional reservation."""

    def __init__(self, database: str | Path) -> None:
        path = Path(database)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(path, isolation_level=None)
        self._connection.execute("PRAGMA busy_timeout=5000")
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS executor_idempotency (
                executor_id TEXT NOT NULL,
                contract_digest TEXT NOT NULL,
                idempotency_key TEXT NOT NULL,
                state TEXT NOT NULL,
                observation_json TEXT,
                PRIMARY KEY (executor_id, contract_digest, idempotency_key)
            )
            """
        )

    def reserve(self, key: tuple[str, str, str]) -> tuple[str, ExecutionObservation | None]:
        with self._connection:
            row = self._connection.execute(
                "SELECT state, observation_json FROM executor_idempotency WHERE executor_id = ? AND contract_digest = ? AND idempotency_key = ?",
                key,
            ).fetchone()
            if row is None:
                self._connection.execute(
                    "INSERT INTO executor_idempotency VALUES (?, ?, ?, 'pending', NULL)", key
                )
                return "reserved", None
        if row[0] == "pending":
            return "pending", None
        return "completed", self._decode_observation(row[1])

    def complete(self, key: tuple[str, str, str], observation: ExecutionObservation) -> None:
        encoded = json.dumps(
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
        with self._connection:
            self._connection.execute(
                "UPDATE executor_idempotency SET state = 'completed', observation_json = ? WHERE executor_id = ? AND contract_digest = ? AND idempotency_key = ?",
                (encoded, *key),
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

    def execute(self, intent: ActionIntent, capability: Capability) -> ExecutionObservation:
        try:
            self._capability_verifier.validate(capability, intent, executor_id=self.executor_id)
        except CapabilityRejected as exc:
            return self._authorization_failure(str(exc))
        key = (self.executor_id, capability.contract_digest, intent.idempotency_key)
        state, cached = self._idempotency_store.reserve(key)
        if state == "completed" and cached is not None:
            return replace(
                cached,
                metadata={**cached.metadata, "idempotency_replay": True, "executor_id": self.executor_id},
            )
        if state == "pending":
            return self._authorization_failure("idempotency key is already executing")
        try:
            self._capability_verifier.validate_and_consume(
                capability, intent, executor_id=self.executor_id
            )
        except CapabilityRejected as exc:
            observation = self._authorization_failure(str(exc))
            self._idempotency_store.complete(key, observation)
            return observation
        observation = self._raw_executor.execute(intent)
        observation = replace(
            observation,
            metadata={**observation.metadata, "executor_id": self.executor_id, "capability_verified": True},
        )
        self._idempotency_store.complete(key, observation)
        return observation

    def bind_run(self, run_id: str) -> None:
        """Forward the controller run binding to raw executors that need it."""

        binder = getattr(self._raw_executor, "bind_run", None)
        if callable(binder):
            binder(run_id)

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
