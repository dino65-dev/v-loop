"""Executor interfaces.

No executor receives the policy signing key. Capability consumption happens in
the trusted orchestrator before the executor is invoked.
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence

from .canonical import digest
from .models import ActionIntent, ExecutionObservation


class Executor(Protocol):
    def execute(self, intent: ActionIntent) -> ExecutionObservation: ...


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
