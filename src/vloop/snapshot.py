"""Canonical, evaluator-owned workspace snapshots for schema-v2 receipts.

Snapshots are taken from a supplied immutable workspace/mount.  Callers should
create a copy-on-write or content-addressed checkout before invoking this
module; hashing a live editable workspace cannot itself eliminate TOCTOU.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
import stat
import subprocess
from typing import Mapping

from .canonical import digest


SNAPSHOT_SCHEMA = "vloop.workspace.v1"


@dataclass(frozen=True, slots=True)
class SnapshotPolicy:
    excluded_relative_paths: tuple[str, ...] = (".git", ".vloop", "__pycache__")

    @property
    def digest(self) -> str:
        return digest({"schema": SNAPSHOT_SCHEMA, "excluded_relative_paths": self.excluded_relative_paths})


@dataclass(frozen=True, slots=True)
class WorkspaceSnapshot:
    schema_version: str
    root_digest: str
    entries: tuple[Mapping[str, str | int], ...]
    git_commit: str | None
    git_dirty: bool | None
    git_untracked: tuple[str, ...]
    dependency_lock_digests: Mapping[str, str]
    toolchain_digest: str
    environment_digest: str
    exclusion_policy_digest: str

    @property
    def workspace_snapshot_digest(self) -> str:
        return self.root_digest

    @property
    def dependency_lock_digest(self) -> str:
        return digest(dict(self.dependency_lock_digests))


class CanonicalWorkspaceSnapshotter:
    """Hashes normalized path/type/mode/content leaves into a stable tree root."""

    def __init__(self, policy: SnapshotPolicy = SnapshotPolicy()) -> None:
        self.policy = policy

    def snapshot(
        self,
        root: str | Path,
        *,
        dependency_locks: tuple[str | Path, ...] = (),
        toolchain_digest: str,
        environment_digest: str,
    ) -> WorkspaceSnapshot:
        if not toolchain_digest or not environment_digest:
            raise ValueError("canonical snapshots need toolchain and environment digests")
        workspace = Path(root).resolve(strict=True)
        if not workspace.is_dir() or workspace.is_symlink():
            raise ValueError("workspace snapshot root must be a real directory")
        entries: list[dict[str, str | int]] = []
        for path in sorted(workspace.rglob("*"), key=lambda item: item.as_posix()):
            relative = PurePosixPath(path.relative_to(workspace).as_posix())
            if self._excluded(relative):
                continue
            info = path.lstat()
            mode = stat.S_IMODE(info.st_mode)
            if path.is_symlink():
                kind, payload = "symlink", os.readlink(path)
            elif path.is_file():
                kind, payload = "file", self._file_digest(path)
            elif path.is_dir():
                continue
            else:
                raise ValueError(f"unsupported workspace entry type: {relative}")
            entries.append(
                {
                    "path": relative.as_posix(),
                    "type": kind,
                    "mode": mode,
                    "payload_digest": digest(payload),
                    "leaf_digest": digest({"path": relative.as_posix(), "type": kind, "mode": mode, "payload": payload}),
                }
            )
        locks = self._lock_digests(workspace, dependency_locks)
        commit, dirty, untracked = self._git_state(workspace)
        root_digest = digest(
            {
                "schema": SNAPSHOT_SCHEMA,
                "entries": entries,
                "git_commit": commit,
                "git_dirty": dirty,
                "git_untracked": untracked,
                "dependency_lock_digests": locks,
                "toolchain_digest": toolchain_digest,
                "environment_digest": environment_digest,
                "exclusion_policy_digest": self.policy.digest,
            }
        )
        return WorkspaceSnapshot(
            SNAPSHOT_SCHEMA,
            root_digest,
            tuple(entries),
            commit,
            dirty,
            tuple(untracked),
            locks,
            toolchain_digest,
            environment_digest,
            self.policy.digest,
        )

    def _excluded(self, relative: PurePosixPath) -> bool:
        return any(relative.parts and relative.parts[0] == excluded for excluded in self.policy.excluded_relative_paths)

    @staticmethod
    def _file_digest(path: Path) -> str:
        hasher = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    def _lock_digests(self, workspace: Path, locks: tuple[str | Path, ...]) -> dict[str, str]:
        result: dict[str, str] = {}
        for candidate in locks:
            path = (workspace / candidate).resolve() if not Path(candidate).is_absolute() else Path(candidate).resolve()
            if path != workspace and workspace not in path.parents:
                raise ValueError("dependency lock escapes snapshot workspace")
            if not path.is_file() or path.is_symlink():
                raise ValueError("dependency lock must be a regular snapshot file")
            result[path.relative_to(workspace).as_posix()] = self._file_digest(path)
        return result

    @staticmethod
    def _git_state(workspace: Path) -> tuple[str | None, bool | None, tuple[str, ...]]:
        try:
            commit = subprocess.run(
                ["git", "-C", str(workspace), "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
                timeout=5,
            ).stdout.strip()
            status = subprocess.run(
                ["git", "-C", str(workspace), "status", "--porcelain=v1"],
                check=True,
                capture_output=True,
                text=True,
                timeout=5,
            ).stdout.splitlines()
        except (FileNotFoundError, subprocess.SubprocessError):
            return None, None, ()
        untracked = tuple(sorted(line[3:] for line in status if line.startswith("?? ")))
        return commit, bool(status), untracked
