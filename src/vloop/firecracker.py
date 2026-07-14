"""Firecracker microVM job contract for untrusted agent execution.

The privileged Firecracker supervisor is deliberately outside this package's
trusted controller process. This module creates only immutable launch data and
accepts a result through a narrow supervisor protocol. A guest image must
contain a V-Loop guest agent that reads the job manifest from the writable job
drive and writes a result document back to that same drive.

The config shape follows Firecracker's config-file API. Firecracker requires
KVM access and, in production, its jailer execution environment.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence
from uuid import uuid4

from .canonical import digest
from .models import ActionIntent, Effect, ExecutionObservation


class FirecrackerConfigurationError(ValueError):
    """The requested guest execution cannot be safely represented."""


class FirecrackerSupervisor(Protocol):
    """Privileged service that owns KVM, jailer, drives, and VM lifecycle."""

    def run(self, launch: "FirecrackerLaunch") -> "GuestExecutionResult": ...


@dataclass(frozen=True, slots=True)
class FirecrackerAssets:
    kernel_image: Path
    rootfs: Path
    job_drive: Path

    def validate(self) -> None:
        for label, path in (
            ("kernel image", self.kernel_image),
            ("rootfs", self.rootfs),
            ("job drive", self.job_drive),
        ):
            if not path.is_absolute() or not path.is_file() or path.is_symlink():
                raise FirecrackerConfigurationError(
                    f"{label} must be an absolute regular non-symlink file"
                )


@dataclass(frozen=True, slots=True)
class MicroVMResources:
    vcpu_count: int = 1
    mem_size_mib: int = 256
    timeout_seconds: int = 60

    def __post_init__(self) -> None:
        if not 1 <= self.vcpu_count <= 8:
            raise FirecrackerConfigurationError("vcpu_count must be in [1, 8]")
        if not 128 <= self.mem_size_mib <= 4096:
            raise FirecrackerConfigurationError("mem_size_mib must be in [128, 4096]")
        if not 1 <= self.timeout_seconds <= 600:
            raise FirecrackerConfigurationError("timeout_seconds must be in [1, 600]")


@dataclass(frozen=True, slots=True)
class FirecrackerLaunch:
    job_id: str
    config: Mapping[str, Any]
    manifest: Mapping[str, Any]
    config_digest: str
    manifest_digest: str


@dataclass(frozen=True, slots=True)
class GuestExecutionResult:
    """Untrusted guest output, bound to a host-created manifest digest."""

    manifest_digest: str
    success: bool
    exit_code: int | None
    stdout: str
    stderr: str
    artifact_digests: Mapping[str, str]
    result_path: str


class FirecrackerJobBuilder:
    """Creates a networkless, two-drive Firecracker config for one action."""

    def __init__(self, assets: FirecrackerAssets, resources: MicroVMResources = MicroVMResources()) -> None:
        assets.validate()
        self.assets = assets
        self.resources = resources

    def build(self, intent: ActionIntent) -> FirecrackerLaunch:
        if intent.tool != "command.run" or intent.effect is not Effect.EXECUTE:
            raise FirecrackerConfigurationError("Firecracker only executes command.run intents")
        command = intent.arguments.get("command")
        if not isinstance(command, list) or not command or not all(isinstance(item, str) for item in command):
            raise FirecrackerConfigurationError("guest command must be a non-empty argv list")
        if any("\x00" in item for item in command):
            raise FirecrackerConfigurationError("guest command contains a NUL byte")
        requested_network = intent.arguments.get("network", False)
        if requested_network is not False:
            raise FirecrackerConfigurationError("network is disabled for untrusted V0 jobs")
        requested_timeout = intent.arguments.get("timeout_seconds", self.resources.timeout_seconds)
        if requested_timeout != self.resources.timeout_seconds:
            raise FirecrackerConfigurationError("timeout must match the supervisor-issued resource profile")

        job_id = str(uuid4())
        manifest = {
            "schema_version": 1,
            "job_id": job_id,
            "intent_digest": intent.intent_digest,
            "argv": command,
            "working_directory": "/workspace",
            "timeout_seconds": self.resources.timeout_seconds,
            "network_enabled": False,
            "result_path": "/job/vloop-result.json",
        }
        config = {
            "boot-source": {
                "kernel_image_path": str(self.assets.kernel_image),
                "boot_args": "console=ttyS0 reboot=k panic=1 quiet",
            },
            "machine-config": {
                "vcpu_count": self.resources.vcpu_count,
                "mem_size_mib": self.resources.mem_size_mib,
                "smt": False,
                "track_dirty_pages": True,
            },
            "drives": [
                {
                    "drive_id": "rootfs",
                    "path_on_host": str(self.assets.rootfs),
                    "is_root_device": True,
                    "is_read_only": True,
                },
                {
                    "drive_id": "job",
                    "path_on_host": str(self.assets.job_drive),
                    "is_root_device": False,
                    "is_read_only": False,
                },
            ],
        }
        return FirecrackerLaunch(
            job_id=job_id,
            config=config,
            manifest=manifest,
            config_digest=digest(config),
            manifest_digest=digest(manifest),
        )


class FirecrackerExecutor:
    """Executor adapter requiring a separately deployed privileged supervisor."""

    def __init__(self, builder: FirecrackerJobBuilder, supervisor: FirecrackerSupervisor) -> None:
        self._builder = builder
        self._supervisor = supervisor

    def execute(self, intent: ActionIntent) -> ExecutionObservation:
        try:
            launch = self._builder.build(intent)
        except FirecrackerConfigurationError as exc:
            return ExecutionObservation(False, None, "", str(exc), metadata={"executor": "firecracker"})
        result = self._supervisor.run(launch)
        if result.manifest_digest != launch.manifest_digest:
            return ExecutionObservation(
                False,
                result.exit_code,
                result.stdout,
                "guest result is not bound to the launched manifest",
                metadata={
                    "executor": "firecracker",
                    "isolation": "microvm",
                    "job_id": launch.job_id,
                    "config_digest": launch.config_digest,
                },
            )
        return ExecutionObservation(
            success=result.success,
            exit_code=result.exit_code,
            stdout=result.stdout[-16_384:],
            stderr=result.stderr[-16_384:],
            artifact_digests=dict(result.artifact_digests),
            metadata={
                "executor": "firecracker",
                "isolation": "microvm",
                "job_id": launch.job_id,
                "config_digest": launch.config_digest,
                "guest_manifest_digest": launch.manifest_digest,
                "guest_result_path": result.result_path,
                "rootfs_read_only": True,
                "network_enabled": False,
            },
        )
