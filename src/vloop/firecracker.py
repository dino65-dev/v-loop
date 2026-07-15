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
import os
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence
from uuid import uuid4

from .canonical import digest
from .models import ActionIntent, Effect, ExecutionObservation
from .receipts import EvaluationReceipt, ReceiptRejected, ReceiptVerifier


class FirecrackerConfigurationError(ValueError):
    """The requested guest execution cannot be safely represented."""


class FirecrackerSupervisor(Protocol):
    """Privileged service that owns KVM, jailer, drives, and VM lifecycle."""

    def run(self, launch: "FirecrackerLaunch") -> "GuestExecutionResult": ...


@dataclass(frozen=True, slots=True)
class FirecrackerRuntime:
    """Deployment-owned paths required by a privileged supervisor."""

    firecracker_binary: Path
    jailer_binary: Path
    chroot_base: Path
    kvm_device: Path = Path("/dev/kvm")

    def __post_init__(self) -> None:
        for label, path in (
            ("Firecracker binary", self.firecracker_binary),
            ("jailer binary", self.jailer_binary),
            ("jailer chroot base", self.chroot_base),
            ("KVM device", self.kvm_device),
        ):
            if not path.is_absolute():
                raise FirecrackerConfigurationError(f"{label} must be an absolute path")


@dataclass(frozen=True, slots=True)
class FirecrackerPreflightReport:
    ready: bool
    checks: Mapping[str, str]

    def require_ready(self) -> None:
        if not self.ready:
            failed = ", ".join(name for name, status in self.checks.items() if status != "ready")
            raise FirecrackerConfigurationError(f"Firecracker supervisor preflight failed: {failed}")


class FirecrackerPreflight:
    """Fail-closed host prerequisite validation for a Firecracker supervisor.

    This checks only deployer-owned files and KVM access.  It neither starts a
    VM nor grants the controller permission to operate Firecracker directly.
    """

    @staticmethod
    def check(runtime: FirecrackerRuntime, assets: "FirecrackerAssets") -> FirecrackerPreflightReport:
        checks: dict[str, str] = {}
        for label, path in (
            ("firecracker_binary", runtime.firecracker_binary),
            ("jailer_binary", runtime.jailer_binary),
        ):
            checks[label] = (
                "ready"
                if path.is_file() and not path.is_symlink() and os.access(path, os.X_OK)
                else "missing-or-not-executable"
            )
        checks["jailer_chroot_base"] = (
            "ready"
            if runtime.chroot_base.is_dir()
            and not runtime.chroot_base.is_symlink()
            and os.access(runtime.chroot_base, os.W_OK | os.X_OK)
            else "missing-or-not-writable"
        )
        checks["kvm_device"] = (
            "ready"
            if runtime.kvm_device.is_char_device()
            and not runtime.kvm_device.is_symlink()
            and os.access(runtime.kvm_device, os.R_OK | os.W_OK)
            else "unavailable"
        )
        try:
            assets.validate()
        except FirecrackerConfigurationError:
            checks["guest_assets"] = "invalid"
        else:
            checks["guest_assets"] = "ready"
        return FirecrackerPreflightReport(
            ready=all(status == "ready" for status in checks.values()), checks=checks
        )


@dataclass(frozen=True, slots=True)
class JailerLaunchSpec:
    """Shell-free supervisor launch plan after it materializes immutable files."""

    job_id: str
    jailer_argv: tuple[str, ...]
    config_path: Path
    manifest_path: Path
    config_digest: str
    manifest_digest: str


class FirecrackerSupervisorPlan:
    """Builds an auditable jailer request; lifecycle stays in the supervisor."""

    def __init__(self, runtime: FirecrackerRuntime, *, uid: int, gid: int) -> None:
        if uid < 0 or gid < 0:
            raise FirecrackerConfigurationError("jailer uid and gid must be non-negative")
        self.runtime = runtime
        self.uid = uid
        self.gid = gid

    def build(self, launch: "FirecrackerLaunch", *, staging_directory: Path) -> JailerLaunchSpec:
        if not staging_directory.is_absolute() or staging_directory.is_symlink():
            raise FirecrackerConfigurationError("supervisor staging directory must be absolute and non-symlink")
        config_path = staging_directory / f"{launch.job_id}.firecracker.json"
        manifest_path = staging_directory / f"{launch.job_id}.manifest.json"
        argv = (
            str(self.runtime.jailer_binary),
            "--id",
            launch.job_id,
            "--exec-file",
            str(self.runtime.firecracker_binary),
            "--uid",
            str(self.uid),
            "--gid",
            str(self.gid),
            "--chroot-base-dir",
            str(self.runtime.chroot_base),
            "--",
            "--config-file",
            str(config_path),
        )
        return JailerLaunchSpec(
            job_id=launch.job_id,
            jailer_argv=argv,
            config_path=config_path,
            manifest_path=manifest_path,
            config_digest=launch.config_digest,
            manifest_digest=launch.manifest_digest,
        )


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
    supervisor_receipt: Mapping[str, Any] | None = None


class FirecrackerJobBuilder:
    """Creates a networkless, two-drive Firecracker config for one action."""

    def __init__(self, assets: FirecrackerAssets, resources: MicroVMResources = MicroVMResources()) -> None:
        assets.validate()
        self.assets = assets
        self.resources = resources

    def build(
        self,
        intent: ActionIntent,
        *,
        run_id: str = "unbound",
        contract_digest: str | None = None,
    ) -> FirecrackerLaunch:
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
            "run_id": run_id,
            "contract_digest": contract_digest or "unbound",
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

    def __init__(
        self,
        builder: FirecrackerJobBuilder,
        supervisor: FirecrackerSupervisor,
        supervisor_receipt_verifier: ReceiptVerifier | None = None,
    ) -> None:
        self._builder = builder
        self._supervisor = supervisor
        self._supervisor_receipt_verifier = supervisor_receipt_verifier
        self._run_id: str | None = None
        self._contract_digest: str | None = None

    @property
    def supervisor_receipt_verifier(self) -> ReceiptVerifier | None:
        return self._supervisor_receipt_verifier

    def bind_run(self, run_id: str, contract_digest: str | None = None) -> None:
        self._run_id = run_id
        self._contract_digest = contract_digest

    def execute(self, intent: ActionIntent) -> ExecutionObservation:
        try:
            launch = self._builder.build(
                intent,
                run_id=self._run_id or "unbound",
                contract_digest=self._contract_digest,
            )
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
        receipt_metadata: dict[str, Any] = {}
        if self._supervisor_receipt_verifier is not None:
            receipt_error = self._validate_supervisor_receipt(launch, intent, result)
            if receipt_error is not None:
                return ExecutionObservation(
                    False,
                    result.exit_code,
                    result.stdout[-16_384:],
                    receipt_error,
                    artifact_digests=dict(result.artifact_digests),
                    metadata={"executor": "firecracker", "isolation": "microvm"},
                )
            receipt_metadata = {"evaluator_receipts": {"firecracker-supervisor": result.supervisor_receipt}}
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
                **receipt_metadata,
            },
        )

    def _validate_supervisor_receipt(
        self,
        launch: FirecrackerLaunch,
        intent: ActionIntent,
        result: GuestExecutionResult,
    ) -> str | None:
        if self._run_id is None or result.supervisor_receipt is None:
            return "missing supervisor-signed execution receipt"
        try:
            receipt = EvaluationReceipt.from_mapping(result.supervisor_receipt)
            self._supervisor_receipt_verifier.validate(
                receipt,
                receipt_type="firecracker-supervisor",
                run_id=self._run_id,
                intent_digest=intent.intent_digest,
                artifact_digests=result.artifact_digests,
                contract_digest=self._contract_digest,
            )
        except (KeyError, TypeError, ValueError, ReceiptRejected):
            return "supervisor-signed execution receipt was rejected"
        claims = receipt.claims
        if (
            claims.get("job_id") != launch.job_id
            or claims.get("manifest_digest") != launch.manifest_digest
            or claims.get("fresh_job_drive") is not True
            or claims.get("job_drive_destroyed") is not True
        ):
            return "supervisor receipt lacks required job lifecycle attestations"
        return None
