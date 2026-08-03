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

from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Any, Mapping, Protocol
from uuid import uuid4

from .canonical import digest
from .models import ActionIntent, Effect, ExecutionObservation, PreparedExecution
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
    """Local supervisor assets plus opaque identities for remote execution.

    Paths are used only by an in-process/test supervisor.  A remote privileged
    supervisor receives ``remote_asset_request`` and resolves these reviewed
    identifiers from its own registry; it never receives controller paths.
    """

    kernel_image: Path
    rootfs: Path
    job_drive: Path
    kernel_image_id: str = ""
    kernel_image_digest: str = ""
    rootfs_image_id: str = ""
    rootfs_digest: str = ""
    resource_profile_id: str = ""
    workspace_snapshot_id: str = ""
    workspace_snapshot_digest: str = ""

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

    @property
    def remote_asset_request(self) -> Mapping[str, str]:
        """The complete allowlisted identity tuple required by a remote service."""

        values = {
            "kernel_image_id": self.kernel_image_id,
            "kernel_image_digest": self.kernel_image_digest,
            "rootfs_image_id": self.rootfs_image_id,
            "rootfs_digest": self.rootfs_digest,
            "resource_profile_id": self.resource_profile_id,
            "workspace_snapshot_id": self.workspace_snapshot_id,
            "workspace_snapshot_digest": self.workspace_snapshot_digest,
        }
        if not all(value.strip() for value in values.values()):
            return {}
        for digest_field in ("kernel_image_digest", "rootfs_digest", "workspace_snapshot_digest"):
            value = values[digest_field]
            if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
                raise FirecrackerConfigurationError(f"{digest_field} must be a SHA-256 hex digest")
        return values


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
    remote_asset_request: Mapping[str, str] = field(default_factory=dict)
    remote_execution_spec: Mapping[str, str] = field(default_factory=dict)
    remote_execution_spec_digest: str = ""


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
    result_file_digest: str = ""
    usage: Mapping[str, int] = field(default_factory=dict)


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
        operation_id: str | None = None,
        graph_digest: str = "",
        graph_node_id: str = "",
    ) -> FirecrackerLaunch:
        if bool(graph_digest) != bool(graph_node_id):
            raise FirecrackerConfigurationError("Firecracker graph digest and node must be supplied together")
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

        job_id = operation_id or str(uuid4())
        remote_asset_request = self.assets.remote_asset_request
        remote_execution_spec = (
            {
                "operation_id": job_id,
                "kernel_image_id": remote_asset_request["kernel_image_id"],
                "kernel_digest": remote_asset_request["kernel_image_digest"],
                "rootfs_image_id": remote_asset_request["rootfs_image_id"],
                "rootfs_digest": remote_asset_request["rootfs_digest"],
                "resource_profile_id": remote_asset_request["resource_profile_id"],
                "workspace_snapshot_id": remote_asset_request["workspace_snapshot_id"],
                "workspace_snapshot_digest": remote_asset_request["workspace_snapshot_digest"],
                "run_id": run_id,
                "contract_digest": contract_digest or "unbound",
                "intent_digest": intent.intent_digest,
                "graph_digest": graph_digest or "unbound",
                "graph_node_id": graph_node_id or "unbound",
            }
            if remote_asset_request
            else {}
        )
        remote_execution_spec_digest = digest(remote_execution_spec) if remote_execution_spec else ""
        manifest = {
            "schema_version": 1,
            "job_id": job_id,
            "operation_id": job_id,
            "run_id": run_id,
            "contract_digest": contract_digest or "unbound",
            "intent_digest": intent.intent_digest,
            "graph_digest": graph_digest or "unbound",
            "graph_node_id": graph_node_id or "unbound",
            "argv": command,
            "working_directory": "/workspace",
            "timeout_seconds": self.resources.timeout_seconds,
            "network_enabled": False,
            "result_path": "/job/vloop-result.json",
            "remote_execution_spec_digest": remote_execution_spec_digest,
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
            remote_asset_request=remote_asset_request,
            remote_execution_spec=remote_execution_spec,
            remote_execution_spec_digest=remote_execution_spec_digest,
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
        self._prepared_launches: dict[str, FirecrackerLaunch] = {}

    @property
    def supervisor_receipt_verifier(self) -> ReceiptVerifier | None:
        return self._supervisor_receipt_verifier

    @property
    def supervisor(self) -> FirecrackerSupervisor:
        """The narrow service client; it never grants host VM authority."""

        return self._supervisor

    @property
    def remote_asset_request(self) -> Mapping[str, str]:
        """Reviewed asset identities available to a remote supervisor."""

        return self._builder.assets.remote_asset_request

    def bind_run(self, run_id: str, contract_digest: str | None = None) -> None:
        self._run_id = run_id
        self._contract_digest = contract_digest

    def prepare_execution(
        self,
        intent: ActionIntent,
        *,
        run_id: str,
        contract_digest: str,
        iteration: int,
        operation_id: str,
        executor_id: str,
        graph_digest: str = "",
        graph_node_id: str = "",
    ) -> PreparedExecution:
        del iteration
        launch = self._builder.build(
            intent,
            run_id=run_id,
            contract_digest=contract_digest,
            operation_id=operation_id,
            graph_digest=graph_digest,
            graph_node_id=graph_node_id,
        )
        if not launch.remote_execution_spec_digest:
            raise FirecrackerConfigurationError("remote Firecracker preparation requires a canonical execution spec")
        self._prepared_launches[operation_id] = launch
        return PreparedExecution(
            operation_id=operation_id,
            executor_id=executor_id,
            intent_digest=intent.intent_digest,
            request_digest=launch.remote_execution_spec_digest,
            remote_job_id=launch.job_id,
            graph_digest=graph_digest,
            graph_node_id=graph_node_id,
        )

    def execute(self, intent: ActionIntent) -> ExecutionObservation:
        try:
            launch = self._builder.build(
                intent,
                run_id=self._run_id or "unbound",
                contract_digest=self._contract_digest,
            )
        except FirecrackerConfigurationError as exc:
            return ExecutionObservation(False, None, "", str(exc), metadata={"executor": "firecracker"})
        return self._execute_launch(intent, launch, prepared_execution=None)

    def execute_prepared(
        self, intent: ActionIntent, prepared_execution: PreparedExecution
    ) -> ExecutionObservation:
        launch = self._prepared_launches.pop(prepared_execution.operation_id, None)
        if launch is None:
            return ExecutionObservation(
                False,
                None,
                "",
                "prepared Firecracker operation is unavailable; reconciliation is required",
                metadata={
                    "executor": "firecracker",
                    "operation_id": prepared_execution.operation_id,
                    "request_digest": prepared_execution.request_digest,
                },
            )
        if (
            launch.job_id != prepared_execution.remote_job_id
            or launch.remote_execution_spec_digest != prepared_execution.request_digest
            or intent.intent_digest != prepared_execution.intent_digest
        ):
            return ExecutionObservation(False, None, "", "prepared Firecracker operation binding mismatch")
        return self._execute_launch(intent, launch, prepared_execution=prepared_execution)

    def _execute_launch(
        self,
        intent: ActionIntent,
        launch: FirecrackerLaunch,
        *,
        prepared_execution: PreparedExecution | None,
    ) -> ExecutionObservation:
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
            receipt_error = self._validate_supervisor_receipt(
                launch, intent, result, prepared_execution=prepared_execution
            )
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
                **(
                    {
                        "operation_id": prepared_execution.operation_id,
                        "request_digest": prepared_execution.request_digest,
                        "remote_job_id": prepared_execution.remote_job_id,
                        "graph_digest": prepared_execution.graph_digest,
                        "graph_node_id": prepared_execution.graph_node_id,
                    }
                    if prepared_execution is not None
                    else {}
                ),
                **receipt_metadata,
            },
        )

    def _validate_supervisor_receipt(
        self,
        launch: FirecrackerLaunch,
        intent: ActionIntent,
        result: GuestExecutionResult,
        *,
        prepared_execution: PreparedExecution | None,
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
                graph_digest=prepared_execution.graph_digest if prepared_execution is not None else None,
                graph_node_id=prepared_execution.graph_node_id if prepared_execution is not None else None,
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
        if prepared_execution is not None and (
            claims.get("operation_id") != prepared_execution.operation_id
            or claims.get("execution_spec_digest") != prepared_execution.request_digest
            or (
                bool(prepared_execution.graph_digest)
                and (
                    claims.get("graph_digest") != prepared_execution.graph_digest
                    or claims.get("graph_node_id") != prepared_execution.graph_node_id
                )
            )
        ):
            return "supervisor receipt is not bound to the prepared operation"
        expected_result = "pass" if result.success and result.exit_code == 0 else "fail"
        if receipt.result != expected_result:
            return "supervisor receipt result disagrees with guest execution outcome"
        expected_claims = {
            "exit_code": result.exit_code,
            "result_path": result.result_path,
            "stdout_digest": digest(result.stdout),
            "stderr_digest": digest(result.stderr),
            "result_file_digest": result.result_file_digest,
        }
        if any(claims.get(name) != value for name, value in expected_claims.items()):
            return "supervisor receipt does not bind the exact guest result"
        if not isinstance(claims.get("job_drive_digest"), str) or not claims["job_drive_digest"]:
            return "supervisor receipt lacks a job-drive digest"
        for name in ("wall_time_ms", "cpu_time_ms", "memory_peak_bytes"):
            if not isinstance(claims.get(name), int) or claims[name] < 0:
                return "supervisor receipt has invalid resource measurements"
        if not isinstance(claims.get("timed_out"), bool) or not isinstance(claims.get("oom_killed"), bool):
            return "supervisor receipt lacks timeout/OOM attestations"
        if result.success and (claims["timed_out"] or claims["oom_killed"]):
            return "successful guest result conflicts with timeout/OOM attestation"
        return None

    def reconcile_prepared(
        self,
        intent: ActionIntent,
        prepared_execution: PreparedExecution,
    ) -> ExecutionObservation:
        """Recover only a previously prepared remote operation by signed receipt."""

        reconcile = getattr(self._supervisor, "reconcile", None)
        if not callable(reconcile):
            raise FirecrackerConfigurationError("supervisor does not support signed operation reconciliation")
        result = reconcile(prepared_execution)
        if not isinstance(result, GuestExecutionResult):
            raise FirecrackerConfigurationError("supervisor reconciliation returned an invalid result")
        receipt_error = self._validate_reconciliation_receipt(intent, prepared_execution, result)
        if receipt_error is not None:
            raise PermissionError(receipt_error)
        return ExecutionObservation(
            success=result.success,
            exit_code=result.exit_code,
            stdout=result.stdout[-16_384:],
            stderr=result.stderr[-16_384:],
            artifact_digests=dict(result.artifact_digests),
            metadata={
                "executor": "firecracker",
                "isolation": "microvm",
                "operation_id": prepared_execution.operation_id,
                "request_digest": prepared_execution.request_digest,
                "remote_job_id": prepared_execution.remote_job_id,
                "graph_digest": prepared_execution.graph_digest,
                "graph_node_id": prepared_execution.graph_node_id,
                "reconciliation_signed": True,
                "evaluator_receipts": {"firecracker-supervisor": result.supervisor_receipt},
            },
        )

    def _validate_reconciliation_receipt(
        self,
        intent: ActionIntent,
        prepared_execution: PreparedExecution,
        result: GuestExecutionResult,
    ) -> str | None:
        if self._run_id is None or result.supervisor_receipt is None or self._supervisor_receipt_verifier is None:
            return "missing supervisor-signed reconciliation receipt"
        try:
            receipt = EvaluationReceipt.from_mapping(result.supervisor_receipt)
            self._supervisor_receipt_verifier.validate(
                receipt,
                receipt_type="firecracker-supervisor",
                run_id=self._run_id,
                intent_digest=intent.intent_digest,
                artifact_digests=result.artifact_digests,
                contract_digest=self._contract_digest,
                graph_digest=prepared_execution.graph_digest,
                graph_node_id=prepared_execution.graph_node_id,
            )
        except (KeyError, TypeError, ValueError, ReceiptRejected):
            return "supervisor-signed reconciliation receipt was rejected"
        claims = receipt.claims
        if (
            claims.get("operation_id") != prepared_execution.operation_id
            or claims.get("execution_spec_digest") != prepared_execution.request_digest
            or (
                bool(prepared_execution.graph_digest)
                and (
                    claims.get("graph_digest") != prepared_execution.graph_digest
                    or claims.get("graph_node_id") != prepared_execution.graph_node_id
                )
            )
            or claims.get("reconciliation") is not True
            or claims.get("fresh_job_drive") is not True
            or claims.get("job_drive_destroyed") is not True
        ):
            return "reconciliation receipt lacks prepared-operation lifecycle attestations"
        expected_result = "pass" if result.success and result.exit_code == 0 else "fail"
        if receipt.result != expected_result:
            return "reconciliation receipt result disagrees with guest execution outcome"
        expected_claims = {
            "exit_code": result.exit_code,
            "result_path": result.result_path,
            "stdout_digest": digest(result.stdout),
            "stderr_digest": digest(result.stderr),
            "result_file_digest": result.result_file_digest,
        }
        if any(claims.get(name) != value for name, value in expected_claims.items()):
            return "reconciliation receipt does not bind the exact guest result"
        return None


@dataclass(frozen=True, slots=True)
class FirecrackerEffectReconciler:
    """Production-only reconciler backed by the signed Firecracker supervisor."""

    executor: FirecrackerExecutor
    executor_id: str

    def __post_init__(self) -> None:
        if not self.executor_id.strip():
            raise ValueError("Firecracker reconciler needs an executor identity")
        if self.executor.supervisor_receipt_verifier is None:
            raise ValueError("Firecracker reconciler needs a supervisor receipt verifier")
        if not callable(getattr(self.executor.supervisor, "reconcile", None)):
            raise ValueError("Firecracker reconciler needs a reconciliation-capable supervisor")

    def reconcile(
        self,
        *,
        run_id: str,
        contract: object,
        intent: ActionIntent,
        executor_id: str,
        prepared_execution: PreparedExecution,
    ) -> ExecutionObservation:
        if executor_id != self.executor_id or prepared_execution.executor_id != executor_id:
            raise PermissionError("reconciliation request targets another executor")
        contract_digest = getattr(contract, "contract_digest", None)
        if not isinstance(contract_digest, str):
            raise TypeError("reconciliation needs a task contract")
        self.executor.bind_run(run_id, contract_digest)
        return self.executor.reconcile_prepared(intent, prepared_execution)
