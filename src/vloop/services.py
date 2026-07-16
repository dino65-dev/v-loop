"""Narrow authenticated clients for externally privileged V-Loop services.

These clients submit immutable work descriptions; they never expose KVM,
jailer, evaluator signing, or ledger-anchor credentials to the planner or
guest. Service implementations must authenticate the caller key, deduplicate
the supplied idempotency key, and issue separately trusted receipts.
"""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Mapping
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen
from uuid import uuid4

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from .canonical import canonical_json, digest
from .firecracker import FirecrackerLaunch, GuestExecutionResult
from .models import PreparedExecution
from .ledger import LedgerAnchorRecord


class RemoteServiceError(RuntimeError):
    """A privileged service response was unavailable or violated its contract."""


class ServiceRequestSigner:
    """Deployment-held caller identity for one constrained remote service role."""

    def __init__(self, private_key: bytes | Ed25519PrivateKey, *, key_id: str) -> None:
        if not key_id.strip():
            raise ValueError("service request signer needs a key id")
        self._key = (
            Ed25519PrivateKey.from_private_bytes(private_key)
            if isinstance(private_key, bytes)
            else private_key
        )
        self.key_id = key_id

    @property
    def public_key_bytes(self) -> bytes:
        return self._key.public_key().public_bytes(
            serialization.Encoding.Raw, serialization.PublicFormat.Raw
        )

    def sign(self, *, method: str, path: str, payload: Mapping[str, Any], timestamp: str, nonce: str) -> str:
        message = canonical_json(
            {
                "method": method,
                "path": path,
                "payload": payload,
                "timestamp": timestamp,
                "nonce": nonce,
            }
        ).encode("utf-8")
        return base64.urlsafe_b64encode(self._key.sign(message)).decode("ascii")


class AuthenticatedHTTPSClient:
    """Signed JSON client; plain HTTP is restricted to explicit loopback tests."""

    def __init__(
        self,
        base_url: str,
        signer: ServiceRequestSigner,
        *,
        bearer_token: str | None = None,
        timeout_seconds: float = 15.0,
        maximum_response_bytes: int = 1_000_000,
        allow_insecure_loopback: bool = False,
    ) -> None:
        parsed = urlparse(base_url)
        loopback = parsed.hostname in {"127.0.0.1", "::1", "localhost"}
        if parsed.scheme not in {"https", "http"} or not parsed.netloc:
            raise ValueError("privileged service URL must be absolute HTTP(S)")
        if parsed.scheme != "https" and not (allow_insecure_loopback and loopback):
            raise ValueError("privileged services require HTTPS outside explicit loopback tests")
        if timeout_seconds <= 0 or maximum_response_bytes < 1:
            raise ValueError("service timeout and response limit must be positive")
        self.base_url = base_url.rstrip("/")
        self.signer = signer
        self.bearer_token = bearer_token
        self.timeout_seconds = timeout_seconds
        self.maximum_response_bytes = maximum_response_bytes

    def post(
        self,
        path: str,
        payload: Mapping[str, Any],
        *,
        idempotency_key: str,
    ) -> Mapping[str, Any]:
        if not path.startswith("/") or not idempotency_key.strip():
            raise ValueError("service requests need an absolute path and idempotency key")
        timestamp = datetime.now(UTC).isoformat()
        nonce = str(uuid4())
        signature = self.signer.sign(
            method="POST", path=path, payload=payload, timestamp=timestamp, nonce=nonce
        )
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Idempotency-Key": idempotency_key,
            "X-VLoop-Key-Id": self.signer.key_id,
            "X-VLoop-Timestamp": timestamp,
            "X-VLoop-Nonce": nonce,
            "X-VLoop-Signature": signature,
        }
        if self.bearer_token:
            headers["Authorization"] = f"Bearer {self.bearer_token}"
        request = Request(
            self.base_url + path,
            data=canonical_json(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:  # nosec B310: URL checked above
                encoded = response.read(self.maximum_response_bytes + 1)
        except (HTTPError, URLError, TimeoutError) as exc:
            raise RemoteServiceError("privileged service request failed") from exc
        if len(encoded) > self.maximum_response_bytes:
            raise RemoteServiceError("privileged service response exceeds its configured limit")
        try:
            decoded = json.loads(encoded.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RemoteServiceError("privileged service returned invalid JSON") from exc
        if not isinstance(decoded, Mapping):
            raise RemoteServiceError("privileged service response must be an object")
        return decoded


@dataclass(frozen=True, slots=True)
class FirecrackerSupervisorHTTPClient:
    """Controller-side client for a separately privileged Firecracker service."""

    client: AuthenticatedHTTPSClient
    endpoint: str = "/v1/firecracker/jobs"
    name: str = "firecracker-supervisor"

    def run(self, launch: FirecrackerLaunch) -> GuestExecutionResult:
        if not launch.remote_execution_spec or not launch.remote_execution_spec_digest:
            raise RemoteServiceError("remote Firecracker jobs require a canonical execution specification")
        response = self.client.post(
            self.endpoint,
            {
                # The privileged service resolves its own allowlisted asset
                # IDs and creates the writable job drive itself. Never send
                # controller-visible kernel/rootfs/drive paths over the API.
                "execution_spec": dict(launch.remote_execution_spec),
                "execution_spec_digest": launch.remote_execution_spec_digest,
                "manifest": dict(launch.manifest),
                "manifest_digest": launch.manifest_digest,
            },
            idempotency_key=launch.remote_execution_spec["operation_id"],
        )
        result = self._decode_result(response)
        if result.manifest_digest != launch.manifest_digest:
            raise RemoteServiceError("Firecracker supervisor response is bound to another manifest")
        return result

    def reconcile(self, prepared_execution: PreparedExecution) -> GuestExecutionResult:
        """Query an existing operation; never submit or replay a new effect."""

        response = self.client.post(
            f"{self.endpoint}/{prepared_execution.operation_id}/reconcile",
            {
                "operation_id": prepared_execution.operation_id,
                "execution_spec_digest": prepared_execution.request_digest,
            },
            idempotency_key=prepared_execution.operation_id,
        )
        if (
            response.get("operation_id") != prepared_execution.operation_id
            or response.get("execution_spec_digest") != prepared_execution.request_digest
        ):
            raise RemoteServiceError("Firecracker reconciliation response is bound to another operation")
        return self._decode_result(response)

    @staticmethod
    def _decode_result(response: Mapping[str, Any]) -> GuestExecutionResult:
        try:
            artifacts = response.get("artifact_digests", {})
            receipt = response.get("supervisor_receipt")
            usage = response.get("usage", {})
            if not isinstance(artifacts, Mapping) or not all(
                isinstance(key, str) and isinstance(value, str) for key, value in artifacts.items()
            ):
                raise ValueError("invalid artifact digest map")
            if receipt is not None and not isinstance(receipt, Mapping):
                raise ValueError("invalid supervisor receipt")
            if not isinstance(usage, Mapping) or not all(
                isinstance(key, str) and isinstance(value, int) and value >= 0 for key, value in usage.items()
            ):
                raise ValueError("invalid guest resource usage")
            success = response["success"]
            exit_code = response.get("exit_code")
            if not isinstance(success, bool) or (
                exit_code is not None and (not isinstance(exit_code, int) or isinstance(exit_code, bool))
            ):
                raise ValueError("invalid success or exit status")
            return GuestExecutionResult(
                manifest_digest=str(response["manifest_digest"]),
                success=success,
                exit_code=exit_code,
                stdout=str(response.get("stdout", "")),
                stderr=str(response.get("stderr", "")),
                artifact_digests=dict(artifacts),
                result_path=str(response["result_path"]),
                supervisor_receipt=dict(receipt) if receipt is not None else None,
                result_file_digest=str(response.get("result_file_digest", "")),
                usage=dict(usage),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise RemoteServiceError("Firecracker supervisor response violated its schema") from exc


@dataclass(frozen=True, slots=True)
class ProtectedEvaluatorHTTPClient:
    """Submits evaluator work and returns only its signed receipt mapping."""

    client: AuthenticatedHTTPSClient
    endpoint: str = "/v1/evaluations"
    name: str = "protected-evaluator"

    def evaluate(
        self,
        *,
        run_id: str,
        contract_digest: str,
        intent_digest: str,
        receipt_type: str,
        artifact_digests: Mapping[str, str],
        workspace_snapshot_digest: str,
        evaluator_image_digest: str,
        test_suite_digest: str,
    ) -> Mapping[str, Any]:
        payload = {
            "run_id": run_id,
            "contract_digest": contract_digest,
            "intent_digest": intent_digest,
            "receipt_type": receipt_type,
            "artifact_digests": dict(artifact_digests),
            "workspace_snapshot_digest": workspace_snapshot_digest,
            "evaluator_image_digest": evaluator_image_digest,
            "test_suite_digest": test_suite_digest,
        }
        response = self.client.post(
            self.endpoint,
            payload,
            idempotency_key=digest({"run_id": run_id, "intent_digest": intent_digest, "receipt_type": receipt_type}),
        )
        receipt = response.get("receipt")
        if not isinstance(receipt, Mapping):
            raise RemoteServiceError("protected evaluator response lacks a receipt")
        return dict(receipt)


@dataclass(frozen=True, slots=True)
class LedgerAnchorHTTPClient:
    """Externally anchors immutable evidence-ledger heads with a signed request."""

    client: AuthenticatedHTTPSClient
    endpoint: str = "/v1/ledger-anchors"
    name: str = "ledger-anchor"

    def anchor(self, record: LedgerAnchorRecord) -> None:
        response = self.client.post(
            self.endpoint,
            {
                "event_hash": record.event_hash,
                "sequence": record.sequence,
                "occurred_at": record.occurred_at.isoformat(),
            },
            idempotency_key=record.event_hash,
        )
        response_hash = response.get("event_hash")
        if response_hash is not None and response_hash != record.event_hash:
            raise RemoteServiceError("ledger anchor response is bound to another evidence head")
