"""Signed proof-of-execution certificates over a causal graph trace."""

from __future__ import annotations

import base64
from dataclasses import asdict, dataclass, replace
from typing import Iterable

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey

from .canonical import canonical_json, digest
from .graph import GraphManifest
from .graph_events import CausalEvent
from .graph_monitor import TransitionMonitor


@dataclass(frozen=True, slots=True)
class ExecutionCertificate:
    run_id: str
    contract_digest: str
    graph_digest: str
    realised_graph_digest: str
    harness_bundle_digest: str
    initial_workspace_digest: str
    final_workspace_digest: str
    causal_trace_root: str
    authority_proof_root: str
    evidence_proof_root: str
    final_decision: str
    validator_version: str
    key_id: str
    signature: str = ""

    def __post_init__(self) -> None:
        if not self.run_id.strip() or not self.final_decision.strip() or not self.validator_version.strip() or not self.key_id.strip():
            raise ValueError("execution certificates need run, decision, validator, and key identities")
        for value in (
            self.contract_digest, self.graph_digest, self.realised_graph_digest, self.harness_bundle_digest,
            self.initial_workspace_digest, self.final_workspace_digest, self.causal_trace_root,
            self.authority_proof_root, self.evidence_proof_root,
        ):
            if len(value) != 64:
                raise ValueError("execution certificate digests must be SHA-256 hex")

    def payload(self) -> bytes:
        return canonical_json({**asdict(self), "signature": ""}).encode("utf-8")


class ExecutionCertificateSigner:
    def __init__(self, private_key: bytes | Ed25519PrivateKey | None = None, *, key_id: str = "execution-validator") -> None:
        self.key_id = key_id
        self._key = Ed25519PrivateKey.generate() if private_key is None else (
            Ed25519PrivateKey.from_private_bytes(private_key) if isinstance(private_key, bytes) else private_key
        )

    @property
    def public_key_bytes(self) -> bytes:
        return self._key.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)

    def issue(self, certificate: ExecutionCertificate) -> ExecutionCertificate:
        if certificate.key_id != self.key_id:
            raise ValueError("certificate key id does not match signer")
        return replace(certificate, signature=base64.urlsafe_b64encode(self._key.sign(certificate.payload())).decode("ascii"))


class ExecutionCertificateValidator:
    """Checks trace consistency before accepting a controller-independent proof."""

    def __init__(self, public_key: bytes | Ed25519PublicKey, *, key_id: str = "execution-validator") -> None:
        self._key = Ed25519PublicKey.from_public_bytes(public_key) if isinstance(public_key, bytes) else public_key
        self.key_id = key_id

    def validate(
        self,
        certificate: ExecutionCertificate,
        events: Iterable[CausalEvent],
        *,
        manifest: GraphManifest | None = None,
    ) -> None:
        if certificate.key_id != self.key_id:
            raise PermissionError("execution certificate key is not trusted")
        try:
            self._key.verify(base64.urlsafe_b64decode(certificate.signature.encode("ascii")), certificate.payload())
        except (InvalidSignature, ValueError) as exc:
            raise PermissionError("execution certificate signature is invalid") from exc
        trace = tuple(events)
        if not trace or any(event.run_id != certificate.run_id or event.graph_digest != certificate.graph_digest for event in trace):
            raise PermissionError("certificate trace belongs to another run or graph")
        by_id = {event.event_id: event for event in trace}
        if len(by_id) != len(trace) or any(parent not in by_id for event in trace for parent in event.causal_parents):
            raise PermissionError("certificate trace has missing causal parents")
        if digest([event.event_digest for event in trace]) != certificate.causal_trace_root:
            raise PermissionError("certificate causal trace root does not match events")
        if digest(sorted(event.node_instance_id for event in trace)) != certificate.realised_graph_digest:
            raise PermissionError("certificate realised graph does not match events")
        if manifest is not None:
            if manifest.graph_digest != certificate.graph_digest:
                raise PermissionError("certificate manifest does not match the certified graph")
            monitor = TransitionMonitor(manifest, joins=manifest.joins)
            states = {}
            for event in trace:
                state = states.get(event.iteration, monitor.initial_state(iteration=event.iteration))
                try:
                    # External producers are represented by a durable
                    # reservation followed by their signed completion.  Replay
                    # the same two-stage state machine; treating a completion
                    # as a fresh transition would let a certificate validate a
                    # trace that the scheduler itself could never admit.
                    if event.payload.get("lifecycle") == "started":
                        states[event.iteration] = monitor.reserve(state, event.template_node_id, event.payload)
                    elif event.template_node_id in state.started:
                        states[event.iteration] = monitor.complete(state, event.template_node_id, event.payload)
                    else:
                        states[event.iteration] = monitor.advance(state, event.template_node_id, event.payload)
                except PermissionError as exc:
                    raise PermissionError("certificate trace violates the compiled transition graph") from exc
        expected_terminal = {
            "accept": "decision.accept",
            "escalate": "decision.escalate",
        }.get(certificate.final_decision)
        if expected_terminal is not None and (not trace or trace[-1].template_node_id != expected_terminal):
            raise PermissionError("certificate terminal decision is absent from the causal trace")


def certificate_from_trace(
    *,
    run_id: str,
    contract_digest: str,
    graph_digest: str,
    harness_bundle_digest: str,
    initial_workspace_digest: str,
    final_workspace_digest: str,
    final_decision: str,
    events: Iterable[CausalEvent],
    validator_version: str = "vloop.execution-certificate.v1",
    key_id: str = "execution-validator",
) -> ExecutionCertificate:
    trace = tuple(events)
    return ExecutionCertificate(
        run_id=run_id,
        contract_digest=contract_digest,
        graph_digest=graph_digest,
        realised_graph_digest=digest(sorted(event.node_instance_id for event in trace)),
        harness_bundle_digest=harness_bundle_digest,
        initial_workspace_digest=initial_workspace_digest,
        final_workspace_digest=final_workspace_digest,
        causal_trace_root=digest([event.event_digest for event in trace]),
        authority_proof_root=digest(sorted(event.authorization_ref for event in trace if event.authorization_ref)),
        evidence_proof_root=digest(sorted(reference for event in trace for reference in event.receipt_refs)),
        final_decision=final_decision,
        validator_version=validator_version,
        key_id=key_id,
    )
