"""Bounded live smoke test for the advisory neural verifier.

It sends only a synthetic task, digests, and hard-check status to the external
model. No repository source, secrets, tool output, or credentials are sent.
"""

from __future__ import annotations

import json

from .models import (
    ActionIntent,
    ActionRule,
    CheckResult,
    CheckStatus,
    Effect,
    ExecutionObservation,
    Provenance,
    TaskContract,
    VerificationReport,
)
from .neural_verifier import OpenAICompatibleDiagnosticBackend, ShadowNeuralVerifier


def main() -> None:
    contract = TaskContract(
        goal="Assess a synthetic successful isolated command",
        success_conditions=("execution and isolation checks pass",),
        allowed_actions=(ActionRule("command.run", Effect.EXECUTE, "/workspace"),),
    )
    intent = ActionIntent(
        tool="command.run",
        effect=Effect.EXECUTE,
        target="/workspace/demo",
        arguments={"command": ["/bin/true"]},
        provenance=(Provenance.USER,),
        explanation="synthetic bounded smoke case",
        contract_id=contract.contract_id,
        contract_version=contract.version,
    )
    observation = ExecutionObservation(
        True,
        0,
        "",
        "",
        {"result": "synthetic-sha256"},
        {
            "executor": "firecracker",
            "isolation": "microvm",
            "config_digest": "synthetic-config",
            "guest_manifest_digest": "synthetic-manifest",
            "rootfs_read_only": True,
            "network_enabled": False,
        },
    )
    report = VerificationReport(
        CheckStatus.PASS,
        CheckStatus.PASS,
        CheckStatus.PASS,
        CheckStatus.PASS,
        (
            CheckResult("execution", CheckStatus.PASS, {}),
            CheckResult("firecracker-isolation", CheckStatus.PASS, {}),
        ),
    )
    diagnostic = ShadowNeuralVerifier(OpenAICompatibleDiagnosticBackend()).diagnose(
        contract=contract,
        intent=intent,
        observation=observation,
        hard_report=report,
    )
    print(json.dumps(diagnostic.ledger_payload(), sort_keys=True))


if __name__ == "__main__":
    main()
