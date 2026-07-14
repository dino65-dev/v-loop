"""Self-contained verified-run demo with no external model or shell execution."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

from .completion import RequiredChecksFinalVerifier
from .controller import VerifiedLoop
from .ledger import EvidenceLedger
from .models import (
    ActionIntent,
    ActionRule,
    CheckResult,
    CheckStatus,
    Effect,
    ExecutionObservation,
    Provenance,
    TaskContract,
)
from .policy import PolicyGate
from .verifiers import CallableVerifier, ExecutionVerifier, HybridVerifier


class DemoPlanner:
    def propose(self, *, contract: TaskContract, history: tuple[dict, ...]) -> ActionIntent:
        return ActionIntent(
            tool="demo.check",
            effect=Effect.READ,
            target="/demo/input",
            arguments={},
            provenance=(Provenance.USER,),
            explanation="Read the bounded demo input.",
            contract_id=contract.contract_id,
            contract_version=contract.version,
        )


class DemoExecutor:
    def execute(self, intent: ActionIntent) -> ExecutionObservation:
        return ExecutionObservation(True, 0, "checked", "", {"result": "demo"})


def main() -> None:
    contract = TaskContract(
        goal="Verify a bounded demo operation",
        success_conditions=("deterministic check passes",),
        allowed_actions=(ActionRule("demo.check", Effect.READ, "/demo"),),
    )
    quality = CallableVerifier(
        "quality",
        "quality",
        lambda _contract, _observation: CheckResult("quality", CheckStatus.PASS, {"metric": 1}),
    )
    with TemporaryDirectory() as directory:
        ledger = EvidenceLedger(Path(directory) / "ledger.db")
        loop = VerifiedLoop(
            contract=contract,
            planner=DemoPlanner(),
            gate=PolicyGate(contract, signing_key=b"d" * 32),
            executor=DemoExecutor(),
            verifier=HybridVerifier([ExecutionVerifier(), quality]),
            ledger=ledger,
            final_verifier=RequiredChecksFinalVerifier(
                {"deterministic check passes": ("execution",)}
            ),
        )
        print(f"decision={loop.run().value}; ledger_valid={ledger.verify_chain()}")


if __name__ == "__main__":
    main()
