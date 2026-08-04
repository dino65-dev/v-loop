"""Public domain-package imports remain stable during internal refactors."""

from vloop.control import PolicyGate, VerifiedLoop
from vloop.evidence import EvidenceLedger, ValidatedNodeCompletion
from vloop.execution import CapabilityEnforcingExecutor, FirecrackerExecutor
from vloop.governance import HarnessRegistry, TopologyRun
from vloop.intelligence import ContextEngine, ShadowNeuralVerifier


def test_domain_namespaces_expose_their_owned_public_api() -> None:
    assert all(
        value is not None
        for value in (
            CapabilityEnforcingExecutor,
            ContextEngine,
            EvidenceLedger,
            FirecrackerExecutor,
            HarnessRegistry,
            PolicyGate,
            ShadowNeuralVerifier,
            TopologyRun,
            ValidatedNodeCompletion,
            VerifiedLoop,
        )
    )
