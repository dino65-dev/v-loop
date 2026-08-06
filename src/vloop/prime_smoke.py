"""Live, bounded smoke test for the experimental RLM intelligence plane.

Only synthetic data is sent to the configured model endpoint.  The result is
an advisory reasoning artifact and no V-Loop executor, ledger, or promotion
surface is constructed.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from .canonical import digest
from .context import ContextEngine, ContextItem, ContextTrust, EnvironmentFingerprint
from .models import ActionRule, Effect, TaskContract
from .programmable_context import ProgrammableContextStore
from .reasoning_sessions import ReasoningSessionStore
from .rlm_protocol import RLMReasoningRequest
from .rlm_reasoning_node import RLMNodePolicy, RLMReasoningNode
from .rlm_worker import OpenAICompatibleRLMWorker, RLMWorkerPolicy


def main() -> None:
    contract = TaskContract(
        "Summarize a synthetic, read-only technical note", ("accurate summary",),
        (ActionRule("repository.read", Effect.READ, "/workspace"),),
    )
    engine = ContextEngine()
    engine.add(ContextItem("synthetic-note", "note", "V-Loop treats model output as untrusted advisory data.", ContextTrust.TRUSTED_REPOSITORY))
    package = engine.package(contract=contract, environment=EnvironmentFingerprint({"scenario": "synthetic-rlm-smoke"}))
    context = ProgrammableContextStore.from_package(package)
    manifest = context.manifest()
    worker = OpenAICompatibleRLMWorker(policy=RLMWorkerPolicy(production_enabled=True))
    database = Path(os.environ.get("VLOOP_RLM_SMOKE_DB", "/tmp/vloop-prime-smoke.db"))
    sessions = ReasoningSessionStore(database)
    node_instance_id = digest({"smoke": "rlm-node"})
    session = sessions.create_root(
        run_id="prime-smoke", contract_digest=contract.contract_digest, graph_digest=digest({"graph": "prime-smoke"}),
        node_instance_id=node_instance_id, model_digest=worker.model_digest, context_root_digest=manifest.manifest_digest,
        token_budget=10_000, call_budget=20,
    )
    request = RLMReasoningRequest(
        run_id="prime-smoke", contract_digest=contract.contract_digest, graph_digest=digest({"graph": "prime-smoke"}),
        node_instance_id=node_instance_id, context_manifest_digest=manifest.manifest_digest,
        allowed_context_handles=manifest.handles, maximum_recursive_calls=2, maximum_tokens=1_000,
        timeout_seconds=60, model_digest=worker.model_digest, harness_digest=digest({"harness": "prime-smoke-v1"}),
        session_id=session.session_id,
    )
    artifact = RLMReasoningNode(worker, sessions, RLMNodePolicy(enabled=True)).execute(request, context)
    print(json.dumps({
        "artifact_digest": artifact.artifact_digest, "model_calls": artifact.model_calls,
        "authority_ceiling": artifact.authority_ceiling.name.lower(), "summary": artifact.final_summary,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
