# V-Loop

V-Loop is a Verified Adaptive Loop Agent. The V0 implementation keeps the
model outside the trusted computing base:

    planner -> ActionIntent -> PolicyGate -> short-lived Capability -> Executor
                                                    |                  |
                                                    +---- Ledger <-----+
                                                           |
                                                       Verifiers
                                                           |
                                                 accept | repair | stop

The controller may propose an action, but only the deterministic policy gate
can authorize it and only independently recorded evidence can satisfy a task.

## What is implemented

- versioned task contracts and per-action budgets;
- default-deny reference monitor, restrictive overlapping rules, and
  Ed25519-signed, audience-bound executor capabilities;
- executor-side one-time nonce consumption and idempotency reservation;
- hash-chained SQLite evidence ledger with a transactional singleton head;
- structured pass/fail/inconclusive verifier results and deterministic
  acceptance rules, plus an explicit final-goal receipt before acceptance;
- L0 structural validation, phased verifier execution, and signed evaluator
  receipts bound to run, intent, and actual executor artifact digests;
- failure diagnosis, retry de-duplication, and terminal budget handling;
- evidence-bound memory-record schema and promotion gate;
- canonical verified-memory ledger, L0 working-state store, scoped retrieval,
  and pluggable hot/associative index adapters;
- fail-closed Bubblewrap executor interface and a development-only local
  command executor;
- Firecracker microVM launch contracts, supervisor-signed lifecycle receipts,
  and Firecracker isolation/benchmark evidence verifiers;
- advisory neural-verifier diagnostics in shadow mode, recorded separately
  from deterministic verification results;
- deterministic repair directives that consume neural output as advice only;
- bounded adversarial probes registered and executed only by protected
  evaluator code;
- specialist dispatcher requiring prior verified same-budget improvement,
  server-side role registration, and a fixed total budget;
- offline trace sanitization and model promotion gates for shadow, canary, and
  production rollout;
- authority-bounded task-contract compilation from server-owned tool catalogs;
- context and state packaging with bounded retrieval, environment fingerprints,
  and explicit untrusted-data provenance;
- optional OpenAI-compatible planner wiring for the configured BazaarLink
  endpoint and DeepSeek model.

## Safety boundary

LocalCommandExecutor is development-only; it is not an isolation boundary.
Production actions must place every raw executor behind
`CapabilityEnforcingExecutor`. The policy service retains the Ed25519 private
key; the executor holds only the public verification key plus a durable nonce
and idempotency store. It verifies audience, expiry, exact intent binding and
signature immediately before the side effect. Raw executors are not policy
enforcement points and must not be exposed to untrusted callers.

Use BubblewrapExecutor (or another separately deployed executor) with separate
credentials, read-only evaluator assets, constrained mounts, no network unless
explicitly required, and no direct ledger access.

For untrusted code, use FirecrackerExecutor with a privileged supervisor
service. The microVM rootfs is read-only, each job receives a new writable job
drive, and V0 disables network access. The rootfs must contain a guest agent
that reads the host-created manifest and writes a result document. Do not use
host shell commands or SSH keys as the guest-control plane.

`FirecrackerPreflight` checks the deployer-owned Firecracker/jailer binaries,
writable jailer chroot, guest assets, and usable `/dev/kvm` before deployment.
`FirecrackerSupervisorPlan` produces a shell-free jailer launch request, but
does not start a VM: a separately privileged supervisor must materialize the
manifest/config, prepare a fresh job drive, manage VM lifecycle, hash the
result after shutdown, destroy the drive, and return a signed supervisor
receipt. When `FirecrackerExecutor` is configured with a receipt verifier, it
rejects a result unless the receipt binds the run, intent, actual artifact,
job/manifest, fresh drive, and destruction attestation. This is intentional;
the controller never holds KVM or jailer authority.

## Quick start

    uv run --extra dev pytest
    uv run python -m vloop.demo

The optional model planner needs a secret outside the repository:

    export VLOOP_API_KEY='...'
    export VLOOP_MODEL='deepseek/deepseek-v4-flash'
    export VLOOP_MODEL_BASE_URL='https://bazaarlink.ai/api/v1'

The planner receives no authority from this configuration. Every parsed model
proposal still passes through the contract and policy gate.

## Final-goal verification

Every `VerifiedLoop` run now requires an explicit `FinalVerifier` before it can
return `accept`. The controller accumulates action evidence over the whole run.
`RequiredChecksFinalVerifier` binds each immutable success condition to named
protected checks and accepts only checks fresh for the final source-state
digest; a check from before the final source change cannot satisfy completion.
Production systems can instead supply a separate end-to-end evaluator. A
passing action report or neural diagnostic alone is never enough to complete a
task.

## Neural verifier

ShadowNeuralVerifier sends a redacted diagnostic packet to a configured model
and records only its structured output. It cannot accept a result, choose a
repair, write memory, or authorize a tool. Run the bounded live smoke test with:

    VLOOP_API_KEY='...' uv run --extra model python -m vloop.model_smoke

If the secret already exists under a differently named environment variable,
do not copy it into source code. Point V-Loop to that name for the current
process instead:

    VLOOP_API_KEY_ENV='existing_secret_variable_name' uv run --extra model python -m vloop.model_smoke

## Repair and delegation

RepairController maps hard verification categories to local repair, probing,
replanning, or escalation. The neural verifier may add evidence gaps and a
suggested stage, but cannot change that mapping. DelegationGate stays disabled
unless a recorded experiment shows a specialist beats the single-agent baseline
under the required total budget. When enabled, SpecialistDispatcher can call
only deployment-registered roles, supplies no credentials or capabilities, and
records output digests rather than raw specialist text.

ProtectedProbeRunner is the active verifier-side probe layer. It selects only
pre-registered edge-case, mutation, counterexample, or consistency probes from
hard-check categories. When configured, it also runs before acceptance and its
failure or uncertainty becomes authoritative verifier evidence. The planner
cannot provide executable probe code.

## Offline improvement

TraceDatasetBuilder exports only independently verified, sanitized run traces.
An accepted trace additionally needs a `final-goal.completed` receipt with a
passing status. Probe and context receipts are retained as labelled evidence,
not as model authority.
ModelPromotionGate requires multiple cross-domain evaluation slices with bounded
false-allow and prompt-injection escape rates before a candidate moves from
offline to shadow, shadow to canary, or canary to production. A failed safety
rate triggers rollback rather than a further self-update.

## Task contracts

TaskContractCompiler accepts requested actions only when they are a strict
subset of the server-owned ToolAuthority catalog. A model may draft the
request, but cannot broaden a path, remove an approval requirement, or modify
the forbidden-action list.

## Context and state

ContextEngine packages repository facts, tool observations, and verified memory
as provenance-labelled data under a fixed context budget. Memory conditions,
source run, confidence, expiry, and supersession metadata remain visible to the
planner. `OpenAICompatiblePlanner.propose_with_context` passes trusted and
untrusted material in separate data blocks, while runtime code—not model JSON—
assigns action provenance. Untrusted retrieval is conservatively propagated to
PolicyGate when a controller context provider is configured. The immutable
contract remains outside this package.

## Verified memory

WorkingStateStore holds only task-local L0 state. MemoryLedger owns canonical,
verified records that cite evidence-ledger hashes. MemoryService filters by
configured authorized project scope and sensitivity, then expiry and
supersession before querying a hot
index. External LightRAG and HippoRAG adapters return only memory IDs; the
ledger rehydrates and filters them, so an external index cannot inject a claim
or bypass memory-write verification.

VerifiedMemoryCommitter wires this gate into a completed controller run: it
requires a passing final-goal receipt and verifies that every memory citation is
an event hash from that run. DiagnosedFailureMemoryGate separately admits only
hard correctness or policy failures; free-form model reflections remain
non-reusable.
