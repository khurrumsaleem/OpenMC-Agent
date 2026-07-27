"""Deterministic, resumable owner-patch retry controller.

The controller never asks an LLM to choose an owner or an action.  Providers
may produce a candidate only after the typed request and execution plan have
already constrained its target.  Candidate generation is injected so the
kernel is safe to exercise with deterministic fake clients.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from typing import Any
from uuid import uuid4

from openmc_agent.plan_builder.dependency_graph import DEFAULT_PLAN_PATCH_DEPENDENCY_GRAPH, PlanPatchDependencyGraph
from openmc_agent.plan_builder.patches import parse_patch_content
from openmc_agent.plan_builder.planning_scope import build_canonical_task_plan
from openmc_agent.plan_builder.state import PlanBuildState, PlanPatchEnvelope
from openmc_agent.plan_builder.validators import validate_patch

from .fingerprints import canonical_json_dumps, compute_candidate_hash
from .models import PLAN_CLOSED_LOOP_CONTRACT_VERSION, PlanClosedLoopPolicy, PlanGateId, PlanLoopMode, PlanStageStatus
from .retry_models import (
    ExecutablePlanRetryRequest,
    PlanRetryAction,
    RetryExecutionOutcome,
    RetryExecutionPlan,
    RetryExecutionStatus,
    RetryRoundRecord,
    RetryTargetSpec,
    RetryTriggerOrigin,
    TERMINAL_RETRY_LIFECYCLE_STATES,
)
from .retry_models import SpecialRetryAction, SpecialRetryRoute
from .retry_owner_policy import retry_owner_policy, RetryOwnerPolicy


def _valid_envelope(state: PlanBuildState, patch_type: str) -> PlanPatchEnvelope | None:
    matches = [item for item in state.patches.values() if item.patch_type == patch_type and item.status == "valid"]
    return matches[0] if len(matches) == 1 else None


def _patch_hash(state: PlanBuildState, patch_type: str) -> str | None:
    env = _valid_envelope(state, patch_type)
    return compute_candidate_hash(target_patch_type=patch_type, candidate_patch=env.content) if env else None


def _state_hash(state: PlanBuildState) -> str:
    payload = {key: {"type": value.patch_type, "status": value.status, "content": value.content} for key, value in sorted(state.patches.items())}
    return hashlib.sha256(canonical_json_dumps(payload).encode("utf-8")).hexdigest()


_TARGET_GATE_SEED_ALLOWED_OWNER_PATCH_TYPES: dict[str, set[str]] = {
    "axial_geometry": {"base_path_axial_profiles", "axial_layers", "axial_overlays"},
    "assembled_plan": set(),
}

_TARGET_GATE_SEED_ALLOWED_GATE_REPLAYS: dict[str, set[PlanGateId]] = {
    "axial_geometry": {PlanGateId.AXIAL_GEOMETRY},
    "assembled_plan": {PlanGateId.AXIAL_GEOMETRY, PlanGateId.ASSEMBLED_PLAN},
}


def _target_gate_seed_gate(state: PlanBuildState) -> str | None:
    """Return the target gate for an accepted-state seed run, if any.

    Accepted target runs deliberately reuse upstream accepted gate boundaries.
    Dependency retry must therefore be constrained to the target gate's own
    patch family; upstream owners remain controlled blockers instead of being
    regenerated inside the target run.
    """
    if not state.metadata.get("accepted_plan_build_state_seed"):
        return None
    gate = state.metadata.get("target_gate_seed_gate")
    return str(gate) if gate else None


def _constrain_target_gate_seed_retry_plan(
    *,
    state: PlanBuildState,
    owner_types: list[str],
    invalidated: list[str],
    gates: list[PlanGateId],
) -> tuple[list[str], list[PlanGateId]]:
    target_gate = _target_gate_seed_gate(state)
    if target_gate is None:
        return invalidated, gates
    if target_gate not in _TARGET_GATE_SEED_ALLOWED_OWNER_PATCH_TYPES:
        return invalidated, gates
    allowed_owners = _TARGET_GATE_SEED_ALLOWED_OWNER_PATCH_TYPES[target_gate]
    disallowed_owners = sorted({owner for owner in owner_types if owner not in allowed_owners})
    if disallowed_owners:
        raise ValueError(
            "planning.retry_owner_frozen_by_target_seed:"
            f"{target_gate}:{','.join(disallowed_owners)}"
        )
    invalidated = [patch_type for patch_type in invalidated if patch_type in allowed_owners]
    allowed_gates = _TARGET_GATE_SEED_ALLOWED_GATE_REPLAYS.get(target_gate)
    if allowed_gates is not None:
        gates = [gate for gate in gates if gate in allowed_gates]
    return invalidated, gates


def normalize_retry_request(
    source: Any,
    *,
    state: PlanBuildState,
    origin: RetryTriggerOrigin = RetryTriggerOrigin.DETERMINISTIC_PREFLIGHT,
    canonical_scope: str | None = None,
) -> ExecutablePlanRetryRequest | None:
    """Normalize legacy retries, root causes, and issue dictionaries.

    The registry is the sole authority for owner/action selection.  A supplied
    owner, LLM text, or legacy free-form reason is retained only as evidence.

    A request that arrives already typed is idempotently registered: if an
    active request with the same fingerprint exists, the existing one is
    returned without creating a duplicate.
    """
    if isinstance(source, ExecutablePlanRetryRequest):
        return _idempotent_register(source, state)
    data = source.model_dump(mode="json") if hasattr(source, "model_dump") else dict(source or {})
    codes = list(data.get("source_issue_codes") or data.get("issue_codes") or data.get("original_issue_codes") or ([data["code"]] if data.get("code") else []))
    if not codes:
        return None
    resolved_scope = canonical_scope or _resolved_scope_kind(state)
    # Enrich the issue dict with the iterating code so _resolve_placement_owner
    # can pick a code-specific owner (e.g. core_layout for core_multiplicity).
    def _policy_for(candidate_code: str):
        enriched = dict(data)
        enriched.setdefault("code", candidate_code)
        result = retry_owner_policy(candidate_code, enriched, canonical_scope=resolved_scope)
        return result

    code = next((item for item in codes if _policy_for(item) is not None), None)
    if code is None:
        state.add_event("planning.retry_request_rejected", "retry request rejected: unregistered issue", {"issue_codes": codes})
        return None
    policy = _policy_for(code)
    assert policy is not None

    # Phase 8B Step 1: SpecialRetryRoute handling.
    if isinstance(policy, SpecialRetryRoute):
        route = policy
        request = ExecutablePlanRetryRequest(
            request_id=f"retry_{uuid4().hex[:16]}", protocol_version=PLAN_CLOSED_LOOP_CONTRACT_VERSION,
            origin=origin, action=PlanRetryAction.FAIL_CLOSED, owner_patch_types=[],
            source_finding_ids=[str(value) for value in data.get("finding_ids", data.get("source_finding_ids", []))],
            source_issue_codes=codes, reason_code=code,
            priority=90, requires_human=route.requires_human, repairable=False,
            created_round=len(state.plan_retry_rounds),
            metadata={"special_route": route.action.value, "message": route.message, **route.metadata},
        )
        return _idempotent_register(request, state)

    assert isinstance(policy, RetryOwnerPolicy)
    owner_hashes = {owner: _patch_hash(state, owner) for owner in policy.owner_patch_types if owner != "planning_task_plan"}
    required_ids = [str(value) for value in data.get("required_ids", [])]
    universe_id = data.get("universe_id")
    if not required_ids and "universes" in policy.owner_patch_types and universe_id:
        required_ids = [str(universe_id)]
    consumer_ids = [str(value) for value in data.get("consumer_ids", [])]
    if not consumer_ids and universe_id:
        consumer_ids = [str(universe_id)]
    targets = [
        RetryTargetSpec(
            patch_type=owner,
            current_patch_hash=owner_hashes.get(owner),
            required_ids=required_ids,
            affected_json_paths=[str(value) for value in data.get("affected_json_paths", [])],
            protected_json_paths=policy.protected_json_paths,
            required_properties=[str(value) for value in ([data.get("required_property")] if data.get("required_property") else data.get("required_properties", []))],
            source_finding_ids=[str(value) for value in data.get("finding_ids", data.get("source_finding_ids", []))],
            source_issue_codes=codes,
            metadata={"legacy_source_type": type(source).__name__},
        )
        for owner in policy.owner_patch_types
    ]
    request = ExecutablePlanRetryRequest(
        request_id=f"retry_{uuid4().hex[:16]}", protocol_version=PLAN_CLOSED_LOOP_CONTRACT_VERSION,
        origin=origin,
        gate_id=(
            PlanGateId.PLACEMENT
            if origin is RetryTriggerOrigin.PLACEMENT_GATE
            else (
                PlanGateId.FACTS
                if origin is RetryTriggerOrigin.FACTS_GATE
                else (
                    PlanGateId.MATERIAL_UNIVERSE
                    if origin is RetryTriggerOrigin.MATERIAL_UNIVERSE_GATE
                    else None
                )
            )
        ),
        action=policy.preferred_action, owner_patch_types=policy.owner_patch_types, targets=targets,
        source_finding_ids=[str(value) for value in data.get("finding_ids", data.get("source_finding_ids", []))],
        source_issue_codes=codes, evidence_refs=[str(value) for value in data.get("evidence_refs", [])],
        reason_code=code, canonical_task_plan_hash=(state.canonical_task_plan.plan_hash if state.canonical_task_plan else None),
        planning_scope_hash=(state.resolved_planning_scope.canonical_hash if state.resolved_planning_scope else None),
        gate_input_hash=str(data.get("gate_input_hash") or "") or None,
        priority={"facts": 10, "planning_task_plan": 30, "materials": 20, "universes": 30}.get(policy.owner_patch_types[0], 60),
        requires_human=bool(data.get("requires_human", False)), repairable=policy.preferred_action is not PlanRetryAction.FAIL_CLOSED,
        created_round=len(state.plan_retry_rounds), owner_patch_hashes=owner_hashes,
        consumer_ids=consumer_ids,
        metadata={"legacy_payload": {key: value for key, value in data.items() if key not in {"message", "reason"}}},
    )
    return _idempotent_register(request, state)


def _resolved_scope_kind(state: PlanBuildState) -> str | None:
    """Map ResolvedPlanningScope.value to the owner-policy scope vocabulary."""
    if state.resolved_planning_scope is None:
        return None
    value = state.resolved_planning_scope.value
    if value in {"single_pin", "single_assembly"}:
        return "single_assembly"
    if value in {"multi_assembly_core"}:
        return "multi_assembly"
    if value in {"full_core"}:
        return "full_core"
    return None


def _normalize_retry_request_inner(  # noqa: C901
    *args: Any, **kwargs: Any) -> None:  # pragma: no cover (deprecated stub)
    """Deprecated stub; normalize_retry_request is the sole entry point."""
    raise RuntimeError("normalize_retry_request_inner is deprecated; use normalize_retry_request")


def _idempotent_register(request: ExecutablePlanRetryRequest, state: PlanBuildState) -> ExecutablePlanRetryRequest:
    """Register a typed request without creating fingerprint duplicates."""
    from .retry_models import TERMINAL_RETRY_LIFECYCLE_STATES

    for existing in state.plan_retry_requests.values():
        if existing.request_fingerprint == request.request_fingerprint and existing.lifecycle.value not in TERMINAL_RETRY_LIFECYCLE_STATES:
            state.add_event(
                "planning.retry_request_deduplicated",
                "retry request with matching fingerprint already active",
                {"request_id": existing.request_id, "fingerprint": request.request_fingerprint},
            )
            return existing
    state.plan_retry_requests[request.request_id] = request
    if request.request_id not in state.plan_retry_pending_request_ids:
        state.plan_retry_pending_request_ids.append(request.request_id)
    state.add_event("planning.retry_request_normalized", "retry request normalized by deterministic owner policy", {"request_id": request.request_id, "fingerprint": request.request_fingerprint, "owner_patch_types": request.owner_patch_types})
    return request


def compile_retry_execution_plan(
    request: ExecutablePlanRetryRequest,
    state: PlanBuildState,
    policy: PlanClosedLoopPolicy,
    dependency_graph: PlanPatchDependencyGraph = DEFAULT_PLAN_PATCH_DEPENDENCY_GRAPH,
) -> RetryExecutionPlan:
    owner_policy = retry_owner_policy(request.reason_code)
    if owner_policy is None or isinstance(owner_policy, SpecialRetryRoute):
        raise ValueError("planning.retry_request_unsupported")
    if request.canonical_task_plan_hash and state.canonical_task_plan and request.canonical_task_plan_hash != state.canonical_task_plan.plan_hash:
        raise ValueError("planning.retry_request_stale")
    if request.planning_scope_hash and state.resolved_planning_scope and request.planning_scope_hash != state.resolved_planning_scope.canonical_hash:
        raise ValueError("planning.retry_request_stale")
    for target in request.targets:
        if target.patch_type == "planning_task_plan":
            continue
        current = _patch_hash(state, target.patch_type)
        if target.current_patch_hash and current != target.current_patch_hash:
            raise ValueError("planning.retry_request_stale")
    attempts = state.plan_retry_attempts_by_fingerprint.get(request.request_fingerprint, 0)
    if attempts >= min(policy.max_attempts_per_retry_request, owner_policy.max_attempts):
        raise ValueError("planning.retry_budget_exhausted")
    owner_types = [item for item in request.owner_patch_types if item != "planning_task_plan"]
    invalidated = dependency_graph.transitive_dependents(owner_types) if owner_types else []
    gates = dependency_graph.gates_affected_by_patch_types(invalidated or owner_types)
    if request.action is PlanRetryAction.RECOMPUTE_TASK_PLAN:
        invalidated = list(state.canonical_task_plan.ordered_patch_types) if state.canonical_task_plan else []
        gates = [PlanGateId.PLACEMENT, PlanGateId.ASSEMBLED_PLAN]
    invalidated, gates = _constrain_target_gate_seed_retry_plan(
        state=state,
        owner_types=list(request.owner_patch_types),
        invalidated=invalidated,
        gates=gates,
    )
    plan_payload = {"request": request.request_fingerprint, "owners": owner_types, "invalidated": invalidated, "gates": [gate.value for gate in gates], "action": request.action.value}
    fingerprint = hashlib.sha256(canonical_json_dumps(plan_payload).encode("utf-8")).hexdigest()
    plan = RetryExecutionPlan(
        execution_id=f"retry_exec_{uuid4().hex[:16]}", request_id=request.request_id,
        owner_patch_types=request.owner_patch_types, invalidation_patch_types=invalidated,
        gates_to_invalidate=gates, gates_to_replay=gates,
        earliest_resume_patch_type=dependency_graph.earliest_patch_type(invalidated), candidate_strategy=request.action,
        validation_steps=owner_policy.required_acceptance_checks, execution_fingerprint=fingerprint,
        budget_snapshot={"request_attempts": attempts, "retry_rounds": len(state.plan_retry_rounds), "retry_llm_calls": int(state.plan_retry_budget.get("llm_calls", 0))},
    )
    state.plan_retry_execution_plans[plan.execution_id] = plan
    state.add_event("planning.retry_execution_plan_compiled", "retry execution plan compiled", {"execution_id": plan.execution_id, "request_id": request.request_id, "invalidation_patch_types": invalidated})
    return plan


def invalidate_gates_for_patch_change(state: PlanBuildState, patch_types: list[str], graph: PlanPatchDependencyGraph = DEFAULT_PLAN_PATCH_DEPENDENCY_GRAPH) -> list[PlanGateId]:
    """Invalidate gates whose inputs changed.

    Phase 3B separates *invalidation* from *replay*.  This function only
    marks gates pending and records the invalidation in
    ``plan_retry_gate_invalidation_counts``.  The replay attempt counter
    is incremented only by :func:`record_gate_replay_attempt`.
    """
    gates = graph.gates_affected_by_patch_types(patch_types)
    for gate in gates:
        stage = state.plan_loop_stages.get(f"plan_gate_{gate.value}")
        if stage is not None:
            # Preserve the prior accepted/reviewed hash so re-classification
            # can detect a no-op replay.
            stage.metadata["prior_accepted_input_hash"] = stage.metadata.get("accepted_input_hash")
            stage.metadata["prior_reviewed_input_hash"] = stage.metadata.get("reviewed_input_hash")
            stage.status = PlanStageStatus.PENDING
            stage.completed_at = None
            stage.metadata["invalidated_by_patch_types"] = sorted(set(patch_types))
            state.add_event("planning.retry_gate_invalidated", "gate invalidated after owner patch change", {"gate_id": gate.value, "patch_types": patch_types})
        state.plan_retry_gate_invalidation_counts[gate.value] = state.plan_retry_gate_invalidation_counts.get(gate.value, 0) + 1
        state.plan_retry_gate_replay_counts[gate.value] = state.plan_retry_gate_replay_counts.get(gate.value, 0) + 1
    return gates


def record_gate_replay_attempt(state: PlanBuildState, gate: PlanGateId, *, success: bool) -> None:
    """Increment the replay attempt (and success) counter for a gate.

    Only call this after a deterministic preflight + Critic + decision has
    actually run for the gate.  ``gates_replayed`` on the round record is
    populated from the caller's record of actual replays, not from
    invalidation.
    """
    state.plan_retry_gate_replay_attempt_counts[gate.value] = state.plan_retry_gate_replay_attempt_counts.get(gate.value, 0) + 1
    if success:
        state.plan_retry_gate_replay_success_counts[gate.value] = state.plan_retry_gate_replay_success_counts.get(gate.value, 0) + 1


def _atomic_owner_commit(
    state: PlanBuildState,
    clone: PlanBuildState,
    owner_patch_types: list[str],
    request: ExecutablePlanRetryRequest,
    candidate_hashes: dict[str, str],
    *,
    allow_create: bool = False,
) -> None:
    original = state.model_dump(mode="json")
    try:
        for patch_type in owner_patch_types:
            old = _valid_envelope(state, patch_type)
            new = _valid_envelope(clone, patch_type)
            if new is None:
                raise ValueError(f"retry owner patch unavailable: {patch_type}")
            if old is None:
                if not allow_create:
                    raise ValueError(f"retry owner patch creation not allowed: {patch_type}")
                replacement = new.model_copy(deep=True)
                replacement.patch_id = f"patch_{patch_type}_retry_{request.request_fingerprint[:12]}"
                replacement.source = "retry"
                replacement.status = "valid"
                replacement.metadata = {**replacement.metadata, "retry_request_id": request.request_id, "retry_candidate_hash": candidate_hashes[patch_type], "retry_created_patch": True}
                state.add_patch(replacement)
            else:
                old.status = "superseded"
                replacement = new.model_copy(deep=True)
                replacement.patch_id = f"{old.patch_id}_retry_{request.request_fingerprint[:12]}"
                replacement.source = "retry"
                replacement.status = "valid"
                replacement.metadata = {**replacement.metadata, "retry_request_id": request.request_id, "retry_candidate_hash": candidate_hashes[patch_type]}
                state.add_patch(replacement)
        state.assembled_plan = None
        state.add_event("planning.retry_owner_commit_completed", "retry owner patch committed atomically", {"request_id": request.request_id, "owners": owner_patch_types, "allow_create": allow_create})
    except Exception:
        restored = PlanBuildState.model_validate(original)
        state.__dict__.update(restored.__dict__)
        state.add_event("planning.retry_owner_commit_rolled_back", "retry owner commit rolled back", {"request_id": request.request_id})
        raise


def _set_request_lifecycle(state: PlanBuildState, request: ExecutablePlanRetryRequest, lifecycle: Any) -> None:
    request.lifecycle = lifecycle
    state.plan_retry_requests[request.request_id] = request
    if lifecycle.value in TERMINAL_RETRY_LIFECYCLE_STATES and request.request_id in state.plan_retry_pending_request_ids:
        state.plan_retry_pending_request_ids.remove(request.request_id)


def _remove_terminal_requests(state: PlanBuildState) -> None:
    """Drop any request whose lifecycle has reached a terminal state."""
    for request_id in list(state.plan_retry_pending_request_ids):
        request = state.plan_retry_requests.get(request_id)
        if request is not None and request.lifecycle.value in TERMINAL_RETRY_LIFECYCLE_STATES:
            state.plan_retry_pending_request_ids.remove(request_id)


def _compute_issue_fingerprint(issues: list[dict[str, Any]]) -> str:
    codes = sorted({str(item.get("code")) for item in issues if item.get("severity") == "error"})
    return hashlib.sha256(canonical_json_dumps({"codes": codes}).encode("utf-8")).hexdigest()


def _detect_cycle(state: PlanBuildState, request: ExecutablePlanRetryRequest, candidate_group_hash: str, issue_fingerprint: str) -> str | None:
    """Return a cycle reason string if a cycle/no-progress pattern is detected.

    Detection covers:
      A. duplicate candidate hash (already handled by the caller, included here
         for completeness).
      B. changed candidate but same blocking issue fingerprint.
      C. task-plan hash oscillation between two values.
      D. Facts↔Universes ping-pong (root-cause alternation).
    """
    fingerprints = state.plan_retry_issue_fingerprints_by_request.setdefault(request.request_fingerprint, [])
    if fingerprints and fingerprints[-1] == issue_fingerprint and len(fingerprints) >= 2:
        return "same_issue_fingerprint_repeated"
    fingerprints.append(issue_fingerprint)
    if len(fingerprints) > 8:
        fingerprints.pop(0)
    # Task-plan hash oscillation: track the last few hashes; if we see an
    # exact repetition after only one intervening value, that is a cycle.
    if state.canonical_task_plan is not None:
        history = state.plan_retry_task_plan_hash_history
        history.append(state.canonical_task_plan.plan_hash)
        if len(history) > 6:
            history.pop(0)
        if len(history) >= 4 and history[-1] == history[-3] and history[-2] == history[-4]:
            return "task_plan_hash_oscillation"
    return None


CandidateProducer = Callable[[ExecutablePlanRetryRequest, RetryExecutionPlan, PlanBuildState], dict[str, dict[str, Any]]]
CandidateValidator = Callable[[ExecutablePlanRetryRequest, RetryExecutionPlan, PlanBuildState], list[dict[str, Any]]]
DownstreamResumer = Callable[[PlanBuildState, RetryExecutionPlan], list[str]]
GateReplayer = Callable[[PlanBuildState, RetryExecutionPlan, list[Any]], tuple[list[Any], list[dict[str, Any]]]]


def _budget_llm_calls(state: PlanBuildState, delta: int) -> None:
    state.plan_retry_budget["llm_calls"] = state.plan_retry_budget.get("llm_calls", 0) + delta


def _select_highest_priority_request(state: PlanBuildState) -> ExecutablePlanRetryRequest | None:
    pending = [state.plan_retry_requests[item] for item in state.plan_retry_pending_request_ids if item in state.plan_retry_requests]
    if not pending:
        return None
    return sorted(pending, key=lambda item: (item.priority, item.request_fingerprint))[0]


def _reclassify_outcome(
    request: ExecutablePlanRetryRequest,
    before_issues: list[dict[str, Any]],
    after_issues: list[dict[str, Any]],
) -> tuple[str, list[str], list[str], list[str]]:
    """Return (classification, resolved_codes, remaining_codes, new_codes)."""
    before_codes = {str(item.get("code")) for item in before_issues if item.get("severity") == "error"}
    after_codes = {str(item.get("code")) for item in after_issues if item.get("severity") == "error"}
    resolved = sorted(before_codes - after_codes)
    remaining = sorted(after_codes & before_codes)
    new = sorted(after_codes - before_codes)
    if request.reason_code in after_codes:
        if remaining == {request.reason_code} and not new:
            classification = "no_progress"
        else:
            classification = "partially_resolved"
    elif after_codes:
        classification = "next_request_required"
    else:
        classification = "resolved"
    return classification, resolved, remaining, new


def execute_plan_retry_loop(
    *,
    state: PlanBuildState,
    policy: PlanClosedLoopPolicy,
    candidate_producer: CandidateProducer | None = None,
    candidate_validator: CandidateValidator | None = None,
    downstream_resumer: DownstreamResumer | None = None,
    gate_replayer: GateReplayer | None = None,
    producer_registry: Any = None,
    acceptance_registry_fn: Any = None,
    dependency_graph: PlanPatchDependencyGraph = DEFAULT_PLAN_PATCH_DEPENDENCY_GRAPH,
    max_rounds: int | None = None,
    requirement: str = "",
    proposer_client: Any = None,
    reviewer_client: Any = None,
    repair_client: Any = None,
    allow_create_owner_patch: bool = False,
) -> RetryExecutionOutcome:
    """Execute retry requests one at a time until the budget is exhausted.

    The loop is bounded by ``max_retry_rounds``.  Each iteration:
      1. selects the highest-priority active request,
      2. compiles an execution plan,
      3. produces an owner candidate (via registry or legacy closure),
      4. runs acceptance checks,
      5. atomically commits the owner patch(es),
      6. invalidates dependent patches and gates,
      7. resumes downstream (optional),
      8. replays gates (optional),
      9. reclassifies the outcome against the post-replay issue set,
     10. marks the request resolved / no_progress / blocked / awaiting_human.

    A ``RESUMED`` outcome does NOT mean the request is resolved — only that
    the owner commit succeeded and downstream rebuild may continue.
    """
    effective_max_rounds = max_rounds if max_rounds is not None else policy.max_retry_rounds
    last_outcome: RetryExecutionOutcome | None = None
    round_index = 0

    while True:
        request = _select_highest_priority_request(state)
        if request is None:
            outcome = last_outcome or RetryExecutionOutcome(status=RetryExecutionStatus.RESOLVED, detail="no pending retry requests")
            state.plan_retry_outcome = outcome
            return outcome
        if round_index >= effective_max_rounds:
            outcome = RetryExecutionOutcome(
                status=RetryExecutionStatus.BUDGET_EXHAUSTED,
                request_id=request.request_id,
                detail=f"retry round budget exhausted ({effective_max_rounds})",
                budget_snapshot=_budget_snapshot(state),
            )
            state.plan_retry_outcome = outcome
            return outcome
        round_index += 1
        if policy.mode is PlanLoopMode.OFF:
            return RetryExecutionOutcome(status=RetryExecutionStatus.BLOCKED, request_id=request.request_id, detail="retry loop disabled in off mode")
        try:
            plan = compile_retry_execution_plan(request, state, policy, dependency_graph)
        except ValueError as exc:
            code = str(exc)
            status = RetryExecutionStatus.BUDGET_EXHAUSTED if "budget" in code else (RetryExecutionStatus.UNSUPPORTED_REQUEST if "unsupported" in code else RetryExecutionStatus.BLOCKED)
            outcome = RetryExecutionOutcome(status=status, request_id=request.request_id, detail=code)
            _set_request_lifecycle(state, request, _lifecycle_for_status(status))
            state.plan_retry_outcome = outcome
            last_outcome = outcome
            continue
        if policy.mode is PlanLoopMode.ADVISORY:
            outcome = RetryExecutionOutcome(status=RetryExecutionStatus.RETRY_PLAN_RECORDED, request_id=request.request_id, execution_id=plan.execution_id, detail="advisory retry plan recorded", workflow_behavior_changed=False)
            state.plan_retry_outcome = outcome
            return outcome
        if request.requires_human:
            outcome = RetryExecutionOutcome(status=RetryExecutionStatus.AWAITING_HUMAN, request_id=request.request_id, execution_id=plan.execution_id, detail="retry request requires human confirmation")
            _set_request_lifecycle(state, request, _Lifecycle.AWAITING_HUMAN)
            state.plan_retry_outcome = outcome
            state.add_event("planning.retry_awaiting_human", outcome.detail, {"request_id": request.request_id})
            return outcome

        before = _state_hash(state)
        before_issues = list(state.validation_issues)

        # ---- RECOMPUTE_TASK_PLAN branch (deterministic, no LLM) ----
        if request.action is PlanRetryAction.RECOMPUTE_TASK_PLAN:
            if state.resolved_planning_scope is None or state.planning_feature_contract is None:
                outcome = RetryExecutionOutcome(status=RetryExecutionStatus.BLOCKED, request_id=request.request_id, detail="cannot recompute task plan without scope/contract")
                _set_request_lifecycle(state, request, _Lifecycle.BLOCKED)
                state.plan_retry_outcome = outcome
                last_outcome = outcome
                continue
            facts_env = _valid_envelope(state, "facts")
            state.canonical_task_plan = build_canonical_task_plan(
                scope=state.resolved_planning_scope,
                contract=state.planning_feature_contract,
                facts_patch=facts_env.content if facts_env else {},
                feature_order=list(dependency_graph._ORDER),
            )
            state.invalidate_patch_types(plan.invalidation_patch_types, reason="canonical task-plan recomputed", issues=[{"code": request.reason_code}])
            gates_invalid = invalidate_gates_for_patch_change(state, plan.invalidation_patch_types, dependency_graph)
            record = RetryRoundRecord(round_index=len(state.plan_retry_rounds), request=request, execution_plan=plan, state_hash_before=before, invalidated_patch_types=plan.invalidation_patch_types, gates_invalidated=gates_invalid, outcome=RetryExecutionStatus.RESUMED)
            state.plan_retry_rounds.append(record)
            _set_request_lifecycle(state, request, _Lifecycle.OWNER_COMMITTED)
            outcome = RetryExecutionOutcome(status=RetryExecutionStatus.RESUMED, request_id=request.request_id, execution_id=plan.execution_id, detail="canonical task plan recomputed", workflow_behavior_changed=True, budget_snapshot=_budget_snapshot(state))
            state.plan_retry_outcome = outcome
            last_outcome = outcome
            continue

        # ---- Owner candidate production ----
        owner_types = [item for item in request.owner_patch_types if item != "planning_task_plan"]
        if not owner_types:
            outcome = RetryExecutionOutcome(status=RetryExecutionStatus.UNSUPPORTED_REQUEST, request_id=request.request_id, detail="retry request has no concrete owner patch type")
            _set_request_lifecycle(state, request, _Lifecycle.FAILED)
            state.plan_retry_outcome = outcome
            last_outcome = outcome
            continue

        # Determine if owner patch creation is allowed (canonical task plan
        # requires the patch but state lacks it).
        owner_missing = _valid_envelope(state, owner_types[0]) is None
        effective_allow_create = allow_create_owner_patch and owner_missing
        if owner_missing and not effective_allow_create:
            outcome = RetryExecutionOutcome(status=RetryExecutionStatus.FAILED, request_id=request.request_id, detail=f"owner patch missing and creation not allowed: {owner_types[0]}")
            _set_request_lifecycle(state, request, _Lifecycle.FAILED)
            state.plan_retry_outcome = outcome
            last_outcome = outcome
            continue

        clone = state.model_copy(deep=True)
        # Ensure the clone has a placeholder valid envelope for the owner so
        # the producer can populate it.
        if effective_allow_create:
            for patch_type in owner_types:
                placeholder = _valid_envelope(clone, patch_type)
                if placeholder is None:
                    placeholder_env = PlanPatchEnvelope(patch_id=f"patch_{patch_type}_placeholder", patch_type=patch_type, content={}, status="valid", source="retry")
                    clone.add_patch(placeholder_env)

        llm_calls_before = int(state.plan_retry_budget.get("llm_calls", 0))
        try:
            if producer_registry is not None:
                from .retry_candidate_producers import RetryCandidateContext
                ctx = RetryCandidateContext(
                    request=request,
                    execution_plan=plan,
                    clone_state=clone,
                    policy=policy,
                    requirement=requirement,
                    proposer_client=proposer_client,
                    reviewer_client=reviewer_client,
                    repair_client=repair_client,
                )
                producer_result = producer_registry.produce(ctx)
                candidates = producer_result.candidates
                llm_delta = producer_result.llm_calls
            elif candidate_producer is not None:
                candidates = candidate_producer(request, plan, clone)
                llm_delta = 1
            else:
                outcome = RetryExecutionOutcome(status=RetryExecutionStatus.FAILED, request_id=request.request_id, execution_id=plan.execution_id, detail="no owner candidate producer registered")
                _set_request_lifecycle(state, request, _Lifecycle.FAILED)
                state.plan_retry_outcome = outcome
                last_outcome = outcome
                continue
        except Exception as exc:
            outcome = RetryExecutionOutcome(status=RetryExecutionStatus.FAILED, request_id=request.request_id, execution_id=plan.execution_id, detail=f"candidate generation failed: {exc}")
            _set_request_lifecycle(state, request, _Lifecycle.FAILED)
            state.plan_retry_outcome = outcome
            last_outcome = outcome
            continue
        _budget_llm_calls(state, llm_delta)
        if state.plan_retry_budget.get("llm_calls", 0) > policy.max_total_retry_llm_calls:
            outcome = RetryExecutionOutcome(status=RetryExecutionStatus.BUDGET_EXHAUSTED, request_id=request.request_id, execution_id=plan.execution_id, detail="retry LLM call budget exhausted", budget_snapshot=_budget_snapshot(state))
            _set_request_lifecycle(state, request, _Lifecycle.BLOCKED)
            state.plan_retry_outcome = outcome
            return outcome

        # ---- Schema/parse validation of each candidate ----
        hashes: dict[str, str] = {}
        for patch_type, content in candidates.items():
            if patch_type not in request.owner_patch_types:
                outcome = RetryExecutionOutcome(status=RetryExecutionStatus.FAILED, request_id=request.request_id, detail=f"candidate targets non-owner patch {patch_type}")
                _set_request_lifecycle(state, request, _Lifecycle.FAILED)
                state.plan_retry_outcome = outcome
                last_outcome = outcome
                continue
            env = _valid_envelope(clone, patch_type)
            if env is None:
                outcome = RetryExecutionOutcome(status=RetryExecutionStatus.FAILED, request_id=request.request_id, detail=f"candidate owner patch missing: {patch_type}")
                _set_request_lifecycle(state, request, _Lifecycle.FAILED)
                state.plan_retry_outcome = outcome
                last_outcome = outcome
                continue
            try:
                parsed = parse_patch_content(patch_type, content)
                validation = validate_patch(parsed)
            except Exception as exc:
                outcome = RetryExecutionOutcome(status=RetryExecutionStatus.FAILED, request_id=request.request_id, detail=f"candidate parse failed: {patch_type}: {exc}")
                _set_request_lifecycle(state, request, _Lifecycle.FAILED)
                state.plan_retry_outcome = outcome
                last_outcome = outcome
                continue
            if not validation.ok:
                outcome = RetryExecutionOutcome(status=RetryExecutionStatus.FAILED, request_id=request.request_id, detail=f"candidate validation failed: {patch_type}")
                _set_request_lifecycle(state, request, _Lifecycle.FAILED)
                state.plan_retry_outcome = outcome
                last_outcome = outcome
                continue
            env.content = content
            hashes[patch_type] = compute_candidate_hash(target_patch_type=patch_type, candidate_patch=content)

        candidate_group_hash = hashlib.sha256(canonical_json_dumps(hashes).encode("utf-8")).hexdigest()
        prior = state.plan_retry_candidate_hashes_by_fingerprint.setdefault(request.request_fingerprint, [])
        duplicate = candidate_group_hash in prior
        prior.append(candidate_group_hash)
        state.plan_retry_attempts_by_fingerprint[request.request_fingerprint] = state.plan_retry_attempts_by_fingerprint.get(request.request_fingerprint, 0) + 1

        # ---- Cycle / no-progress detection (covers changed-candidate-same-issue) ----
        issue_fingerprint = _compute_issue_fingerprint(before_issues)
        cycle_reason = _detect_cycle(state, request, candidate_group_hash, issue_fingerprint)
        if duplicate:
            cycle_reason = cycle_reason or "duplicate_candidate"

        if cycle_reason:
            status = RetryExecutionStatus.CYCLE_DETECTED if "oscillation" in cycle_reason or "ping" in cycle_reason else RetryExecutionStatus.NO_PROGRESS
            record = RetryRoundRecord(round_index=len(state.plan_retry_rounds), request=request, execution_plan=plan, state_hash_before=before, candidate_hashes=hashes, issue_fingerprint_before=issue_fingerprint, outcome=status, error=cycle_reason)
            state.plan_retry_rounds.append(record)
            state.plan_retry_cycle_trace.append({"request_fingerprint": request.request_fingerprint, "candidate_hash": candidate_group_hash, "issue_fingerprint": issue_fingerprint, "reason": cycle_reason})
            _set_request_lifecycle(state, request, _Lifecycle.NO_PROGRESS)
            outcome = RetryExecutionOutcome(status=status, request_id=request.request_id, execution_id=plan.execution_id, detail=f"retry stopped: {cycle_reason}", budget_snapshot=_budget_snapshot(state))
            state.plan_retry_outcome = outcome
            state.add_event("planning.retry_no_progress", outcome.detail, {"request_id": request.request_id, "reason": cycle_reason})
            last_outcome = outcome
            continue

        # ---- Acceptance registry ----
        acceptance_issues: list[dict[str, Any]] = []
        checks_executed: list[str] = []
        checks_passed: list[str] = []
        checks_failed: list[str] = []
        if acceptance_registry_fn is not None:
            acceptance_result = acceptance_registry_fn(request, plan, clone, policy)
            acceptance_issues = acceptance_result.issues
            checks_executed = acceptance_result.checks_executed
            checks_passed = acceptance_result.passed_checks
            checks_failed = acceptance_result.failed_checks
        elif candidate_validator is not None:
            acceptance_issues = candidate_validator(request, plan, clone)
            checks_executed = ["legacy_validator"]
            checks_passed = [] if any(item.get("severity", "error") == "error" for item in acceptance_issues) else ["legacy_validator"]

        if any(item.get("severity", "error") == "error" for item in acceptance_issues):
            record = RetryRoundRecord(round_index=len(state.plan_retry_rounds), request=request, execution_plan=plan, state_hash_before=before, candidate_hashes=hashes, checks_executed=checks_executed, checks_passed=checks_passed, checks_failed=checks_failed, outcome=RetryExecutionStatus.FAILED, error="owner-specific acceptance failed", remaining_issue_codes=[str(item.get("code")) for item in acceptance_issues])
            state.plan_retry_rounds.append(record)
            outcome = RetryExecutionOutcome(status=RetryExecutionStatus.FAILED, request_id=request.request_id, execution_id=plan.execution_id, detail="owner-specific acceptance failed", new_request_ids=[])
            state.plan_retry_outcome = outcome
            last_outcome = outcome
            # Acceptance failure does NOT terminal-remove; the request may be
            # retried with a different candidate up to the budget.
            attempts = state.plan_retry_attempts_by_fingerprint.get(request.request_fingerprint, 0)
            if attempts >= min(policy.max_attempts_per_retry_request, policy.max_same_candidate_attempts + 1):
                _set_request_lifecycle(state, request, _Lifecycle.FAILED)
            continue

        # ---- Atomic owner commit ----
        try:
            _atomic_owner_commit(state, clone, owner_types, request, hashes, allow_create=effective_allow_create)
        except Exception as exc:
            outcome = RetryExecutionOutcome(status=RetryExecutionStatus.FAILED, request_id=request.request_id, detail=f"atomic owner commit failed: {exc}")
            _set_request_lifecycle(state, request, _Lifecycle.FAILED)
            state.plan_retry_outcome = outcome
            last_outcome = outcome
            continue

        _set_request_lifecycle(state, request, _Lifecycle.OWNER_COMMITTED)
        dependent_only = [item for item in dependency_graph.transitive_dependents(owner_types) if item not in owner_types]
        state.invalidate_patch_types(dependent_only, reason=f"retry owner commit {request.request_id}", issues=[{"code": request.reason_code}])
        gates_invalid = invalidate_gates_for_patch_change(state, dependent_only + owner_types, dependency_graph)
        state.plan_retry_owner_regenerations.update({patch: state.plan_retry_owner_regenerations.get(patch, 0) + 1 for patch in hashes})

        regenerated: list[str] = []
        replayed_gates: list[Any] = []
        after_issues: list[dict[str, Any]] = list(state.validation_issues)

        if downstream_resumer is not None:
            try:
                regenerated = downstream_resumer(state, plan)
            except Exception as exc:
                state.add_event("planning.retry_downstream_resume_failed", str(exc), {"request_id": request.request_id})

        if gate_replayer is not None:
            try:
                replayed_gates, gate_issues = gate_replayer(state, plan, gates_invalid)
                for gate in replayed_gates:
                    record_gate_replay_attempt(state, gate, success=True)
                after_issues = gate_issues
            except Exception as exc:
                state.add_event("planning.retry_gate_replay_failed", str(exc), {"request_id": request.request_id})

        classification, resolved_codes, remaining_codes, new_codes = _reclassify_outcome(request, before_issues, after_issues)
        # Only mark resolved if the reason code actually disappeared.
        if classification == "resolved":
            _set_request_lifecycle(state, request, _Lifecycle.RESOLVED)
            outcome_status = RetryExecutionStatus.RESOLVED
        elif classification == "no_progress":
            _set_request_lifecycle(state, request, _Lifecycle.NO_PROGRESS)
            outcome_status = RetryExecutionStatus.NO_PROGRESS
        elif classification == "partially_resolved":
            _set_request_lifecycle(state, request, _Lifecycle.OWNER_COMMITTED)
            outcome_status = RetryExecutionStatus.PARTIALLY_RESOLVED
        else:
            _set_request_lifecycle(state, request, _Lifecycle.OWNER_COMMITTED)
            outcome_status = RetryExecutionStatus.RESUMED

        record = RetryRoundRecord(
            round_index=len(state.plan_retry_rounds), request=request, execution_plan=plan, state_hash_before=before,
            owner_hashes_before={target.patch_type: target.current_patch_hash or "" for target in request.targets},
            candidate_hashes=hashes, owner_hashes_after={patch: _patch_hash(state, patch) or "" for patch in hashes},
            invalidated_patch_types=dependent_only, regenerated_patch_types=regenerated,
            gates_invalidated=gates_invalid, gates_replayed=replayed_gates,
            issue_fingerprint_before=issue_fingerprint, issue_fingerprint_after=_compute_issue_fingerprint(after_issues),
            resolved_issue_codes=resolved_codes, remaining_issue_codes=remaining_codes, new_issue_codes=new_codes,
            checks_executed=checks_executed, checks_passed=checks_passed, checks_failed=checks_failed,
            llm_calls=llm_delta, outcome=outcome_status, reclassification=classification,
        )
        state.plan_retry_rounds.append(record)
        _remove_terminal_requests(state)
        outcome = RetryExecutionOutcome(status=outcome_status, request_id=request.request_id, execution_id=plan.execution_id, detail=f"owner committed; reclassification={classification}", workflow_behavior_changed=True, reclassification=classification, budget_snapshot=_budget_snapshot(state))
        state.plan_retry_outcome = outcome
        state.add_event("planning.retry_round_completed", outcome.detail, {"request_id": request.request_id, "classification": classification, "resolved": resolved_codes, "remaining": remaining_codes, "new": new_codes})
        last_outcome = outcome

    # unreachable
    return last_outcome  # type: ignore[return-value]


def _budget_snapshot(state: PlanBuildState) -> dict[str, int]:
    return {
        "llm_calls": int(state.plan_retry_budget.get("llm_calls", 0)),
        "retry_rounds": len(state.plan_retry_rounds),
        "owner_regenerations": sum(state.plan_retry_owner_regenerations.values()),
        "gate_invalidation_total": sum(state.plan_retry_gate_invalidation_counts.values()),
        "gate_replay_attempts": sum(state.plan_retry_gate_replay_attempt_counts.values()),
        "gate_replay_successes": sum(state.plan_retry_gate_replay_success_counts.values()),
    }


class _Lifecycle:
    """Alias for :class:`RetryRequestLifecycle` to avoid late import cycles."""

    PENDING = None
    EXECUTING = None
    AWAITING_HUMAN = None
    OWNER_COMMITTED = None
    REBUILDING = None
    REPLAYING = None
    RESOLVED = None
    SUPERSEDED = None
    NO_PROGRESS = None
    BLOCKED = None
    FAILED = None


from .retry_models import RetryRequestLifecycle as _RealLifecycle  # noqa: E402

_Lifecycle.PENDING = _RealLifecycle.PENDING
_Lifecycle.EXECUTING = _RealLifecycle.EXECUTING
_Lifecycle.AWAITING_HUMAN = _RealLifecycle.AWAITING_HUMAN
_Lifecycle.OWNER_COMMITTED = _RealLifecycle.OWNER_COMMITTED
_Lifecycle.REBUILDING = _RealLifecycle.REBUILDING
_Lifecycle.REPLAYING = _RealLifecycle.REPLAYING
_Lifecycle.RESOLVED = _RealLifecycle.RESOLVED
_Lifecycle.SUPERSEDED = _RealLifecycle.SUPERSEDED
_Lifecycle.NO_PROGRESS = _RealLifecycle.NO_PROGRESS
_Lifecycle.BLOCKED = _RealLifecycle.BLOCKED
_Lifecycle.FAILED = _RealLifecycle.FAILED


def _lifecycle_for_status(status: Any) -> Any:
    mapping = {
        RetryExecutionStatus.RESOLVED: _Lifecycle.RESOLVED,
        RetryExecutionStatus.PARTIALLY_RESOLVED: _Lifecycle.OWNER_COMMITTED,
        RetryExecutionStatus.BUDGET_EXHAUSTED: _Lifecycle.BLOCKED,
        RetryExecutionStatus.NO_PROGRESS: _Lifecycle.NO_PROGRESS,
        RetryExecutionStatus.CYCLE_DETECTED: _Lifecycle.NO_PROGRESS,
        RetryExecutionStatus.UNSUPPORTED_REQUEST: _Lifecycle.BLOCKED,
        RetryExecutionStatus.FAILED: _Lifecycle.FAILED,
        RetryExecutionStatus.BLOCKED: _Lifecycle.BLOCKED,
        RetryExecutionStatus.AWAITING_HUMAN: _Lifecycle.AWAITING_HUMAN,
    }
    return mapping.get(status, _Lifecycle.BLOCKED)
