"""Plan build state for incremental (patch-based) plan generation (Phase 1).

Provides the external state container that future patch-based planning phases
(Phase 2+) will use to track component tasks, patch envelopes, assembled plan,
validation issues, and a structured build log.

Design constraints
------------------
* **Pure data.**  No OpenMC, no renderer, no LLM dependencies.
* **JSON serializable.**  Can be written to transcript / trace / debug files.
* **Future-friendly.**  Helpers (``add_event``, ``add_task``, ``add_patch``,
  ``mark_patch_status``, ``get_valid_patches``, ``to_summary``) support the
  patch generation → assembly → validation cycle that later phases will add.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import Field

from openmc_agent.schemas import AgentBaseModel

from .mode import PlanningModeDecision
from .closed_loop.models import (
    ConfirmedPlanFactRecord,
    ConfirmedFactRecord,
    HumanPlanAnswer,
    HumanPlanQuestion,
    PlanLoopMode,
    PlanReviewDecision,
    PlanReviewFinding,
    PlanStageState,
)
from .planning_scope import CanonicalPatchTaskPlan, ResolvedPlanningScope, PlanningFeatureContract
from .closed_loop.retry_models import (
    ExecutablePlanRetryRequest,
    RetryExecutionPlan,
    RetryExecutionOutcome,
    RetryRoundRecord,
)


# ---------------------------------------------------------------------------
# Stable event / issue codes
# ---------------------------------------------------------------------------

EVENT_PLANNING_MODE_SELECTED: str = "planning.incremental_mode_selected"
EVENT_MONOLITHIC_MODE_SELECTED: str = "planning.monolithic_mode_selected"
EVENT_INCREMENTAL_RECOMMENDED_NOT_EXECUTED: str = "planning.incremental_recommended_but_not_executed"
EVENT_INCREMENTAL_EXECUTOR_NOT_IMPLEMENTED: str = "planning.incremental_executor_not_implemented"
EVENT_BUILD_STATE_INITIALIZED: str = "planning.build_state_initialized"
EVENT_COMPONENT_TASKS_INITIALIZED: str = "planning.component_tasks_initialized"
EVENT_PATCH_PARSED: str = "planning.patch_parsed"
EVENT_PATCH_VALIDATED: str = "planning.patch_validated"
EVENT_PATCH_INVALID: str = "planning.patch_invalid"
EVENT_ASSEMBLY_STARTED: str = "planning.assembly_started"
EVENT_ASSEMBLY_COMPLETED: str = "planning.assembly_completed"
EVENT_ASSEMBLY_FAILED: str = "planning.assembly_failed"
EVENT_PATCH_GENERATION_STARTED: str = "planning.patch_generation_started"
EVENT_PATCH_GENERATED: str = "planning.patch_generated"
EVENT_PATCH_GENERATION_FAILED: str = "planning.patch_generation_failed"
EVENT_CLOSED_LOOP_INITIALIZED: str = "planning.closed_loop_initialized"
EVENT_CLOSED_LOOP_DISABLED: str = "planning.closed_loop_disabled"
EVENT_GATE_INITIALIZED: str = "planning.gate_initialized"
EVENT_GATE_TRANSITIONED: str = "planning.gate_transitioned"
EVENT_REVIEW_FINDING_RECORDED: str = "planning.review_finding_recorded"
EVENT_REVIEW_DECISION_RECORDED: str = "planning.review_decision_recorded"
EVENT_NO_PROGRESS_DETECTED: str = "planning.no_progress_detected"
EVENT_CLOSED_LOOP_BUDGET_EXHAUSTED: str = "planning.closed_loop_budget_exhausted"
EVENT_CLOSED_LOOP_ARTIFACT_WRITTEN: str = "planning.closed_loop_artifact_written"
EVENT_CLOSED_LOOP_ARTIFACT_WARNING: str = "planning.closed_loop_artifact_warning"


# ---------------------------------------------------------------------------
# Core data models
# ---------------------------------------------------------------------------


class BuildEvent(AgentBaseModel):
    """A single event in the plan build log."""

    timestamp: str | None = None
    event_type: str
    message: str
    data: dict[str, Any] = Field(default_factory=dict)


class PlanComponentTask(AgentBaseModel):
    """A pending task for building a plan component (materials, universes, ...).

    In Phase 1 these are shallow placeholders.  Phase 2+ will flesh out the
    decomposer that populates and executes them.
    """

    task_id: str
    patch_type: str
    description: str = ""
    status: Literal["pending", "running", "valid", "invalid", "skipped", "blocked"] = (
        "pending"
    )
    dependencies: list[str] = Field(default_factory=list)
    issues: list[dict[str, Any]] = Field(default_factory=list)


class PlanPatchEnvelope(AgentBaseModel):
    """A single patch envelope produced by an LLM, deterministic code, or user.

    Phase 1 only stores patches; Phase 2+ will define patch schemas, validators,
    and the assembler that merges valid patches into ``assembled_plan``.
    """

    patch_id: str
    patch_type: str
    schema_version: str = "v1"
    content: dict[str, Any] = Field(default_factory=dict)
    source: Literal["llm", "deterministic", "user", "fixture", "repair", "retry"] = "llm"
    status: Literal["pending", "valid", "invalid", "repaired", "superseded", "skipped"] = "pending"
    issues: list[dict[str, Any]] = Field(default_factory=list)
    raw_text: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class PlanBuildState(AgentBaseModel):
    """External state container for incremental plan building.

    Holds the requirement, detected features, component task list, patches,
    assembled plan, validation issues, and a structured build log.  Designed
    to be JSON-serialized into transcript / trace / debug reports without
    depending on OpenMC or any renderer.
    """

    state_id: str
    requirement_text: str
    planning_mode: Literal["incremental"] = "incremental"

    benchmark_id: str | None = None
    selected_variant: str | None = None

    extracted_facts: dict[str, Any] = Field(default_factory=dict)
    confirmed_facts: dict[str, Any] = Field(default_factory=dict)

    component_tasks: list[PlanComponentTask] = Field(default_factory=list)
    patches: dict[str, PlanPatchEnvelope] = Field(default_factory=dict)
    patch_status: dict[str, str] = Field(default_factory=dict)

    assembled_plan: dict[str, Any] | None = None
    material_composition_report: dict[str, Any] | None = None
    material_species_resolution_report: dict[str, Any] | None = None
    validation_issues: list[dict[str, Any]] = Field(default_factory=list)
    build_log: list[BuildEvent] = Field(default_factory=list)

    # Validation-driven patch repair is deliberately independent of graph
    # retries.  These JSON-serializable ledgers prevent duplicate candidates
    # from being proposed again after resume/checkpoint restore.
    validation_repair_history: list[dict[str, Any]] = Field(default_factory=list)
    validation_repair_attempts_by_fingerprint: dict[str, int] = Field(default_factory=dict)
    validation_repair_candidate_hashes: dict[str, list[str]] = Field(default_factory=dict)
    validation_full_patch_regenerations_by_fingerprint: dict[str, int] = Field(default_factory=dict)

    # Patch-generation candidates can fail before a patch model exists (for
    # example, a schema-invalid LLM response).  Persist their hashes by the
    # upstream-context fingerprint so resume does not repeatedly call an LLM
    # for an identical rejected JSON candidate.
    patch_generation_candidate_hashes_by_context: dict[str, list[str]] = Field(default_factory=dict)
    patch_generation_attempts_by_context: dict[str, int] = Field(default_factory=dict)

    # Phase-0 plan closed-loop state.  These ledgers are deliberately
    # namespaced from validation/runtime repair so checkpoint restore preserves
    # the future review protocol without changing existing retry semantics.
    plan_loop_mode: PlanLoopMode = PlanLoopMode.OFF
    # Older persisted ledgers deliberately remain valid.  The controller
    # upgrades behaviour lazily instead of rewriting checkpoint files.
    plan_loop_contract_version: str = "0.5"
    plan_loop_policy: dict[str, Any] = Field(default_factory=dict)
    plan_loop_stages: dict[str, PlanStageState] = Field(default_factory=dict)
    plan_review_findings: dict[str, PlanReviewFinding] = Field(default_factory=dict)
    plan_review_decisions: dict[str, PlanReviewDecision] = Field(default_factory=dict)
    plan_human_questions: dict[str, HumanPlanQuestion] = Field(default_factory=dict)
    plan_loop_issue_attempts_by_fingerprint: dict[str, int] = Field(default_factory=dict)
    plan_loop_candidate_hashes_by_fingerprint: dict[str, list[str]] = Field(default_factory=dict)
    plan_loop_additional_llm_calls: int = 0
    plan_loop_no_progress_events: list[dict[str, Any]] = Field(default_factory=list)
    plan_loop_artifacts: list[str] = Field(default_factory=list)
    plan_human_answers: dict[str, HumanPlanAnswer] = Field(default_factory=dict)
    plan_confirmed_fact_records: dict[str, ConfirmedFactRecord] = Field(default_factory=dict)
    facts_review_history: list[dict[str, Any]] = Field(default_factory=list)
    facts_revision_history: list[dict[str, Any]] = Field(default_factory=list)
    placement_review_history: list[dict[str, Any]] = Field(default_factory=list)
    placement_revision_history: list[dict[str, Any]] = Field(default_factory=list)
    placement_dependency_requests: list[dict[str, Any]] = Field(default_factory=list)
    plan_confirmed_plan_fact_records: dict[str, ConfirmedPlanFactRecord] = Field(default_factory=dict)

    # Phase-1C canonical planning contract.  These fields replace neither the
    # original detector metadata nor FactsPatch; they persist their resolved
    # reconciliation so every downstream consumer uses one scope truth.
    planning_feature_contract: PlanningFeatureContract | None = None
    resolved_planning_scope: ResolvedPlanningScope | None = None
    canonical_task_plan: CanonicalPatchTaskPlan | None = None
    root_cause_retry_history: list[dict[str, Any]] = Field(default_factory=list)
    root_cause_attempts_by_fingerprint: dict[str, int] = Field(default_factory=dict)
    root_cause_candidate_hashes: dict[str, list[str]] = Field(default_factory=dict)

    # Phase-3 executable retry ledger.  It is deliberately typed and kept
    # distinct from legacy validation/runtime retry metadata.
    plan_retry_requests: dict[str, ExecutablePlanRetryRequest] = Field(default_factory=dict)
    plan_retry_execution_plans: dict[str, RetryExecutionPlan] = Field(default_factory=dict)
    plan_retry_rounds: list[RetryRoundRecord] = Field(default_factory=list)
    plan_retry_outcome: RetryExecutionOutcome | None = None
    plan_retry_attempts_by_fingerprint: dict[str, int] = Field(default_factory=dict)
    plan_retry_candidate_hashes_by_fingerprint: dict[str, list[str]] = Field(default_factory=dict)
    plan_retry_owner_regenerations: dict[str, int] = Field(default_factory=dict)
    plan_retry_gate_replay_counts: dict[str, int] = Field(default_factory=dict)
    plan_retry_cycle_trace: list[dict[str, Any]] = Field(default_factory=list)
    plan_retry_budget: dict[str, int] = Field(default_factory=dict)
    plan_retry_pending_request_ids: list[str] = Field(default_factory=list)
    # Phase-3B finer-grained ledgers.  Gate invalidation, gate replay attempts
    # and gate replay successes are kept separate so the artifacts can prove
    # that a replay actually ran, not merely that the gate was invalidated.
    plan_retry_gate_invalidation_counts: dict[str, int] = Field(default_factory=dict)
    plan_retry_gate_replay_attempt_counts: dict[str, int] = Field(default_factory=dict)
    plan_retry_gate_replay_success_counts: dict[str, int] = Field(default_factory=dict)
    # Per-fingerprint issue trace so that cycles and no-progress on changed
    # candidates with identical blocking issue fingerprints are detectable.
    plan_retry_issue_fingerprints_by_request: dict[str, list[str]] = Field(default_factory=dict)
    plan_retry_task_plan_hash_history: list[str] = Field(default_factory=list)
    plan_retry_human_questions: dict[str, Any] = Field(default_factory=dict)

    metadata: dict[str, Any] = Field(default_factory=dict)

    # Phase-8A Step 1 plan-investigation slots.  These are deliberately
    # inert here: no graph node, gate, patch generator or renderer reads or
    # writes them.  They exist so a later step can populate them from the
    # plan_investigation package outputs without another state migration.
    # Central access helpers live in
    # openmc_agent.plan_investigation.state_compat; callers MUST go through
    # those helpers rather than hand-writing metadata keys.
    planning_source_manifest: dict[str, Any] | None = None
    planning_evidence_ledger: dict[str, Any] | None = None
    plan_investigation_schema_version: str | None = None

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------

    def add_event(
        self,
        event_type: str,
        message: str,
        data: dict[str, Any] | None = None,
    ) -> None:
        """Append a build event to ``build_log``."""
        self.build_log.append(
            BuildEvent(
                timestamp=datetime.now(timezone.utc).isoformat(),
                event_type=event_type,
                message=message,
                data=dict(data or {}),
            )
        )

    def add_task(self, task: PlanComponentTask) -> None:
        """Add or replace a component task by ``task_id``."""
        existing = next(
            (t for t in self.component_tasks if t.task_id == task.task_id), None
        )
        if existing is not None:
            idx = self.component_tasks.index(existing)
            self.component_tasks[idx] = task
        else:
            self.component_tasks.append(task)

    def add_patch(self, patch: PlanPatchEnvelope) -> None:
        """Add a patch envelope and initialize its status tracking."""
        if patch.patch_type == "materials" and patch.status == "valid":
            from .materials_patch_normalization import normalize_materials_patch_content

            normalization = normalize_materials_patch_content(patch.content, state=self)
            if normalization.changed:
                patch.content = normalization.content
                patch.metadata.setdefault("deterministic_normalizations", [])
                patch.metadata["deterministic_normalizations"].extend(
                    normalization.operations
                )
                self.metadata.setdefault("materials_deterministic_normalizations", [])
                self.metadata["materials_deterministic_normalizations"].extend(
                    normalization.operations
                )
                self.add_event(
                    event_type="planning.materials_deterministic_normalization_applied",
                    message="materials patch normalized using source-declared unit semantics",
                    data={
                        "patch_id": patch.patch_id,
                        "operation_count": len(normalization.operations),
                        "operations": [
                            {
                                "operation": op.get("operation"),
                                "material_id": op.get("material_id"),
                                "target_ppm_by_weight": op.get("target_ppm_by_weight"),
                                "previous_boron_mass_fraction": op.get("previous_boron_mass_fraction"),
                                "relative_error": op.get("relative_error"),
                            }
                            for op in normalization.operations
                        ],
                    },
                )

        if patch.patch_type == "universes" and patch.status == "valid":
            from .universes_patch_normalization import normalize_universes_patch_content

            normalization = normalize_universes_patch_content(patch.content, state=self)
            if normalization.changed:
                patch.content = normalization.content
                patch.metadata.setdefault("deterministic_normalizations", [])
                patch.metadata["deterministic_normalizations"].extend(
                    normalization.operations
                )
                self.metadata.setdefault("universes_deterministic_normalizations", [])
                self.metadata["universes_deterministic_normalizations"].extend(
                    normalization.operations
                )
                self.add_event(
                    event_type="planning.universes_deterministic_normalization_applied",
                    message="universes patch normalized (implicit universe merge)",
                    data={
                        "patch_id": patch.patch_id,
                        "operation_count": len(normalization.operations),
                        "operations": [
                            {
                                "operation": op.get("operation"),
                                "implicit_universe_id": op.get("implicit_universe_id"),
                                "host_universe_id": op.get("host_universe_id"),
                                "cells_merged": op.get("cells_merged"),
                            }
                            for op in normalization.operations
                        ],
                    },
                )

        if patch.patch_type == "pin_map" and patch.status == "valid":
            patch.content = _normalize_pin_map_intent_bindings(patch.content, self)
        self.patches[patch.patch_id] = patch
        self.patch_status[patch.patch_id] = patch.status

    def mark_patch_status(
        self,
        patch_id: str,
        status: str,
        issues: list[dict[str, Any]] | None = None,
    ) -> None:
        """Update a patch's status (and optionally its issues list)."""
        if patch_id not in self.patches:
            return
        patch = self.patches[patch_id]
        # Use model_validate to update; Pydantic v2 doesn't allow direct field
        # assignment on frozen models, but AgentBaseModel is not frozen.
        patch.status = status  # type: ignore[assignment]
        self.patch_status[patch_id] = status
        if issues is not None:
            patch.issues = issues

    def get_valid_patches(
        self,
        patch_type: str | None = None,
    ) -> list[PlanPatchEnvelope]:
        """Return all patches with ``status='valid'``, optionally filtered by type."""
        result = [p for p in self.patches.values() if p.status == "valid"]
        if patch_type is not None:
            result = [p for p in result if p.patch_type == patch_type]
        return result

    def invalidate_patch_types(
        self,
        patch_types: list[str],
        *,
        reason: str = "",
        issues: list[dict[str, Any]] | None = None,
    ) -> list[str]:
        """Mark valid patches of the given types invalid and clear stale assembly."""
        target_types = set(patch_types)
        invalidated: list[str] = []
        issue_payload = list(issues or [])
        for patch_id, patch in self.patches.items():
            if patch.patch_type not in target_types or patch.status != "valid":
                continue
            patch.status = "invalid"  # type: ignore[assignment]
            patch.issues = issue_payload
            patch.metadata["invalidated_reason"] = reason
            self.patch_status[patch_id] = "invalid"
            invalidated.append(patch_id)

        if invalidated:
            for task in self.component_tasks:
                if task.patch_type in target_types:
                    task.status = "pending"  # type: ignore[assignment]
                    task.issues = issue_payload
            self.assembled_plan = None
            self.material_composition_report = None
            self.material_species_resolution_report = None
            self.validation_issues = []
            self.add_event(
                event_type="planning.patch_invalidated_for_plan_repair",
                message=(
                    "invalidated patch type(s) for plan-level validation repair: "
                    f"{sorted(target_types)}"
                ),
                data={
                    "patch_types": sorted(target_types),
                    "patch_ids": invalidated,
                    "reason": reason,
                },
            )
        return invalidated

    def to_summary(self) -> dict[str, Any]:
        """Return a compact summary suitable for transcript / debug output."""
        return {
            "state_id": self.state_id,
            "planning_mode": self.planning_mode,
            "benchmark_id": self.benchmark_id,
            "selected_variant": self.selected_variant,
            "task_count": len(self.component_tasks),
            "task_statuses": {
                s: sum(1 for t in self.component_tasks if t.status == s)
                for s in ("pending", "running", "valid", "invalid", "skipped", "blocked")
            },
            "tasks": [
                {
                    "task_id": t.task_id,
                    "patch_type": t.patch_type,
                    "status": t.status,
                }
                for t in self.component_tasks
            ],
            "patch_count": len(self.patches),
            "valid_patch_count": len(self.get_valid_patches()),
            "has_assembled_plan": self.assembled_plan is not None,
            "validation_issue_count": len(self.validation_issues),
            "validation_repair_count": len(self.validation_repair_history),
            "build_log_events": len(self.build_log),
        }


# ---------------------------------------------------------------------------
# Initialization helpers
# ---------------------------------------------------------------------------


def _generate_state_id() -> str:
    """Generate a stable state id."""
    from uuid import uuid4

    return f"pbs_{uuid4().hex[:16]}"


def initialize_plan_build_state(
    requirement: str,
    decision: PlanningModeDecision,
    confirmed_facts: dict[str, Any] | None = None,
    benchmark_id: str | None = None,
    selected_variant: str | None = None,
) -> PlanBuildState:
    """Initialize a :class:`PlanBuildState` from a planning mode decision.

    Records ``planning_mode_selected`` and ``build_state_initialized`` events.
    If the decision is monolithic, the state is still created (for debugging)
    but the mode is noted in metadata.

    Parameters
    ----------
    requirement
        The original requirement text.
    decision
        The :class:`PlanningModeDecision` from
        :func:`~openmc_agent.plan_builder.mode.should_use_incremental_planning`.
    confirmed_facts
        Optional pre-confirmed facts (e.g. from expert feedback).
    benchmark_id
        Optional benchmark identifier (e.g. ``"VERA3"``).
    selected_variant
        Optional variant identifier (e.g. ``"3B"``).

    Returns
    -------
    PlanBuildState
        The initialized build state.
    """
    state = PlanBuildState(
        state_id=_generate_state_id(),
        requirement_text=requirement,
        planning_mode="incremental",
        benchmark_id=benchmark_id,
        selected_variant=selected_variant,
        confirmed_facts=dict(confirmed_facts or {}),
        metadata={
            "planning_mode_decision": decision.model_dump(mode="json"),
        },
    )

    mode_event_type = (
        EVENT_PLANNING_MODE_SELECTED
        if decision.mode == "incremental"
        else EVENT_MONOLITHIC_MODE_SELECTED
    )
    state.add_event(
        event_type=mode_event_type,
        message=f"planning mode: {decision.mode}",
        data={
            "mode": decision.mode,
            "triggers": decision.triggers,
            "reasons": decision.reasons,
            "confidence": decision.confidence,
        },
    )
    state.add_event(
        event_type=EVENT_BUILD_STATE_INITIALIZED,
        message=f"plan build state initialized (state_id={state.state_id})",
        data={
            "benchmark_id": benchmark_id,
            "selected_variant": selected_variant,
            "feature_summary": decision.feature_summary,
        },
    )

    # Initialize shallow component tasks based on feature summary.
    tasks = create_initial_component_tasks(decision.feature_summary)
    for task in tasks:
        state.add_task(task)
    if tasks:
        state.add_event(
            event_type=EVENT_COMPONENT_TASKS_INITIALIZED,
            message=f"initialized {len(tasks)} component tasks",
            data={"task_ids": [t.task_id for t in tasks]},
        )

    return state


# ---------------------------------------------------------------------------
# Shallow component-task skeleton
# ---------------------------------------------------------------------------

# Task type ordering for a complex assembly.  ``facts`` and ``materials`` come
# first; ``settings`` is last and can be deterministic.
_DEFAULT_TASK_SEQUENCE: tuple[tuple[str, str], ...] = (
    ("facts", "Extract and confirm modeling facts from requirement"),
    ("materials", "Define materials (fuel, coolant, cladding, structural)"),
    ("universes", "Define universes (pin cells, guide tubes, special pins)"),
    ("pin_map", "Build lattice pin map with universe assignments"),
    ("assembly_catalog", "Define assembly type templates with per-type sparse pin maps"),
    ("axial_layers", "Define core.axial_layers (z-segmentation)"),
    ("axial_overlays", "Define core.axial_overlays (spacer grids, bands)"),
    ("core_layout", "Define core-level assembly placement pattern"),
    ("settings", "Define execution settings (batches, particles, source)"),
)

# Dependencies between task types.
_DEFAULT_DEPENDENCIES: dict[str, list[str]] = {
    "materials": ["task_facts"],
    "universes": ["task_materials"],
    "pin_map": ["task_universes"],
    "assembly_catalog": ["task_universes"],
    "axial_layers": ["task_facts"],
    "axial_overlays": ["task_axial_layers"],
    "core_layout": ["task_assembly_catalog"],
    "settings": [],
}


def create_initial_component_tasks(
    feature_summary: dict[str, Any],
) -> list[PlanComponentTask]:
    """Create a shallow component-task skeleton from detected features.

    Parameters
    ----------
    feature_summary
        The ``feature_summary`` dict from a :class:`PlanningModeDecision`.

    Returns
    -------
    list[PlanComponentTask]
        Ordered list of pending tasks.  May be empty for a trivial feature set.
    """
    has_axial = bool(feature_summary.get("has_axial_geometry"))
    has_spacer = bool(feature_summary.get("has_spacer_grid"))
    has_special_pin = bool(feature_summary.get("has_special_pin_map"))
    has_large_lattice = feature_summary.get("large_lattice_dimension") is not None
    has_variants = bool(feature_summary.get("has_benchmark_variant"))
    is_multi_assembly = bool(
        feature_summary.get("multi_assembly_core")
        or feature_summary.get("core_lattice")
    )

    # If nothing complex is detected, return an empty task list.
    if not any([has_axial, has_spacer, has_special_pin, has_large_lattice, has_variants, is_multi_assembly]):
        return []

    tasks: list[PlanComponentTask] = []
    for patch_type, description in _DEFAULT_TASK_SEQUENCE:
        # Skip axial_layers if no axial geometry detected.
        if patch_type == "axial_layers" and not has_axial:
            continue
        # Skip axial_overlays if no spacer grid and no axial geometry.
        if patch_type == "axial_overlays" and not (has_spacer or has_axial):
            continue
        # Single-assembly: skip assembly_catalog and core_layout.
        if patch_type in ("assembly_catalog", "core_layout") and not is_multi_assembly:
            continue
        # Multi-assembly: skip top-level pin_map.
        if patch_type == "pin_map" and is_multi_assembly:
            continue
        # Single-assembly without special pins: skip pin_map.
        if patch_type == "pin_map" and not is_multi_assembly and not (has_special_pin or has_large_lattice):
            continue

        tasks.append(
            PlanComponentTask(
                task_id=f"task_{patch_type}",
                patch_type=patch_type,
                description=description,
                status="pending",
                dependencies=list(_DEFAULT_DEPENDENCIES.get(patch_type, [])),
            )
        )

    return tasks


# ---------------------------------------------------------------------------
# Patch integration helpers (Phase 2)
# ---------------------------------------------------------------------------


def add_validated_patch_to_state(
    state: PlanBuildState,
    envelope: PlanPatchEnvelope,
    parsed_patch: Any,
    validation: Any,
) -> PlanBuildState:
    """Integrate a validated patch into a :class:`PlanBuildState`.

    Parameters
    ----------
    state
        The build state to update (mutated in place and returned).
    envelope
        The :class:`PlanPatchEnvelope` wrapping the patch content.
    parsed_patch
        The parsed patch model (from :func:`parse_patch_content`).
    validation
        The :class:`~openmc_agent.plan_builder.validators.PatchValidationResult`.

    Returns
    -------
    PlanBuildState
        The updated state (same object, for chaining).
    """
    patch_type = getattr(parsed_patch, "patch_type", envelope.patch_type)
    state.add_event(
        event_type=EVENT_PATCH_PARSED,
        message=f"patch {envelope.patch_id!r} parsed as {patch_type}",
        data={"patch_id": envelope.patch_id, "patch_type": patch_type},
    )

    issue_dicts = [issue.model_dump(mode="json") for issue in validation.issues]
    envelope.issues = issue_dicts

    if validation.ok:
        envelope.status = "valid"  # type: ignore[assignment]
        state.add_patch(envelope)
        state.add_event(
            event_type=EVENT_PATCH_VALIDATED,
            message=f"patch {envelope.patch_id!r} validated ({patch_type})",
            data={
                "patch_id": envelope.patch_id,
                "patch_type": patch_type,
                "issue_count": len(validation.issues),
                "summary": validation.summary,
            },
        )
    else:
        envelope.status = "invalid"  # type: ignore[assignment]
        state.add_patch(envelope)
        error_codes = [
            i.code for i in validation.issues if i.severity == "error"
        ]
        state.add_event(
            event_type=EVENT_PATCH_INVALID,
            message=f"patch {envelope.patch_id!r} invalid ({patch_type}): {error_codes}",
            data={
                "patch_id": envelope.patch_id,
                "patch_type": patch_type,
                "error_codes": error_codes,
                "summary": validation.summary,
            },
        )

    return state


def assemble_state_if_ready(
    state: PlanBuildState,
    *,
    strict: bool = True,
    material_policy: Any = None,
    resolved_planning_scope: Any = None,
) -> PlanBuildState:
    """Attempt to assemble all valid patches in ``state`` into a SimulationPlan.

    Reads ``state.patches`` (status='valid'), parses them, calls the
    deterministic assembler, and stores the result in ``state.assembled_plan``.

    Parameters
    ----------
    state
        The build state (mutated in place).
    strict
        Forwarded to the assembler.
    material_policy
        Optional material composition policy forwarded to the assembler.
        Accepts the enum, a string value, or None (assembler default).

    Returns
    -------
    PlanBuildState
        The updated state (same object, for chaining).
    """
    from .patches import parse_patch_content
    from .assembler import assemble_simulation_plan_from_patches

    valid_envelopes = [
        env for env in state.patches.values() if env.status == "valid"
    ]
    if not valid_envelopes:
        state.add_event(
            event_type=EVENT_ASSEMBLY_FAILED,
            message="no valid patches to assemble",
            data={"valid_patch_count": 0},
        )
        return state

    parsed_patches: list[Any] = []
    parse_errors: list[str] = []
    for env in valid_envelopes:
        try:
            parsed = parse_patch_content(env.patch_type, env.content)
            parsed_patches.append(parsed)
        except Exception as exc:
            parse_errors.append(f"{env.patch_id}: {exc}")

    if parse_errors:
        state.add_event(
            event_type=EVENT_ASSEMBLY_FAILED,
            message=f"failed to parse {len(parse_errors)} patch(es)",
            data={"parse_errors": parse_errors},
        )
        return state

    state.add_event(
        event_type=EVENT_ASSEMBLY_STARTED,
        message=f"assembling {len(parsed_patches)} patches",
        data={"patch_types": [getattr(p, "patch_type", "?") for p in parsed_patches]},
    )

    assembler_kwargs: dict[str, Any] = {"strict": strict}
    if material_policy is not None:
        assembler_kwargs["material_policy"] = material_policy
    if resolved_planning_scope is not None:
        assembler_kwargs["resolved_planning_scope"] = resolved_planning_scope
    result = assemble_simulation_plan_from_patches(parsed_patches, **assembler_kwargs)

    if result.ok and result.plan is not None:
        state.assembled_plan = result.plan.model_dump(mode="json")
        if result.material_composition_report is not None:
            state.material_composition_report = (
                result.material_composition_report.model_dump(mode="json")
            )
        if result.material_species_resolution_report is not None:
            state.material_species_resolution_report = result.material_species_resolution_report
        state.add_event(
            event_type=EVENT_ASSEMBLY_COMPLETED,
            message=f"assembly completed ({len(result.summary)} summary entries)",
            data=result.summary,
        )
    else:
        error_codes = [i.code for i in result.issues if i.severity == "error"]
        state.validation_issues.extend(
            issue.model_dump(mode="json") for issue in result.issues
        )
        state.add_event(
            event_type=EVENT_ASSEMBLY_FAILED,
            message=f"assembly failed: {error_codes}",
            data={
                "error_codes": error_codes,
                "summary": result.summary,
            },
        )

    return state


def generate_and_add_patch_to_state(
    state: PlanBuildState,
    patch_type: str,
    requirement: str,
    context: Any | None = None,
    llm_client: Any | None = None,
    max_attempts: int = 2,
) -> PlanBuildState:
    """Generate a patch via LLM and add it to the build state.

    Calls :func:`~openmc_agent.plan_builder.patch_generator.generate_patch`,
    then integrates the result into ``state``.

    On success:
    * Adds the valid patch envelope to state.
    * Records ``planning.patch_generation_started`` and
      ``planning.patch_generated`` events.

    On failure:
    * Records an invalid envelope if raw output exists.
    * Records ``planning.patch_generation_failed`` event.
    * **Does NOT touch already-valid patches.**

    Parameters
    ----------
    state
        The build state (mutated in place).
    patch_type
        The patch type to generate.
    requirement
        The requirement text for the prompt.
    context
        Optional :class:`PatchGenerationContext`.
    llm_client
        A callable ``(prompt: str) -> str``.
    max_attempts
        Maximum LLM call attempts.

    Returns
    -------
    PlanBuildState
        The updated state (same object, for chaining).
    """
    from .patch_generator import generate_patch

    state.add_event(
        event_type=EVENT_PATCH_GENERATION_STARTED,
        message=f"generating {patch_type} patch via LLM",
        data={"patch_type": patch_type, "max_attempts": max_attempts},
    )

    result = generate_patch(
        patch_type=patch_type,
        requirement=requirement,
        state=state,
        context=context,
        llm_client=llm_client,
        max_attempts=max_attempts,
    )

    if result.ok and result.envelope is not None:
        state.add_patch(result.envelope)
        state.add_event(
            event_type=EVENT_PATCH_GENERATED,
            message=f"{patch_type} patch generated and validated",
            data={
                "patch_id": result.envelope.patch_id,
                "patch_type": patch_type,
                "attempts": len(result.attempts),
            },
        )
    else:
        # Record failure — preserve existing valid patches.
        error_codes = [
            i.get("code", "unknown")
            for i in result.issues
            if i.get("severity") == "error"
        ]
        # Phase 7C: save attempt raw/prompt for artifact diagnostics.
        patch_attempts = state.metadata.setdefault("patch_attempt_artifacts", {})
        for att in result.attempts:
            att_key = f"{patch_type}_attempt_{att.attempt_index + 1}"
            patch_attempts[att_key] = {
                "patch_type": patch_type,
                "attempt_index": att.attempt_index,
                "raw_chars": att.raw_chars,
                "raw_text": (att.raw_text or "")[:5000],  # cap for state size
                "prompt_text": (att.prompt_text or "")[:3000],
                "issues": att.issues,
                "output_mode_used": att.output_mode_used,
                "error": att.error,
            }
        state.add_event(
            event_type=EVENT_PATCH_GENERATION_FAILED,
            message=f"{patch_type} patch generation failed: {error_codes}",
            data={
                "patch_type": patch_type,
                "error_codes": error_codes,
                "attempts": len(result.attempts),
            },
        )

    return state


# ---------------------------------------------------------------------------
# Resume helpers (Phase 7D)
# ---------------------------------------------------------------------------


def save_plan_build_state(state: PlanBuildState, path: str | Path) -> None:
    """Save a PlanBuildState to a JSON file."""
    import json
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        json.dumps(state.model_dump(mode="json"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_plan_build_state(path: str | Path) -> PlanBuildState:
    """Load a PlanBuildState from a JSON file.

    Raises FileNotFoundError if the file does not exist.
    """
    import json
    p = Path(path)
    data = json.loads(p.read_text(encoding="utf-8"))
    return PlanBuildState.model_validate(data)


__all__ = [
    "BuildEvent",
    "PlanBuildState",
    "PlanComponentTask",
    "PlanPatchEnvelope",
    "add_validated_patch_to_state",
    "assemble_state_if_ready",
    "generate_and_add_patch_to_state",
    "save_plan_build_state",
    "load_plan_build_state",
    "initialize_plan_build_state",
    "create_initial_component_tasks",
    "EVENT_PLANNING_MODE_SELECTED",
    "EVENT_MONOLITHIC_MODE_SELECTED",
    "EVENT_INCREMENTAL_RECOMMENDED_NOT_EXECUTED",
    "EVENT_INCREMENTAL_EXECUTOR_NOT_IMPLEMENTED",
    "EVENT_BUILD_STATE_INITIALIZED",
    "EVENT_COMPONENT_TASKS_INITIALIZED",
    "EVENT_PATCH_PARSED",
    "EVENT_PATCH_VALIDATED",
    "EVENT_PATCH_INVALID",
    "EVENT_ASSEMBLY_STARTED",
    "EVENT_ASSEMBLY_COMPLETED",
    "EVENT_ASSEMBLY_FAILED",
    "EVENT_PATCH_GENERATION_STARTED",
    "EVENT_PATCH_GENERATED",
    "EVENT_PATCH_GENERATION_FAILED",
]


def _normalize_pin_map_intent_bindings(
    content: dict[str, Any],
    state: PlanBuildState,
) -> dict[str, Any]:
    """Fill null anchor_z_cm / axial_profile_id in localized_insert_intents.

    The LLM frequently leaves these fields null in the pin_map even though
    Facts declares explicit values.  Placement preflight flags the mismatch.
    This normalizer copies values from the matching Facts requirement when
    the insert_kind unambiguously matches.
    """
    import copy as _copy

    intents = content.get("localized_insert_intents")
    if not isinstance(intents, list) or not intents:
        return content

    facts_reqs: list[dict[str, Any]] = []
    for env in state.patches.values():
        if (
            getattr(env, "patch_type", None) == "facts"
            and getattr(env, "status", None) == "valid"
        ):
            facts_reqs = list(env.content.get("localized_insert_requirements") or [])
            break
    if not facts_reqs:
        return content

    kind_to_req: dict[str, dict[str, Any]] = {}
    for req in facts_reqs:
        if not isinstance(req, dict):
            continue
        kind = str(req.get("insert_kind", "")).lower().strip()
        if kind:
            kind_to_req.setdefault(kind, req)

    changed = False
    new_intents = []
    for intent in intents:
        if not isinstance(intent, dict):
            new_intents.append(intent)
            continue
        kind = str(intent.get("insert_kind", "")).lower().strip()
        req = kind_to_req.get(kind)
        if req is None:
            new_intents.append(intent)
            continue
        patched = _copy.deepcopy(intent)
        if patched.get("anchor_z_cm") is None and req.get("anchor_z_cm") is not None:
            patched["anchor_z_cm"] = req["anchor_z_cm"]
            changed = True
        if not patched.get("axial_profile_id") and req.get("required_profile_id"):
            patched["axial_profile_id"] = req["required_profile_id"]
            changed = True
        new_intents.append(patched)

    if changed:
        new_content = _copy.deepcopy(content)
        new_content["localized_insert_intents"] = new_intents
        return new_content
    return content
