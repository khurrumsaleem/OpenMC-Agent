"""Facts-only RFC6902 proposal validation and clone evaluation."""

from __future__ import annotations

import json
from typing import Any, Literal

from pydantic import Field

from openmc_agent.plan_builder.patches import parse_patch_content
from openmc_agent.plan_builder.validation_repair import PatchRepairOperation
from openmc_agent.plan_builder.validators import validate_patch
from openmc_agent.repair_proposal import apply_json_patch_to_clone
from openmc_agent.schemas import AgentBaseModel

from .fingerprints import compute_candidate_hash, compute_issue_fingerprint
from .models import FactsRevisionProposal, PlanReviewFinding

__all__ = [
    "FactsRevisionEvaluation",
    "FactsDeterministicRepair",
    "allowed_paths_for_findings",
    "normalize_facts_revision",
    "evaluate_facts_revision",
    "targeted_facts_repair",
    "run_clone_validation",
    "check_facts_repair_completeness",
    "required_coverage_paths_for_patch",
    "facts_revision_closure_fingerprint",
    "MAX_FACTS_REVISION_CLOSURE_ROUNDS",
    "REQUIRED_COVERAGE_PATHS",
]


# ---------------------------------------------------------------------------
# Phase 8B Step 3: repair coverage completeness
# ---------------------------------------------------------------------------

# Core FactsPatch fields considered by the repair coverage contract.  Whether
# a field is required depends on the current source-derived model scope and
# feature flags; for example a single-assembly case must not be forced to
# invent assembly-type counts, and a case with no special pin map must not be
# forced to invent localized insert requirements.
REQUIRED_COVERAGE_PATHS: tuple[str, ...] = (
    "/model_scope",
    "/assembly_count",
    "/core_lattice_size",
    "/assembly_type_counts",
    "/fuel_variant_requirements",
    "/localized_insert_requirements",
    "/has_spacer_grids",
)

# Internal safety cap; no CLI tuning is exposed for this closure loop.
MAX_FACTS_REVISION_CLOSURE_ROUNDS = 3


def facts_revision_closure_fingerprint(findings: list[PlanReviewFinding]) -> str:
    """Fingerprint unresolved Facts errors without relying on critic prose."""
    rows = sorted(
        (
            item.code,
            tuple(sorted(item.affected_json_paths)),
            tuple(sorted(excerpt.evidence_hash for excerpt in item.source_evidence)),
            item.requires_human,
            item.repairable_by_llm,
        )
        for item in findings
    )
    return compute_issue_fingerprint(
        gate_id="facts",
        code="facts_revision.unresolved_closure",
        affected_patch_type="facts",
        actual=rows,
    )


def _is_empty_coverage_value(value: Any) -> bool:
    return (
        value is None
        or value == ""
        or value == []
        or value == {}
        or value == "unknown"
    )


def required_coverage_paths_for_patch(candidate: dict[str, Any]) -> tuple[str, ...]:
    """Return coverage fields required for this FactsPatch shape.

    The contract stays reactor-neutral, but it must be conditional on source
    scope.  A required field here is both shown in the revision prompt and
    accepted by the revision evaluator.
    """

    scope = str(candidate.get("model_scope") or "").strip().lower()
    assembly_count = candidate.get("assembly_count")
    core_lattice_size = candidate.get("core_lattice_size")
    assembly_type_counts = candidate.get("assembly_type_counts")
    has_special_pin_map = candidate.get("has_special_pin_map")
    localized_insert_requirements = candidate.get("localized_insert_requirements")
    has_declared_insert_count = any(
        (candidate.get(key) or 0) > 0
        for key in ("expected_pyrex_count", "expected_thimble_plug_count")
        if isinstance(candidate.get(key), (int, float))
    )

    paths: list[str] = [
        "/model_scope",
        "/assembly_count",
        "/fuel_variant_requirements",
        "/has_spacer_grids",
    ]
    is_multi_assembly = (
        scope == "multi_assembly_core"
        or (isinstance(assembly_count, int) and assembly_count > 1)
        or not _is_empty_coverage_value(core_lattice_size)
    )
    if is_multi_assembly:
        paths.append("/assembly_type_counts")
    if (
        has_special_pin_map is True
        or has_declared_insert_count
        or not _is_empty_coverage_value(localized_insert_requirements)
    ):
        paths.append("/localized_insert_requirements")
    return tuple(dict.fromkeys(paths))


def check_facts_repair_completeness(candidate: dict[str, Any]) -> list[str]:
    """Return the list of required coverage paths still empty in ``candidate``.

    An empty list means the candidate is complete.  Used by
    :func:`evaluate_facts_revision` to reject repairs that leave required
    fields empty (``planning.facts_repair_incomplete``).
    """

    missing: list[str] = []
    for pointer in required_coverage_paths_for_patch(candidate):
        key = pointer.lstrip("/")
        value = candidate.get(key)
        if _is_empty_coverage_value(value):
            missing.append(pointer)
    return missing


class FactsRevisionEvaluation(AgentBaseModel):
    accepted: bool = False
    candidate: dict[str, Any] | None = None
    candidate_hash: str | None = None
    reasons: list[str] = Field(default_factory=list)


def _pointer_value(value: Any, pointer: str) -> Any:
    """Resolve RFC6901 pointer without treating a nested confirmation as a
    top-level key.  Missing values remain distinct from JSON null."""
    current = value
    for token in pointer.lstrip("/").split("/"):
        token = token.replace("~1", "/").replace("~0", "~")
        if isinstance(current, dict) and token in current:
            current = current[token]
        elif isinstance(current, list) and token.isdigit() and int(token) < len(current):
            current = current[int(token)]
        else:
            return _MISSING
    return current


_MISSING = object()

_FACTS_REVISION_LIST_PATHS = {
    "/missing_facts",
    "/assumptions",
    "/source_notes",
    "/fuel_variant_requirements",
    "/localized_insert_requirements",
}

_FACTS_REVISION_SOURCE_NOTE_ONLY_PATHS = {
    "/operating_conditions",
    "/operating_state_parameters",
    "/operating_state_conditions",
    "/coolant_conditions",
}


def _schema_out_metadata_operation_to_source_note(
    operation: PatchRepairOperation,
    *,
    drop_if_source_note_already_present: bool,
) -> PatchRepairOperation | None:
    """Map known schema-out Facts metadata edits to source_notes.

    FactsPatch intentionally has no structured operating-conditions object.
    Review repair LLMs may still propose one because the information is real
    and source-backed.  Keep the data as a note, but never accept the schema-out
    field itself.
    """

    if operation.path not in _FACTS_REVISION_SOURCE_NOTE_ONLY_PATHS:
        return None
    if drop_if_source_note_already_present:
        return None
    value = json.dumps(operation.value, ensure_ascii=False, sort_keys=True)
    return operation.model_copy(
        update={
            "op": "add",
            "path": "/source_notes/-",
            "value": f"{operation.path.lstrip('/').replace('_', ' ')}: {value}",
        }
    )


def _prepare_facts_revision_operations(
    facts_patch: dict[str, Any],
    operations: list[PatchRepairOperation],
) -> list[PatchRepairOperation]:
    """Normalize safe JSON Patch edge cases before application.

    Review-time repair agents often use ``replace`` for an optional top-level
    FactsPatch field that is absent from the current JSON.  RFC6902 rejects
    that, but semantically the safe operation is ``add``.  Limit the conversion
    to single-token top-level paths; nested or array edits still obey strict
    JSON Patch semantics.
    """

    prepared: list[PatchRepairOperation] = []
    for operation in operations:
        if (
            operation.op == "replace"
            and operation.path.count("/") == 1
            and _pointer_value(facts_patch, operation.path) is _MISSING
        ):
            prepared.append(operation.model_copy(update={"op": "add"}))
        elif (
            operation.op == "add"
            and operation.path.endswith("/-")
            and (parent_path := operation.path[:-2]) in _FACTS_REVISION_LIST_PATHS
            and _pointer_value(facts_patch, parent_path) is _MISSING
        ):
            prepared.append(operation.model_copy(update={"path": parent_path, "value": [operation.value]}))
        else:
            prepared.append(operation)
    return prepared


def _confirmed_records(confirmed_facts: dict[str, Any]) -> list[tuple[str, Any]]:
    """Supports the stable typed record store and the legacy namespaced map."""
    records = confirmed_facts.get("plan_closed_loop_records", []) if isinstance(confirmed_facts, dict) else []
    result: list[tuple[str, Any]] = []
    for record in records if isinstance(records, list) else []:
        if isinstance(record, dict) and isinstance(record.get("json_path"), str):
            result.append((record["json_path"], record.get("value")))
    facts = confirmed_facts.get("plan_closed_loop", {}).get("facts", {}) if isinstance(confirmed_facts, dict) else {}
    if isinstance(facts, dict):
        for path, value in facts.items():
            result.append((path if str(path).startswith("/") else "/" + str(path), value))
    return result


def allowed_paths_for_findings(findings: list[PlanReviewFinding], facts_patch: dict[str, Any] | None = None) -> list[str]:
    paths = {
        "/missing_facts",
        "/assumptions",
        "/source_notes",
        *REQUIRED_COVERAGE_PATHS,
    }
    contract_paths = {
        "facts.model_scope_conflicts_with_planning_features": {"/model_scope", "/assembly_count", "/core_lattice_size", "/assembly_type_counts"},
        "facts.multi_assembly_contract_incomplete": {"/model_scope", "/assembly_count", "/core_lattice_size", "/assembly_type_counts"},
        "facts.localized_insert_contract_missing": {"/localized_insert_requirements", "/has_special_pin_map"},
        "facts.localized_insert_profile_contract_missing": {"/localized_insert_requirements"},
        "facts.spacer_grid_contract_missing": {"/has_spacer_grids", "/expected_spacer_grid_count"},
        "facts.fuel_variant_contract_missing": {"/fuel_variant_requirements"},
    }
    for finding in findings:
        if finding.repairable_by_llm and not finding.requires_human:
            paths.update(contract_paths.get(finding.code, set()))
            for path in finding.affected_json_paths:
                paths.add(path)
    return sorted(paths)


def _extract_facts_revision_payload(raw: str | dict[str, Any]) -> dict[str, Any]:
    """Extract a FactsRevisionProposal payload from an LLM response.

    Handles three response shapes:

    1. ``dict``: passed through (caller already parsed).
    2. JSON-only string: parsed via ``json.loads``.
    3. "Prose + JSON" string (the common failure mode for thinking-mode
       providers): scanned for the last embedded JSON object declaring an
       ``operations`` field.  Mirrors the recovery logic that the patch
       generator and the structured review caller already use.

    Raises ``json.JSONDecodeError`` if no usable payload is found, so the
    caller can treat the failure uniformly.
    """
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str):
        raise json.JSONDecodeError("facts revision raw response was not JSON", str(raw), 0)
    text = raw.strip()
    if not text:
        raise json.JSONDecodeError("facts revision raw response was empty", "", 0)
    try:
        payload = json.loads(text)
        if isinstance(payload, dict):
            return payload
    except json.JSONDecodeError:
        pass
    # Fallback: scan for embedded JSON objects (prose-wrapped or multiple
    # candidates).  Prefer the last object declaring ``operations``.
    decoder = json.JSONDecoder()
    candidates: list[dict[str, Any]] = []
    for offset, char in enumerate(text):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[offset:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            candidates.append(value)
    for candidate in reversed(candidates):
        if "operations" in candidate:
            return candidate
    if candidates:
        return candidates[-1]
    raise json.JSONDecodeError(
        "facts revision raw response contained no parseable JSON object",
        text, 0,
    )


def normalize_facts_revision(raw: str | dict[str, Any]) -> FactsRevisionProposal:
    payload = _extract_facts_revision_payload(raw)
    proposal = FactsRevisionProposal.model_validate(payload)
    operations = [PatchRepairOperation.model_validate(item) for item in proposal.operations]
    if not operations:
        raise ValueError("facts revision requires operations")
    if any(operation.op not in {"add", "replace", "remove"} for operation in operations):
        raise ValueError("facts revision permits only add, replace, and remove")
    proposal.operations = operations
    return proposal


def evaluate_facts_revision(*, facts_patch: dict[str, Any], proposal: FactsRevisionProposal,
                            findings: list[PlanReviewFinding], confirmed_facts: dict[str, Any],
                            prior_candidate_hashes: list[str]) -> FactsRevisionEvaluation:
    allowed = allowed_paths_for_findings(findings, facts_patch)
    operations = [PatchRepairOperation.model_validate(item) for item in proposal.operations]
    has_source_note_edit = any(
        operation.path == "/source_notes" or operation.path.startswith("/source_notes/")
        for operation in operations
    )
    operations_to_apply: list[PatchRepairOperation] = []
    for operation in operations:
        if operation.path == "" or operation.path == "/patch_type" or not any(operation.path == path or operation.path.startswith(path + "/") for path in allowed):
            converted = _schema_out_metadata_operation_to_source_note(
                operation,
                drop_if_source_note_already_present=has_source_note_edit,
            )
            if converted is None:
                if operation.path in _FACTS_REVISION_SOURCE_NOTE_ONLY_PATHS and has_source_note_edit:
                    continue
                return FactsRevisionEvaluation(reasons=["facts_revision.path_out_of_scope"])
            operations_to_apply.append(converted)
            continue
        operations_to_apply.append(operation)
    if not operations_to_apply:
        return FactsRevisionEvaluation(reasons=["facts_revision.no_applicable_operations"])
    try:
        prepared_operations = _prepare_facts_revision_operations(facts_patch, operations_to_apply)
        applied = apply_json_patch_to_clone(facts_patch, [operation.model_dump(mode="json", exclude_none=True) for operation in prepared_operations])
        if not applied.ok:
            return FactsRevisionEvaluation(reasons=[f"facts_revision.apply_failed: {applied.error}"])
        candidate = applied.plan
    except Exception as exc:
        return FactsRevisionEvaluation(reasons=[f"facts_revision.apply_failed: {exc}"])
    if not isinstance(candidate, dict):
        return FactsRevisionEvaluation(reasons=["facts_revision.root_replacement_forbidden"])
    candidate_hash = compute_candidate_hash(target_patch_type="facts", candidate_patch=candidate)
    if candidate_hash in prior_candidate_hashes:
        return FactsRevisionEvaluation(candidate_hash=candidate_hash, reasons=["facts_revision.duplicate_candidate"])
    try:
        parsed = parse_patch_content("facts", candidate)
        validation = validate_patch(parsed)
    except Exception as exc:
        return FactsRevisionEvaluation(candidate_hash=candidate_hash, reasons=[f"facts_revision.schema_invalid: {exc}"])
    if not validation.ok:
        return FactsRevisionEvaluation(candidate_hash=candidate_hash, reasons=["facts_revision.validator_failed"])
    for path, value in _confirmed_records(confirmed_facts):
        if _pointer_value(candidate, path) != value:
            return FactsRevisionEvaluation(candidate_hash=candidate_hash, reasons=["facts_revision.confirmed_fact_changed"])
    # Phase 8B Step 3: enforce required coverage field completeness.
    missing = check_facts_repair_completeness(candidate)
    if missing:
        return FactsRevisionEvaluation(
            candidate_hash=candidate_hash,
            reasons=[f"facts_revision.incomplete_coverage: {', '.join(missing)}"],
        )
    return FactsRevisionEvaluation(accepted=True, candidate=candidate, candidate_hash=candidate_hash)


# ---------------------------------------------------------------------------
# Phase 8B Step 2: targeted facts repair and clone validation
# ---------------------------------------------------------------------------


class FactsDeterministicRepair(AgentBaseModel):
    repair_id: str = ""
    target_path: str
    expected_value: Any = None
    actual_value: Any = None
    operation_kind: Literal["replace", "add", "remove"] = "replace"
    source_claim_id: str | None = None
    resolved: bool = False
    error: str | None = None


def targeted_facts_repair(
    facts_patch: dict[str, Any],
    skeleton: Any,
    findings: list[Any],
) -> tuple[dict[str, Any] | None, list[FactsDeterministicRepair]]:
    """Deterministically repair a facts patch from skeleton + findings.

    Returns (repaired_patch_or_None, repairs_applied).
    Repairs are deterministic: model_scope, feature flags, assembly_count,
    fuel_variant_requirements, and localized_insert_requirements values
    from the skeleton are forced into the patch when findings indicate
    they are wrong.
    """
    if skeleton is None:
        return None, [FactsDeterministicRepair(
            repair_id="skeleton_missing",
            target_path="",
            error="FactsRequirementSkeleton is None; cannot repair",
        )]

    repairs: list[FactsDeterministicRepair] = []
    candidate = dict(facts_patch)

    finding_codes = {getattr(f, "code", None) or getattr(f, "get", lambda k: None)("code") for f in findings}

    # Repair model_scope
    if any("model_scope" in (c or "") for c in finding_codes):
        if skeleton.model_scope is not None and skeleton.model_scope.status in ("human_confirmed", "source_backed"):
            expected = skeleton.model_scope.value
            actual = candidate.get("model_scope")
            if actual != expected:
                candidate["model_scope"] = expected
                repairs.append(FactsDeterministicRepair(
                    repair_id="fix_model_scope",
                    target_path="/model_scope",
                    expected_value=expected,
                    actual_value=actual,
                    operation_kind="replace",
                    resolved=True,
                ))

    # Repair feature flags
    if skeleton.features is not None:
        for flag in ("has_axial_geometry", "has_spacer_grids", "has_special_pin_map"):
            expected_val = getattr(skeleton.features, flag, None)
            if expected_val is not None:
                actual_val = candidate.get(flag)
                if actual_val != expected_val:
                    candidate[flag] = expected_val
                    repairs.append(FactsDeterministicRepair(
                        repair_id=f"fix_{flag}",
                        target_path=f"/{flag}",
                        expected_value=expected_val,
                        actual_value=actual_val,
                        operation_kind="replace",
                        resolved=True,
                    ))

    # Repair assembly_count
    if skeleton.assembly_layout is not None and skeleton.assembly_layout.assembly_count is not None:
        expected = skeleton.assembly_layout.assembly_count
        actual = candidate.get("assembly_count")
        if actual != expected:
            candidate["assembly_count"] = expected
            repairs.append(FactsDeterministicRepair(
                repair_id="fix_assembly_count",
                target_path="/assembly_count",
                expected_value=expected,
                actual_value=actual,
                operation_kind="replace",
                resolved=True,
            ))
        if skeleton.assembly_layout.core_lattice_size is not None:
            expected_cls = list(skeleton.assembly_layout.core_lattice_size)
            actual_cls = candidate.get("core_lattice_size")
            if actual_cls != expected_cls:
                candidate["core_lattice_size"] = expected_cls
                repairs.append(FactsDeterministicRepair(
                    repair_id="fix_core_lattice_size",
                    target_path="/core_lattice_size",
                    expected_value=expected_cls,
                    actual_value=actual_cls,
                    operation_kind="replace",
                    resolved=True,
                ))

    # Repair fuel variants
    if skeleton.fuel_variant_slots and any("fuel_variant" in (c or "") for c in finding_codes):
        required_variants = {s.variant_id: s for s in skeleton.fuel_variant_slots}
        existing = {v.get("variant_id"): v for v in candidate.get("fuel_variant_requirements", []) if isinstance(v, dict)}
        for vid, slot in required_variants.items():
            if vid not in existing:
                candidate.setdefault("fuel_variant_requirements", []).append({
                    "variant_id": slot.variant_id,
                    "enrichment_wt_percent": slot.enrichment_wt_percent,
                    "density_g_cm3": slot.density_g_cm3,
                    "assembly_type_ids": list(slot.assembly_type_ids),
                })
                repairs.append(FactsDeterministicRepair(
                    repair_id=f"add_fuel_variant_{vid}",
                    target_path="/fuel_variant_requirements",
                    expected_value=vid,
                    operation_kind="add",
                    resolved=True,
                ))

    # Repair localized inserts
    if skeleton.localized_insert_slots and any("localized_insert" in (c or "") for c in finding_codes):
        required_ids = {s.requirement_id for s in skeleton.localized_insert_slots}
        existing_ids = {r.get("requirement_id") for r in candidate.get("localized_insert_requirements", []) if isinstance(r, dict)}
        for slot in skeleton.localized_insert_slots:
            if slot.requirement_id not in existing_ids:
                candidate.setdefault("localized_insert_requirements", []).append({
                    "requirement_id": slot.requirement_id,
                    "insert_kind": slot.insert_kind,
                    "assembly_type_ids": list(slot.assembly_type_ids),
                })
                repairs.append(FactsDeterministicRepair(
                    repair_id=f"add_insert_{slot.requirement_id}",
                    target_path="/localized_insert_requirements",
                    expected_value=slot.requirement_id,
                    operation_kind="add",
                    resolved=True,
                ))

    return candidate, repairs


def run_clone_validation(
    candidate: dict[str, Any] | None,
    skeleton: Any,
    reviewer: Any = None,
) -> FactsRevisionEvaluation:
    """Validate a facts clone against skeleton + optional reviewer.

    Checks:
    - Schema valid
    - Skeleton preflight errors = 0
    - Consistency errors = 0
    - Reviewer coverage complete (when reviewer provided)
    """
    if candidate is None:
        return FactsRevisionEvaluation(reasons=["clone is None"])

    reasons: list[str] = []

    # Schema validation
    try:
        parsed = parse_patch_content("facts", candidate)
        validation = validate_patch(parsed)
        if not validation.ok:
            reason = "facts_revision.validator_failed"
            reasons.append(reason)
    except Exception as exc:
        reasons.append(f"facts_revision.schema_invalid: {exc}")

    # Skeleton preflight
    if skeleton is not None:
        try:
            from openmc_agent.plan_builder.facts_evidence_contract import (
                run_facts_skeleton_preflight,
            )
            preflight = run_facts_skeleton_preflight(skeleton, candidate)
            if not preflight.ok:
                for issue in preflight.issues:
                    if issue.get("severity") == "error":
                        reasons.append(f"facts_revision.skeleton_preflight: {issue.get('code')}")
        except Exception:
            pass

    # Consistency check
    try:
        from openmc_agent.plan_builder.closed_loop.facts_consistency import (
            run_facts_consistency_preflight,
        )
        from openmc_agent.plan_builder.planning_scope import planning_feature_contract
        contract = planning_feature_contract({"feature_summary": {}})
        consistency = run_facts_consistency_preflight(
            feature_contract=contract,
            facts_patch=candidate,
        )
        if not consistency.ok:
            reasons.append("facts_revision.consistency_errors")
    except Exception:
        pass

    # Reviewer coverage
    if reviewer is not None:
        try:
            from openmc_agent.plan_builder.closed_loop.facts_reviewer import (
                FactsReviewCoverageSummary as CovSummary,
            )
            coverage = getattr(reviewer, "coverage_summary", None)
            if coverage is not None and hasattr(coverage, "reviewed_source_excerpt_count"):
                if coverage.reviewed_source_excerpt_count == 0:
                    reasons.append("facts_revision.reviewer_coverage_incomplete")
        except Exception:
            pass

    candidate_hash = compute_candidate_hash(
        target_patch_type="facts",
        candidate_patch=candidate,
    )

    if reasons:
        return FactsRevisionEvaluation(
            candidate_hash=candidate_hash,
            reasons=reasons,
        )
    return FactsRevisionEvaluation(
        accepted=True,
        candidate=candidate,
        candidate_hash=candidate_hash,
    )
