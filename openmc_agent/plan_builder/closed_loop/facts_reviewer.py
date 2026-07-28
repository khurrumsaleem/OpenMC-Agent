"""Independent Facts Critic invocation and strict Python normalization."""

from __future__ import annotations

from openmc_agent.structured_output import canonical_payload_hash

import re
from typing import Any

from pydantic import Field

from openmc_agent.schemas import AgentBaseModel

from .facts_review_prompts import build_facts_review_prompt, build_facts_review_schema_retry_prompt
from .models import (
    FactsReviewFindingDraft, FactsReviewModelOutput, PlanClosedLoopPolicy, PlanEvidencePack, PlanFindingCategory,
    PlanFindingSeverity, PlanGateId, PlanReviewFinding, SourceExcerpt,
)
from .review_io import (
    StructuredReviewCallSpec,
    _ACCEPTED_REVIEW_STATUSES,
    run_structured_review_call,
)


class FactsReviewResult(AgentBaseModel):
    ok: bool = False
    findings: list[PlanReviewFinding] = Field(default_factory=list)
    rejected: list[dict[str, Any]] = Field(default_factory=list)
    outputs: list[dict[str, Any]] = Field(default_factory=list)
    coverage_complete: bool = False
    reviewer_calls: int = 0
    schema_retries: int = 0
    error: str = ""
    raw_outputs: list[str] = Field(default_factory=list)
    call_metadata: list[dict[str, Any]] = Field(default_factory=list)
    failure_code: str = ""


class FactsFindingNormalization(AgentBaseModel):
    """Deterministic firewall result for one untrusted Facts reviewer finding."""

    draft: FactsReviewFindingDraft | None = None
    rejected: dict[str, Any] | None = None
    classification_override: dict[str, Any] | None = None


_FACTS_RECORDING_METADATA_ROOTS = (
    "/missing_facts",
    "/assumptions",
    "/source_notes",
)


_FACTS_REVIEW_PATH_PREFIXES: tuple[tuple[str, str], ...] = (
    ("facts_subset.", ""),
    ("/facts_subset/", ""),
    ("/relevant_patches.facts.", ""),
    ("/relevant_patches/facts/", ""),
)

_SOURCE_NOTE_FACT_CODES: frozenset[str] = frozenset({
    "missing_material_densities",
    "facts.missing_material_densities",
    "missing_pyrex_composition",
    "facts.missing_pyrex_composition",
    "missing_moderator_specs",
    "facts.missing_moderator_specs",
    "missing_operating_condition",
    "facts.missing_operating_condition",
    "missing_operating_conditions",
    "facts.missing_operating_conditions",
})

_DOWNSTREAM_DETAIL_FACT_CODES: frozenset[str] = frozenset({
    "missing_axial_layers",
    "facts.missing_axial_layers",
    "missing_standard_pin_radii",
    "facts.missing_standard_pin_radii",
    "missing_nozzle_core_plate_compositions",
    "facts.missing_nozzle_core_plate_compositions",
    "missing_spacer_grid_equivalent_geometry",
    "facts.missing_spacer_grid_equivalent_geometry",
    "missing_axial_overlaps",
    "facts.missing_axial_overlaps",
    "pyrex_bottom_plug_ambiguity",
    "facts.pyrex_bottom_plug_ambiguity",
})


def _normalized_text(value: Any) -> str:
    return re.sub(r"[^a-z0-9.+-]+", " ", str(value).lower()).strip()


def _collect_source_note_text(value: Any, *, key_hint: str = "") -> list[str]:
    """Collect user-visible FactsPatch note carriers recursively.

    FactsPatch deliberately does not define structured slots for every
    downstream fact (e.g. material density tables or insert composition
    vectors).  Those source-backed details are carried through
    ``source_notes``, nested ``source_note`` fields, and ``assumptions``.
    Reviewer findings that demand a non-existent structured Facts field must
    be checked against these carriers before they can block the gate.
    """

    if isinstance(value, dict):
        out: list[str] = []
        for key, child in value.items():
            child_key = str(key)
            if child_key in {"source_note", "source_notes", "assumption", "assumptions"}:
                if isinstance(child, (list, tuple, dict)):
                    out.extend(_collect_source_note_text(child, key_hint=child_key))
                elif child is not None:
                    out.append(str(child))
            elif key_hint in {"source_note", "source_notes", "assumption", "assumptions"}:
                out.extend(_collect_source_note_text(child, key_hint=key_hint))
            elif isinstance(child, (dict, list, tuple)):
                out.extend(_collect_source_note_text(child))
        return out
    if isinstance(value, (list, tuple)):
        out: list[str] = []
        for child in value:
            out.extend(_collect_source_note_text(child, key_hint=key_hint))
        return out
    if key_hint in {"source_note", "source_notes", "assumption", "assumptions"} and value is not None:
        return [str(value)]
    return []


def _number_tokens(text: str) -> set[str]:
    tokens: set[str] = set()
    for match in re.findall(r"(?<![a-z0-9])[-+]?\d+(?:\.\d+)?(?:e[-+]?\d+)?(?![a-z0-9])", text.lower()):
        normalized = match.lstrip("+")
        tokens.add(normalized)
        if "." in normalized:
            tokens.add(normalized.rstrip("0").rstrip("."))
    return {token for token in tokens if token not in {"0", "1"}}


def _format_float_token(value: float) -> str:
    return f"{value:.12g}".rstrip("0").rstrip(".")


def _number_token_covered(token: str, haystack: str) -> bool:
    if token in haystack:
        return True
    try:
        value = float(token)
    except ValueError:
        return False
    equivalents = {
        _format_float_token(value),
        _format_float_token(value * 100.0),
        _format_float_token(value / 100.0),
    }
    return any(item and item in haystack for item in equivalents)


def _source_note_schema_boundary_covered(
    draft: FactsReviewFindingDraft,
    facts_patch: dict[str, Any] | None,
) -> tuple[FactsReviewFindingDraft, dict[str, Any] | None]:
    """Reject false-positive requests for non-existent structured Facts fields.

    A reviewer may correctly notice that a source-backed material or operating
    detail is not a dedicated FactsPatch field, then incorrectly label that as
    a blocking Facts error even though the detail is already recorded in
    source notes for downstream gates.  This is a schema-boundary false
    positive, not a repair request.
    """

    if not facts_patch or draft.severity is not PlanFindingSeverity.ERROR:
        return draft, None
    code = draft.code.strip().lower()
    message = _normalized_text(" ".join([
        draft.code,
        draft.message,
        str(draft.expected_value or ""),
        str(draft.current_value or ""),
    ]))
    paths = tuple(draft.affected_json_paths)
    path_in_note_boundary = (
        not paths
        or all(
            path in {"/", ""}
            or _is_facts_recording_metadata_path(path)
            or path.endswith("/source_note")
            or "/source_notes/" in path
            for path in paths
        )
    )
    mentions_schema_boundary = any(
        marker in message
        for marker in (
            "source note",
            "source notes",
            "source_note",
            "not structured",
            "structured field",
            "dedicated field",
            "only recorded",
        )
    )
    is_known_source_note_fact = code in _SOURCE_NOTE_FACT_CODES
    if not ((is_known_source_note_fact or mentions_schema_boundary) and path_in_note_boundary):
        return draft, None

    note_text = _normalized_text(" ".join(_collect_source_note_text(facts_patch)))
    if not note_text:
        return draft, None
    required_numbers = _number_tokens(message)
    if required_numbers and not all(_number_token_covered(token, note_text) for token in required_numbers):
        return draft, None
    topic_markers = {
        "density": ("density", "densities", "g cm3", "g/cc"),
        "pyrex": ("pyrex", "b10", "b 10", "b11", "b 11", "boron", "b2o3"),
        "moderator": ("moderator", "coolant", "boron", "ppm", "pressure", "temperature"),
        "operating": ("operating", "coolant", "boron", "pressure", "power"),
        "material": ("material", "density", "composition", "isotope", "isotopic"),
    }
    topic_hits = [
        topic
        for topic, markers in topic_markers.items()
        if any(marker in message for marker in markers)
    ]
    if topic_hits and not any(
        any(marker in note_text for marker in topic_markers[topic])
        for topic in topic_hits
    ):
        return draft, None

    original = {
        "reason": "facts_source_note_schema_boundary_covered",
        "covered_by": "facts_patch.source_notes",
        "original_severity": draft.severity.value,
        "original_repairable_by_llm": draft.repairable_by_llm,
        "original_requires_human": draft.requires_human,
    }
    return draft.model_copy(update={
        "severity": PlanFindingSeverity.WARNING,
        "repairable_by_llm": False,
        "requires_human": False,
    }), original


def _normalize_downstream_detail_scope_classification(
    draft: FactsReviewFindingDraft,
) -> tuple[FactsReviewFindingDraft, dict[str, Any] | None]:
    """Keep schema-unrepresentable detailed geometry/material facts advisory.

    FactsPatch records high-level source contracts.  Detailed pin radii,
    spacer-grid equivalent geometry, full axial layer tables, nozzle/core-plate
    homogenization, and overlap semantics are represented in downstream patch
    families.  A Facts reviewer may flag them while reading the source; that
    finding is useful routing information but must not block Facts acceptance
    unless it points to a required FactsPatch contract field that is actually
    missing (scope, counts, variants, feature flags, or insert requirement
    existence).
    """

    if draft.severity is not PlanFindingSeverity.ERROR:
        return draft, None
    code = draft.code.strip().lower()
    message = _normalized_text(" ".join([
        draft.code,
        draft.message,
        str(draft.expected_value or ""),
        str(draft.current_value or ""),
    ]))
    downstream_markers = (
        "pin radi",
        "pellet radius",
        "clad outer",
        "guide tube wall",
        "instrument tube wall",
        "axial layer",
        "z range",
        "z ranges",
        "z-ranges",
        "plenum",
        "end plug",
        "bottom plug",
        "top plug",
        "nozzle",
        "core plate",
        "spacer grid",
        "equivalent geometry",
        "homogenized",
        "overlap",
        "downstream critical",
    )
    if code not in _DOWNSTREAM_DETAIL_FACT_CODES and not any(marker in message for marker in downstream_markers):
        return draft, None

    # Keep true FactsPatch contract omissions blocking.  The markers below are
    # the fields FactsPatch owns structurally; downstream-detail findings that
    # mention these only as already-present context should not block.
    hard_contract_markers = (
        "model_scope",
        "assembly_count",
        "core_lattice_size",
        "assembly_type_counts",
        "fuel_variant_requirements",
        "localized_insert_requirements",
        "has_spacer_grids",
        "expected_spacer_grid_count",
    )
    missing_contract_markers = (
        "missing model_scope",
        "missing assembly_count",
        "missing core_lattice_size",
        "missing assembly_type_counts",
        "missing fuel_variant_requirements",
        "missing localized_insert_requirements",
        "missing has_spacer_grids",
        "missing expected_spacer_grid_count",
        "omits model_scope",
        "omits assembly_count",
        "omits core_lattice_size",
        "omits assembly_type_counts",
        "omits fuel_variant_requirements",
        "omits localized_insert_requirements",
        "omits has_spacer_grids",
        "omits expected_spacer_grid_count",
    )
    if any(marker in message for marker in missing_contract_markers):
        return draft, None
    if code == "facts_patch_incomplete":
        # A generic incompleteness finding is only demoted when the prose is
        # exclusively about downstream-detail topics, not when it identifies a
        # missing hard FactsPatch contract field.
        mentions_hard_contract = any(marker in message for marker in hard_contract_markers)
        if mentions_hard_contract and not any(marker in message for marker in downstream_markers):
            return draft, None

    downstream_impact = list(dict.fromkeys([
        *draft.downstream_impact,
        "materials_contract",
        "universes_contract",
        "axial_geometry_contract",
        "assembled_plan_contract",
    ]))
    original = {
        "reason": "facts_downstream_detail_scope",
        "owner_route": "downstream_patch_family",
        "original_severity": draft.severity.value,
        "original_repairable_by_llm": draft.repairable_by_llm,
        "original_requires_human": draft.requires_human,
    }
    return draft.model_copy(update={
        "severity": PlanFindingSeverity.WARNING,
        "repairable_by_llm": False,
        "requires_human": False,
        "downstream_impact": downstream_impact,
    }), original


def _canonicalize_facts_review_path(path: str) -> str:
    """Canonicalize reviewer path drift to FactsPatch JSON Pointer style."""

    p = str(path).strip()
    for prefix, replacement in _FACTS_REVIEW_PATH_PREFIXES:
        if p.startswith(prefix):
            p = replacement + p[len(prefix):]
            break
    if p in {"relevant_patches.facts", "/relevant_patches.facts", "/relevant_patches/facts", "facts_subset"}:
        p = ""
    if not p.startswith("/"):
        p = "/" + p
    p = re.sub(r"\[([0-9]+)\]", r"/\1", p)
    # Reviewers often emit dotted object/list paths.  Convert those to JSON
    # Pointer tokens after known FactsPatch roots, while preserving ordinary
    # field names that do not contain nesting.
    if "." in p:
        p = "/" + "/".join(token for token in p.strip("/").split(".") if token)
    p = re.sub(r"/+", "/", p)
    return p


def _is_facts_recording_metadata_path(path: str) -> bool:
    return any(path == root or path.startswith(f"{root}/") for root in _FACTS_RECORDING_METADATA_ROOTS)


def _normalize_recording_metadata_classification(
    draft: FactsReviewFindingDraft,
) -> tuple[FactsReviewFindingDraft, dict[str, Any] | None]:
    # Metadata repairs record an evidence-backed gap; they never infer a value.
    paths = draft.affected_json_paths
    if not paths or not all(_is_facts_recording_metadata_path(path) for path in paths):
        return draft, None
    original = {
        "reason": "facts_metadata_recording",
        "original_repairable_by_llm": draft.repairable_by_llm,
        "original_requires_human": draft.requires_human,
    }
    return draft.model_copy(update={"repairable_by_llm": True, "requires_human": False}), original


def _normalize_confirmation_not_error_classification(
    draft: FactsReviewFindingDraft,
) -> tuple[FactsReviewFindingDraft, dict[str, Any] | None]:
    """Downgrade self-contradictory reviewer confirmations.

    Real reviewers occasionally emit an error-severity finding whose own
    message explicitly says the condition is a coverage confirmation rather
    than an error.  Treating that as a non-repairable blocker prevents the
    revision loop from fixing genuinely remaining repairable findings.
    Keep the finding visible as a warning with provenance instead of
    suppressing the code.
    """
    if (
        draft.severity is not PlanFindingSeverity.ERROR
        or draft.category is not PlanFindingCategory.SOURCE_COVERAGE
        or draft.requires_human
        or draft.repairable_by_llm
    ):
        return draft, None
    message = draft.message.lower()
    confirmation_markers = (
        "recorded as a coverage confirmation",
        "not an error",
    )
    if not all(marker in message for marker in confirmation_markers):
        return draft, None
    original = {
        "reason": "facts_reviewer_confirmation_not_error",
        "original_severity": draft.severity.value,
        "original_repairable_by_llm": draft.repairable_by_llm,
        "original_requires_human": draft.requires_human,
    }
    return draft.model_copy(update={"severity": PlanFindingSeverity.WARNING}), original


def _normalize_downstream_material_scope_classification(
    draft: FactsReviewFindingDraft,
) -> tuple[FactsReviewFindingDraft, dict[str, Any] | None]:
    """Keep Materials/MU-owned fuel details from blocking the Facts Gate.

    Facts fuel variants identify source-declared variants and coarse attributes
    such as enrichment/density when supplied.  Isotope vectors, oxygen
    stoichiometry, and material composition are enforced by Materials /
    Material-Universe contracts.  A reviewer may still mention that downstream
    gap while looking at ``/fuel_variant_requirements``; preserve it as a
    warning with owner metadata instead of treating it as a Facts repair.
    """

    code = draft.code.strip().lower()
    paths = tuple(draft.affected_json_paths)
    message = draft.message.lower()
    material_detail_markers = (
        "isotope",
        "isotopic",
        "composition",
        "stoichiometry",
        "o-16",
        "u-234",
        "u-235",
        "u-236",
        "u-238",
    )
    is_known_code = code in {
        "fuel_variant_missing_isotope_composition",
        "facts.fuel_variant_missing_isotope_composition",
    }
    is_fuel_variant_material_detail = (
        any(path == "/fuel_variant_requirements" or path.startswith("/fuel_variant_requirements/") for path in paths)
        and any(marker in message for marker in material_detail_markers)
    )
    if not (is_known_code or is_fuel_variant_material_detail):
        return draft, None

    downstream_impact = list(dict.fromkeys([
        *draft.downstream_impact,
        "materials_contract",
        "material_universe_contract",
    ]))
    original = {
        "reason": "facts_downstream_material_scope",
        "owner_route": "materials",
        "original_severity": draft.severity.value,
        "original_repairable_by_llm": draft.repairable_by_llm,
        "original_requires_human": draft.requires_human,
    }
    return draft.model_copy(update={
        "severity": PlanFindingSeverity.WARNING,
        "repairable_by_llm": False,
        "requires_human": False,
        "downstream_impact": downstream_impact,
    }), original


def _normalize_blank_operating_state_classification(
    draft: FactsReviewFindingDraft,
) -> tuple[FactsReviewFindingDraft, dict[str, Any] | None]:
    """Treat an explicitly blank operating state as a recorded base-state gap.

    VERA-style inputs may carry an empty operating_state field to mean the case
    has no named operating variant.  That is not a human-required Facts
    ambiguity when the source itself is explicitly blank.  Downstream code uses
    canonical base-state labels for control-state IDs; the Facts Gate should
    keep this visible without blocking.
    """

    code = draft.code.strip().lower()
    message = draft.message.lower()
    paths = tuple(draft.affected_json_paths)
    blank_markers = (
        '""',
        "blank",
        "empty",
        "omitted",
        "unspecified",
    )
    is_operating_state_code = code in {
        "missing_operating_state",
        "missing_operating_state_unrecorded",
        "facts.missing_operating_state",
    }
    is_selected_variant_path = any(
        path == "/selected_variant" or path.startswith("/selected_variant/")
        for path in paths
    )
    mentions_operating_state = "operating state" in message or "operating_state" in message
    mentions_blank_source = any(marker in message for marker in blank_markers)
    if not (is_operating_state_code and (is_selected_variant_path or mentions_operating_state) and mentions_blank_source):
        return draft, None

    original = {
        "reason": "facts_blank_operating_state_base_gap",
        "canonical_value": "base",
        "original_severity": draft.severity.value,
        "original_repairable_by_llm": draft.repairable_by_llm,
        "original_requires_human": draft.requires_human,
    }
    return draft.model_copy(update={
        "severity": PlanFindingSeverity.WARNING,
        "repairable_by_llm": False,
        "requires_human": False,
    }), original


def _normalize_non_error_human_requirement(
    draft: FactsReviewFindingDraft,
) -> tuple[FactsReviewFindingDraft, dict[str, Any] | None]:
    """Non-error findings may be advisory, but must not remain human blockers."""

    if draft.severity is PlanFindingSeverity.ERROR or not draft.requires_human:
        return draft, None
    original = {
        "reason": "facts_non_error_human_advisory",
        "original_severity": draft.severity.value,
        "original_requires_human": draft.requires_human,
        "original_repairable_by_llm": draft.repairable_by_llm,
    }
    return draft.model_copy(update={"requires_human": False, "repairable_by_llm": False}), original


def _is_excerpt_limited_finding(draft: FactsReviewFindingDraft) -> bool:
    """Reject chunk-local missing-evidence claims.

    Facts review can be split across source chunks.  A reviewer for one chunk
    may say a value is "not provided in this source excerpt" even though a
    later chunk contains the source-backed value.  That is not a valid
    whole-source ambiguity and must not be promoted to a human blocker.
    """

    code = draft.code.strip().lower()
    message = draft.message.lower()
    expected = str(draft.expected_value or "").lower()
    if draft.category not in {
        PlanFindingCategory.SOURCE_COVERAGE,
        PlanFindingCategory.UNSUPPORTED_INFERENCE,
    }:
        return False
    excerpt_markers = (
        "source excerpt",
        "provided text",
        "provided evidence",
        "supplied excerpt",
        "supplied source",
        "supplied source excerpt",
        "provided source excerpt",
        "assigned excerpt",
        "evidence pack",
    )
    scope_limited = (
        any(marker in message for marker in excerpt_markers)
        or "not provided in source excerpt" in expected
        or "not provided in the source excerpt" in expected
    )
    if draft.requires_human:
        return scope_limited
    return (
        code.startswith("unsupported_inference.")
        or code.endswith("_unevidenced")
    ) and scope_limited


def normalize_facts_review_finding(
    draft: FactsReviewFindingDraft,
    *,
    facts_patch: dict[str, Any] | None = None,
) -> FactsFindingNormalization:
    """Normalize one reviewer finding before it can affect the Facts Gate.

    This is the deterministic firewall for reviewer drift.  It canonicalizes
    paths, routes/downshifts known owner-boundary findings, rejects chunk-local
    missing-evidence claims, and fail-closes malformed blocking contracts.
    """

    if not draft.code.strip():
        return FactsFindingNormalization(
            rejected={"code": "facts_review.invalid_finding_contract", "reason": "blank code"}
        )

    normalized_paths = [_canonicalize_facts_review_path(path) for path in draft.affected_json_paths]
    if normalized_paths != draft.affected_json_paths:
        draft = draft.model_copy(update={"affected_json_paths": normalized_paths})

    if any(not path.startswith("/") or path.startswith("/materials") or path.startswith("/universes") for path in draft.affected_json_paths):
        return FactsFindingNormalization(
            draft=draft,
            rejected={"code": "facts_review.path_out_of_scope", "finding_code": draft.code},
        )

    classification_override = None
    for normalizer in (
        _normalize_downstream_detail_scope_classification,
        _normalize_recording_metadata_classification,
        _normalize_confirmation_not_error_classification,
        _normalize_downstream_material_scope_classification,
        _normalize_blank_operating_state_classification,
        lambda item: _source_note_schema_boundary_covered(item, facts_patch),
        _normalize_non_error_human_requirement,
    ):
        if classification_override is not None:
            break
        draft, classification_override = normalizer(draft)

    if _is_excerpt_limited_finding(draft):
        return FactsFindingNormalization(
            draft=draft.model_copy(update={"severity": PlanFindingSeverity.WARNING, "requires_human": False}),
            rejected={
                "code": "facts_review.excerpt_limited_finding",
                "finding_code": draft.code,
                "reason": "chunk-local missing-evidence claim is not a whole-source human blocker",
            },
        )

    if draft.requires_human and draft.repairable_by_llm:
        return FactsFindingNormalization(
            draft=draft,
            rejected={"code": "facts_review.invalid_finding_contract", "finding_code": draft.code},
        )

    if draft.severity is PlanFindingSeverity.ERROR and draft.category is not PlanFindingCategory.PHYSICAL_AMBIGUITY:
        # An error must be both evidence-grounded and actionable against
        # a specific FactsPatch field.  This rejects self-contradictory
        # critic output such as "this is consistent; no issue" labelled
        # as an error with no affected path; accepting such a finding
        # would fail-close a valid plan without a repairable contract.
        if not draft.evidence_hashes or not draft.affected_json_paths:
            return FactsFindingNormalization(
                draft=draft,
                rejected={"code": "facts_review.invalid_finding_contract", "finding_code": draft.code},
            )

    return FactsFindingNormalization(draft=draft, classification_override=classification_override)


def _normalize(output: FactsReviewModelOutput, pack: PlanEvidencePack) -> tuple[list[PlanReviewFinding], list[dict[str, Any]]]:
    evidence = {item.evidence_hash: item for item in pack.source_excerpts}
    accepted: list[PlanReviewFinding] = []
    rejected: list[dict[str, Any]] = []
    for index, draft in enumerate(output.findings):
        unknown = set(draft.evidence_hashes) - set(evidence)
        if unknown:
            rejected.append({"code": "facts_review.unknown_evidence_hash", "finding_code": draft.code, "unknown": sorted(unknown)})
            continue
        normalized = normalize_facts_review_finding(
            draft,
            facts_patch=pack.relevant_patches.get("facts", {}),
        )
        if normalized.draft is not None:
            output.findings[index] = normalized.draft
        if normalized.rejected is not None:
            rejected.append(normalized.rejected)
            continue
        if normalized.draft is None:
            continue
        draft = normalized.draft
        excerpts = [evidence[key] for key in draft.evidence_hashes]
        finding = PlanReviewFinding(
            gate_id=PlanGateId.FACTS, code=draft.code, severity=draft.severity,
            category=draft.category, message=draft.message, source_evidence=excerpts,
            affected_patch_types=["facts"], affected_json_paths=draft.affected_json_paths,
            repairable_by_llm=draft.repairable_by_llm, requires_human=draft.requires_human,
            confidence=draft.confidence,
            metadata={"expected_value": draft.expected_value, "current_value": draft.current_value,
                      "candidate_interpretations": [item.model_dump(mode="json") for item in draft.candidate_interpretations],
                      "downstream_impact": draft.downstream_impact,
                      **({"classification_override": normalized.classification_override} if normalized.classification_override else {})},
        )
        accepted.append(finding)
    # Finding identity excludes wording and unions evidence under the same semantic fingerprint.
    merged: dict[str, PlanReviewFinding] = {item.finding_id: item for item in accepted}
    return list(merged.values()), rejected


def _aggregate_coverage(outputs: list[dict[str, Any]], *, expected_stage_count: int | None = None) -> bool:
    """Aggregate review-stage outputs into a single coverage decision.

    Rules (fail-closed):

    * Every stage ``review_status`` must be in ``_ACCEPTED_REVIEW_STATUSES``.
    * No finding may have ``severity == "error"`` (blocking).
    * The number of outputs must match *expected_stage_count* when provided.
    """
    if not outputs:
        return False
    if expected_stage_count is not None and len(outputs) != expected_stage_count:
        return False
    statuses = [
        str(item.get("output", {}).get("review_status", "")).strip()
        for item in outputs
    ]
    if any(status not in _ACCEPTED_REVIEW_STATUSES for status in statuses):
        return False
    has_blocking = any(
        str(finding.get("severity", "")).lower() == "error"
        for item in outputs
        for finding in item.get("output", {}).get("findings", [])
    )
    return not has_blocking


def run_facts_review(*, evidence_packs: list[PlanEvidencePack], reviewer_client: Any, state: Any, policy: PlanClosedLoopPolicy) -> FactsReviewResult:
    # Phase 8B Step 3: stage-split path.
    if getattr(policy, "facts_review_stage_split", False) and evidence_packs:
        return _run_facts_review_staged(
            evidence_packs=evidence_packs,
            reviewer_client=reviewer_client,
            state=state,
            policy=policy,
        )
    result = FactsReviewResult()
    all_findings: list[PlanReviewFinding] = []
    all_rejected: list[dict[str, Any]] = []
    for pack in evidence_packs:
        call = run_structured_review_call(
            client=reviewer_client, initial_prompt=build_facts_review_prompt(pack),
            retry_prompt_builder=lambda raw, error: build_facts_review_schema_retry_prompt(pack, error, raw),
            output_model=FactsReviewModelOutput,
            call_spec=StructuredReviewCallSpec(
                role_id="facts_review", gate_id=PlanGateId.FACTS,
                schema_name="FactsReviewModelOutput", json_schema=FactsReviewModelOutput.model_json_schema(),
                artifact_prefix="facts_review",
            input_payload_hash=canonical_payload_hash(pack)
            ), state=state, stage=state.plan_loop_stages.get("plan_gate_facts"), policy=policy,
        )
        result.reviewer_calls += call.call_count
        result.schema_retries += call.schema_retry_count
        result.raw_outputs.extend(call.raw_outputs)
        for attempt in call.attempts:
            result.call_metadata.append({"pack_id": pack.evidence_pack_id, **attempt.model_dump(mode="json", exclude={"raw_text"})})
        if not call.ok or call.parsed_output is None:
            # Phase 8B Step 3: classify the failure precisely.
            result.error = f"facts_review.schema_invalid: {call.error_detail}"
            result.failure_code = _classify_review_failure(call)
            return result
        output = FactsReviewModelOutput.model_validate(call.parsed_output)
        findings, rejected = _normalize(output, pack)
        all_findings.extend(findings)
        all_rejected.extend(rejected)
        result.outputs.append({"pack_id": pack.evidence_pack_id, "output": output.model_dump(mode="json")})
    result.findings = list({finding.finding_id: finding for finding in all_findings}.values())
    result.rejected = all_rejected
    expected = {item.evidence_hash for pack in evidence_packs for item in pack.source_excerpts}
    reviewed: set[str] = set()
    for out_entry in result.outputs:
        reviewed.update(out_entry["output"].get("reviewed_evidence_hashes", []))
        for finding in out_entry["output"].get("findings", []):
            reviewed.update(finding.get("evidence_hashes", []))
    result.coverage_complete = (
        _aggregate_coverage(result.outputs, expected_stage_count=len(evidence_packs))
        and not any(item.severity is PlanFindingSeverity.ERROR for item in result.findings)
    )
    if not result.coverage_complete:
        result.failure_code = "facts_review.coverage_incomplete"
    result.ok = not result.error
    return result


# ---------------------------------------------------------------------------
# Phase 8B Step 3: staged review path
# ---------------------------------------------------------------------------


def _run_facts_review_staged(
    *,
    evidence_packs: list[PlanEvidencePack],
    reviewer_client: Any,
    state: Any,
    policy: PlanClosedLoopPolicy,
) -> FactsReviewResult:
    """Run per-topic stage calls instead of one monolithic per-pack call.

    Each stage sees only its FactsPatch subset + the source excerpts,
    producing a much smaller and more focused prompt.
    """

    from .facts_review_stages import (
        STAGE_ORDER,
        FactsReviewStageRequest,
        build_stage_review_prompt,
        build_stage_schema_retry_prompt,
        extract_facts_subset,
    )

    result = FactsReviewResult()
    all_findings: list[PlanReviewFinding] = []
    all_rejected: list[dict[str, Any]] = []

    # Use the first pack as the evidence source.  Stage-split mode does
    # not iterate per-pack; it iterates per-stage using the consolidated
    # evidence from the first pack.
    base_pack = evidence_packs[0]
    facts_patch = base_pack.relevant_patches.get("facts", {})
    base_excerpts = [
        s.model_dump(mode="json") if hasattr(s, "model_dump") else s
        for s in base_pack.source_excerpts
    ]
    confirmed_summary = base_pack.metadata.get("facts_summary", {})
    consistency_issues = base_pack.metadata.get("facts_consistency_issues", [])

    for stage in STAGE_ORDER:
        facts_subset = extract_facts_subset(facts_patch, stage)
        stage_request = FactsReviewStageRequest(
            stage=stage,
            target_fields=tuple(facts_subset.keys()),
            facts_subset=facts_subset,
            evidence_excerpts=base_excerpts,
            confirmed_facts_summary=confirmed_summary,
            consistency_issues=consistency_issues,
        )
        call = run_structured_review_call(
            client=reviewer_client,
            initial_prompt=build_stage_review_prompt(stage_request, base_pack),
            retry_prompt_builder=lambda raw, error, sr=stage_request, bp=base_pack: build_stage_schema_retry_prompt(sr, bp, error, raw),
            output_model=FactsReviewModelOutput,
            call_spec=StructuredReviewCallSpec(
                role_id="facts_review",
                gate_id=PlanGateId.FACTS,
                schema_name="FactsReviewModelOutput",
                json_schema=FactsReviewModelOutput.model_json_schema(),
                artifact_prefix=f"facts_review_{stage.value}",
                input_payload_hash=canonical_payload_hash(
                    {"stage": stage.value, "facts_subset": facts_subset}
                ),
            ),
            state=state,
            stage=state.plan_loop_stages.get("plan_gate_facts"),
            policy=policy,
        )
        result.reviewer_calls += call.call_count
        result.schema_retries += call.schema_retry_count
        result.raw_outputs.extend(call.raw_outputs)
        for attempt in call.attempts:
            result.call_metadata.append(
                {"stage": stage.value, **attempt.model_dump(mode="json", exclude={"raw_text"})}
            )
        if not call.ok or call.parsed_output is None:
            result.error = f"facts_review.schema_invalid[{stage.value}]: {call.error_detail}"
            result.failure_code = _classify_review_failure(call)
            return result
        output = FactsReviewModelOutput.model_validate(call.parsed_output)
        # Build a synthetic pack for _normalize that carries the same
        # source excerpts but a pruned facts subset.
        synthetic_pack = base_pack.model_copy(
            update={"relevant_patches": {"facts": facts_subset}}
        )
        findings, rejected = _normalize(output, synthetic_pack)
        all_findings.extend(findings)
        all_rejected.extend(rejected)
        result.outputs.append(
            {"stage": stage.value, "output": output.model_dump(mode="json")}
        )

    result.findings = list({finding.finding_id: finding for finding in all_findings}.values())
    result.rejected = all_rejected
    result.coverage_complete = _aggregate_coverage(result.outputs, expected_stage_count=len(STAGE_ORDER))
    if not result.coverage_complete:
        result.failure_code = "facts_review.coverage_incomplete"
    result.ok = not result.error
    return result


# ---------------------------------------------------------------------------
# Phase 8B Step 3: failure classification
# ---------------------------------------------------------------------------

# Phrases that indicate a free-text "approve" (no JSON structure).
_FREE_TEXT_APPROVE_PHRASES: tuple[str, ...] = (
    "looks good",
    "no issues",
    "no issues found",
    "everything looks correct",
    "i approve",
    "approved",
    "accepted",
    "no findings",
    "no errors",
    "all correct",
    "consistent with the source",
    "patch is correct",
)


def _classify_review_failure(call: Any) -> str:
    """Classify a failed structured review call precisely.

    Distinguishes:
    * ``facts_review.budget_exhausted`` — LLM budget ran out.
    * ``facts.reviewer_empty_response`` — all attempts returned empty content.
    * ``facts.reviewer_free_text_approve`` — prose-only "approve" without JSON.
    * ``facts_review.schema_invalid`` — JSON was present but malformed.
    """

    if call.error_code == "planning.closed_loop.budget_exhausted":
        return "facts_review.budget_exhausted"

    # If the call was successful, empty raw_text is a data propagation bug.
    if call.ok is True:
        return call.error_code or "facts_review.schema_invalid"

    # Inspect raw outputs captured at the provider boundary (P1 fix).
    # Fall back to attempt.raw_text for callers that haven't been updated.
    raw_texts = list(getattr(call, "raw_outputs", []) or [])
    if not raw_texts:
        raw_texts = [
            getattr(a, "raw_text", "") or "" for a in (call.attempts or [])
        ]
    # Empty response: all attempts produced empty content.
    if raw_texts and all(not text.strip() for text in raw_texts):
        return "facts.reviewer_empty_response"
    # Free-text approve: the response is short prose that matches an
    # approval phrase but contains no JSON structure.
    for text in raw_texts:
        lower = text.strip().lower()
        if lower and len(lower) < 200 and "{" not in lower:
            if any(phrase in lower for phrase in _FREE_TEXT_APPROVE_PHRASES):
                return "facts.reviewer_free_text_approve"
    return call.error_code or "facts_review.schema_invalid"
