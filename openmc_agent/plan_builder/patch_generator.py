"""LLM patch generator for incremental plan building (Phase 4).

Calls an LLM to generate **exactly one** patch at a time.  Each patch is a
small JSON object (a few hundred to a few thousand characters) — never a 25 KB
monolithic SimulationPlan.  Parse failures and validation failures trigger a
targeted retry for the *current patch only*; already-valid patches in the
:class:`PlanBuildState` are never touched.

Design constraints
------------------
* **No full plan.**  The generator never outputs a SimulationPlan.
* **No full lattice.**  PinMapPatch contains only special coordinates.
* **Fake-LLM friendly.**  The ``llm_client`` is a simple callable
  ``(prompt: str) -> str``; tests inject a :class:`FakePatchLLM`.
* **No OpenMC, no renderer.**
"""

from __future__ import annotations

import json
import re
from typing import Any, Literal

from pydantic import Field

from openmc_agent.schemas import AgentBaseModel

from .patches import (
    PatchParseError,
    parse_patch_content,
    get_patch_allowed_top_level_keys,
    get_patch_forbidden_top_level_keys,
    get_patch_json_schema,
)
from .patch_prompts import build_patch_prompt, build_retry_prompt
from .state import PlanBuildState, PlanPatchEnvelope
from .closed_loop.fingerprints import compute_candidate_hash
from .material_resolution import infer_material_aliases, resolve_material_id
from .validators import (
    PatchValidationContext,
    PatchValidationResult,
    validate_patch,
)


# ---------------------------------------------------------------------------
# Context / attempt / result models
# ---------------------------------------------------------------------------


class ContextFactValue(AgentBaseModel):
    field_path: str
    value: Any = None
    provenance_kind: Literal[
        "source_backed",
        "human_confirmed",
        "deterministically_derived",
        "accepted_upstream",
        "unresolved",
        "unknown",
        "compatibility_default",
    ] = "unresolved"
    source_claim_ids: list[str] = Field(default_factory=list)
    source_span_ids: list[str] = Field(default_factory=list)
    derivation_code: str | None = None
    confidence: float = 0.0
    authoritative: bool = False
    value_hash: str = ""


class PatchGenerationContext(AgentBaseModel):
    """Context passed to the patch generator and prompt builder."""

    benchmark_id: str | None = None
    selected_variant: str | None = None
    confirmed_facts: dict[str, Any] = Field(default_factory=dict)
    extracted_facts: dict[str, Any] = Field(default_factory=dict)
    validated_patch_summaries: dict[str, Any] = Field(default_factory=dict)
    reference_summary: dict[str, Any] = Field(default_factory=dict)
    strict_benchmark: bool = False
    expected_counts: dict[str, int] = Field(default_factory=dict)
    expected_counts_complete: bool = False
    reference_expected_counts: dict[str, int] = Field(default_factory=dict)
    known_material_ids: list[str] = Field(default_factory=list)
    material_aliases: dict[str, str] = Field(default_factory=dict)
    known_universe_ids: list[str] = Field(default_factory=list)
    known_lattice_ids: list[str] = Field(default_factory=list)
    active_fuel_region_cm: tuple[float, float] | None = None
    axial_domain_cm: tuple[float, float] | None = None
    few_shot_case_ids: list[str] = Field(default_factory=list)
    known_cell_ids: list[str] = Field(default_factory=list)
    cell_owner_universe_ids: dict[str, list[str]] = Field(default_factory=dict)
    material_roles_by_id: dict[str, str] = Field(default_factory=dict)
    known_overlay_summaries: list[dict[str, Any]] = Field(default_factory=list)
    has_spacer_grids: bool = False
    expected_spacer_grid_count: int | None = None
    # P2-FULLCORE-1: multi-assembly context fields
    model_scope: str | None = None
    assembly_count: int | None = None
    core_lattice_size: tuple[int, int] | None = None
    assembly_type_counts: dict[str, int] = Field(default_factory=dict)
    known_assembly_type_ids: list[str] = Field(default_factory=list)
    assembly_pitch_cm: float | None = None
    scoped_expected_counts: list[dict[str, Any]] = Field(default_factory=list)
    # P2-FULLCORE-2C-A: localized insert profile context
    known_insert_profile_ids: list[str] = Field(default_factory=list)
    insert_profile_summaries: list[dict[str, Any]] = Field(default_factory=list)
    movable_insert_facts: dict[str, Any] = Field(default_factory=dict)
    # P2-FULLCORE-2D-B: fuel variant source contract
    fuel_variant_requirements: list[dict[str, Any]] = Field(default_factory=list)
    material_summaries: list[dict[str, Any]] = Field(default_factory=list)
    universe_summaries: list[dict[str, Any]] = Field(default_factory=list)
    assembly_fuel_binding_summaries: list[dict[str, Any]] = Field(default_factory=list)
    # P2-FULLCORE-2C-C: localized insert placement contract
    localized_insert_requirements: list[dict[str, Any]] = Field(default_factory=list)
    localized_insert_universe_summaries: list[dict[str, Any]] = Field(default_factory=list)
    assembly_insert_binding_summaries: list[dict[str, Any]] = Field(default_factory=list)
    # Phase-8A Step 3: structured evidence claims produced by the optional
    # investigation stage.  Each entry is the canonical claim payload
    # produced by ``collect_evidence_for_patch_prompt`` (claim_id, subject,
    # predicate, value, status, criticality, source_spans).  Empty list
    # (the default) means no investigation was run; existing patch prompts
    # are unchanged in that case.
    investigation_evidence: list[dict[str, Any]] = Field(default_factory=list)
    # Phase-8A Step 6: derived planning constraints (distinct from
    # source-backed EvidenceClaims).  Populated from the
    # GeometryComponentInventory + Material/Universe/Placement/Axial
    # requirement sets.  ``derivation_status=deterministically_derived``
    # entries must NOT be confused with source quotations (the P0-4 bug).
    # When non-empty, the prompt renderer emits a separate "Derived
    # Planning Constraints" section.
    planning_constraints: list[dict[str, Any]] = Field(default_factory=list)
    # Phase-8B Step 2: context facts with provenance tracking.
    context_facts: dict[str, ContextFactValue] = Field(default_factory=dict)
    # Phase-8B Step 3: FactsRequirementSkeleton — compiled from evidence
    # ledger and feature contract to constrain the Facts patch LLM.
    facts_requirement_skeleton: dict[str, Any] | None = None
    facts_evidence_contract: dict[str, Any] | None = None


class RetryPatchGenerationContext(AgentBaseModel):
    """Wrapper that carries typed retry constraints alongside the base context.

    When passed to :func:`generate_patch` as ``context``, the generator
    unwraps the base context and exposes the retry fields to the prompt
    builder so the LLM sees required IDs, protected invariants and prior
    failure codes.
    """

    base_context: PatchGenerationContext | None = None
    retry_request_id: str = ""
    reason_code: str = ""
    source_issue_codes: list[str] = Field(default_factory=list)
    required_ids: list[str] = Field(default_factory=list)
    required_properties: list[str] = Field(default_factory=list)
    affected_json_paths: list[str] = Field(default_factory=list)
    protected_invariants: list[str] = Field(default_factory=list)
    accepted_upstream_summaries: dict[str, Any] = Field(default_factory=dict)
    confirmed_records: list[dict[str, Any]] = Field(default_factory=list)
    prior_candidate_hashes: list[str] = Field(default_factory=list)
    prior_failure_codes: list[str] = Field(default_factory=list)

    def unwrap(self) -> PatchGenerationContext:
        return self.base_context or PatchGenerationContext()


def _retry_context_requirement_block(retry_context: RetryPatchGenerationContext) -> str:
    """Render retry constraints for the first generation attempt.

    ``build_retry_prompt`` only runs after a candidate has failed validation.
    Owner retry requests need their exact IDs and protected invariants in the
    first attempt as well, otherwise the producer can regenerate a plausible
    patch that does not target the dependency issue.
    """

    lines = [
        "=== Executable retry context ===",
        f"Retry request: {retry_context.retry_request_id or '(unknown)'}",
        f"Reason code: {retry_context.reason_code or '(unknown)'}",
        f"Source issue codes: {', '.join(retry_context.source_issue_codes) or '(none)'}",
        f"Required IDs: {', '.join(retry_context.required_ids) or '(none)'}",
        f"Required properties: {', '.join(retry_context.required_properties) or '(none)'}",
        f"Affected JSON paths: {', '.join(retry_context.affected_json_paths) or '(none)'}",
        f"Protected invariants: {', '.join(retry_context.protected_invariants) or '(none)'}",
        "Constraints:",
        "- Generate ONLY the requested owner patch; do NOT generate a full plan.",
        "- Include every required ID exactly as listed; do NOT substitute similar IDs.",
        "- Preserve every protected invariant and every already-valid unrelated object.",
        "- Do NOT modify patches outside the requested owner patch.",
    ]
    if retry_context.prior_failure_codes:
        lines.append(f"- Prior failure codes: {', '.join(retry_context.prior_failure_codes)}")
    return "\n".join(lines)


class PatchGenerationAttempt(AgentBaseModel):
    """Record of a single LLM call attempt."""

    attempt_index: int
    patch_type: str = ""
    prompt_text: str | None = None
    raw_text: str | None = None
    raw_chars: int = 0
    parsed: bool = False
    parsed_content: dict[str, Any] | None = None
    candidate_hash: str | None = None
    validated: bool = False
    issues: list[dict[str, Any]] = Field(default_factory=list)
    error: str | None = None
    contains_full_plan_markers: bool = False
    contains_full_lattice_suspected: bool = False
    output_mode_used: str = ""
    semantic_normalizations: list[dict[str, Any]] = Field(default_factory=list)
    retry_mode: str | None = None
    retry_allowed_paths: list[str] = Field(default_factory=list)
    retry_drift: list[dict[str, Any]] = Field(default_factory=list)
    # P0-LARGE-STRUCTURED-PATCH telemetry fields.
    finish_reason: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    reasoning_tokens: int | None = None
    total_tokens: int | None = None
    structured_fallback_used: bool = False
    structured_fallback_reasons: list[str] = Field(default_factory=list)


class PatchGenerationResult(AgentBaseModel):
    """Result of :func:`generate_patch`."""

    ok: bool = False
    patch_type: str
    envelope: PlanPatchEnvelope | None = None
    parsed_patch: dict[str, Any] | None = None
    validation: dict[str, Any] | None = None
    attempts: list[PatchGenerationAttempt] = Field(default_factory=list)
    issues: list[dict[str, Any]] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Output diagnostics
# ---------------------------------------------------------------------------

# Markers that are definitive evidence of a full SimulationPlan output.
# If ANY of these appear in a patch response (outside the expected patch_type
# field), the response is a forbidden full-plan output.
_FULL_PLAN_MARKERS: tuple[str, ...] = (
    '"complex_model"',
    '"simulation_plan"',
    '"materials.xml"',
    '"geometry.xml"',
    '"settings.xml"',
    '"capability_report"',
    '"execution_check"',
    '"plot_specs"',
    '"schema_version"',
)

# Fields that are part of SimulationPlan but NOT part of any patch.
# Their presence means the LLM returned a full plan.
_PLAN_ONLY_FIELDS: tuple[str, ...] = (
    "complex_model",
    "capability_report",
    "execution_check",
    "plot_specs",
    "schema_version",
    "model_spec",
)

# Thresholds for full-lattice suspicion in pin_map output.  The check is
# *structural* (see ``_detect_full_lattice``): a single coordinate list whose
# size approaches ``lattice_size[0] x lattice_size[1]`` is a near-full dump.
# We deliberately do NOT reject merely because the raw text is long or has many
# brackets — a legitimate multi-assembly special-coordinate map (guide tubes +
# pyrex + thimble plugs + RCCA, each its own list) can legitimately be several
# KB and contain ~100 coords in aggregate while no single list exceeds a
# handful of cells.
# A single list covering this fraction of all lattice cells is a near-full
# dump.  50% stays clear of reactor designs with many legitimate special
# paths (e.g. BWR part-length rods / water rods can be ~30-40% of cells).
_FULL_LATTICE_FRAC_THRESHOLD: float = 0.5
# Absolute fallback when lattice_size is absent/unparseable (~a full 17x17).
_FULL_LATTICE_ABS_LIST_THRESHOLD: int = 200
# Pure DoS backstop; far above any legitimate special-coordinate pin_map.
_FULL_LATTICE_RAW_CHARS_BACKSTOP: int = 50_000


def _detect_full_plan_markers(raw: str) -> bool:
    """Check if raw output contains markers of a full SimulationPlan."""
    return any(marker in raw for marker in _FULL_PLAN_MARKERS)


def _detect_full_plan_in_parsed(content: dict[str, Any], patch_type: str) -> bool:
    """Check if parsed JSON contains SimulationPlan-only fields.

    A valid patch should never have top-level keys like ``complex_model``,
    ``capability_report``, ``execution_check``, etc.
    """
    for field in _PLAN_ONLY_FIELDS:
        if field in content:
            return True
    # Also check for nested complex_model content that doesn't belong.
    if patch_type != "materials" and "materials" in content and isinstance(
        content["materials"], list
    ) and len(content.get("materials", [])) > 0:
        # materials list is valid in MaterialsPatch, but if we also see
        # other plan-level structure, it's likely a full plan.
        if any(f in content for f in ("core", "universes", "lattices")):
            return True
    return False


def _pin_map_largest_coord_list(content: dict[str, Any]) -> int:
    """Return the size of the largest single coordinate list in a pin_map.

    Checks the base path groups, the legacy coord fields, and each localized
    insert intent's ``coordinates``.  Returns the max length of any *single*
    list (not the sum) — summing lists across sections was the source of the
    old false-positive heuristic.
    """
    largest = 0
    base_fields = (
        "guide_tube_coords",
        "instrument_tube_coords",
        "water_cell_coords",
        "pyrex_rod_coords",
        "thimble_plug_coords",
    )
    for fld in base_fields:
        lst = content.get(fld)
        if isinstance(lst, list):
            largest = max(largest, len(lst))
    inserts = content.get("localized_insert_intents")
    if isinstance(inserts, list):
        for intent in inserts:
            if isinstance(intent, dict):
                coords = intent.get("coordinates")
                if isinstance(coords, list):
                    largest = max(largest, len(coords))
    return largest


def _pin_map_lattice_cells(content: dict[str, Any]) -> int:
    """Return lattice_size[0] x lattice_size[1], or 0 if unknown."""
    lattice_size = content.get("lattice_size")
    if isinstance(lattice_size, (list, tuple)) and len(lattice_size) >= 2:
        try:
            cells = int(lattice_size[0]) * int(lattice_size[1])
            if cells > 0:
                return cells
        except (TypeError, ValueError):
            pass
    return 0


def _detect_full_lattice(raw: str, patch_type: str) -> bool:
    """Return True if a pin_map output declares a full/near-full lattice.

    This is a *structural* check on the parsed JSON: a single coordinate list
    whose size approaches ``lattice_size[0] x lattice_size[1]`` (>= 50% of
    cells) is treated as a near-full dump.  Forbidden full-lattice *field
    names* (``universe_pattern`` etc.) are caught separately by
    :func:`validate_patch_contract`, so this function only inspects list sizes.

    Raw text is never rejected for being long or bracket-dense — a legitimate
    special-coordinate map for a multi-assembly core can be several KB and
    contain ~100 coords in aggregate while no single list is large.
    """
    if patch_type != "pin_map":
        return False
    # Pathological-size backstop only (DoS guard).
    if len(raw) > _FULL_LATTICE_RAW_CHARS_BACKSTOP:
        return True

    # Best-effort parse, tolerant of markdown fences / surrounding prose.
    # On parse failure, defer to the dedicated json_parse_error path.
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    try:
        content = json.loads(text)
    except json.JSONDecodeError:
        match = _JSON_OBJECT_RE.search(raw)
        if not match:
            return False
        try:
            content = json.loads(match.group(0))
        except json.JSONDecodeError:
            return False
    if not isinstance(content, dict):
        return False

    largest_list = _pin_map_largest_coord_list(content)
    cells = _pin_map_lattice_cells(content)
    if cells > 0:
        return largest_list >= _FULL_LATTICE_FRAC_THRESHOLD * cells
    return largest_list > _FULL_LATTICE_ABS_LIST_THRESHOLD


# ---------------------------------------------------------------------------
# Fake LLM for testing
# ---------------------------------------------------------------------------


class FakePatchLLM:
    """Simple fake LLM that returns pre-scripted responses.

    Each call to ``__call__`` pops the next response from the list.
    """

    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.prompts: list[str] = []

    def __call__(self, prompt: str) -> str:
        self.prompts.append(prompt)
        if not self.responses:
            return '{"patch_type": "settings"}'
        return self.responses.pop(0)


# ---------------------------------------------------------------------------
# JSON extraction helper
# ---------------------------------------------------------------------------

# Matches a JSON object starting with { and ending with the matching }.
_JSON_OBJECT_RE = re.compile(r"\{[\s\S]*\}", re.MULTILINE)


def _scan_json_objects(text: str) -> list[dict[str, Any]]:
    """Find all valid top-level JSON objects anywhere in *text*.

    Unlike a greedy regex, this tries ``json.JSONDecoder().raw_decode`` at
    every ``{`` position, so it recovers a JSON object that is buried after
    chain-of-thought prose — which reasoning models emit before the answer
    and which may itself contain stray ``{`` characters that defeat a regex.
    Trailing-comma / other syntax errors skip that position and keep scanning.
    """
    decoder = json.JSONDecoder()
    found: list[dict[str, Any]] = []
    i = 0
    n = len(text)
    while i < n:
        if text[i] != "{":
            i += 1
            continue
        try:
            obj, end = decoder.raw_decode(text, i)
        except json.JSONDecodeError:
            i += 1
            continue
        if isinstance(obj, dict):
            found.append(obj)
        # Advance past the consumed object (raw_decode returns the index
        # one past the last char it consumed).
        i = end if isinstance(end, int) and end > i else i + 1
    return found


def _pick_best_json_object(
    objs: list[dict[str, Any]], patch_type: str
) -> dict[str, Any] | None:
    """Return the last JSON object declaring the expected ``patch_type``.

    A reasoning model's final answer comes after its chain-of-thought, so the
    latest object with a matching ``patch_type`` is the real patch. Returns
    ``None`` when no object declares the expected type — callers then fall
    back to the greedy-regex path rather than risk recovering a stray CoT
    example (which lacks the right ``patch_type``).
    """
    for o in reversed(objs):
        if o.get("patch_type") == patch_type:
            return o
    return None


# Characteristic content keys for each patch type.  Used to auto-inject
# patch_type when the LLM omits it (common with DS reasoning models that
# output the content object directly after chain-of-thought).
_PATCH_CHARACTERISTIC_KEYS: dict[str, frozenset[str]] = {
    "facts": frozenset({"feature_summary", "expected_counts", "confirmed_facts"}),
    "materials": frozenset({"materials"}),
    "universes": frozenset({"universes"}),
    "pin_map": frozenset({"lattice_size", "default_universe_id", "guide_tube_coords"}),
    "axial_layers": frozenset({"layers", "lattice_loadings", "axial_domain_cm"}),
    "axial_overlays": frozenset({"overlays"}),
    "assembly_catalog": frozenset({"assembly_types"}),
    "core_layout": frozenset({"core_lattice", "core_pattern", "assembly_placements"}),
    "settings": frozenset({"source_strategy", "plot_strategy"}),
    "localized_insert_profiles": frozenset({"insert_profiles"}),
}


def _maybe_inject_patch_type(obj: dict[str, Any], patch_type: str) -> dict[str, Any]:
    """Inject ``patch_type`` if the object has characteristic content keys
    for the expected patch type but is missing the ``patch_type`` field.

    This is safe because:
    1. The prompt explicitly asks for this specific patch type.
    2. The content keys uniquely identify the patch type.
    3. The contract validator still validates the injected value.
    """
    if "patch_type" in obj:
        return obj
    char_keys = _PATCH_CHARACTERISTIC_KEYS.get(patch_type)
    if char_keys and char_keys & set(obj.keys()):
        obj["patch_type"] = patch_type
    return obj


def parse_llm_patch_json(raw_text: str, patch_type: str) -> dict[str, Any]:
    """Extract a JSON dict from raw LLM output.

    Handles markdown fences and surrounding text, including chain-of-thought
    prose emitted by reasoning models before the JSON answer.  Returns the
    parsed dict.  Raises ``PatchParseError`` if no valid JSON can be extracted.
    """
    if not raw_text or not raw_text.strip():
        raise PatchParseError(patch_type, "empty LLM response")

    text = raw_text.strip()

    # Strip markdown fences.
    if text.startswith("```"):
        lines = text.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    # Try direct parse first (clean JSON, no surrounding prose).
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return _maybe_inject_patch_type(obj, patch_type)
    except json.JSONDecodeError:
        pass

    # Robust scan: recover a JSON object buried after reasoning prose. This
    # also handles the case where the greedy regex below would span from a
    # stray '{' in the prose to the answer's closing '}' and fail to parse.
    best = _pick_best_json_object(_scan_json_objects(text), patch_type)
    if best is not None:
        return best

    # Fallback: extract the first {...} block (greedy regex).
    match = _JSON_OBJECT_RE.search(raw_text)
    if match:
        candidate = match.group(0)
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                return _maybe_inject_patch_type(obj, patch_type)
        except json.JSONDecodeError:
            pass
        # Try stripping trailing commas (common LLM mistake).
        cleaned = re.sub(r",\s*([}\]])", r"\1", candidate)
        try:
            obj = json.loads(cleaned)
            if isinstance(obj, dict):
                return _maybe_inject_patch_type(obj, patch_type)
        except json.JSONDecodeError:
            pass

    # Last-resort structural repair: some LLMs drop an object opener inside a
    # homogeneous array, emitting `,"<value>","<key2>":` where they meant
    # `},{"<key1>":"<value>","<key2>":`. Learn key1 from a valid sibling and
    # re-insert the opener.
    for candidate_text in _dropped_opener_repair_candidates(text, raw_text):
        repaired = _repair_dropped_object_openers(candidate_text)
        if repaired is None:
            continue
        try:
            obj = json.loads(repaired)
            if isinstance(obj, dict):
                return _maybe_inject_patch_type(obj, patch_type)
        except json.JSONDecodeError:
            continue

    raise PatchParseError(
        patch_type,
        f"could not extract valid JSON from LLM response",
        details=raw_text[:200],
    )


def _dropped_opener_repair_candidates(text: str, raw_text: str) -> list[str]:
    """Text candidates to try the dropped-opener repair on, best first."""

    candidates = [text]
    match = _JSON_OBJECT_RE.search(raw_text)
    if match:
        candidates.append(match.group(0))
    # Deduplicate while preserving order.
    seen: set[str] = set()
    unique: list[str] = []
    for item in candidates:
        if item not in seen:
            seen.add(item)
            unique.append(item)
    return unique


def _repair_dropped_object_openers(text: str) -> str | None:
    """Repair LLM JSON that dropped object openers in a homogeneous array.

    When an LLM emits an array of similar objects it sometimes drops the
    opening ``{`` and first key for one or more entries, producing
    ``,"<value>","<key2>":`` (after the previous object's ``}``) instead of
    ``},{"<key1>":"<value>","<key2>":``. This learns the ``key2 -> key1``
    mapping from a valid sibling object and re-inserts the opener.

    Conservative: it only re-inserts an opener when the broken entry sits in
    an array context (immediately after ``}`` or ``]``) and the result parses
    as valid JSON. Returns the repaired text or ``None``.
    """

    if not text:
        return None
    # Learn (first_key, second_key) shapes from valid object openers.
    shapes = re.findall(
        r'\{\s*"([A-Za-z_][A-Za-z0-9_]*)"\s*:\s*"[^"]*"\s*,\s*"([A-Za-z_][A-Za-z0-9_]*)"\s*:',
        text,
    )
    if not shapes:
        return None
    second_to_first: dict[str, str] = {}
    for first_key, second_key in shapes:
        second_to_first.setdefault(second_key, first_key)
    # Broken entry: after } or ] (array of objects), `,"<value>","<key2>":`.
    broken_re = re.compile(
        r'([}\]])\s*,\s*"([^",{}]+)"\s*,\s*"([A-Za-z_][A-Za-z0-9_]*)"\s*:'
    )

    def _repl(match: re.Match[str]) -> str:
        prev_close, value, second_key = match.group(1), match.group(2), match.group(3)
        first_key = second_to_first.get(second_key)
        if not first_key:
            return match.group(0)
        return f'{prev_close},{{"{first_key}":"{value}","{second_key}":'

    repaired, count = broken_re.subn(_repl, text)
    if count == 0:
        return None
    return repaired


def _looks_truncated(raw_text: str) -> bool:
    """Heuristic: does the output look like a JSON object cut off mid-stream?

    True when there is at least one ``{`` but curly or square openers outnumber
    closers — i.e. a JSON object started but never balanced. Pure-prose outputs
    (no braces) return ``False`` so they still report a generic parse error.
    Used only to produce a clearer error message, never to reject valid JSON.
    """
    if not raw_text or "{" not in raw_text:
        return False
    return (
        raw_text.count("{") > raw_text.count("}")
        or raw_text.count("[") > raw_text.count("]")
    )


# ---------------------------------------------------------------------------
# Patch contract validation (Phase 7C)
# ---------------------------------------------------------------------------


def validate_patch_contract(
    *,
    patch_type: str,
    parsed_json: dict[str, Any],
) -> list[dict[str, Any]]:
    """Check the parsed JSON against the patch contract before model parse.

    Returns a list of issue dicts (empty if contract is satisfied).
    """
    issues: list[dict[str, Any]] = []
    allowed = get_patch_allowed_top_level_keys(patch_type)
    forbidden = get_patch_forbidden_top_level_keys(patch_type)

    # 1. patch_type field check.
    actual_pt = parsed_json.get("patch_type")
    if actual_pt is None:
        issues.append({
            "code": "patch_generation.patch_type_missing",
            "severity": "error",
            "message": f'top-level "patch_type" key is missing; expected patch_type="{patch_type}"',
        })
    elif actual_pt != patch_type:
        issues.append({
            "code": "patch_generation.patch_type_mismatch",
            "severity": "error",
            "message": f'patch_type is {actual_pt!r}, expected {patch_type!r}',
        })

    # 2. Forbidden SimulationPlan-only fields.
    present_forbidden = forbidden & set(parsed_json.keys())
    if present_forbidden:
        is_plan_level = any(k in _PLAN_ONLY_FIELDS_MARKER for k in present_forbidden)
        is_lattice = any(k in _PIN_MAP_FORBIDDEN_MARKER for k in present_forbidden)
        if is_lattice:
            issues.append({
                "code": "patch_generation.pin_map_full_lattice_forbidden",
                "severity": "error",
                "message": (
                    f"forbidden full-lattice fields present: {sorted(present_forbidden)}. "
                    "PinMapPatch must only contain special coordinates, not a full lattice."
                ),
            })
        else:
            issues.append({
                "code": "patch_generation.full_plan_output_forbidden",
                "severity": "error",
                "message": (
                    f"forbidden SimulationPlan-only fields present: {sorted(present_forbidden)}. "
                    f"This is a full plan, not a {patch_type} patch."
                ),
            })

    # 3. Extra unknown keys (warning, not error).
    if allowed:
        extra = set(parsed_json.keys()) - allowed
        harmless_extra = extra - forbidden
        if harmless_extra:
            issues.append({
                "code": "patch_generation.extra_top_level_keys",
                "severity": "warning",
                "message": f"unexpected top-level keys: {sorted(harmless_extra)}",
            })

    return issues


_PLAN_ONLY_FIELDS_MARKER = frozenset({
    "complex_model", "capability_report", "execution_check",
    "plot_specs", "schema_version", "model_spec",
    "core", "surfaces", "regions", "assemblies",
    "reflectors", "control_rods", "trisos", "pebbles",
    "lattice_loadings", "packed_spheres",
})
_PIN_MAP_FORBIDDEN_MARKER = frozenset({
    "universe_pattern", "full_map", "lattice_map", "rows",
})


# ---------------------------------------------------------------------------
# Structured client call helper (Phase 7C)
# ---------------------------------------------------------------------------


def _call_llm_for_patch(
    llm_client: Any,
    *,
    prompt: str,
    patch_type: str,
    max_tokens: int | None = None,
) -> tuple[str, str]:
    """Call the LLM client, preferring structured output if available.

    Returns ``(raw_text, output_mode_used)``.
    """
    resp = _call_llm_for_patch_with_meta(llm_client, prompt=prompt, patch_type=patch_type, max_tokens=max_tokens)
    return resp.content, resp.output_mode_used or "plain_prompt"


def _call_llm_for_patch_with_meta(
    llm_client: Any,
    *,
    prompt: str,
    patch_type: str,
    max_tokens: int | None = None,
) -> PatchLLMResponse:
    """Call the LLM client and return a ``PatchLLMResponse`` with telemetry.

    Prefers ``generate_patch_json_with_meta`` (rich telemetry), falls back to
    ``generate_patch_json`` (string only), then to plain callable.
    """
    from openmc_agent.plan_builder.llm_adapter import PatchLLMResponse, normalize_patch_llm_response

    # Rich telemetry path.
    if hasattr(llm_client, "generate_patch_json_with_meta"):
        try:
            json_schema = get_patch_json_schema(patch_type)
            return llm_client.generate_patch_json_with_meta(
                prompt=prompt,
                patch_type=patch_type,
                json_schema=json_schema,
                max_tokens=max_tokens,
            )
        except Exception:
            pass  # fall through to legacy path

    # Legacy structured path (returns str).
    if hasattr(llm_client, "generate_patch_json"):
        try:
            json_schema = get_patch_json_schema(patch_type)
            raw = llm_client.generate_patch_json(
                prompt=prompt,
                patch_type=patch_type,
                json_schema=json_schema,
                max_tokens=max_tokens,
            )
            mode = str(getattr(llm_client, "last_output_mode_used", "structured"))
            return PatchLLMResponse(content=raw, output_mode_used=mode)
        except Exception:
            pass  # fall through to plain callable

    # Plain callable (FakePatchLLM or bare function).
    raw = llm_client(prompt)
    return normalize_patch_llm_response(raw)


# ---------------------------------------------------------------------------
# Validation context adapter
# ---------------------------------------------------------------------------


def _to_validation_context(
    gen_context: PatchGenerationContext | None,
) -> PatchValidationContext:
    """Adapt a PatchGenerationContext to a PatchValidationContext."""
    if gen_context is None:
        return PatchValidationContext()
    return PatchValidationContext(
        benchmark_id=gen_context.benchmark_id,
        selected_variant=gen_context.selected_variant,
        expected_counts=gen_context.expected_counts,
        expected_counts_complete=gen_context.expected_counts_complete,
        reference_expected_counts=gen_context.reference_expected_counts,
        expected_material_roles=[
            r for r in (gen_context.material_roles_by_id.values())
        ],
        known_material_ids=gen_context.known_material_ids,
        material_aliases=gen_context.material_aliases,
        known_universe_ids=gen_context.known_universe_ids,
        known_lattice_ids=gen_context.known_lattice_ids,
        axial_domain_cm=gen_context.axial_domain_cm,
        active_fuel_region_cm=gen_context.active_fuel_region_cm,
        strict_benchmark=gen_context.strict_benchmark,
        known_cell_ids=gen_context.known_cell_ids,
        cell_owner_universe_ids=gen_context.cell_owner_universe_ids,
        material_roles_by_id=gen_context.material_roles_by_id,
        known_overlay_summaries=gen_context.known_overlay_summaries,
        has_spacer_grids=gen_context.has_spacer_grids,
        expected_spacer_grid_count=gen_context.expected_spacer_grid_count,
        model_scope=gen_context.model_scope,
        assembly_count=gen_context.assembly_count,
        core_lattice_size=gen_context.core_lattice_size,
        assembly_type_counts=gen_context.assembly_type_counts,
        known_assembly_type_ids=gen_context.known_assembly_type_ids,
        assembly_pitch_cm=gen_context.assembly_pitch_cm,
        scoped_expected_counts=gen_context.scoped_expected_counts,
        known_insert_profile_ids=gen_context.known_insert_profile_ids,
        insert_profile_summaries=gen_context.insert_profile_summaries,
        movable_insert_facts=gen_context.movable_insert_facts,
        fuel_variant_requirements=gen_context.fuel_variant_requirements,
        material_summaries=gen_context.material_summaries,
        universe_summaries=gen_context.universe_summaries,
        assembly_fuel_binding_summaries=gen_context.assembly_fuel_binding_summaries,
        localized_insert_requirements=gen_context.localized_insert_requirements,
        localized_insert_universe_summaries=gen_context.localized_insert_universe_summaries,
        assembly_insert_binding_summaries=gen_context.assembly_insert_binding_summaries,
    )


def _canonical_universe_id(value: str, known_ids: list[str]) -> str | None:
    """Return a unique separator-insensitive match from known universe IDs."""
    normalized = re.sub(r"[^a-z0-9]", "", value.lower())
    matches = [
        universe_id
        for universe_id in known_ids
        if re.sub(r"[^a-z0-9]", "", universe_id.lower()) == normalized
    ]
    return matches[0] if len(matches) == 1 else None


def _normalize_known_universe_references(
    patch_type: str,
    content: dict[str, Any],
    context: PatchGenerationContext,
) -> None:
    """Normalize unambiguous LLM spelling variants in lattice transformations.

    This intentionally never creates a new universe or guesses between multiple
    IDs. It only maps variants such as ``fuel_pin_endplug`` to the sole known
    ``fuel_pin_end_plug`` before schema/semantic validation.
    """
    if patch_type != "axial_layers" or not context.known_universe_ids:
        return
    for loading in content.get("lattice_loadings", []):
        if not isinstance(loading, dict):
            continue
        for transformation in loading.get("transformations", []):
            if not isinstance(transformation, dict):
                continue
            for field in ("replacement_universe_id", "source_universe_id"):
                value = transformation.get(field)
                if isinstance(value, str):
                    canonical = _canonical_universe_id(value, context.known_universe_ids)
                    if canonical is not None:
                        transformation[field] = canonical
            values = transformation.get("source_universe_ids")
            if isinstance(values, list):
                normalized_values: list[Any] = []
                for value in values:
                    canonical = (
                        _canonical_universe_id(value, context.known_universe_ids)
                        if isinstance(value, str)
                        else None
                    )
                    normalized_values.append(canonical if canonical is not None else value)
                transformation["source_universe_ids"] = normalized_values


_AXIAL_LAYER_ROLE_ALIASES: dict[str, str] = {
    # These are lexical separator variants of schema literals, not alternate
    # physical roles.  Keeping this mapping deliberately finite prevents a
    # spelling normalizer from becoming an unreviewed semantic repair path.
    "lower_endplug": "lower_end_plug",
    "upper_endplug": "upper_end_plug",
}


def _normalize_axial_schema_defaults(content: dict[str, Any]) -> list[dict[str, Any]]:
    """Normalize only unambiguous axial JSON spelling/default variants.

    Pydantic assigns ``priority=0`` when the field is absent.  A model-emitted
    ``null`` therefore means the same *unspecified* ordering, while the wire
    schema quite correctly rejects null for an integer.  We drop that null
    rather than inventing an ordering.  Every change is returned for attempt
    artifacts; unknown roles and all other invalid values remain hard errors.
    """
    normalizations: list[dict[str, Any]] = []

    for index, layer in enumerate(content.get("layers", [])):
        if not isinstance(layer, dict):
            continue
        role = layer.get("role")
        canonical = _AXIAL_LAYER_ROLE_ALIASES.get(role) if isinstance(role, str) else None
        if canonical is not None:
            layer["role"] = canonical
            normalizations.append({
                "kind": "axial_role_lexical_alias",
                "json_path": f"/layers/{index}/role",
                "from": role,
                "to": canonical,
            })

    for loading_index, loading in enumerate(content.get("lattice_loadings", [])):
        if not isinstance(loading, dict):
            continue
        for transform_index, transformation in enumerate(loading.get("transformations", [])):
            if not isinstance(transformation, dict) or transformation.get("priority") is not None:
                continue
            if "priority" not in transformation:
                continue
            transformation.pop("priority")
            normalizations.append({
                "kind": "axial_priority_default_omitted",
                "json_path": (
                    f"/lattice_loadings/{loading_index}/transformations/"
                    f"{transform_index}/priority"
                ),
                "from": None,
                "to": "schema_default:0",
            })
    return normalizations


_OVERLAY_MATERIAL_REQUIRED_MODES = frozenset({
    "homogenized_open_region",
    "mass_conserving_outer_frame",
})


def _normalize_axial_overlay_material_references(
    content: dict[str, Any],
    context: PatchGenerationContext,
) -> list[dict[str, Any]]:
    """Normalize axial overlay material references without inventing materials.

    If a generated material ID has a unique accepted-material alias, replace it
    with the canonical ID.  If a material-required overlay references an
    unresolved or ambiguous material, degrade that overlay to a skeleton with
    human confirmation.  This preserves the source-backed grid location while
    avoiding a false executable material assignment.
    """
    if not context.known_material_ids:
        return []

    aliases = {
        **infer_material_aliases(
            context.material_summaries,
            set(context.known_material_ids),
        ),
        **context.material_aliases,
    }
    known = set(context.known_material_ids)
    normalizations: list[dict[str, Any]] = []
    for index, overlay in enumerate(content.get("overlays", [])):
        if not isinstance(overlay, dict):
            continue
        mode = overlay.get("geometry_mode")
        material_id = overlay.get("material_id")
        if not isinstance(mode, str) or not isinstance(material_id, str) or not material_id:
            continue

        resolved = resolve_material_id(material_id, known, aliases)
        if resolved.ok and resolved.resolved_id and resolved.resolved_id != material_id:
            overlay["material_id"] = resolved.resolved_id
            normalizations.append({
                "kind": "axial_overlay_material_alias",
                "json_path": f"/overlays/{index}/material_id",
                "from": material_id,
                "to": resolved.resolved_id,
            })
            continue

        if not resolved.ok and mode in _OVERLAY_MATERIAL_REQUIRED_MODES:
            overlay["geometry_mode"] = "skeleton"
            overlay["material_id"] = None
            overlay["requires_human_confirmation"] = True
            if overlay.get("through_path_preserved") is False:
                overlay["through_path_preserved"] = None
            assumptions = overlay.get("assumptions")
            if not isinstance(assumptions, list):
                assumptions = []
            note = (
                f"Generated material reference {material_id!r} is not a unique "
                "accepted material; retained as skeleton overlay pending "
                "source-backed material confirmation."
            )
            if note not in assumptions:
                assumptions.append(note)
            overlay["assumptions"] = assumptions
            normalizations.append({
                "kind": "axial_overlay_unresolved_material_skeleton",
                "json_path": f"/overlays/{index}",
                "from": {"geometry_mode": mode, "material_id": material_id},
                "to": {"geometry_mode": "skeleton", "material_id": None},
                "reason": resolved.reason,
            })
    return normalizations


# ---------------------------------------------------------------------------
# Semantic normalization detection (through_path_preserved derivation audit)
# ---------------------------------------------------------------------------

_THROUGH_PATH_MODES = frozenset({
    "mass_conserving_outer_frame",
    "homogenized_open_region",
})


def _detect_semantic_normalizations(
    patch_type: str,
    original_json: dict[str, Any],
    parsed_model: Any,
) -> list[dict[str, Any]]:
    """Detect fields deterministically derived from absent (None) by model
    validators — e.g. ``through_path_preserved`` derived for
    ``mass_conserving_outer_frame`` overlays.

    Returns a list of audit records for ``semantic_normalizations``.
    """
    normalizations: list[dict[str, Any]] = []
    if patch_type != "axial_overlays":
        return normalizations
    orig_list = original_json.get("overlays")
    if not isinstance(orig_list, list):
        return normalizations
    parsed_list = getattr(parsed_model, "overlays", [])
    for orig, parsed in zip(orig_list, parsed_list):
        if not isinstance(orig, dict):
            continue
        orig_tpp = orig.get("through_path_preserved", None)
        parsed_tpp = getattr(parsed, "through_path_preserved", None)
        mode = getattr(parsed, "geometry_mode", None)
        if (
            orig_tpp is None
            and parsed_tpp is True
            and mode in _THROUGH_PATH_MODES
        ):
            normalizations.append({
                "patch_type": patch_type,
                "overlay_id": getattr(parsed, "overlay_id", ""),
                "field": "through_path_preserved",
                "original_value": None,
                "derived_value": True,
                "derived_from": "geometry_mode",
                "geometry_mode": mode,
            })
    return normalizations


# ---------------------------------------------------------------------------
# Retry drift detection
# ---------------------------------------------------------------------------

# Locked fields for axial_overlays retry drift checks.
# These fields must NOT change during a targeted retry that only fixes
# through_path_preserved or similar single-field issues.
_OVERLAY_LOCKED_FIELDS = frozenset({
    "overlay_id", "overlay_kind", "z_min_cm", "z_max_cm",
    "target_lattice_id", "material_id", "geometry_mode",
    "total_mass_g", "cell_count", "pitch_cm",
    "frame_area_cm2", "frame_thickness_cm",
    "volume_fraction", "effective_density_g_cm3",
    "material_density_source", "mass_tolerance_rel",
    "requires_human_confirmation",
})

# Allowed-change fields for through-path contradiction retry.
_THROUGH_PATH_ALLOWED_FIELDS = frozenset({
    "through_path_preserved", "assumptions", "source_note",
})

_ASSEMBLY_CATALOG_INSERT_ALLOWED_FIELDS = frozenset({
    "localized_insert_intents", "assumptions", "source_note",
})


def _compute_allowed_overlay_fields(issue_codes: list[str]) -> set[str]:
    """Return the set of overlay fields that the retry is allowed to change."""
    allowed = set(_THROUGH_PATH_ALLOWED_FIELDS)
    for code in issue_codes:
        if "total_mass_missing" in code:
            allowed.add("total_mass_g")
        if "material_missing" in code:
            allowed.add("material_id")
        if "target_missing" in code:
            allowed.add("target_lattice_id")
    return allowed


def _detect_retry_drift(
    patch_type: str,
    previous: dict[str, Any],
    retry: dict[str, Any],
    issue_codes: list[str],
) -> list[dict[str, Any]]:
    """Compare previous and retry patch JSON to detect unexpected field changes.

    Only checks overlay-level fields for ``axial_overlays`` patches.
    Returns a list of drift records.  Empty means no unexpected drift.
    """
    if patch_type == "assembly_catalog" and any(
        code.startswith("localized_insert.") for code in issue_codes
    ):
        previous_types = {
            item.get("assembly_type_id"): item
            for item in previous.get("assembly_types", [])
            if isinstance(item, dict) and isinstance(item.get("assembly_type_id"), str)
        }
        retry_types = {
            item.get("assembly_type_id"): item
            for item in retry.get("assembly_types", [])
            if isinstance(item, dict) and isinstance(item.get("assembly_type_id"), str)
        }
        if previous_types.keys() != retry_types.keys():
            return [{
                "json_path": "assembly_types",
                "reason": "assembly type IDs changed during localized-insert retry",
                "triggering_issue_codes": issue_codes,
            }]
        drift: list[dict[str, Any]] = []
        for type_id, before in previous_types.items():
            after = retry_types[type_id]
            for field, previous_value in before.items():
                if field in _ASSEMBLY_CATALOG_INSERT_ALLOWED_FIELDS:
                    continue
                if after.get(field) != previous_value:
                    drift.append({
                        "json_path": f"assembly_types[{type_id}].{field}",
                        "previous": previous_value,
                        "new": after.get(field),
                        "allowed_paths": sorted(_ASSEMBLY_CATALOG_INSERT_ALLOWED_FIELDS),
                        "triggering_issue_codes": issue_codes,
                    })
        return drift
    if patch_type != "axial_overlays":
        return []
    prev_ovs = previous.get("overlays", [])
    new_ovs = retry.get("overlays", [])
    if not isinstance(prev_ovs, list) or not isinstance(new_ovs, list):
        return []
    drift: list[dict[str, Any]] = []
    if len(prev_ovs) != len(new_ovs):
        drift.append({
            "json_path": "overlays.length",
            "previous": len(prev_ovs),
            "new": len(new_ovs),
            "reason": "overlay count changed during targeted retry",
        })
        return drift
    allowed = _compute_allowed_overlay_fields(issue_codes)
    for i, (po, no) in enumerate(zip(prev_ovs, new_ovs)):
        if not isinstance(po, dict) or not isinstance(no, dict):
            continue
        overlay_id = no.get("overlay_id", po.get("overlay_id", f"[{i}]"))
        for field in _OVERLAY_LOCKED_FIELDS:
            if field in allowed:
                continue
            pv = po.get(field)
            nv = no.get(field)
            if not _retry_drift_values_equal(field, pv, nv):
                drift.append({
                    "json_path": f"overlays[{overlay_id}].{field}",
                    "previous": pv,
                    "new": nv,
                    "allowed_paths": sorted(allowed),
                    "triggering_issue_codes": issue_codes,
                })
    return drift


def _retry_drift_values_equal(field: str, previous_value: Any, new_value: Any) -> bool:
    """Return whether two retry values are semantically equivalent.

    Providers often echo explicit schema defaults on retry.  That should not
    count as semantic drift when the previous raw candidate omitted the field
    and the parsed model would have supplied the same default.
    """

    if field == "requires_human_confirmation" and previous_value is None and new_value is False:
        return True
    return previous_value == new_value


# ---------------------------------------------------------------------------
# Main generate_patch API
# ---------------------------------------------------------------------------


def _finalize_failed_patch_generation(
    *,
    patch_type: str,
    attempts: list[PatchGenerationAttempt],
    max_attempts: int,
) -> PatchGenerationResult:
    """Build a consistent failed result for exhausted and early-stop attempts.

    A duplicate candidate is a no-progress terminal condition, so callers
    should not spend another LLM call.  It is still useful to say explicitly
    when that terminal attempt also consumed the configured generation budget.
    """
    all_issues = [issue for attempt in attempts for issue in attempt.issues]
    if len(attempts) >= max_attempts:
        all_issues.append({
            "code": "patch_generation.max_attempts_exceeded",
            "severity": "error",
            "message": (
                f"{patch_type} generation failed after {len(attempts)} "
                "attempt(s)"
            ),
        })
    return PatchGenerationResult(
        ok=False,
        patch_type=patch_type,
        attempts=attempts,
        issues=all_issues,
    )


def generate_patch(
    *,
    patch_type: str,
    requirement: str,
    state: PlanBuildState | None = None,
    context: PatchGenerationContext | RetryPatchGenerationContext | None = None,
    llm_client: Any | None = None,
    max_attempts: int = 2,
    max_tokens: int | None = None,
) -> PatchGenerationResult:
    """Generate a single patch via LLM with targeted retry.

    Parameters
    ----------
    patch_type
        One of the :data:`~openmc_agent.plan_builder.patches.PatchType` values.
    requirement
        The user/benchmark requirement text.
    state
        Optional build state for context (validated patch summaries).
    context
        Generation context with confirmed facts, expected counts, etc.
    llm_client
        A callable ``(prompt: str) -> str``.  If ``None``, returns an error
        result with ``patch_generation.no_llm_client``.
    max_attempts
        Maximum number of LLM calls (including the first).

    Returns
    -------
    PatchGenerationResult
    """
    if llm_client is None:
        return PatchGenerationResult(
            ok=False,
            patch_type=patch_type,
            issues=[{
                "code": "patch_generation.no_llm_client",
                "severity": "error",
                "message": "llm_client is None; cannot generate patch",
            }],
        )

    # Unwrap retry context if present so the rest of the generator sees the
    # base context.  Retry-specific fields are available on ``retry_context``.
    retry_context: RetryPatchGenerationContext | None = None
    if isinstance(context, RetryPatchGenerationContext):
        retry_context = context
        context = retry_context.unwrap()
        retry_block = _retry_context_requirement_block(retry_context)
        if retry_block and "=== Executable retry context ===" not in requirement:
            requirement = f"{requirement}\n\n{retry_block}"

    # Enrich context with validated patch summaries from state.
    effective_context = context or PatchGenerationContext()
    if state is not None:
        for env in state.patches.values():
            if env.status == "valid":
                effective_context.validated_patch_summaries.setdefault(
                    env.patch_type, {"status": "valid", "patch_id": env.patch_id}
                )

    val_context = _to_validation_context(effective_context)
    attempts: list[PatchGenerationAttempt] = []
    last_issues: list[dict[str, Any]] = []
    previous_parsed_content: dict[str, Any] | None = None
    persisted_candidate_hashes = set(
        retry_context.prior_candidate_hashes if retry_context is not None else []
    )
    current_generation_candidate_hashes: set[str] = set()

    for attempt_idx in range(max_attempts):
        attempt = PatchGenerationAttempt(attempt_index=attempt_idx, patch_type=patch_type)

        # Build prompt.
        if attempt_idx == 0:
            prompt = build_patch_prompt(patch_type, requirement, effective_context)
        else:
            prompt = build_retry_prompt(
                patch_type, requirement, effective_context,
                last_issues, attempt_idx,
                previous_patch=previous_parsed_content,
            )

        # Call LLM (Phase 7C: prefer structured output if available).
        try:
            llm_resp = _call_llm_for_patch_with_meta(
                llm_client, prompt=prompt, patch_type=patch_type,
                max_tokens=max_tokens,
            )
            raw = llm_resp.content
            output_mode = llm_resp.output_mode_used or "plain_prompt"
        except Exception as exc:
            attempt.error = str(exc)
            attempt.issues.append({
                "code": "patch_generation.llm_error",
                "severity": "error",
                "message": f"LLM client raised: {exc}",
            })
            attempts.append(attempt)
            last_issues = attempt.issues
            continue

        attempt.raw_text = raw
        attempt.raw_chars = len(raw)
        attempt.prompt_text = prompt
        attempt.output_mode_used = output_mode
        # Populate telemetry from PatchLLMResponse.
        attempt.finish_reason = llm_resp.finish_reason
        attempt.prompt_tokens = llm_resp.prompt_tokens
        attempt.completion_tokens = llm_resp.completion_tokens
        attempt.reasoning_tokens = llm_resp.reasoning_tokens
        attempt.total_tokens = llm_resp.total_tokens
        attempt.structured_fallback_used = llm_resp.structured_fallback_used
        attempt.structured_fallback_reasons = list(llm_resp.structured_fallback_reasons)

        # Output diagnostics: detect forbidden patterns (Phase 7B: errors).
        attempt.contains_full_plan_markers = _detect_full_plan_markers(raw)
        if attempt.contains_full_plan_markers:
            attempt.issues.append({
                "code": "patch_generation.full_plan_output_forbidden",
                "severity": "error",
                "message": (
                    "raw output contains SimulationPlan markers "
                    "(complex_model, capability_report, etc.). "
                    "This is a forbidden full-plan output, not a patch."
                ),
            })
            attempts.append(attempt)
            last_issues = attempt.issues
            continue

        attempt.contains_full_lattice_suspected = _detect_full_lattice(raw, patch_type)
        if attempt.contains_full_lattice_suspected:
            attempt.issues.append({
                "code": "patch_generation.pin_map_full_lattice_forbidden",
                "severity": "error",
                "message": (
                    "pin_map output enumerates a near-full lattice (one "
                    "coordinate list approaches rows x cols). Only special "
                    "coordinates are allowed; the default universe fills the "
                    "rest."
                ),
            })
            attempts.append(attempt)
            last_issues = attempt.issues
            continue

        # Parse JSON.
        try:
            content = parse_llm_patch_json(raw, patch_type)
        except PatchParseError as exc:
            attempt.error = str(exc)
            # Prioritize finish_reason for truncation detection.  Only fall
            # back to the bracket heuristic when the provider does not report
            # a finish_reason.
            is_truncated = False
            truncation_source = ""
            if llm_resp.finish_reason:
                fr = llm_resp.finish_reason.lower()
                if fr in ("length", "max_tokens"):
                    is_truncated = True
                    truncation_source = "finish_reason"
                elif "context" in fr:
                    is_truncated = True
                    truncation_source = "finish_reason"
            if not is_truncated and _looks_truncated(raw):
                is_truncated = True
                truncation_source = "heuristic"
            if is_truncated:
                attempt.issues.append({
                    "code": "patch_generation.json_truncated",
                    "severity": "error",
                    "message": (
                        f"LLM output appears truncated (detected via {truncation_source}) — "
                        "the JSON object started but never completed. This "
                        "usually means the output token budget (including "
                        "reasoning tokens) was exhausted. Raise "
                        "PATCH_MAX_TOKENS for this patch type, lower "
                        "reasoning_effort, or use fragmented generation."
                    ),
                    "finish_reason": llm_resp.finish_reason,
                    "truncation_source": truncation_source,
                })
            else:
                attempt.issues.append({
                    "code": "patch_generation.json_parse_error",
                    "severity": "error",
                    "message": str(exc),
                })
            attempts.append(attempt)
            last_issues = attempt.issues
            continue

        _normalize_known_universe_references(
            patch_type, content, effective_context
        )
        schema_normalizations: list[dict[str, Any]] = []
        if patch_type == "materials":
            from .materials_patch_normalization import normalize_materials_patch_content

            normalization = normalize_materials_patch_content(
                content,
                requirement_text=prompt,
                state=state,
            )
            if normalization.changed:
                content = normalization.content
                schema_normalizations = normalization.operations
        elif patch_type == "axial_layers":
            schema_normalizations = _normalize_axial_schema_defaults(content)
        elif patch_type == "axial_overlays":
            schema_normalizations = _normalize_axial_overlay_material_references(
                content,
                effective_context,
            )
        attempt.semantic_normalizations = schema_normalizations

        # Record a canonical candidate before model parsing.  Schema-invalid
        # candidates are still candidates: retaining this hash is what lets a
        # resumed controlled run stop a repeated bad response before another
        # whole planning pass is scheduled.
        attempt.parsed_content = content
        attempt.candidate_hash = compute_candidate_hash(
            target_patch_type=patch_type,
            candidate_patch=content,
        )
        duplicate_scope: str | None = None
        if attempt.candidate_hash in persisted_candidate_hashes:
            duplicate_scope = "persisted_context"
        elif attempt.candidate_hash in current_generation_candidate_hashes:
            duplicate_scope = "current_generation"

        if duplicate_scope is not None:
            attempt.semantic_normalizations = schema_normalizations
            attempt.issues.append({
                "code": "patch_generation.no_progress_duplicate_candidate",
                "severity": "error",
                "message": (
                    f"{patch_type} returned a candidate JSON already rejected "
                    "for the same upstream patch context"
                ),
                "candidate_hash": attempt.candidate_hash,
                "duplicate_scope": duplicate_scope,
            })
            attempts.append(attempt)
            return _finalize_failed_patch_generation(
                patch_type=patch_type,
                attempts=attempts,
                max_attempts=max_attempts,
            )
        current_generation_candidate_hashes.add(attempt.candidate_hash)

        # Phase 7C: validate patch contract (patch_type, allowed/forbidden keys).
        contract_issues = validate_patch_contract(
            patch_type=patch_type, parsed_json=content,
        )
        contract_errors = [i for i in contract_issues if i.get("severity") == "error"]
        if contract_errors:
            attempt.issues.extend(contract_issues)
            attempts.append(attempt)
            last_issues = contract_errors
            continue
        # Warnings from contract are recorded but don't block.
        attempt.issues.extend(contract_issues)

        # Parse into patch model.
        try:
            parsed_model = parse_patch_content(patch_type, content)
        except PatchParseError as exc:
            attempt.error = str(exc)
            attempt.issues.append({
                "code": "patch_generation.schema_error",
                "severity": "error",
                "message": str(exc),
            })
            attempts.append(attempt)
            last_issues = attempt.issues
            continue

        attempt.parsed = True

        # Detect semantic normalizations (e.g. through_path_preserved derived
        # from None by model_validator).
        normalizations = _detect_semantic_normalizations(
            patch_type, content, parsed_model,
        )
        attempt.semantic_normalizations = [*schema_normalizations, *normalizations]

        # Retry drift detection: if this is a retry and we have the previous
        # parsed patch, check that only allowed fields changed.
        if previous_parsed_content is not None and attempt_idx > 0:
            issue_codes = [i.get("code", "") for i in last_issues]
            drift = _detect_retry_drift(
                patch_type,
                previous_parsed_content,
                content,
                issue_codes,
            )
            attempt.retry_drift = drift
            attempt.retry_allowed_paths = sorted(
                _compute_allowed_overlay_fields(issue_codes)
            )
            if drift:
                for d in drift:
                    attempt.issues.append({
                        "code": "patch_retry.unexpected_semantic_drift",
                        "severity": "error",
                        "message": (
                            f"retry changed {d['json_path']}: "
                            f"{d.get('previous')!r} → {d.get('new')!r} "
                            f"(allowed: {attempt.retry_allowed_paths})"
                        ),
                        **d,
                    })
                attempts.append(attempt)
                last_issues = attempt.issues
                previous_parsed_content = content
                continue

        # Validate.
        val_result: PatchValidationResult = validate_patch(parsed_model, val_context)
        attempt.validated = val_result.ok
        attempt.issues = [i.model_dump(mode="json") for i in val_result.issues]
        attempts.append(attempt)

        if val_result.ok:
            # Success — build envelope and return.
            envelope = PlanPatchEnvelope(
                patch_id=f"patch_{patch_type}_{attempt_idx}",
                patch_type=patch_type,
                content=parsed_model.model_dump(mode="json"),
                source="llm",
                status="valid",
                issues=[i.model_dump(mode="json") for i in val_result.issues if i.severity != "info"],
            )
            return PatchGenerationResult(
                ok=True,
                patch_type=patch_type,
                envelope=envelope,
                parsed_patch=parsed_model.model_dump(mode="json"),
                validation=val_result.model_dump(mode="json"),
                attempts=attempts,
                issues=[i for a in attempts for i in a.issues],
            )

        # Validation failed — prepare retry issues.
        last_issues = [
            i.model_dump(mode="json")
            for i in val_result.issues
            if i.severity == "error"
        ]
        if not last_issues:
            last_issues = attempt.issues
        previous_parsed_content = content

    return _finalize_failed_patch_generation(
        patch_type=patch_type,
        attempts=attempts,
        max_attempts=max_attempts,
    )


__all__ = [
    "PatchGenerationContext",
    "PatchGenerationAttempt",
    "PatchGenerationResult",
    "FakePatchLLM",
    "generate_patch",
    "parse_llm_patch_json",
]
