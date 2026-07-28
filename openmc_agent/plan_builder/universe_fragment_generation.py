"""Large structured patch generation: universe fragment pipeline.

This module implements the fragmented universe generation pipeline:

    accepted Facts/Materials
    → requirement inventory
    → universe manifest (deterministic, with per-item contract hash)
    → one-universe fragments (LLM, one call each)
    → fragment qualification (deterministic, structured)
    → checkpoint with hash/contract/qualification integrity
    → deterministic structured merge (pure Python, no LLM)
    → targeted fragment replay when merge reports fragment-scoped issues
    → standard UniversesPatch validation
    → one authoritative UniversesPatch envelope

The pipeline is designed to avoid truncation on large universe patches
(e.g. multi-assembly cores with many distinct universes).  Each LLM call
produces only one universe, keeping the structured output small enough
to complete within provider token limits.

Key design principles:
- Thinking/reasoning mode is NOT disabled.  Reliability comes from
  shrinking each structured output, not from larger token budgets.
- Checkpoint/resume: completed fragments are never re-generated, but
  their integrity (data + hash + contract hash + qualification) is
  re-verified on resume.
- Partial fragments are never exposed to downstream gates.
- Fragment acceptance requires deterministic qualification against the
  manifest contract — not just JSON parseability.
- The final output is a single standard UniversesPatch that passes
  existing validators unchanged.
"""

from __future__ import annotations

import hashlib
from typing import Any, Literal

from pydantic import Field

from openmc_agent.schemas import AgentBaseModel

from .closed_loop.fingerprints import canonical_json_dumps
from .llm_adapter import PatchLLMResponse, normalize_patch_llm_response
from .patches import parse_patch_content
from .validators import PatchValidationContext, validate_patch


# ---------------------------------------------------------------------------
# Requirement inventory
# ---------------------------------------------------------------------------


class UniverseGenerationRequirement(AgentBaseModel):
    """A single universe generation requirement extracted from accepted upstream context."""
    requirement_id: str
    universe_id: str = ""
    kind: str = ""
    required_cell_roles: list[str] = Field(default_factory=list)
    required_material_ids: list[str] = Field(default_factory=list)
    required_material_roles: list[str] = Field(default_factory=list)
    fuel_variant_id: str | None = None
    localized_insert_requirement_id: str | None = None
    localized_insert_requirement_ids: list[str] = Field(default_factory=list)
    base_path_component_profile_id: str | None = None
    protected_through_path_roles: list[str] = Field(default_factory=list)
    required_layer_roles: list[str] = Field(default_factory=list)
    source_requirement_ids: list[str] = Field(default_factory=list)
    dependency_ids: list[str] = Field(default_factory=list)
    resolved: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


class UniverseGenerationRequirementSet(AgentBaseModel):
    """The complete set of universe generation requirements."""
    requirements: list[UniverseGenerationRequirement] = Field(default_factory=list)
    unresolved_requirements: list[UniverseGenerationRequirement] = Field(default_factory=list)
    input_hash: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


def extract_universe_requirements(
    *,
    facts: Any | None = None,
    materials: Any | None = None,
    canonical_task_plan: Any | None = None,
    confirmed_records: dict[str, Any] | None = None,
) -> UniverseGenerationRequirementSet:
    """Deterministically extract universe requirements from accepted upstream context.

    Sources (only these are allowed):
    - Facts: localized_insert_requirements, fuel_variant_requirements,
      planning_feature_contract.
    - Materials: material IDs and roles.
    - Canonical task plan: patch type ordering.
    - Confirmed human facts.

    Benchmark names, VERA3/VERA4 strings, and fixture names must NOT
    influence the requirement list.
    """
    requirements: list[UniverseGenerationRequirement] = []
    unresolved: list[UniverseGenerationRequirement] = []
    material_ids: set[str] = set()
    material_roles: set[str] = set()
    if materials is not None:
        for m in getattr(materials, "materials", []) or []:
            material_ids.add(m.material_id)
            if hasattr(m, "role") and m.role:
                material_roles.add(m.role)
    # Fuel variant requirements.
    if facts is not None:
        for variant in getattr(facts, "fuel_variant_requirements", []) or []:
            requirements.append(UniverseGenerationRequirement(
                requirement_id=f"fuel_variant:{variant.variant_id}",
                kind="fuel_pin",
                required_cell_roles=["fuel"],
                required_material_roles=["fuel"],
                fuel_variant_id=variant.variant_id,
                source_requirement_ids=[f"fuel_variant:{variant.variant_id}"],
                resolved=True,
            ))
        # Localized insert requirements.
        for req in getattr(facts, "localized_insert_requirements", []) or []:
            kind_map = {
                "control_rod": "control_rod",
                "absorber_insert": "control_rod",
                "pyrex_rod": "pyrex_rod",
                "thimble_plug": "thimble_plug",
            }
            mapped_kind = kind_map.get(getattr(req, "insert_kind", ""), "custom")
            requirements.append(UniverseGenerationRequirement(
                requirement_id=f"localized_insert:{req.requirement_id}",
                kind=mapped_kind,
                required_material_roles=["absorber"] if mapped_kind == "control_rod" else (["poison"] if mapped_kind == "pyrex_rod" else ["structural"]),
                localized_insert_requirement_id=req.requirement_id,
                source_requirement_ids=[f"localized_insert:{req.requirement_id}"],
                resolved=True,
            ))
    # Guide tube and instrument tube are standard requirements for assembly models.
    if facts is not None and getattr(facts, "planning_feature_contract", None):
        fc = facts.planning_feature_contract
        if getattr(fc, "expected_guide_tube_count", None):
            requirements.append(UniverseGenerationRequirement(
                requirement_id="implicit:guide_tube",
                kind="guide_tube",
                required_cell_roles=["wall", "coolant"],
                required_material_roles=["structural", "coolant"],
                source_requirement_ids=["implicit:guide_tube"],
                resolved=True,
            ))
        if getattr(fc, "expected_instrument_tube_count", None):
            requirements.append(UniverseGenerationRequirement(
                requirement_id="implicit:instrument_tube",
                kind="instrument_tube",
                required_cell_roles=["coolant"],
                required_material_roles=["coolant"],
                source_requirement_ids=["implicit:instrument_tube"],
                resolved=True,
            ))
    # Auxiliary universes needed for axial layer transitions.  When the
    # facts patch declares axial geometry, the downstream axial_layers
    # patch will reference these cross-sectional profiles for non-active-
    # fuel regions (end plugs, plenum, shoulder gaps).  Declaring them
    # as implicit requirements ensures the universes patch generates them
    # rather than leaving axial_layers to reference non-existent IDs.
    _has_axial = bool(getattr(facts, "has_axial_geometry", False)) if facts else False
    if _has_axial:
        _aux_specs = [
            ("implicit:end_plug_lower", "custom", ["structural"], ["end_plug"]),
            ("implicit:end_plug_upper", "custom", ["structural"], ["end_plug"]),
            ("implicit:gas_gap", "custom", ["coolant", "structural"], ["gas_gap"]),
            ("implicit:water_pin", "water_cell", ["coolant"], ["moderator"]),
        ]
        for req_id, kind, roles, cell_roles in _aux_specs:
            requirements.append(UniverseGenerationRequirement(
                requirement_id=req_id,
                kind=kind,
                required_cell_roles=cell_roles,
                required_material_roles=roles,
                source_requirement_ids=[req_id],
                resolved=True,
            ))
    payload = canonical_json_dumps({
        "reqs": [r.model_dump(mode="json") for r in requirements],
        "material_ids": sorted(material_ids),
    })
    input_hash = hashlib.sha256(payload.encode()).hexdigest()[:16]
    return UniverseGenerationRequirementSet(
        requirements=requirements,
        unresolved_requirements=unresolved,
        input_hash=input_hash,
        metadata={"material_ids": sorted(material_ids), "material_roles": sorted(material_roles)},
    )


# ---------------------------------------------------------------------------
# Inventory-driven requirement conversion (controlled mode)
# ---------------------------------------------------------------------------

_COMPONENT_KIND_TO_UNIVERSE_KIND: dict[str, str] = {
    "fuel_pin": "fuel_pin",
    "fuel": "fuel_pin",
    "guide_tube": "guide_tube",
    "instrument_tube": "instrument_tube",
    "control_rod": "control_rod",
    "absorber_insert": "control_rod",
    "poison_insert": "pyrex_rod",
    "pyrex_rod": "pyrex_rod",
    "thimble_plug": "thimble_plug",
    "water_pin": "water_cell",
    "moderator_region": "water_cell",
    "end_plug": "custom",
    "gas_gap": "custom",
    "plenum": "custom",
    "spacer_grid": "custom",
    "support_plate": "custom",
    "nozzle": "custom",
    "core_plate": "custom",
    "dashpot": "custom",
    "reflector": "custom",
    "vessel_or_boundary": "custom",
    "custom": "custom",
}


def convert_inventory_to_generation_requirements(
    inventory_set: Any,
) -> UniverseGenerationRequirementSet:
    """Convert ``InventoryUniverseRequirementSet`` → ``UniverseGenerationRequirementSet``.

    This is the controlled-mode path: no ``implicit:*`` requirements are
    emitted.  Every requirement is source-backed by an inventory profile.

    The inventory_set argument may be a Pydantic model or a plain dict
    (as stored in ``state.metadata``); both are accepted.
    """
    # Accept both model instances and plain dicts.
    if isinstance(inventory_set, dict):
        requirements_data = inventory_set.get("requirements", [])
        unresolved_data = inventory_set.get("unresolved_requirements", [])
        inv_hash = inventory_set.get("inventory_hash", "")
        mat_hash = inventory_set.get("material_requirement_set_hash", "")
    else:
        requirements_data = [r.model_dump(mode="json") for r in inventory_set.requirements]
        unresolved_data = [r.model_dump(mode="json") for r in inventory_set.unresolved_requirements]
        inv_hash = inventory_set.inventory_hash
        mat_hash = inventory_set.material_requirement_set_hash

    requirements: list[UniverseGenerationRequirement] = []
    for rd in requirements_data:
        component_kind = rd.get("component_kind", "custom")
        profile_kind = rd.get("profile_kind", "")
        kind = _COMPONENT_KIND_TO_UNIVERSE_KIND.get(component_kind, "custom")
        geometry_profile_id = rd.get("geometry_profile_id", "")
        fuel_variant_id = rd.get("fuel_variant_id")
        # Build a deterministic universe_id from the profile id.
        universe_id = geometry_profile_id or rd.get("requirement_id", "").replace(":", "_")
        if fuel_variant_id and kind == "fuel_pin":
            universe_id = f"u_fuel_{fuel_variant_id}"
        elif kind == "guide_tube":
            universe_id = "u_guide_tube"
        elif kind == "instrument_tube":
            universe_id = "u_instrument_tube"
        localized_insert_ids = list(rd.get("localized_insert_requirement_ids", []) or [])
        singular_insert_id = rd.get("localized_insert_requirement_id")
        if singular_insert_id and singular_insert_id not in localized_insert_ids:
            localized_insert_ids.append(singular_insert_id)
        localized_insert_ids = sorted(set(str(item) for item in localized_insert_ids if item))
        source_ids = [rd.get("requirement_id", universe_id)] + list(rd.get("source_claim_ids", []))
        requirements.append(UniverseGenerationRequirement(
            requirement_id=rd.get("requirement_id", universe_id),
            universe_id=universe_id,
            kind=kind,
            required_cell_roles=list(rd.get("required_cell_roles", [])),
            required_material_roles=list(rd.get("required_material_roles", [])),
            fuel_variant_id=fuel_variant_id,
            localized_insert_requirement_id=localized_insert_ids[0] if localized_insert_ids else None,
            localized_insert_requirement_ids=localized_insert_ids,
            base_path_component_profile_id=geometry_profile_id,
            protected_through_path_roles=list(rd.get("protected_through_path_roles", [])),
            required_layer_roles=list(rd.get("required_layer_roles", [])),
            source_requirement_ids=source_ids,
            resolved=rd.get("resolved", True),
            metadata={
                "component_kind": component_kind,
                "profile_kind": profile_kind,
                "required_layer_roles": list(rd.get("required_layer_roles", [])),
                "source_span_ids": list(rd.get("source_span_ids", [])),
                "requirement_source": "inventory",
                "localized_insert_requirement_ids": localized_insert_ids,
            },
        ))

    unresolved: list[UniverseGenerationRequirement] = []
    for rd in unresolved_data:
        unresolved.append(UniverseGenerationRequirement(
            requirement_id=rd.get("requirement_id", ""),
            universe_id=rd.get("geometry_profile_id", ""),
            kind=_COMPONENT_KIND_TO_UNIVERSE_KIND.get(rd.get("component_kind", "custom"), "custom"),
            required_material_roles=list(rd.get("required_material_roles", [])),
            resolved=False,
            metadata={"unresolved_fields": list(rd.get("unresolved_fields", []))},
        ))

    payload = canonical_json_dumps({
        "reqs": [r.model_dump(mode="json") for r in requirements],
        "inv_hash": inv_hash,
        "mat_hash": mat_hash,
        "source": "inventory",
    })
    input_hash = hashlib.sha256(payload.encode()).hexdigest()[:16]
    return UniverseGenerationRequirementSet(
        requirements=requirements,
        unresolved_requirements=unresolved,
        input_hash=input_hash,
        metadata={
            "requirement_source": "inventory",
            "inventory_hash": inv_hash,
            "material_requirement_set_hash": mat_hash,
        },
    )


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------


# Fields that define a manifest item's contract.  These are canonicalized
# (sorted) and hashed to produce a stable per-item ``contract_hash``.  The
# hash is independent of item ordering in the manifest.
_MANIFEST_CONTRACT_FIELDS: tuple[str, ...] = (
    "universe_id",
    "kind",
    "required_cell_roles",
    "required_material_ids",
    "required_material_roles",
    "fuel_variant_id",
    "localized_insert_requirement_id",
    "localized_insert_requirement_ids",
    "base_path_component_profile_id",
    "protected_through_path_roles",
    "required_layer_roles",
    "source_requirement_ids",
    "dependency_ids",
)


def compute_manifest_item_contract_hash(item_data: dict[str, Any]) -> str:
    """Deterministically hash a manifest item's contract fields.

    Only the fields listed in :data:`_MANIFEST_CONTRACT_FIELDS` are hashed,
    so changes to metadata, ``expected_cell_count``, ``assumptions_allowed``,
    or item ordering in the manifest do NOT change the per-item contract.
    """
    payload = {
        field: item_data.get(field)
        for field in _MANIFEST_CONTRACT_FIELDS
    }
    return hashlib.sha256(
        canonical_json_dumps(payload).encode("utf-8")
    ).hexdigest()[:16]


class UniverseManifestItem(AgentBaseModel):
    """One entry in the universe manifest: describes a single universe to generate.

    Contract fields are hashed into :attr:`contract_hash` via
    :func:`compute_manifest_item_contract_hash`.  Mutation of any contract
    field invalidates the hash; downstream code must recompute it.
    """

    universe_id: str
    kind: str = "custom"
    required_cell_roles: list[str] = Field(default_factory=list)
    required_material_ids: list[str] = Field(default_factory=list)
    required_material_roles: list[str] = Field(default_factory=list)
    fuel_variant_id: str | None = None
    expected_cell_count: int | None = None
    # --- Contract binding fields (participate in contract_hash) ---
    protected_through_path_roles: list[str] = Field(default_factory=list)
    required_layer_roles: list[str] = Field(default_factory=list)
    source_requirement_ids: list[str] = Field(default_factory=list)
    dependency_ids: list[str] = Field(default_factory=list)
    localized_insert_requirement_id: str | None = None
    localized_insert_requirement_ids: list[str] = Field(default_factory=list)
    base_path_component_profile_id: str | None = None
    # --- Non-contract fields ---
    contract_hash: str = ""
    assumptions_allowed: bool = True
    metadata: dict[str, Any] = Field(default_factory=dict)

    def recompute_contract_hash(self) -> str:
        """Recompute and store :attr:`contract_hash` from contract fields."""
        data = self.model_dump(mode="json")
        self.contract_hash = compute_manifest_item_contract_hash(data)
        return self.contract_hash


class UniverseManifest(AgentBaseModel):
    """The complete manifest of universes to generate."""
    manifest_id: str = ""
    input_hash: str = ""
    expected_universe_count: int = 0
    items: list[UniverseManifestItem] = Field(default_factory=list)
    unresolved_requirements: list[dict[str, Any]] = Field(default_factory=list)
    generation_order: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


def validate_manifest(
    manifest: UniverseManifest,
    requirement_set: UniverseGenerationRequirementSet,
    *,
    known_material_ids: set[str] | None = None,
) -> list[str]:
    """Validate that the manifest covers all requirements. Returns list of error codes."""
    errors: list[str] = []
    # Check for duplicate universe IDs.
    ids = [item.universe_id for item in manifest.items]
    if len(ids) != len(set(ids)):
        errors.append("manifest.duplicate_universe_id")
    # Check that all resolved requirements are covered.
    req_ids = {r.requirement_id for r in requirement_set.requirements if r.resolved}
    covered_ids = set()
    for item in manifest.items:
        covered_ids.update(item.source_requirement_ids)
    missing = req_ids - covered_ids
    if missing:
        errors.append("manifest.missing_required_universe")
    # Check that material IDs are known.
    if known_material_ids is not None:
        for item in manifest.items:
            for mid in item.required_material_ids:
                if mid not in known_material_ids:
                    errors.append(f"manifest.unknown_material_id:{mid}")
                    break
    # Check generation_order covers all items.
    order_set = set(manifest.generation_order)
    item_set = set(ids)
    if order_set != item_set:
        errors.append("manifest.generation_order_mismatch")
    return errors


def build_manifest_from_requirements(
    requirement_set: UniverseGenerationRequirementSet,
    *,
    known_material_ids: set[str] | None = None,
) -> UniverseManifest:
    """Build a manifest from the requirement set (deterministic, no LLM).

    Each requirement's contract fields (kind, cell/material roles, source
    IDs, profile bindings, protected-through-path roles, dependency IDs)
    are preserved verbatim on the resulting manifest item, and a stable
    per-item :attr:`UniverseManifestItem.contract_hash` is computed.
    """
    items: list[UniverseManifestItem] = []
    for req in requirement_set.requirements:
        if not req.resolved:
            continue
        uid = req.universe_id or req.requirement_id.replace(":", "_")
        item = UniverseManifestItem(
            universe_id=uid,
            kind=req.kind,
            required_cell_roles=list(req.required_cell_roles),
            required_material_ids=list(req.required_material_ids),
            required_material_roles=list(req.required_material_roles),
            fuel_variant_id=req.fuel_variant_id,
            localized_insert_requirement_id=req.localized_insert_requirement_id,
            localized_insert_requirement_ids=list(req.localized_insert_requirement_ids),
            base_path_component_profile_id=req.base_path_component_profile_id,
            protected_through_path_roles=list(req.protected_through_path_roles),
            required_layer_roles=list(req.required_layer_roles),
            source_requirement_ids=list(req.source_requirement_ids or [req.requirement_id]),
            dependency_ids=list(req.dependency_ids),
        )
        item.recompute_contract_hash()
        items.append(item)
    manifest_hash = hashlib.sha256(
        canonical_json_dumps([i.model_dump(mode="json") for i in items]).encode("utf-8")
    ).hexdigest()[:16]
    return UniverseManifest(
        manifest_id=f"manifest:{manifest_hash}",
        input_hash=requirement_set.input_hash,
        expected_universe_count=len(items),
        items=items,
        generation_order=[i.universe_id for i in items],
    )


# ---------------------------------------------------------------------------
# Fragment
# ---------------------------------------------------------------------------


class UniverseDefinitionFragment(AgentBaseModel):
    """A single universe definition fragment from one LLM call.

    ``fragment_hash`` is the canonical hash recomputed from
    :attr:`universe` (not the LLM-claimed hash).  ``manifest_contract_hash``
    binds the fragment to the manifest item whose contract it must satisfy.
    """
    fragment_type: Literal["universe_definition"] = "universe_definition"
    universe_id: str
    universe: dict[str, Any] = Field(default_factory=dict)
    fragment_hash: str = ""
    manifest_contract_hash: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Session / checkpoint
# ---------------------------------------------------------------------------


# Qualification status of a fragment against its manifest item.
FragmentQualificationStatus = Literal["pending", "passed", "failed"]


class AcceptedFragmentRecord(AgentBaseModel):
    """Typed checkpoint record of an accepted fragment.

    Stored on :attr:`LargePatchGenerationSession.accepted_fragments` so
    resume can deterministically verify that (i) the data exists, (ii) the
    hash matches, (iii) the manifest contract hash is unchanged, and
    (iv) the qualification record is still passing.

    A stale/corrupt record causes only that fragment to be regenerated —
    other accepted fragments remain usable.
    """

    universe_id: str
    universe: dict[str, Any] = Field(default_factory=dict)
    fragment_hash: str = ""
    manifest_contract_hash: str = ""
    qualification_status: FragmentQualificationStatus = "passed"
    qualification_issues: list[dict[str, Any]] = Field(default_factory=list)
    accepted_at_attempt: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)


class FragmentStatus(AgentBaseModel):
    """Per-universe status inside a checkpoint session.

    On resume, ``status == accepted`` alone is NOT sufficient: the resume
    path must verify ``fragment_hash``, ``manifest_contract_hash``,
    ``qualification_status`` and the presence of an
    :class:`AcceptedFragmentRecord`.  Any mismatch downgrades this entry to
    ``pending`` (so it gets regenerated) without touching other fragments.
    """

    universe_id: str
    status: Literal["pending", "accepted", "failed", "stale"] = "pending"
    fragment_hash: str = ""
    manifest_contract_hash: str = ""
    qualification_status: FragmentQualificationStatus = "pending"
    qualification_issues: list[dict[str, Any]] = Field(default_factory=list)
    accepted_at_attempt: int | None = None
    issues: list[dict[str, Any]] = Field(default_factory=list)
    llm_calls: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)


class LargePatchGenerationSession(AgentBaseModel):
    """Checkpoint session for large patch generation (universes).

    All integrity-bearing fields are typed and revalidated on resume so a
    single corrupted fragment does not silently re-enter the merge.

    ``accepted_fragments`` is the authoritative checkpoint store.  The
    legacy ``accepted_fragment_hashes`` is kept for backward compatibility
    with older sessions and as a quick lookup map.  The two MUST stay in
    sync after every successful accept/replay.
    """

    session_id: str = ""
    patch_type: str = "universes"
    input_hash: str = ""
    mode: Literal["auto", "monolithic", "fragmented"] = "auto"
    requirement_set_hash: str = ""
    manifest: UniverseManifest | None = None
    manifest_status: Literal["pending", "accepted", "failed"] = "pending"
    fragment_statuses: list[FragmentStatus] = Field(default_factory=list)
    accepted_fragment_hashes: dict[str, str] = Field(default_factory=dict)
    accepted_fragments: dict[str, AcceptedFragmentRecord] = Field(default_factory=dict)
    failed_fragment_issues: dict[str, list[dict[str, Any]]] = Field(default_factory=dict)
    strategy_transitions: list[dict[str, Any]] = Field(default_factory=list)
    llm_call_count: int = 0
    completed: bool = False
    merged_patch_hash: str = ""
    provider_telemetry: list[dict[str, Any]] = Field(default_factory=list)
    # Structured merge history: each entry is a structured ``UniverseMergeResult``
    # dump plus the reason this merge was attempted.
    merge_history: list[dict[str, Any]] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Strategy / admission
# ---------------------------------------------------------------------------

UniversesGenerationMode = Literal["auto", "monolithic", "fragmented"]

# Reactor-neutral thresholds (NOT hardcoded for VERA4 or any specific benchmark).
_DEFAULT_SAFE_FRAGMENT_TOKENS = 4000
_DEFAULT_MAX_MONOLITHIC_UNIVERSES = 6
_DEFAULT_LARGE_PATCH_SAFE_OUTPUT_RATIO = 0.6


def estimate_universes_output_size(
    *,
    universe_count: int,
    avg_cells_per_universe: int = 4,
) -> int:
    """Estimate output token size for a monolithic universes patch."""
    # Empirical: each universe with ~4 cells costs roughly 400-600 JSON tokens.
    # This is deliberately conservative.
    base_overhead = 100  # patch_type, outer braces, etc.
    per_universe = 350 + avg_cells_per_universe * 80
    return base_overhead + universe_count * per_universe


def should_fragment_universes(
    *,
    mode: UniversesGenerationMode,
    universe_count: int,
    provider_max_output_tokens: int | None = None,
    reasoning_enabled: bool = False,
    history_json_truncated: bool = False,
    history_context_exhausted: bool = False,
    history_monolithic_parse_failure: bool = False,
    avg_cells_per_universe: int = 4,
    safe_output_ratio: float = _DEFAULT_LARGE_PATCH_SAFE_OUTPUT_RATIO,
) -> tuple[bool, str]:
    """Decide whether to fragment universe generation.

    Returns (should_fragment, reason).
    """
    if mode == "fragmented":
        return True, "explicit_fragmented_mode"
    if mode == "monolithic":
        return False, "explicit_monolithic_mode"
    # auto mode.
    if history_json_truncated:
        return True, "history_json_truncated"
    if history_context_exhausted:
        return True, "history_context_exhausted"
    estimated = estimate_universes_output_size(universe_count=universe_count, avg_cells_per_universe=avg_cells_per_universe)
    if provider_max_output_tokens is not None:
        budget = int(provider_max_output_tokens * safe_output_ratio)
        if reasoning_enabled:
            budget = int(budget * 0.6)  # reasoning consumes part of the output budget
        if estimated > budget:
            return True, f"estimated_{estimated}_exceeds_budget_{budget}"
    if universe_count > _DEFAULT_MAX_MONOLITHIC_UNIVERSES:
        return True, f"universe_count_{universe_count}_exceeds_threshold_{_DEFAULT_MAX_MONOLITHIC_UNIVERSES}"
    if history_monolithic_parse_failure:
        return True, "history_monolithic_parse_failure"
    return False, "estimated_safe_for_monolithic"


# ---------------------------------------------------------------------------
# Merge
# ---------------------------------------------------------------------------


class UniverseMergeIssue(AgentBaseModel):
    """A single structured issue discovered during fragment merge.

    ``retry_scope`` controls how the caller reacts:

    * ``fragment`` — only the universes in ``universe_id`` (and
      ``metadata.invalid_fragment_ids`` when present) need to be replayed;
      other accepted fragments remain usable.
    * ``manifest`` — the manifest itself is inconsistent (duplicate IDs,
      generation-order mismatch, item count mismatch).  Fail closed; do
      not attempt fragment replay.
    * ``global`` — the merged patch fails patch-level validation for
      reasons that cannot be attributed to a single fragment.  Fail
      closed with the full diagnostic.
    """

    code: str
    severity: Literal["error", "warning"] = "error"
    universe_id: str | None = None
    fragment_hash: str | None = None
    json_path: str | None = None
    message: str
    retry_scope: Literal["fragment", "manifest", "global"] = "global"
    retryable: bool = False
    expected: Any | None = None
    actual: Any | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class UniverseMergeResult(AgentBaseModel):
    """Structured result of merging fragments into a UniversesPatch."""

    ok: bool
    merged_patch: dict[str, Any] | None = None
    issues: list[UniverseMergeIssue] = Field(default_factory=list)
    invalid_fragment_ids: list[str] = Field(default_factory=list)
    merged_patch_hash: str | None = None
    manifest_id: str = ""
    manifest_input_hash: str = ""

    @property
    def top_level_error_code(self) -> str:
        """Backward-compatible top-level error code consumed by the
        existing retry owner policy.  Always ``patch_generation.merge_failed``
        when ``ok is False`` so routing continues to work unchanged."""
        return "patch_generation.merge_failed" if not self.ok else ""

    def to_legacy_tuple(self) -> tuple[dict[str, Any] | None, list[str]]:
        """Convert to the legacy ``(patch, list[str])`` return shape.

        Existing callers and tests that don't yet use the structured result
        can use this for backward compatibility.
        """
        return self.merged_patch, [issue.code for issue in self.issues if issue.severity == "error"]


def _compute_merged_patch_hash(merged_patch: dict[str, Any]) -> str:
    return hashlib.sha256(
        canonical_json_dumps(merged_patch).encode("utf-8")
    ).hexdigest()[:16]


def merge_universe_fragments_structured(
    *,
    manifest: UniverseManifest,
    fragments: list[UniverseDefinitionFragment],
    known_material_ids: set[str] | None = None,
    known_material_roles_by_id: dict[str, str] | None = None,
    qualification_records: dict[str, "AcceptedFragmentRecord"] | None = None,
) -> UniverseMergeResult:
    """Deterministically merge fragments into a structured result.

    Pure Python; no LLM.  Attribute every failure to a specific universe
    ID, fragment hash, and JSON path where possible so the caller can do
    targeted replay instead of a wholesale regeneration.

    Parameters
    ----------
    manifest
        The accepted manifest (its ``generation_order`` is canonical).
    fragments
        Fragments to merge.  May include duplicate/extra/missing IDs; all
        such conditions are reported as structured issues.
    known_material_ids
        Set of accepted material IDs from the upstream MaterialsPatch.
        Used to catch unknown material references before merged-patch
        validation.
    known_material_roles_by_id
        Optional mapping from material ID to material role; used to verify
        that required material roles are actually covered.
    qualification_records
        Optional mapping from universe ID to its last
        :class:`AcceptedFragmentRecord`.  When provided, the merge fails
        fragment-scoped if a fragment's saved qualification status is not
        ``passed`` or its contract hash has drifted.
    """
    issues: list[UniverseMergeIssue] = []
    invalid_fragment_ids: list[str] = []
    qualification_records = qualification_records or {}

    # --- Manifest self-consistency (fail-closed at manifest scope) ---
    order = list(manifest.generation_order)
    if len(order) != len(set(order)):
        dupes = sorted({uid for uid in order if order.count(uid) > 1})
        issues.append(UniverseMergeIssue(
            code="merge.manifest_duplicate_in_order",
            severity="error",
            message=f"manifest generation_order has duplicates: {dupes}",
            retry_scope="manifest",
            metadata={"duplicate_ids": dupes},
        ))
    expected_count = manifest.expected_universe_count
    if expected_count != len(manifest.items):
        issues.append(UniverseMergeIssue(
            code="merge.manifest_count_mismatch",
            severity="error",
            message=(
                f"manifest expected_universe_count={expected_count} but "
                f"items has {len(manifest.items)} entries"
            ),
            retry_scope="manifest",
            metadata={"expected": expected_count, "actual": len(manifest.items)},
        ))
    item_ids = {item.universe_id for item in manifest.items}
    if set(order) != item_ids:
        issues.append(UniverseMergeIssue(
            code="merge.manifest_order_items_mismatch",
            severity="error",
            message=(
                "manifest generation_order does not match item universe_ids: "
                f"order_only={sorted(set(order) - item_ids)} "
                f"items_only={sorted(item_ids - set(order))}"
            ),
            retry_scope="manifest",
        ))

    # --- Fragment index by universe_id (detect duplicates) ---
    frag_by_id: dict[str, UniverseDefinitionFragment] = {}
    for frag in fragments:
        uid = frag.universe_id
        if uid in frag_by_id:
            issues.append(UniverseMergeIssue(
                code="merge.duplicate_fragment",
                severity="error",
                universe_id=uid,
                fragment_hash=frag.fragment_hash or None,
                message=f"duplicate fragment for universe_id {uid!r}",
                retry_scope="fragment",
                retryable=True,
                json_path=f"/universes/{uid}",
            ))
            invalid_fragment_ids.append(uid)
            continue
        frag_by_id[uid] = frag

    # --- Coverage: every manifest item must have exactly one fragment ---
    merged_universes: list[dict[str, Any]] = []
    manifest_item_by_id = {item.universe_id: item for item in manifest.items}

    for uid in order:
        item = manifest_item_by_id.get(uid)
        if item is None:
            # Already reported as manifest inconsistency.
            continue
        frag = frag_by_id.get(uid)
        if frag is None:
            issues.append(UniverseMergeIssue(
                code="merge.missing_fragment",
                severity="error",
                universe_id=uid,
                message=f"missing fragment for universe_id {uid!r}",
                retry_scope="fragment",
                retryable=True,
                json_path=f"/universes/{uid}",
            ))
            invalid_fragment_ids.append(uid)
            continue
        universe_data = frag.universe or {}
        universe_data_id = universe_data.get("universe_id")
        if universe_data_id != uid:
            issues.append(UniverseMergeIssue(
                code="merge.universe_id_mismatch",
                severity="error",
                universe_id=uid,
                fragment_hash=frag.fragment_hash or None,
                message=(
                    f"fragment universe_id={universe_data_id!r} does not match "
                    f"manifest universe_id={uid!r}"
                ),
                retry_scope="fragment",
                retryable=True,
                json_path=f"/universes/{uid}/universe_id",
                actual=universe_data_id,
                expected=uid,
            ))
            invalid_fragment_ids.append(uid)
            continue

        # Kind consistency with manifest.
        fragment_kind = universe_data.get("kind")
        if item.kind and fragment_kind and fragment_kind != item.kind:
            issues.append(UniverseMergeIssue(
                code="merge.kind_mismatch",
                severity="error",
                universe_id=uid,
                fragment_hash=frag.fragment_hash or None,
                message=(
                    f"fragment kind={fragment_kind!r} does not match manifest "
                    f"kind={item.kind!r}"
                ),
                retry_scope="fragment",
                retryable=True,
                json_path=f"/universes/{uid}/kind",
                expected=item.kind,
                actual=fragment_kind,
            ))
            invalid_fragment_ids.append(uid)
            continue

        # Material reference checks (cell-level).
        if known_material_ids is not None:
            for cell in universe_data.get("cells", []) or []:
                mid = cell.get("material_id")
                if mid and mid not in known_material_ids:
                    issues.append(UniverseMergeIssue(
                        code="merge.unknown_material",
                        severity="error",
                        universe_id=uid,
                        fragment_hash=frag.fragment_hash or None,
                        message=(
                            f"cell {cell.get('id')!r} in universe {uid!r} "
                            f"references unknown material_id {mid!r}"
                        ),
                        retry_scope="fragment",
                        retryable=True,
                        json_path=f"/universes/{uid}/cells/{cell.get('id')}/material_id",
                        actual=mid,
                        expected=sorted(known_material_ids),
                    ))
                    if uid not in invalid_fragment_ids:
                        invalid_fragment_ids.append(uid)

        # Qualification/contract hash drift against the saved record.
        rec = qualification_records.get(uid)
        if rec is not None:
            if rec.qualification_status != "passed":
                issues.append(UniverseMergeIssue(
                    code="merge.qualification_not_passed",
                    severity="error",
                    universe_id=uid,
                    fragment_hash=frag.fragment_hash or None,
                    message=(
                        f"fragment {uid!r} qualification_status="
                        f"{rec.qualification_status!r}; cannot enter merge"
                    ),
                    retry_scope="fragment",
                    retryable=True,
                    json_path=f"/universes/{uid}",
                ))
                if uid not in invalid_fragment_ids:
                    invalid_fragment_ids.append(uid)
            elif item.contract_hash and rec.manifest_contract_hash and rec.manifest_contract_hash != item.contract_hash:
                issues.append(UniverseMergeIssue(
                    code="merge.manifest_contract_drift",
                    severity="error",
                    universe_id=uid,
                    fragment_hash=frag.fragment_hash or None,
                    message=(
                        f"fragment {uid!r} was qualified against contract_hash="
                        f"{rec.manifest_contract_hash!r} but the current manifest "
                        f"item has contract_hash={item.contract_hash!r}"
                    ),
                    retry_scope="fragment",
                    retryable=True,
                    json_path=f"/universes/{uid}",
                    expected=item.contract_hash,
                    actual=rec.manifest_contract_hash,
                ))
                if uid not in invalid_fragment_ids:
                    invalid_fragment_ids.append(uid)

        merged_universes.append(universe_data)

    # --- Extra (undeclared) fragments ---
    manifest_ids = set(order)
    extra = set(frag_by_id.keys()) - manifest_ids
    for eid in sorted(extra):
        issues.append(UniverseMergeIssue(
            code="merge.extra_fragment",
            severity="error",
            universe_id=eid,
            fragment_hash=frag_by_id[eid].fragment_hash or None,
            message=(
                f"fragment {eid!r} is not declared in the manifest; "
                f"it cannot enter the merged patch"
            ),
            retry_scope="fragment",
            retryable=False,  # not safe to auto-replay: source of the extra is unclear
            json_path=f"/universes/{eid}",
        ))

    if any(issue.severity == "error" for issue in issues):
        return UniverseMergeResult(
            ok=False,
            merged_patch=None,
            issues=issues,
            invalid_fragment_ids=sorted(set(invalid_fragment_ids)),
            manifest_id=manifest.manifest_id,
            manifest_input_hash=manifest.input_hash,
        )

    patch = {"patch_type": "universes", "universes": merged_universes}
    return UniverseMergeResult(
        ok=True,
        merged_patch=patch,
        issues=issues,  # may carry warnings
        invalid_fragment_ids=[],
        merged_patch_hash=_compute_merged_patch_hash(patch),
        manifest_id=manifest.manifest_id,
        manifest_input_hash=manifest.input_hash,
    )


def merge_universe_fragments(
    *,
    manifest: UniverseManifest,
    fragments: list[UniverseDefinitionFragment],
    known_material_ids: set[str] | None = None,
    known_material_roles_by_id: dict[str, str] | None = None,
    qualification_records: dict[str, AcceptedFragmentRecord] | None = None,
) -> tuple[dict[str, Any] | None, list[str]]:
    """Backward-compatible wrapper around :func:`merge_universe_fragments_structured`.

    Returns the legacy ``(merged_patch, error_codes)`` tuple.  New callers
    should use the structured function directly so they can react to
    fragment-scoped issues with targeted replay.
    """
    result = merge_universe_fragments_structured(
        manifest=manifest,
        fragments=fragments,
        known_material_ids=known_material_ids,
        known_material_roles_by_id=known_material_roles_by_id,
        qualification_records=qualification_records,
    )
    return result.to_legacy_tuple()


def validate_merged_patch(
    patch_dict: dict[str, Any],
    *,
    known_material_ids: set[str] | None = None,
    known_universe_ids: set[str] | None = None,
) -> tuple[bool, list[dict[str, Any]]]:
    """Validate merged patch using existing validators.

    Returns the legacy ``(ok, issues_list)`` tuple so existing callers and
    tests continue to work.  Pure-Python, deterministic, no LLM.
    """
    try:
        parsed = parse_patch_content("universes", patch_dict)
    except Exception as exc:
        return False, [{"code": "merge.schema_invalid", "severity": "error", "message": str(exc)}]
    ctx = PatchValidationContext(
        known_material_ids=list(known_material_ids) if known_material_ids else [],
        known_universe_ids=list(known_universe_ids) if known_universe_ids else [],
    )
    result = validate_patch(parsed, context=ctx)
    issues = [
        {
            "code": i.code,
            "severity": i.severity,
            "message": i.message,
            "path": i.path,
            "expected": i.expected,
            "actual": i.actual,
        }
        for i in result.issues
    ]
    return result.ok, issues


# ---------------------------------------------------------------------------
# Token budget
# ---------------------------------------------------------------------------


def resolve_patch_output_budget(
    *,
    explicit: int | None = None,
    fragment_mode: bool = False,
    provider_max_output: int | None = None,
) -> int:
    """Resolve the output token budget for a patch or fragment call.

    Priority: explicit > fragment-specific > provider capability > conservative default.
    """
    if explicit is not None:
        return explicit
    if fragment_mode:
        return _DEFAULT_SAFE_FRAGMENT_TOKENS
    if provider_max_output is not None:
        return provider_max_output
    return 8000


__all__ = [
    "UniverseGenerationRequirement",
    "UniverseGenerationRequirementSet",
    "extract_universe_requirements",
    "UniverseManifestItem",
    "UniverseManifest",
    "validate_manifest",
    "build_manifest_from_requirements",
    "compute_manifest_item_contract_hash",
    "UniverseDefinitionFragment",
    "AcceptedFragmentRecord",
    "FragmentStatus",
    "FragmentQualificationStatus",
    "LargePatchGenerationSession",
    "UniversesGenerationMode",
    "estimate_universes_output_size",
    "should_fragment_universes",
    "UniverseMergeIssue",
    "UniverseMergeResult",
    "merge_universe_fragments_structured",
    "merge_universe_fragments",
    "validate_merged_patch",
    "resolve_patch_output_budget",
]
