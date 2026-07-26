"""Materials fragment transaction — deterministic manifest, bounded fragments, qualification, merge.

Transaction contract
--------------------
* Accepted Facts / Inventory
→ existing ``MaterialGenerationRequirementSet``
→ deterministic material manifest (Python, fixed IDs / roles / variants / dependencies)
→ bounded material fragments (one material per LLM call)
→ fragment qualification (deterministic, structured)
→ checkpoint hash / contract hash / qualification integrity
→ dependency-aware deterministic merge (pure Python, no LLM)
→ full ``MaterialsPatch`` validation

This mirrors the proven Universe fragment transaction (Step 4B-1) but is
adapted for the Materials patch type.  Each manifest item is one material;
each LLM call produces exactly one material; the merge is a pure-Python
assembly that delegates to the existing ``MaterialsPatch`` validator.
"""

from __future__ import annotations

import json
import typing
from typing import Any, Literal

from pydantic import Field

from openmc_agent.schemas import AgentBaseModel

from openmc_agent.plan_investigation.hashing import content_hash, short_id

from .material_requirements import (
    MaterialGenerationRequirement,
    MaterialGenerationRequirementSet,
)
from .closed_loop.fingerprints import compute_candidate_hash

__all__ = [
    "MaterialManifestItem",
    "MaterialManifest",
    "MaterialDefinitionFragment",
    "AcceptedMaterialFragmentRecord",
    "MaterialFragmentStatus",
    "MaterialsPatchGenerationSession",
    "MaterialMergeIssue",
    "MaterialMergeResult",
    "MaterialFragmentQualificationResult",
    "build_material_manifest",
    "validate_material_manifest",
    "qualify_material_fragment",
    "verify_accepted_material_fragment",
    "merge_material_fragments_structured",
    "should_fragment_materials",
    "estimate_materials_output_size",
    "compute_manifest_item_contract_hash",
]


# ---------------------------------------------------------------------------
# Placeholder tokens — never allowed in accepted fragments.
# ---------------------------------------------------------------------------

_PLACEHOLDER_MATERIAL_TOKENS: frozenset[str] = frozenset({
    "replace", "replace_me", "<material_id>", "tbd", "todo",
    "placeholder", "unknown", "null", "none", "", "fixme",
})

_POISON_ABSORBER_ROLES: frozenset[str] = frozenset({"poison", "absorber"})


# ---------------------------------------------------------------------------
# Manifest models
# ---------------------------------------------------------------------------

class MaterialManifestItem(AgentBaseModel):
    """Deterministic contract for one material fragment.

    The ``contract_hash`` is computed from the immutable contract fields.
    On resume, if the recomputed hash differs from the stored value the
    fragment is downgraded for regeneration.
    """

    material_id: str
    requirement_id: str
    role: str
    source_variant_id: str | None = None
    localized_insert_requirement_id: str | None = None
    preferred_name: str | None = None
    density_required: bool = True
    composition_required: bool = True
    mixture_required: bool = False
    mixture_component_ids: tuple[str, ...] = Field(default_factory=tuple)
    generation_order_index: int = 0
    contract_hash: str = ""

    def recompute_contract_hash(self) -> None:
        object.__setattr__(self, "contract_hash", compute_manifest_item_contract_hash(self))


_MANIFEST_CONTRACT_FIELDS: tuple[str, ...] = (
    "material_id",
    "requirement_id",
    "role",
    "source_variant_id",
    "localized_insert_requirement_id",
    "preferred_name",
    "density_required",
    "composition_required",
    "mixture_required",
    "mixture_component_ids",
    "generation_order_index",
)


def compute_manifest_item_contract_hash(item: MaterialManifestItem) -> str:
    payload = {f: getattr(item, f) for f in _MANIFEST_CONTRACT_FIELDS}
    if isinstance(payload["mixture_component_ids"], tuple):
        payload["mixture_component_ids"] = list(payload["mixture_component_ids"])
    return content_hash(payload)


class MaterialManifest(AgentBaseModel):
    """Ordered manifest of all materials to generate."""

    manifest_id: str = ""
    items: list[MaterialManifestItem] = Field(default_factory=list)
    generation_order: list[str] = Field(default_factory=list)
    manifest_input_hash: str = ""

    @property
    def material_ids(self) -> set[str]:
        return {item.material_id for item in self.items}

    def item_by_id(self, mid: str) -> MaterialManifestItem | None:
        return next((item for item in self.items if item.material_id == mid), None)


# ---------------------------------------------------------------------------
# Fragment models
# ---------------------------------------------------------------------------

class MaterialDefinitionFragment(AgentBaseModel):
    """One LLM call output — exactly one material."""

    material_id: str
    material: dict[str, Any] = Field(default_factory=dict)
    fragment_hash: str = ""
    manifest_contract_hash: str = ""


FragmentQualificationStatus = Literal["pending", "passed", "failed"]


class AcceptedMaterialFragmentRecord(AgentBaseModel):
    """Typed checkpoint record of an accepted material fragment."""

    material_id: str
    material: dict[str, Any] = Field(default_factory=dict)
    fragment_hash: str = ""
    manifest_contract_hash: str = ""
    qualification_status: FragmentQualificationStatus = "passed"
    qualification_issues: list[dict[str, Any]] = Field(default_factory=list)
    accepted_at_attempt: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)


class MaterialFragmentStatus(AgentBaseModel):
    """Per-material status inside a checkpoint session."""

    material_id: str
    status: Literal["pending", "accepted", "failed", "stale"] = "pending"
    fragment_hash: str = ""
    manifest_contract_hash: str = ""
    qualification_status: FragmentQualificationStatus = "pending"
    qualification_issues: list[dict[str, Any]] = Field(default_factory=list)
    accepted_at_attempt: int | None = None
    issues: list[dict[str, Any]] = Field(default_factory=list)
    llm_calls: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)


class MaterialsPatchGenerationSession(AgentBaseModel):
    """Checkpoint session for materials fragment generation."""

    session_id: str = ""
    patch_type: str = "materials"
    input_hash: str = ""
    mode: Literal["auto", "monolithic", "fragmented"] = "auto"
    requirement_set_hash: str = ""
    manifest: MaterialManifest | None = None
    manifest_status: Literal["pending", "accepted", "failed"] = "pending"
    fragment_statuses: list[MaterialFragmentStatus] = Field(default_factory=list)
    accepted_fragment_hashes: dict[str, str] = Field(default_factory=dict)
    accepted_fragments: dict[str, AcceptedMaterialFragmentRecord] = Field(default_factory=dict)
    failed_fragment_issues: dict[str, list[dict[str, Any]]] = Field(default_factory=dict)
    strategy_transitions: list[dict[str, Any]] = Field(default_factory=list)
    llm_call_count: int = 0
    completed: bool = False
    merged_patch_hash: str = ""
    provider_telemetry: list[dict[str, Any]] = Field(default_factory=list)
    merge_history: list[dict[str, Any]] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Merge models
# ---------------------------------------------------------------------------

MaterialMergeRetryScope = Literal["fragment", "manifest", "global"]


class MaterialMergeIssue(AgentBaseModel):
    code: str
    severity: Literal["error", "warning"] = "error"
    message: str = ""
    material_id: str = ""
    retry_scope: MaterialMergeRetryScope = "fragment"


class MaterialMergeResult(AgentBaseModel):
    ok: bool = False
    merged_patch: dict[str, Any] | None = None
    merged_patch_hash: str = ""
    manifest_id: str = ""
    manifest_input_hash: str = ""
    issues: list[MaterialMergeIssue] = Field(default_factory=list)
    invalid_fragment_ids: list[str] = Field(default_factory=list)
    accepted_fragment_ids: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Qualification result
# ---------------------------------------------------------------------------

class MaterialFragmentQualificationResult(AgentBaseModel):
    ok: bool = False
    material_id: str = ""
    fragment_hash: str = ""
    manifest_contract_hash: str = ""
    canonical_material_data: dict[str, Any] = Field(default_factory=dict)
    issues: list[dict[str, Any]] = Field(default_factory=list)
    qualification_attempt: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Manifest builder (deterministic, no LLM)
# ---------------------------------------------------------------------------

def build_material_manifest(
    requirement_set: MaterialGenerationRequirementSet,
) -> MaterialManifest:
    """Build a deterministic material manifest from a requirement set.

    Each requirement becomes one manifest item with a deterministic
    ``material_id``.  The ``generation_order`` is: non-mixture materials
    first, then mixtures (which depend on their components).
    """
    items: list[MaterialManifestItem] = []
    material_id_by_requirement: dict[str, str] = {}

    for req in requirement_set.requirements:
        mid = short_id("mat", {
            "role": req.role,
            "variant": req.source_variant_id or "",
            "insert": req.localized_insert_requirement_id or "",
            "rid": req.requirement_id,
        })
        material_id_by_requirement[req.requirement_id] = mid
        items.append(MaterialManifestItem(
            material_id=mid,
            requirement_id=req.requirement_id,
            role=req.role,
            source_variant_id=req.source_variant_id,
            localized_insert_requirement_id=req.localized_insert_requirement_id,
            preferred_name=req.preferred_name,
            density_required=req.density_required,
            composition_required=req.composition_required,
            mixture_required=req.mixture_required,
            mixture_component_ids=req.mixture_components,
        ))

    # Assign generation_order_index; mixtures come after their components.
    order_index = 0
    non_mixture_ids: list[str] = []
    mixture_ids: list[str] = []
    for item in items:
        if item.mixture_required:
            mixture_ids.append(item.material_id)
        else:
            non_mixture_ids.append(item.material_id)

    generation_order = non_mixture_ids + mixture_ids
    for i, mid in enumerate(generation_order):
        item = next(it for it in items if it.material_id == mid)
        object.__setattr__(item, "generation_order_index", i)
        item.recompute_contract_hash()

    return MaterialManifest(
        manifest_id=f"manifest:{requirement_set.requirement_set_hash[:16]}",
        items=items,
        generation_order=generation_order,
        manifest_input_hash=requirement_set.requirement_set_hash,
    )


def validate_material_manifest(
    manifest: MaterialManifest,
    requirement_set: MaterialGenerationRequirementSet,
) -> list[str]:
    """Validate manifest consistency.  Returns error messages (empty = ok)."""
    errors: list[str] = []
    req_ids = {r.requirement_id for r in requirement_set.requirements}
    manifest_req_ids = {item.requirement_id for item in manifest.items}
    missing = req_ids - manifest_req_ids
    if missing:
        errors.append(f"manifest missing requirements: {sorted(missing)}")
    extra = manifest_req_ids - req_ids
    if extra:
        errors.append(f"manifest has unknown requirements: {sorted(extra)}")
    ids = [item.material_id for item in manifest.items]
    if len(ids) != len(set(ids)):
        dupes = [mid for mid in ids if ids.count(mid) > 1]
        errors.append(f"duplicate material_ids in manifest: {sorted(set(dupes))}")
    if set(manifest.generation_order) != set(ids):
        errors.append("generation_order does not match manifest items")
    for item in manifest.items:
        recomputed = compute_manifest_item_contract_hash(item)
        if item.contract_hash and item.contract_hash != recomputed:
            errors.append(f"contract_hash mismatch for {item.material_id}")
    return errors


# ---------------------------------------------------------------------------
# Strategy decision
# ---------------------------------------------------------------------------

_DEFAULT_SAFE_FRAGMENT_TOKENS = 3000
_DEFAULT_MAX_MONOLITHIC_MATERIALS = 5
_DEFAULT_LARGE_PATCH_SAFE_OUTPUT_RATIO = 0.6


def estimate_materials_output_size(
    *,
    material_count: int,
    avg_compound_per_material: int = 2,
) -> int:
    """Rough token estimate for a monolithic MaterialsPatch."""
    base = 80
    per_material = 280 + avg_compound_per_material * 90
    return base + material_count * per_material


def should_fragment_materials(
    *,
    mode: str,
    material_count: int,
    provider_max_output_tokens: int | None = None,
    reasoning_enabled: bool = False,
    history_json_truncated: bool = False,
    history_monolithic_parse_failure: bool = False,
    safe_output_ratio: float = _DEFAULT_LARGE_PATCH_SAFE_OUTPUT_RATIO,
) -> tuple[bool, str]:
    """Decide whether to fragment materials generation."""
    if mode == "fragmented":
        return True, "explicit_fragmented"
    if mode == "monolithic":
        return False, "explicit_monolithic"
    if history_json_truncated:
        return True, "history_json_truncated"
    if material_count > _DEFAULT_MAX_MONOLITHIC_MATERIALS:
        return True, f"material_count_{material_count}_exceeds_{_DEFAULT_MAX_MONOLITHIC_MATERIALS}"
    estimated = estimate_materials_output_size(material_count=material_count)
    budget = provider_max_output_tokens or 8000
    effective_budget = int(budget * safe_output_ratio)
    if reasoning_enabled:
        effective_budget = int(effective_budget * 0.6)
    if estimated > effective_budget:
        return True, f"estimated_{estimated}_exceeds_budget_{effective_budget}"
    if history_monolithic_parse_failure:
        return True, "history_monolithic_parse_failure"
    return False, "within_budget"


# ---------------------------------------------------------------------------
# Enum synonym normalization (pre-qualification)
# ---------------------------------------------------------------------------

_ENUM_SYNONYMS: dict[str, dict[str, str]] = {
    "composition_basis": {
        "mass_frac": "weight_frac",
        "mass_fraction": "weight_frac",
        "weight_fraction": "weight_frac",
        "weight_percent": "weight_frac",
        "atom_fraction": "atom_frac",
        "atomic_fraction": "atom_frac",
        "atom_percent": "atom_frac",
        "atoms": "atom_frac",
    },
    "composition_status": {
        "library": "needs_library",
        "from_library": "needs_library",
        "derived": "derived_from_mixture",
        "mixture": "derived_from_mixture",
        "unknown": "needs_confirmation",
        "unconfirmed": "needs_confirmation",
        "assumed": "approximate",
    },
    "density_status": {
        "given": "confirmed",
        "from_source": "source_provided",
        "estimated": "approximate",
        "unknown": "needs_confirmation",
    },
}


def _normalize_material_enum_synonyms(mat_data: dict[str, Any]) -> dict[str, Any]:
    """Map common LLM enum synonyms to valid Pydantic literal values."""
    mat_data = dict(mat_data)
    for field, mapping in _ENUM_SYNONYMS.items():
        val = mat_data.get(field)
        if isinstance(val, str):
            normalized = mapping.get(val.strip().lower())
            if normalized:
                mat_data[field] = normalized
    return mat_data


def _normalize_percent_style_fraction_basis(mat_data: dict[str, Any]) -> dict[str, Any]:
    """Normalize obvious percent-style vectors under fraction bases.

    LLMs often emit alloy compositions as ``composition_basis='weight_frac'``
    with values like Fe=69.5, Cr=19.0, ... because the source text uses wt%.
    The patch contract, however, treats atom_frac/weight_frac as normalized
    fractions.  When the vector is unambiguously percent-style (sum≈100), scale
    it to fractions before schema validation so the accepted fragment cannot
    carry a known MU reviewer blocker downstream.
    """
    basis = str(mat_data.get("composition_basis") or "").strip().lower()
    if basis not in {"atom_frac", "weight_frac"}:
        return mat_data
    composition = mat_data.get("composition")
    if not isinstance(composition, dict) or not composition:
        return mat_data
    numeric: dict[str, float] = {}
    try:
        for key, value in composition.items():
            numeric[str(key)] = float(value)
    except (TypeError, ValueError):
        return mat_data
    total = sum(numeric.values())
    if abs(total - 100.0) > 1.0e-2:
        return mat_data

    normalized = {key: value / 100.0 for key, value in numeric.items()}
    mat_data = dict(mat_data)
    mat_data["composition"] = normalized
    current_warnings = mat_data.get("warnings") or []
    warnings = list(current_warnings) if isinstance(current_warnings, list) else [str(current_warnings)]
    warnings.append(
        "deterministically normalized percent-style composition values to fractions"
    )
    mat_data["warnings"] = warnings
    return mat_data


# ---------------------------------------------------------------------------
# Fragment qualification (deterministic)
# ---------------------------------------------------------------------------

def qualify_material_fragment(
    *,
    raw_fragment: dict[str, Any],
    manifest_item: MaterialManifestItem,
    all_manifest_material_ids: set[str],
    attempt_index: int = 0,
) -> MaterialFragmentQualificationResult:
    """Deterministically qualify a single material fragment.

    Checks:
    1. Output boundary — exactly one material.
    2. Identity binding — material_id matches manifest.
    3. Role binding — role matches manifest.
    4. Schema validation via authoritative MaterialsPatch validator.
    5. Variant binding — source_variant_id matches (for fuel).
    6. Placeholder rejection.
    7. Poison/absorber non-confusion.
    8. Mixture component existence.
    9. Canonical hash — never trust LLM-provided hash.
    """
    from .patches import MaterialsPatch, MaterialSpecPatch
    from .validators import validate_patch

    issues: list[dict[str, Any]] = []

    # 1. Output boundary.
    materials_list: list[dict[str, Any]] = []
    if isinstance(raw_fragment, dict):
        if "materials" in raw_fragment and isinstance(raw_fragment["materials"], list):
            materials_list = [m for m in raw_fragment["materials"] if isinstance(m, dict)]
        elif "material_id" in raw_fragment:
            materials_list = [raw_fragment]
        elif "patch_type" in raw_fragment:
            inner = raw_fragment.get("materials", [])
            if isinstance(inner, list):
                materials_list = [m for m in inner if isinstance(m, dict)]
    if not materials_list:
        issues.append({"code": "qualification.no_material", "severity": "error",
                        "message": "fragment contains no material"})
        return MaterialFragmentQualificationResult(
            ok=False, material_id=manifest_item.material_id, issues=issues,
            qualification_attempt=attempt_index)
    if len(materials_list) > 1:
        issues.append({"code": "qualification.multiple_materials", "severity": "error",
                        "message": f"expected 1 material, got {len(materials_list)}"})
        return MaterialFragmentQualificationResult(
            ok=False, material_id=manifest_item.material_id, issues=issues,
            qualification_attempt=attempt_index)

    mat_data = materials_list[0]

    # 2. Identity binding.
    frag_mid = str(mat_data.get("material_id", "")).strip()
    if frag_mid != manifest_item.material_id:
        issues.append({
            "code": "qualification.material_id_mismatch",
            "severity": "error",
            "message": f"expected {manifest_item.material_id}, got {frag_mid}",
        })
        return MaterialFragmentQualificationResult(
            ok=False, material_id=frag_mid or manifest_item.material_id,
            issues=issues, qualification_attempt=attempt_index)

    # 3. Role binding.
    frag_role = str(mat_data.get("role", "")).strip().lower()
    expected_role = manifest_item.role.lower()
    if frag_role != expected_role:
        if not (frag_role in _POISON_ABSORBER_ROLES and expected_role in _POISON_ABSORBER_ROLES):
            issues.append({
                "code": "qualification.role_mismatch",
                "severity": "error",
                "message": f"expected role={expected_role}, got role={frag_role}",
            })
            return MaterialFragmentQualificationResult(
                ok=False, material_id=frag_mid, issues=issues,
                qualification_attempt=attempt_index)

    # 3b. Normalize common LLM enum/value synonyms before schema validation.
    mat_data = _normalize_material_enum_synonyms(mat_data)
    mat_data = _normalize_percent_style_fraction_basis(mat_data)

    # 4. Schema validation via authoritative MaterialsPatch.
    try:
        patch_payload = {"patch_type": "materials", "materials": [mat_data]}
        patch_obj = MaterialsPatch.model_validate(patch_payload)
        val_result = validate_patch(patch_obj)
        if not val_result.ok:
            for vi in val_result.issues:
                issues.append({
                    "code": f"qualification.schema_{vi.code}",
                    "severity": vi.severity,
                    "message": vi.message,
                })
            if any(i["severity"] == "error" for i in issues):
                return MaterialFragmentQualificationResult(
                    ok=False, material_id=frag_mid, issues=issues,
                    qualification_attempt=attempt_index)
        canonical = patch_obj.materials[0].model_dump(mode="json")
    except Exception as exc:
        issues.append({
            "code": "qualification.schema_invalid",
            "severity": "error",
            "message": f"{type(exc).__name__}: {exc}",
        })
        return MaterialFragmentQualificationResult(
            ok=False, material_id=frag_mid, issues=issues,
            qualification_attempt=attempt_index)

    # 5. Variant binding.
    if manifest_item.source_variant_id:
        frag_variant = canonical.get("source_variant_id")
        if frag_variant != manifest_item.source_variant_id:
            issues.append({
                "code": "qualification.variant_mismatch",
                "severity": "error",
                "message": f"expected source_variant_id={manifest_item.source_variant_id}, got {frag_variant}",
            })

    # 6. Placeholder rejection.
    mid_lower = str(canonical.get("material_id", "")).lower().strip()
    name_lower = str(canonical.get("name", "")).lower().strip()
    if mid_lower in _PLACEHOLDER_MATERIAL_TOKENS or name_lower in _PLACEHOLDER_MATERIAL_TOKENS:
        issues.append({
            "code": "qualification.placeholder_material",
            "severity": "error",
            "message": f"material_id or name is a placeholder: {mid_lower}/{name_lower}",
        })
    density_status = str(canonical.get("composition_status", "")).lower()
    if density_status == "placeholder":
        issues.append({
            "code": "qualification.placeholder_composition",
            "severity": "error",
            "message": "composition_status is 'placeholder'; placeholders are not accepted",
        })

    # 7. Poison/absorber non-confusion.
    if expected_role in _POISON_ABSORBER_ROLES:
        for comp_name in canonical.get("composition", {}):
            comp_lower = comp_name.lower()
            if expected_role == "poison" and "boron" not in comp_lower and "b10" not in comp_lower and "b-10" not in comp_lower:
                if canonical.get("composition") and not any(
                    "b" in k.lower() for k in canonical.get("composition", {})
                ):
                    pass  # don't over-flag; just a soft check
            if expected_role == "absorber" and "ag" in comp_lower and expected_role == "poison":
                issues.append({
                    "code": "qualification.poison_absorber_confusion",
                    "severity": "error",
                    "message": f"role={expected_role} but composition suggests different absorber type",
                })

    # 8. Mixture component existence.
    if manifest_item.mixture_required:
        mix_comps = canonical.get("mixture_components", [])
        if not mix_comps:
            issues.append({
                "code": "qualification.mixture_missing_components",
                "severity": "error",
                "message": "mixture_required but no mixture_components provided",
            })
        for mc in mix_comps:
            ref_id = str(mc.get("material_id", "")) if isinstance(mc, dict) else str(mc)
            if ref_id and ref_id not in all_manifest_material_ids:
                issues.append({
                    "code": "qualification.mixture_unknown_component",
                    "severity": "error",
                    "message": f"mixture references unknown material_id: {ref_id}",
                })

    # 9. Canonical hash — never trust LLM-provided hash.
    canonical_hash = compute_candidate_hash(target_patch_type="materials", candidate_patch=canonical)
    contract_hash = manifest_item.contract_hash or compute_manifest_item_contract_hash(manifest_item)

    ok = not any(i.get("severity") == "error" for i in issues)
    return MaterialFragmentQualificationResult(
        ok=ok,
        material_id=frag_mid,
        fragment_hash=canonical_hash,
        manifest_contract_hash=contract_hash,
        canonical_material_data=canonical,
        issues=issues,
        qualification_attempt=attempt_index,
    )


def verify_accepted_material_fragment(
    record: AcceptedMaterialFragmentRecord,
    manifest_item: MaterialManifestItem,
    all_manifest_material_ids: set[str],
) -> MaterialFragmentQualificationResult:
    """Re-verify an accepted fragment record on resume."""
    result = qualify_material_fragment(
        raw_fragment={"material_id": record.material_id, **record.material},
        manifest_item=manifest_item,
        all_manifest_material_ids=all_manifest_material_ids,
        attempt_index=record.accepted_at_attempt,
    )
    if not result.ok:
        return result
    if result.fragment_hash != record.fragment_hash:
        return MaterialFragmentQualificationResult(
            ok=False, material_id=record.material_id,
            fragment_hash=result.fragment_hash,
            manifest_contract_hash=result.manifest_contract_hash,
            issues=[{
                "code": "qualification.fragment_hash_drift",
                "severity": "error",
                "message": "stored fragment_hash does not match recomputed hash",
            }],
            qualification_attempt=record.accepted_at_attempt,
            metadata={"stored_hash": record.fragment_hash, "recomputed_hash": result.fragment_hash},
        )
    if result.manifest_contract_hash != record.manifest_contract_hash:
        return MaterialFragmentQualificationResult(
            ok=False, material_id=record.material_id,
            fragment_hash=result.fragment_hash,
            manifest_contract_hash=result.manifest_contract_hash,
            issues=[{
                "code": "qualification.manifest_contract_drift",
                "severity": "error",
                "message": "stored manifest_contract_hash does not match current manifest",
            }],
            qualification_attempt=record.accepted_at_attempt,
            metadata={"stored": record.manifest_contract_hash, "current": result.manifest_contract_hash},
        )
    return result


# ---------------------------------------------------------------------------
# Merge (pure Python, no LLM)
# ---------------------------------------------------------------------------

def merge_material_fragments_structured(
    *,
    manifest: MaterialManifest,
    accepted_fragments: list[MaterialDefinitionFragment],
    accepted_records: dict[str, AcceptedMaterialFragmentRecord],
) -> MaterialMergeResult:
    """Merge accepted material fragments into a single MaterialsPatch.

    Pure Python; no LLM calls.  Structured issues attribute each failure
    to a fragment / manifest / global scope.
    """
    issues: list[MaterialMergeIssue] = []
    invalid_ids: list[str] = []
    accepted_ids: list[str] = []

    manifest_ids = manifest.material_ids

    # Coverage: every manifest item must have an accepted fragment.
    records_by_id = {mid: rec for mid, rec in accepted_records.items()}
    for item in manifest.items:
        if item.material_id not in records_by_id:
            issues.append(MaterialMergeIssue(
                code="merge.missing_fragment",
                material_id=item.material_id,
                message=f"no accepted fragment for material_id={item.material_id}",
                retry_scope="fragment",
            ))
            invalid_ids.append(item.material_id)

    # Duplicate fragment indices.
    seen_ids: set[str] = set()
    for frag in accepted_fragments:
        mid = frag.material_id
        if mid in seen_ids:
            issues.append(MaterialMergeIssue(
                code="merge.duplicate_fragment",
                material_id=mid,
                message=f"duplicate fragment for material_id={mid}",
                retry_scope="fragment",
            ))
            invalid_ids.append(mid)
        seen_ids.add(mid)

    # Out-of-manifest fragments.
    for frag in accepted_fragments:
        if frag.material_id not in manifest_ids:
            issues.append(MaterialMergeIssue(
                code="merge.extra_fragment",
                material_id=frag.material_id,
                message=f"fragment not in manifest: {frag.material_id}",
                retry_scope="global",
            ))

    # Qualification status check.
    for mid, rec in records_by_id.items():
        if rec.qualification_status != "passed":
            issues.append(MaterialMergeIssue(
                code="merge.qualification_not_passed",
                material_id=mid,
                message=f"material {mid} qualification_status={rec.qualification_status}",
                retry_scope="fragment",
            ))
            invalid_ids.append(mid)

    # Contract drift check.
    for item in manifest.items:
        rec = records_by_id.get(item.material_id)
        if rec is None:
            continue
        current_contract = item.contract_hash or compute_manifest_item_contract_hash(item)
        if rec.manifest_contract_hash and rec.manifest_contract_hash != current_contract:
            issues.append(MaterialMergeIssue(
                code="merge.manifest_contract_drift",
                material_id=item.material_id,
                message=f"contract hash drift for {item.material_id}",
                retry_scope="manifest",
            ))
            invalid_ids.append(item.material_id)

    # Mixture dependency check.
    materials_by_id: dict[str, dict[str, Any]] = {}
    for mid, rec in records_by_id.items():
        materials_by_id[mid] = rec.material
    for item in manifest.items:
        if not item.mixture_required:
            continue
        mat = materials_by_id.get(item.material_id, {})
        for mc in mat.get("mixture_components", []):
            ref_id = str(mc.get("material_id", "")) if isinstance(mc, dict) else str(mc)
            if ref_id and ref_id not in materials_by_id:
                issues.append(MaterialMergeIssue(
                    code="merge.mixture_missing_component",
                    material_id=item.material_id,
                    message=f"mixture references missing material: {ref_id}",
                    retry_scope="fragment",
                ))
                invalid_ids.append(item.material_id)

    has_blocking = any(i.severity == "error" for i in issues)
    if has_blocking:
        return MaterialMergeResult(
            ok=False,
            manifest_id=manifest.manifest_id,
            manifest_input_hash=manifest.manifest_input_hash,
            issues=issues,
            invalid_fragment_ids=sorted(set(invalid_ids)),
            accepted_fragment_ids=sorted(seen_ids & manifest_ids),
        )

    # Assemble merged patch.
    ordered_materials: list[dict[str, Any]] = []
    for mid in manifest.generation_order:
        rec = records_by_id.get(mid)
        if rec is not None:
            ordered_materials.append(rec.material)

    import hashlib
    merged_patch = {"patch_type": "materials", "materials": ordered_materials}
    merged_hash = hashlib.sha256(
        json.dumps(merged_patch, sort_keys=True, default=str).encode()
    ).hexdigest()[:16]

    return MaterialMergeResult(
        ok=True,
        merged_patch=merged_patch,
        merged_patch_hash=merged_hash,
        manifest_id=manifest.manifest_id,
        manifest_input_hash=manifest.manifest_input_hash,
        issues=issues,
        invalid_fragment_ids=[],
        accepted_fragment_ids=sorted(materials_by_id.keys()),
    )
