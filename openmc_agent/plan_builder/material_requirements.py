"""Material generation requirements derived from GeometryComponentInventory.

Replaces the legacy pattern of "let the Materials LLM figure out what
materials it needs from the requirement text" with a deterministic
MaterialGenerationRequirementSet that the Materials patch must satisfy.

Rules:
* Each fuel variant → one ``fuel`` material requirement.
* Each declared material role from the Inventory → one requirement.
* Poison and absorber are NEVER merged.
* Homogenized material requirements need source evidence; absent that,
  they become ``needs_confirmation`` rather than fabricated.
* Unknown compositions become ``needs_library`` requirements, NOT
  invented compositions.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping

from pydantic import Field, field_validator, model_validator

from openmc_agent.schemas import AgentBaseModel

from openmc_agent.plan_investigation.errors import PlanInvestigationIssue
from openmc_agent.plan_investigation.geometry_inventory import GeometryComponentInventory
from openmc_agent.plan_investigation.hashing import content_hash, short_id

__all__ = [
    "MaterialGenerationRequirement",
    "MaterialGenerationRequirementSet",
    "extract_material_requirements_from_inventory",
    "extract_material_requirements_from_facts",
    "validate_materials_against_requirement_set",
    "MaterialValidationReport",
]


# ---------------------------------------------------------------------------
# Requirement models
# ---------------------------------------------------------------------------


class MaterialGenerationRequirement(AgentBaseModel):
    """One material requirement the Materials patch must satisfy."""

    requirement_id: str
    role: str
    preferred_name: str | None = None
    source_variant_id: str | None = None
    localized_insert_requirement_id: str | None = None
    required_by_component_ids: tuple[str, ...] = Field(default_factory=tuple)
    required_by_profile_ids: tuple[str, ...] = Field(default_factory=tuple)
    density_required: bool = True
    temperature_required: bool = False
    composition_required: bool = True
    mixture_required: bool = False
    mixture_components: tuple[str, ...] = Field(default_factory=tuple)
    source_claim_ids: tuple[str, ...] = Field(default_factory=tuple)
    source_span_ids: tuple[str, ...] = Field(default_factory=tuple)
    resolution_status: str = "required"
    unresolved_fields: tuple[str, ...] = Field(default_factory=tuple)
    assumptions_allowed: bool = False


class MaterialGenerationRequirementSet(AgentBaseModel):
    """The complete material requirement set for one Materials patch."""

    requirements: tuple[MaterialGenerationRequirement, ...] = Field(default_factory=tuple)
    requirement_set_hash: str = ""
    inventory_hash: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _compute_hash(self) -> "MaterialGenerationRequirementSet":
        payload = {
            "r": [r.model_dump(mode="json") for r in self.requirements],
            "ih": self.inventory_hash,
        }
        expected = content_hash(payload)
        if not self.requirement_set_hash:
            object.__setattr__(self, "requirement_set_hash", expected)
        elif self.requirement_set_hash != expected:
            raise PlanInvestigationIssue(
                "plan_investigation.material_requirement_set_hash_mismatch",
                "requirement_set_hash does not match the recomputed value",
                details={"expected": expected, "actual": self.requirement_set_hash},
            )
        return self

    @property
    def roles(self) -> tuple[str, ...]:
        seen: list[str] = []
        for req in self.requirements:
            if req.role not in seen:
                seen.append(req.role)
        return tuple(seen)


class MaterialValidationReport(AgentBaseModel):
    """Result of validating a generated MaterialsPatch against requirements."""

    requirement_set_hash: str
    materials_patch_hash: str = ""
    covered_requirement_ids: tuple[str, ...] = Field(default_factory=tuple)
    uncovered_requirement_ids: tuple[str, ...] = Field(default_factory=tuple)
    unmatched_material_ids: tuple[str, ...] = Field(default_factory=tuple)
    fuel_variant_coverage: dict[str, str] = Field(default_factory=dict)
    warnings: tuple[str, ...] = Field(default_factory=tuple)
    errors: tuple[str, ...] = Field(default_factory=tuple)

    @property
    def ok(self) -> bool:
        return not self.uncovered_requirement_ids and not self.errors


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------


def extract_material_requirements_from_inventory(
    inventory: GeometryComponentInventory,
    *,
    accepted_facts: Any | None = None,
) -> MaterialGenerationRequirementSet:
    """Build the MaterialGenerationRequirementSet from the Inventory."""

    requirements: list[MaterialGenerationRequirement] = []

    # Walk the inventory's material_role_requirements and convert each
    # into a MaterialGenerationRequirement.
    requirements_by_key: dict[str, MaterialGenerationRequirement] = {}
    for mrole in inventory.material_role_requirements:
        # De-duplicate by (role, fuel_variant_id, localized_insert_requirement_id).
        # Multiple profiles declaring the same role + variant share one
        # material requirement; merge their required_by_profile_ids.
        dedup_key = content_hash(
            {
                "role": mrole.role,
                "variant": mrole.fuel_variant_id or "",
                "insert": mrole.localized_insert_requirement_id or "",
            }
        )
        if dedup_key in requirements_by_key:
            existing = requirements_by_key[dedup_key]
            merged_profiles = set(existing.required_by_profile_ids)
            merged_profiles.update(mrole.required_by_profile_ids)
            object.__setattr__(
                existing,
                "required_by_profile_ids",
                tuple(sorted(merged_profiles)),
            )
            continue
        requirements_by_key[dedup_key] = MaterialGenerationRequirement(
            requirement_id=short_id(
                "mreq",
                {
                    "role": mrole.role,
                    "variant": mrole.fuel_variant_id or "",
                    "insert": mrole.localized_insert_requirement_id or "",
                },
            ),
            role=mrole.role,
            source_variant_id=mrole.fuel_variant_id,
            localized_insert_requirement_id=mrole.localized_insert_requirement_id,
            required_by_profile_ids=mrole.required_by_profile_ids,
            density_required=True,
            composition_required=True,
            source_claim_ids=mrole.source_claim_ids,
            resolution_status="needs_library"
            if mrole.status == "needs_library"
            else "required",
        )

    requirements = list(requirements_by_key.values())

    # Add fuel-variant-specific requirements from accepted Facts when
    # available (preferred name, enrichment hint, density hint).
    if accepted_facts is not None:
        _augment_with_facts(requirements, accepted_facts)

    return MaterialGenerationRequirementSet(
        requirements=tuple(requirements),
        inventory_hash=inventory.inventory_hash,
        metadata={
            "inventory_id": inventory.inventory_id,
            "source_role_count": len(inventory.material_role_requirements),
        },
    )


def _augment_with_facts(
    requirements: list[MaterialGenerationRequirement], accepted_facts: Any
) -> None:
    """Attach enrichment / density hints to fuel-variant requirements."""

    variants = list(getattr(accepted_facts, "fuel_variant_requirements", []) or [])
    variant_by_id = {v.variant_id: v for v in variants}
    for req in requirements:
        if req.source_variant_id and req.source_variant_id in variant_by_id:
            variant = variant_by_id[req.source_variant_id]
            req.preferred_name = getattr(variant, "source_label", None) or req.preferred_name


def extract_material_requirements_from_facts(
    accepted_facts: Any,
) -> MaterialGenerationRequirementSet:
    """Derive a MaterialGenerationRequirementSet directly from accepted Facts.

    This is the fallback path used when the geometry-component inventory was
    not compiled (so ``planning_material_requirement_set`` is absent), allowing
    the materials fragmented-generation pipeline to activate on the basis of
    the FactsPatch's declared ``material_roles`` and ``fuel_variant_requirements``
    — mirroring how the universes pipeline derives its requirements from facts.

    Each fuel variant yields one ``fuel`` requirement; every other declared
    material role yields one requirement.  The result is a deterministic,
    source-backed manifest that drives per-material fragment generation and
    qualification, avoiding monolithic-truncation failures on large catalogs.
    """

    requirements: list[MaterialGenerationRequirement] = []
    seen_keys: set[tuple[str, str]] = set()

    fuel_variants = list(getattr(accepted_facts, "fuel_variant_requirements", []) or [])
    for variant in fuel_variants:
        variant_id = str(getattr(variant, "variant_id", "") or "")
        key = ("fuel", variant_id)
        if key in seen_keys or not variant_id:
            continue
        seen_keys.add(key)
        requirements.append(MaterialGenerationRequirement(
            requirement_id=short_id("mreq", {"role": "fuel_UO2", "variant": variant_id}),
            role="fuel_UO2",
            source_variant_id=variant_id,
            preferred_name=getattr(variant, "source_label", None),
            density_required=True,
            composition_required=True,
        ))

    material_roles = list(getattr(accepted_facts, "material_roles", []) or [])
    fuel_covered = bool(fuel_variants)
    for role in material_roles:
        if not isinstance(role, str) or not role.strip():
            continue
        role_norm = role.strip()
        role_lower = role_norm.lower()
        # If fuel variants were declared, skip any role that represents fuel
        # (matched by "fuel", "uo2", or an enrichment/wt pattern) so fuel is
        # not double-counted alongside the per-variant fuel requirements.
        if fuel_covered and (
            role_lower.startswith("fuel")
            or "fuel" in role_lower
            or "uo2" in role_lower
        ):
            continue
        key = ("role", role_norm)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        requirements.append(MaterialGenerationRequirement(
            requirement_id=short_id("mreq", {"role": role_norm}),
            role=role_norm,
            density_required=True,
            composition_required=True,
        ))

    return MaterialGenerationRequirementSet(
        requirements=tuple(requirements),
        inventory_hash="facts-derived",
        metadata={
            "source": "facts_fallback",
            "role_count": len(requirements),
        },
    )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_materials_against_requirement_set(
    *,
    materials_patch: Any,
    requirement_set: MaterialGenerationRequirementSet,
) -> MaterialValidationReport:
    """Check that a generated MaterialsPatch covers every requirement.

    ``materials_patch`` is a MaterialsPatch model (or dict) with a
    ``materials`` list of MaterialSpec-like objects exposing
    ``material_id`` and ``role``.
    """

    materials_list: list[Any] = []
    if hasattr(materials_patch, "materials"):
        materials_list = list(materials_patch.materials or [])
    elif isinstance(materials_patch, dict):
        materials_list = list(materials_patch.get("materials", []) or [])

    material_ids: set[str] = set()
    for m in materials_list:
        mid = m.get("material_id", "") if isinstance(m, dict) else getattr(m, "material_id", "")
        if mid:
            material_ids.add(mid)
    materials_by_role: dict[str, list[str]] = {}
    for m in materials_list:
        if isinstance(m, dict):
            role = m.get("role", "") or ""
            mid = m.get("material_id", "")
        else:
            role = getattr(m, "role", "") or ""
            mid = getattr(m, "material_id", "")
        if not role:
            continue
        materials_by_role.setdefault(role, []).append(mid)

    covered: list[str] = []
    uncovered: list[str] = []
    unmatched: list[str] = []
    fuel_variant_coverage: dict[str, str] = {}
    warnings: list[str] = []

    for req in requirement_set.requirements:
        candidates = materials_by_role.get(req.role, [])
        if candidates:
            covered.append(req.requirement_id)
            if req.source_variant_id:
                fuel_variant_coverage[req.source_variant_id] = candidates[0]
        else:
            uncovered.append(req.requirement_id)
            warnings.append(
                f"role '{req.role}' (requirement {req.requirement_id}) has no material"
            )

    # Unmatched materials: those whose role was not requested.
    requested_roles = {req.role for req in requirement_set.requirements}
    for role, ids in materials_by_role.items():
        if role not in requested_roles:
            unmatched.extend(ids)
            warnings.append(f"material(s) {ids} have role '{role}' not declared by Inventory")

    return MaterialValidationReport(
        requirement_set_hash=requirement_set.requirement_set_hash,
        covered_requirement_ids=tuple(sorted(covered)),
        uncovered_requirement_ids=tuple(sorted(uncovered)),
        unmatched_material_ids=tuple(sorted(unmatched)),
        fuel_variant_coverage=fuel_variant_coverage,
        warnings=tuple(warnings),
    )
