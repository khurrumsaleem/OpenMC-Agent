"""Deterministic material normalization.

Only performs transforms whose physical meaning is **unambiguous** given
the declared ``composition_basis``.  Each transform produces a new material
object (never modifies in place) and records the operation for provenance.

Supported transforms:

* ``stoichiometric_ratio`` → ``atom_fraction`` (UO2 fuel: O/U=2 expansion)
* ``ppm_by_weight`` → ``atom_fraction`` (borated water: ppm → atom fraction)
* ``atom_fraction`` / ``weight_fraction`` / ``atom_density_barn_cm``: pass-through (no transform needed)

Ambiguous or undeclared compositions are **not** normalized; they are
returned with ``normalization_status = AMBIGUOUS``.
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

from openmc_agent.material_semantics import (
    MaterialClassification,
    classify_material_semantics,
)
from openmc_agent.schemas import (
    CompositionValueBasis,
    ComplexMaterialSpec,
    MaterialSpec,
    NormalizationStatus,
    NuclideSpec,
)

NORMALIZATION_CONTRACT_VERSION = "1.0.0"

_NA_BARN_CM = 0.6022  # atoms/(barn·cm) per (g/cm³)/(g/mol)
_M_B = 10.81           # g/mol, natural boron molar mass
_M_H2O = 18.015        # g/mol
_B10_NATURAL = 0.199   # natural B-10 atom fraction
_B11_NATURAL = 0.801   # natural B-11 atom fraction

_URANIUM_NUCLIDES = {"U233", "U234", "U235", "U236", "U238"}


# --------------------------------------------------------------------------- #
# Data classes
# --------------------------------------------------------------------------- #


@dataclass
class MaterialNormalizationOperation:
    """A single normalization operation applied to a material."""

    operation: str
    reason: str
    input_summary: dict[str, float]
    output_summary: dict[str, float]
    parameters: dict[str, Any] = field(default_factory=dict)


@dataclass
class MaterialNormalizationResult:
    """Result of normalizing a material."""

    material_id: str
    material_name: str
    original_basis: CompositionValueBasis
    normalized_basis: CompositionValueBasis
    normalization_status: NormalizationStatus
    operations: list[MaterialNormalizationOperation] = field(default_factory=list)
    assumptions: list[str] = field(default_factory=list)
    requires_human_confirmation: bool = False
    original_hash: str = ""
    normalized_hash: str = ""


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #


def normalize_material_semantics(
    material: MaterialSpec | ComplexMaterialSpec,
    *,
    classification: MaterialClassification | None = None,
) -> tuple[MaterialSpec | ComplexMaterialSpec, MaterialNormalizationResult]:
    """Normalize a material composition deterministically.

    Returns ``(new_material, result)`` where ``new_material`` is a copy of
    the input with normalized composition and provenance fields populated.
    The original material is **never** modified.

    If the basis is ambiguous or invalid, the material is returned unchanged
    with ``normalization_status = AMBIGUOUS`` (or ``INVALID``).
    """
    if classification is None:
        classification = classify_material_semantics(material)

    original_hash = _composition_hash(material.composition)

    result = MaterialNormalizationResult(
        material_id=classification.material_id or getattr(material, "name", ""),
        material_name=classification.material_name,
        original_basis=classification.detected_basis,
        normalized_basis=classification.detected_basis,
        normalization_status=NormalizationStatus.NOT_REQUIRED,
        original_hash=original_hash,
    )

    # If basis is explicitly declared and is already renderable, no transform.
    basis = classification.detected_basis

    if basis in (
        CompositionValueBasis.ATOM_FRACTION,
        CompositionValueBasis.WEIGHT_FRACTION,
        CompositionValueBasis.ATOM_DENSITY_BARN_CM,
    ):
        # Already renderable — no normalization needed.
        result.normalization_status = NormalizationStatus.NOT_REQUIRED
        result.normalized_hash = original_hash
        new_mat = _apply_provenance(
            material, result, composition_unchanged=True,
        )
        return new_mat, result

    if basis == CompositionValueBasis.STOICHIOMETRIC_RATIO:
        new_mat, ops = _normalize_stoichiometric_uo2(material)
        result.operations = ops
        result.normalized_basis = CompositionValueBasis.ATOM_FRACTION
        result.normalization_status = NormalizationStatus.DETERMINISTICALLY_NORMALIZED
        result.normalized_hash = _composition_hash(new_mat.composition)
        return _apply_provenance(material, result, new_composition=new_mat.composition), result

    if basis == CompositionValueBasis.PPM_BY_WEIGHT:
        new_mat, ops = _normalize_ppm_borated_water(material)
        result.operations = ops
        result.normalized_basis = CompositionValueBasis.ATOM_FRACTION
        result.normalization_status = NormalizationStatus.DETERMINISTICALLY_NORMALIZED
        result.normalized_hash = _composition_hash(new_mat.composition)
        return _apply_provenance(material, result, new_composition=new_mat.composition), result

    if basis == CompositionValueBasis.UNKNOWN:
        result.normalization_status = NormalizationStatus.AMBIGUOUS
        result.requires_human_confirmation = True
        result.normalized_hash = original_hash
        new_mat = _apply_provenance(material, result, composition_unchanged=True)
        return new_mat, result

    # Other bases (ppm_by_atom, mass_density_component, elemental_enrichment)
    # are recognized but not yet implemented as deterministic transforms.
    result.normalization_status = NormalizationStatus.AMBIGUOUS
    result.requires_human_confirmation = True
    result.normalized_hash = original_hash
    new_mat = _apply_provenance(material, result, composition_unchanged=True)
    return new_mat, result


# --------------------------------------------------------------------------- #
# UO2 stoichiometric normalization
# --------------------------------------------------------------------------- #


def _normalize_stoichiometric_uo2(
    material: MaterialSpec | ComplexMaterialSpec,
) -> tuple[MaterialSpec | ComplexMaterialSpec, list[MaterialNormalizationOperation]]:
    """Expand stoichiometric O/U ratio to full atom fractions.

    Given a uranium isotopic vector summing to ~100 and O16 ≈ 2 (meaning
    O/U = 2), produces atom fractions where O is 2/3 of the total.
    """
    composition = material.composition
    u_entries = [c for c in composition if c.name in _URANIUM_NUCLIDES]
    o_entries = [c for c in composition if c.name == "O16"]

    if not u_entries or not o_entries:
        return material.model_copy(), []

    u_sum = sum(c.percent for c in u_entries)
    o_value = o_entries[0].percent

    # The stoichiometric ratio: O/U.  LLM/material patch inputs appear in two
    # equivalent scales: U isotopes sum to ~100 with O16=2.0 meaning O/U=2, or
    # U isotopes sum to ~100 with O16=200 already expanded onto the same scale.
    if u_sum > 0 and 1.5 <= (o_value / u_sum) <= 2.5:
        o_per_u = o_value / u_sum
    else:
        o_per_u = o_value if u_sum > 50 else (o_value / u_sum if u_sum > 0 else 2.0)

    input_summary = {c.name: c.percent for c in composition}

    # Scale O16 so that it's on the same scale as U (i.e., O16 = o_per_u * u_sum).
    # This produces the correct atom fractions when OpenMC normalizes.
    new_o16 = o_per_u * u_sum

    new_composition = []
    for c in composition:
        if c.name == "O16":
            new_composition.append(c.model_copy(update={"percent": new_o16}))
        else:
            new_composition.append(c.model_copy())

    output_summary = {c.name: c.percent for c in new_composition}

    op = MaterialNormalizationOperation(
        operation="uo2_stoichiometric_expansion",
        reason=f"O/U ratio = {o_per_u:.1f}, U sum = {u_sum:.1f}",
        input_summary=input_summary,
        output_summary=output_summary,
        parameters={
            "o_per_u": o_per_u,
            "uranium_sum": u_sum,
            "original_o16": o_value,
            "normalized_o16": new_o16,
        },
    )

    new_mat = material.model_copy(update={"composition": new_composition})
    return new_mat, [op]


# --------------------------------------------------------------------------- #
# Borated water ppm normalization
# --------------------------------------------------------------------------- #


def _normalize_ppm_borated_water(
    material: MaterialSpec | ComplexMaterialSpec,
) -> tuple[MaterialSpec | ComplexMaterialSpec, list[MaterialNormalizationOperation]]:
    """Convert ppm boron by weight to atom fractions.

    Handles two input formats:
    1. Patch format (``composition_basis='ppm_by_weight'``):
       B10=1066.0 means 1066 ppm total boron; H1=2.0, O16=1.0 are atom ratios.
    2. Legacy format: B10=0.001066 (atom fraction encoding ppm).
    """
    composition = material.composition
    density_value = getattr(material, "density_value", None) or 1.0
    density_unit = getattr(material, "density_unit", "g/cm3")

    nuclide_map = {c.name: c for c in composition}
    b10 = nuclide_map.get("B10")
    if b10 is None:
        return material.model_copy(), []

    # Detect ppm encoding: if B10 > 0.1, it's a raw ppm value (patch format).
    # If B10 < 0.01, it's an atom fraction encoding ppm (legacy format).
    if b10.percent > 0.1:
        ppm_value = b10.percent
    else:
        ppm_value = b10.percent * 1e6

    input_summary = {c.name: c.percent for c in composition}

    # Compute atom fractions from ppm.
    boron_mass_frac = ppm_value * 1e-6
    boron_atom_density = density_value * boron_mass_frac * _NA_BARN_CM / _M_B
    b10_density = boron_atom_density * _B10_NATURAL
    b11_density = boron_atom_density * _B11_NATURAL

    water_mass = density_value * (1 - boron_mass_frac)
    water_atom_density = water_mass * _NA_BARN_CM * 3 / _M_H2O

    total_atom_density = water_atom_density + boron_atom_density
    b10_correct_frac = b10_density / total_atom_density
    b11_correct_frac = b11_density / total_atom_density
    h1_correct_frac = water_atom_density * 2 / 3 / total_atom_density
    o16_correct_frac = water_atom_density / 3 / total_atom_density

    # Reconstruct full composition with correct atom fractions.
    new_composition = []
    for c in composition:
        if c.name == "B10":
            new_composition.append(c.model_copy(update={"percent": b10_correct_frac}))
        elif c.name == "B11":
            new_composition.append(c.model_copy(update={"percent": b11_correct_frac}))
        elif c.name == "H1":
            new_composition.append(c.model_copy(update={"percent": h1_correct_frac}))
        elif c.name == "O16":
            new_composition.append(c.model_copy(update={"percent": o16_correct_frac}))
        else:
            new_composition.append(c.model_copy())

    output_summary = {c.name: c.percent for c in new_composition}

    op = MaterialNormalizationOperation(
        operation="boron_ppm_to_atom_fraction",
        reason=f"{ppm_value:.0f} ppm total boron → atom fraction",
        input_summary=input_summary,
        output_summary=output_summary,
        parameters={
            "ppm_value": ppm_value,
            "density_value": density_value,
            "density_unit": density_unit,
            "b10_fraction": b10_correct_frac,
            "b11_fraction": b11_correct_frac,
            "constants": {
                "N_A_barn_cm": _NA_BARN_CM,
                "M_B": _M_B,
                "M_H2O": _M_H2O,
                "B10_natural": _B10_NATURAL,
                "B11_natural": _B11_NATURAL,
            },
        },
    )

    new_mat = material.model_copy(update={"composition": new_composition})
    return new_mat, [op]


# --------------------------------------------------------------------------- #
# Provenance application
# --------------------------------------------------------------------------- #


def _apply_provenance(
    material: MaterialSpec | ComplexMaterialSpec,
    result: MaterialNormalizationResult,
    *,
    composition_unchanged: bool = False,
    new_composition: list[NuclideSpec] | None = None,
) -> MaterialSpec | ComplexMaterialSpec:
    """Return a copy of the material with provenance fields populated."""
    updates: dict[str, Any] = {
        "composition_basis": result.original_basis,
        "normalization_status": result.normalization_status,
        "normalization_version": NORMALIZATION_CONTRACT_VERSION,
        "normalization_operations": [
            {
                "operation": op.operation,
                "reason": op.reason,
                "input_summary": op.input_summary,
                "output_summary": op.output_summary,
                "parameters": op.parameters,
            }
            for op in result.operations
        ],
        "semantic_assumptions": list(result.assumptions),
        "original_composition": [c.model_copy() for c in material.composition],
    }

    if new_composition is not None:
        updates["composition"] = new_composition
        updates["normalized_composition"] = [c.model_copy() for c in new_composition]
    elif composition_unchanged:
        updates["normalized_composition"] = [c.model_copy() for c in material.composition]

    if result.requires_human_confirmation:
        existing = list(getattr(material, "requires_human_confirmation", []))
        if "composition_basis" not in existing:
            existing.append("composition_basis")
        updates["requires_human_confirmation"] = existing

    return material.model_copy(update=updates)


def _composition_hash(composition: list[NuclideSpec]) -> str:
    """Compute a stable hash of a composition list."""
    data = [
        {"name": c.name, "percent": round(c.percent, 12), "percent_type": c.percent_type}
        for c in composition
    ]
    return hashlib.sha256(
        json.dumps(data, sort_keys=True).encode()
    ).hexdigest()
