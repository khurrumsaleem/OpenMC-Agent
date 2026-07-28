"""Deterministic normalization for MaterialsPatch content.

This module operates on patch dictionaries before they are persisted into
``PlanBuildState``.  It is intentionally input-driven: no reactor/benchmark
constant is hard-coded here.  A soluble-boron correction is applied only when
the requirement/facts text explicitly declares a boron mass concentration and
the generated coolant/moderator material already carries a water-like atom
fraction vector that can be corrected unambiguously.
"""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass
from typing import Any


_MASS = {
    "H": 1.00784,
    "H1": 1.00784,
    "O": 15.999,
    "O16": 15.994,
    "B": 10.81,
    "B10": 10.012937,
    "B11": 11.009305,
}
_REPAIR_RELATIVE_TOLERANCE = 0.01

_COOLANT_ROLE_TOKENS = {"coolant", "moderator"}
_COOLANT_TEXT_RE = re.compile(r"(coolant|moderator|borated\s*water|water|冷却剂|慢化剂|硼水|水)", re.IGNORECASE)
_BORON_TEXT_RE = re.compile(r"(boron|soluble\s+boron|borated|硼|可溶硼)", re.IGNORECASE)
_PPM_RE = re.compile(r"(?P<value>\d+(?:\.\d+)?)\s*ppm", re.IGNORECASE)
_MASS_FRACTION_RE = re.compile(
    r"(?:boron|硼)[^\n]{0,24}(?:mass\s*fraction|质量份额|质量分数)[^\d]{0,12}"
    r"(?P<value>0?\.\d+|\d+(?:\.\d+)?)",
    re.IGNORECASE,
)
_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")


@dataclass(frozen=True)
class SolubleBoronRequirement:
    """Source-declared soluble-boron concentration."""

    mass_fraction: float
    source: str
    b10_atom_fraction: float | None = None

    @property
    def ppm_by_weight(self) -> float:
        return self.mass_fraction * 1_000_000.0


@dataclass(frozen=True)
class MaterialsPatchNormalizationResult:
    """Result of patch-level deterministic normalization."""

    content: dict[str, Any]
    operations: list[dict[str, Any]]

    @property
    def changed(self) -> bool:
        return bool(self.operations)


def normalize_materials_patch_content(
    content: dict[str, Any],
    *,
    state: Any | None = None,
    requirement_text: str | None = None,
) -> MaterialsPatchNormalizationResult:
    """Normalize a MaterialsPatch dict using source-declared unit semantics.

    The current production correction covers a common LLM failure mode:
    soluble boron is source-specified in ppm by mass, but the generated coolant
    material declares tiny normalized atom fractions that correspond to a much
    lower boron loading.  When H/O/B atom fractions are present, the correction
    recomputes the whole water vector from the source mass fraction and the
    candidate isotope split, then records a deterministic operation.
    """

    if not isinstance(content, dict) or content.get("patch_type") != "materials":
        return MaterialsPatchNormalizationResult(content=content, operations=[])

    source_text = _collect_source_text(state=state, requirement_text=requirement_text)
    requirement = extract_soluble_boron_requirement(source_text)
    if requirement is None:
        return MaterialsPatchNormalizationResult(content=content, operations=[])

    normalized = copy.deepcopy(content)
    operations: list[dict[str, Any]] = []
    for material in normalized.get("materials", []) or []:
        if not isinstance(material, dict) or not _is_coolant_material(material):
            continue
        operation = _normalize_coolant_boron_atom_fraction(material, requirement)
        if operation is not None:
            operations.append(operation)

    return MaterialsPatchNormalizationResult(content=normalized, operations=operations)


def normalize_materials_patches_in_state(state: Any) -> list[dict[str, Any]]:
    """Apply production material normalization to valid patches and assembled plan."""

    operations: list[dict[str, Any]] = []
    source_text = _collect_source_text(state=state, requirement_text=None)
    requirement = extract_soluble_boron_requirement(source_text)
    if requirement is None:
        return operations
    patches = getattr(state, "patches", {})
    if not isinstance(patches, dict):
        patches = {}
    else:
        for patch in patches.values():
            if getattr(patch, "patch_type", None) != "materials":
                continue
            if getattr(patch, "status", None) != "valid":
                continue
            content = getattr(patch, "content", None)
            if not isinstance(content, dict):
                continue
            result = normalize_materials_patch_content(content, state=state)
            if not result.changed:
                continue
            patch.content = result.content
            patch.metadata.setdefault("deterministic_normalizations", [])
            patch.metadata["deterministic_normalizations"].extend(result.operations)
            getattr(state, "metadata", {}).setdefault("materials_deterministic_normalizations", [])
            state.metadata["materials_deterministic_normalizations"].extend(result.operations)
            operations.extend(result.operations)

    assembled_ops = _normalize_assembled_plan_materials_in_state(state, requirement)
    if assembled_ops:
        getattr(state, "metadata", {}).setdefault("materials_deterministic_normalizations", [])
        state.metadata["materials_deterministic_normalizations"].extend(assembled_ops)
        operations.extend(assembled_ops)
    if operations and hasattr(state, "add_event"):
        state.add_event(
            event_type="planning.materials_deterministic_normalization_applied",
            message="materials patch normalized using source-declared unit semantics",
            data={
                "operation_count": len(operations),
                "operations": [
                    {
                        "operation": op.get("operation"),
                        "material_id": op.get("material_id"),
                        "target_ppm_by_weight": op.get("target_ppm_by_weight"),
                        "previous_boron_mass_fraction": op.get("previous_boron_mass_fraction"),
                        "relative_error": op.get("relative_error"),
                    }
                    for op in operations
                ],
            },
        )
    return operations


def _normalize_assembled_plan_materials_in_state(
    state: Any,
    requirement: SolubleBoronRequirement,
) -> list[dict[str, Any]]:
    assembled = getattr(state, "assembled_plan", None)
    if not isinstance(assembled, dict):
        return []
    complex_model = assembled.get("complex_model")
    if not isinstance(complex_model, dict):
        return []
    materials = complex_model.get("materials")
    if not isinstance(materials, list):
        return []
    operations: list[dict[str, Any]] = []
    for material in materials:
        if not isinstance(material, dict):
            continue
        composition = material.get("composition")
        if not isinstance(composition, list):
            continue
        composition_dict = {
            str(item.get("name")): item.get("percent")
            for item in composition
            if isinstance(item, dict) and item.get("name")
        }
        patch_like = {
            "material_id": material.get("id") or material.get("material_id"),
            "name": material.get("name"),
            "role": material.get("role") or "",
            "composition": composition_dict,
            "composition_basis": material.get("composition_basis"),
            "composition_status": material.get("composition_status", "needs_confirmation"),
            "source_note": material.get("source_note"),
            "warnings": list(material.get("warnings") or []),
        }
        operation = _normalize_coolant_boron_atom_fraction(patch_like, requirement)
        if operation is None:
            continue
        _apply_composition_dict_to_plan_material(material, patch_like["composition"])
        material["composition_basis"] = "atom_fraction"
        operation["target"] = "assembled_plan"
        operations.append(operation)
    return operations


def _apply_composition_dict_to_plan_material(
    material: dict[str, Any],
    composition_dict: dict[str, float],
) -> None:
    existing = {
        str(item.get("name")): item
        for item in material.get("composition", [])
        if isinstance(item, dict) and item.get("name")
    }
    new_items: list[dict[str, Any]] = []
    for name, percent in composition_dict.items():
        item = dict(existing.get(name) or {"name": name, "percent_type": "ao", "kind": "nuclide"})
        item["name"] = name
        item["percent"] = percent
        item.setdefault("percent_type", "ao")
        item.setdefault("kind", "nuclide")
        new_items.append(item)
    material["composition"] = new_items


def extract_soluble_boron_requirement(text: str) -> SolubleBoronRequirement | None:
    """Extract a source-declared soluble-boron mass fraction from text."""

    if not text:
        return None
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    b10_atom_fraction: float | None = None
    for line in lines:
        if not _BORON_TEXT_RE.search(line):
            continue
        if re.search(r"B-?10|B10", line, re.IGNORECASE) and "%" in line:
            # Avoid interpreting the isotope label itself ("B-10") as the
            # enrichment value.  The percentage is the final numeric token on
            # lines such as "B-10 isotope fraction 19.9 at%".
            values = [float(item.group(0)) for item in _NUMBER_RE.finditer(line)]
            if values:
                value = values[-1]
                if 0.0 < value < 100.0:
                    b10_atom_fraction = value / 100.0

    for line in lines:
        if not _BORON_TEXT_RE.search(line):
            continue
        mass_match = _MASS_FRACTION_RE.search(line)
        if mass_match:
            value = float(mass_match.group("value"))
            if 0.0 < value < 1.0:
                return SolubleBoronRequirement(
                    mass_fraction=value,
                    source=line,
                    b10_atom_fraction=b10_atom_fraction,
                )
        ppm_match = _PPM_RE.search(line)
        if ppm_match:
            ppm = float(ppm_match.group("value"))
            if 0.0 < ppm < 100_000.0:
                return SolubleBoronRequirement(
                    mass_fraction=ppm / 1_000_000.0,
                    source=line,
                    b10_atom_fraction=b10_atom_fraction,
                )
    return None


def _collect_source_text(*, state: Any | None, requirement_text: str | None) -> str:
    chunks: list[str] = []
    if requirement_text:
        chunks.append(str(requirement_text))
    if state is not None:
        chunks.append(str(getattr(state, "requirement_text", "") or ""))
        for attr in ("confirmed_facts", "extracted_facts"):
            value = getattr(state, attr, None)
            if value:
                chunks.append(str(value))
        for envelope in getattr(state, "patches", {}).values():
            if getattr(envelope, "patch_type", None) != "facts":
                continue
            content = getattr(envelope, "content", None)
            if isinstance(content, dict):
                for key in ("source_notes", "material_roles", "missing_facts"):
                    value = content.get(key)
                    if value:
                        chunks.append(str(value))
    return "\n".join(chunks)


def _is_coolant_material(material: dict[str, Any]) -> bool:
    role = str(material.get("role") or "").strip().lower()
    if role in _COOLANT_ROLE_TOKENS:
        return True
    text = " ".join(str(material.get(key) or "") for key in ("material_id", "name", "source_note"))
    return bool(_COOLANT_TEXT_RE.search(text))


def _normalize_coolant_boron_atom_fraction(
    material: dict[str, Any],
    requirement: SolubleBoronRequirement,
) -> dict[str, Any] | None:
    basis = str(material.get("composition_basis") or "").strip().lower()
    if basis not in {"atom_frac", "atom_fraction"}:
        return None
    composition = material.get("composition")
    if not isinstance(composition, dict):
        return None
    keys = {str(key) for key in composition}
    if not ({"H1", "O16"} <= keys or {"H", "O"} <= keys):
        return None
    current_boron_mass_fraction = _boron_mass_fraction_from_atom_fraction(composition)
    if current_boron_mass_fraction <= 0 and requirement.b10_atom_fraction is None:
        return None
    target = requirement.mass_fraction
    relative_error = (
        abs(current_boron_mass_fraction - target) / target
        if current_boron_mass_fraction > 0 else 1.0
    )
    if relative_error <= _REPAIR_RELATIVE_TOLERANCE:
        return None

    b10_split = _candidate_b10_split(composition)
    if b10_split is None:
        b10_split = requirement.b10_atom_fraction
    if b10_split is None:
        return None

    new_comp = _borated_water_atom_fractions(
        boron_mass_fraction=target,
        b10_atom_fraction=b10_split,
        h_key="H1" if "H1" in keys else "H",
        o_key="O16" if "O16" in keys else "O",
    )
    old_comp = dict(composition)
    material["composition"] = new_comp
    material["composition_basis"] = "atom_frac"
    material["composition_status"] = (
        "confirmed"
        if material.get("composition_status") in {"confirmed", "source_provided"}
        else material.get("composition_status", "needs_confirmation")
    )
    warnings = list(material.get("warnings") or [])
    warning = (
        "deterministically normalized coolant boron atom fractions from "
        f"source-declared {requirement.ppm_by_weight:.6g} ppm by mass"
    )
    if warning not in warnings:
        warnings.append(warning)
    material["warnings"] = warnings
    source_note = str(material.get("source_note") or "").strip()
    note = (
        f"coolant boron atom fractions normalized from source-declared "
        f"{requirement.ppm_by_weight:.6g} ppm by mass"
    )
    material["source_note"] = f"{source_note}; {note}" if source_note else note

    return {
        "operation": "coolant_boron_mass_ppm_atom_fraction_repair",
        "material_id": material.get("material_id"),
        "basis": basis,
        "source": requirement.source,
        "target_boron_mass_fraction": target,
        "target_ppm_by_weight": requirement.ppm_by_weight,
        "previous_boron_mass_fraction": current_boron_mass_fraction,
        "relative_error": relative_error,
        "b10_atom_fraction": b10_split,
        "input_composition": old_comp,
        "output_composition": new_comp,
    }


def _candidate_b10_split(composition: dict[str, Any]) -> float | None:
    b10 = _num(composition.get("B10"))
    b11 = _num(composition.get("B11"))
    if b10 is None or b11 is None or b10 < 0 or b11 < 0 or (b10 + b11) <= 0:
        return None
    split = b10 / (b10 + b11)
    if 0.0 < split < 1.0:
        return split
    return None


def _boron_mass_fraction_from_atom_fraction(composition: dict[str, Any]) -> float:
    total_mass = 0.0
    boron_mass = 0.0
    for raw_name, raw_value in composition.items():
        name = str(raw_name)
        value = _num(raw_value)
        if value is None or value < 0:
            continue
        weight = _MASS.get(name)
        if weight is None:
            continue
        contribution = value * weight
        total_mass += contribution
        if name in {"B", "B10", "B11"}:
            boron_mass += contribution
    return boron_mass / total_mass if total_mass > 0 else 0.0


def _borated_water_atom_fractions(
    *,
    boron_mass_fraction: float,
    b10_atom_fraction: float,
    h_key: str,
    o_key: str,
) -> dict[str, float]:
    b11_atom_fraction = 1.0 - b10_atom_fraction
    boron_molar_mass = (
        b10_atom_fraction * _MASS["B10"]
        + b11_atom_fraction * _MASS["B11"]
    )
    boron_moles = boron_mass_fraction / boron_molar_mass
    water_moles = (1.0 - boron_mass_fraction) / 18.015
    amounts = {
        h_key: 2.0 * water_moles,
        o_key: water_moles,
        "B10": boron_moles * b10_atom_fraction,
        "B11": boron_moles * b11_atom_fraction,
    }
    total = sum(amounts.values())
    return {key: value / total for key, value in amounts.items()}


def _num(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value))
    except Exception:
        return None


__all__ = [
    "MaterialsPatchNormalizationResult",
    "SolubleBoronRequirement",
    "extract_soluble_boron_requirement",
    "normalize_materials_patch_content",
    "normalize_materials_patches_in_state",
]
