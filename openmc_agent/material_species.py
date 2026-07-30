"""Deterministic, fail-closed material species resolution.

This module is intentionally independent of OpenMC.  It turns source-level
compound declarations into transport-ready element/nuclide entries and is the
single authority used by patch assembly and material emission.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
import math
import re
from typing import Any, Iterable, Literal
from xml.etree import ElementTree


# Standard symbols and conventional atomic weights (IUPAC interval midpoints
# where appropriate).  They are used only for deterministic stoichiometric
# mass splitting, never as source composition data.
_ATOMIC_MASS: dict[str, float] = {
    "H": 1.008, "He": 4.002602, "Li": 6.94, "Be": 9.0121831,
    "B": 10.81, "C": 12.011, "N": 14.007, "O": 15.999, "F": 18.998403163,
    "Ne": 20.1797, "Na": 22.98976928, "Mg": 24.305, "Al": 26.9815385,
    "Si": 28.085, "P": 30.973761998, "S": 32.06, "Cl": 35.45,
    "Ar": 39.948, "K": 39.0983, "Ca": 40.078, "Sc": 44.955908,
    "Ti": 47.867, "V": 50.9415, "Cr": 51.9961, "Mn": 54.938044,
    "Fe": 55.845, "Co": 58.933194, "Ni": 58.6934, "Cu": 63.546,
    "Zn": 65.38, "Ga": 69.723, "Ge": 72.630, "As": 74.921595,
    "Se": 78.971, "Br": 79.904, "Kr": 83.798, "Rb": 85.4678,
    "Sr": 87.62, "Y": 88.90584, "Zr": 91.224, "Nb": 92.90637,
    "Mo": 95.95, "Tc": 98.0, "Ru": 101.07, "Rh": 102.90550,
    "Pd": 106.42, "Ag": 107.8682, "Cd": 112.414, "In": 114.818,
    "Sn": 118.710, "Sb": 121.760, "Te": 127.60, "I": 126.90447,
    "Xe": 131.293, "Cs": 132.90545196, "Ba": 137.327, "La": 138.90547,
    "Ce": 140.116, "Pr": 140.90766, "Nd": 144.242, "Pm": 145.0,
    "Sm": 150.36, "Eu": 151.964, "Gd": 157.25, "Tb": 158.92535,
    "Dy": 162.500, "Ho": 164.93033, "Er": 167.259, "Tm": 168.93422,
    "Yb": 173.045, "Lu": 174.9668, "Hf": 178.49, "Ta": 180.94788,
    "W": 183.84, "Re": 186.207, "Os": 190.23, "Ir": 192.217,
    "Pt": 195.084, "Au": 196.966569, "Hg": 200.592, "Tl": 204.38,
    "Pb": 207.2, "Bi": 208.98040, "Po": 209.0, "At": 210.0,
    "Rn": 222.0, "Fr": 223.0, "Ra": 226.0, "Ac": 227.0,
    "Th": 232.0377, "Pa": 231.03588, "U": 238.02891, "Np": 237.0,
    "Pu": 244.0, "Am": 243.0, "Cm": 247.0, "Bk": 247.0, "Cf": 251.0,
}

ELEMENT_SYMBOLS = frozenset(_ATOMIC_MASS)
FISSILE_COMPOUND_ELEMENTS = frozenset({"U", "Pu", "Th", "Np", "Am"})
_ELEMENT_RE = re.compile(r"^[A-Z][a-z]?$")
_NUCLIDE_RE = re.compile(r"^([A-Z][a-z]?)-?(\d+)(?:_?m\d*)?$")
_FORMULA_TOKEN_RE = re.compile(r"([A-Z][a-z]?)(\d*)")


@dataclass(frozen=True)
class SpeciesEntry:
    name: str
    fraction: float
    kind: Literal["element", "nuclide"]
    fraction_basis: Literal["weight_frac", "atom_frac", "atom_density_barn_cm"]


@dataclass
class MaterialSpeciesResolution:
    material_id: str
    source_composition: dict[str, float] = field(default_factory=dict)
    source_compound_components: list[dict[str, Any]] = field(default_factory=list)
    parsed_formulas: list[dict[str, Any]] = field(default_factory=list)
    resolved_elements: dict[str, float] = field(default_factory=dict)
    resolved_nuclides: dict[str, float] = field(default_factory=dict)
    fraction_basis: str | None = None
    isotope_policies: list[str] = field(default_factory=list)
    normalization_events: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    mass_balance_before: float = 0.0
    mass_balance_after: float = 0.0

    @property
    def ok(self) -> bool:
        return not self.errors

    @property
    def species(self) -> list[SpeciesEntry]:
        basis = self.fraction_basis or "atom_frac"
        if basis not in {"weight_frac", "atom_frac", "atom_density_barn_cm"}:
            basis = "atom_frac"
        entries = [
            SpeciesEntry(name, fraction, "element", basis)
            for name, fraction in sorted(self.resolved_elements.items())
        ]
        entries += [
            SpeciesEntry(name, fraction, "nuclide", basis)
            for name, fraction in sorted(self.resolved_nuclides.items())
        ]
        return entries

    def report(self) -> dict[str, Any]:
        return {
            "material_id": self.material_id,
            "source_composition": self.source_composition,
            "source_compound_components": self.source_compound_components,
            "parsed_formulas": self.parsed_formulas,
            "resolved_element_fractions": self.resolved_elements,
            "resolved_nuclide_fractions": self.resolved_nuclides,
            "fraction_basis": self.fraction_basis,
            "isotope_policies": self.isotope_policies,
            "merged_species": [entry.__dict__ for entry in self.species],
            "mass_balance_before": self.mass_balance_before,
            "mass_balance_after": self.mass_balance_after,
            "compatibility_normalization": self.normalization_events,
            "warnings": self.warnings,
            "errors": self.errors,
        }


def canonical_nuclide_name(name: str) -> str:
    """Canonicalize accepted GND-style nuclide spelling without guessing."""
    name = name.strip()
    match = _NUCLIDE_RE.fullmatch(name)
    if match is None:
        return name
    suffix = name[match.end(2):].replace("_", "")
    return f"{match.group(1)}{match.group(2)}{suffix}"


def classify_species_name(name: str) -> Literal["element", "nuclide", "compound", "invalid"]:
    """Classify transport species without treating a nuclide as a formula."""
    name = name.strip()
    if name in ELEMENT_SYMBOLS:
        return "element"
    match = _NUCLIDE_RE.fullmatch(name)
    if match is not None and match.group(1) in ELEMENT_SYMBOLS:
        return "nuclide"
    try:
        parsed = parse_empirical_formula(name)
    except ValueError:
        return "invalid"
    return "compound" if len(parsed) >= 2 else "invalid"


def parse_empirical_formula(formula: str) -> list[tuple[str, int]]:
    """Parse only simple empirical formulas, rejecting all richer notation."""
    formula = formula.strip()
    if not formula or any(c in formula for c in "().·+- "):
        raise ValueError("materials.unsupported_compound_formula")
    tokens = _FORMULA_TOKEN_RE.findall(formula)
    if not tokens:
        raise ValueError("materials.unsupported_compound_formula")
    pos = 0
    parsed: list[tuple[str, int]] = []
    for symbol, count_text in tokens:
        if symbol not in ELEMENT_SYMBOLS:
            raise ValueError("materials.unsupported_compound_formula")
        pos += len(symbol) + len(count_text)
        count = int(count_text) if count_text else 1
        if count <= 0:
            raise ValueError("materials.unsupported_compound_formula")
        parsed.append((symbol, count))
    if pos != len(formula) or len({s for s, _ in parsed}) < 2:
        raise ValueError("materials.unsupported_compound_formula")
    return parsed


def _valid_fraction(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value)) and float(value) > 0.0


def _basis_for_component(value: Any) -> str | None:
    if isinstance(value, dict):
        return value.get("fraction_basis")
    return getattr(value, "fraction_basis", None)


def _component_value(value: Any, key: str, default: Any = None) -> Any:
    return value.get(key, default) if isinstance(value, dict) else getattr(value, key, default)


def _add_species(
    target: dict[str, float], name: str, fraction: float,
) -> None:
    target[name] = target.get(name, 0.0) + fraction


def merge_duplicate_species(entries: Iterable[SpeciesEntry]) -> list[SpeciesEntry]:
    """Merge identically typed species deterministically."""
    merged: dict[tuple[str, str, str], float] = defaultdict(float)
    for entry in entries:
        merged[(entry.name, entry.kind, entry.fraction_basis)] += entry.fraction
    return [SpeciesEntry(name, frac, kind, basis) for (name, kind, basis), frac in sorted(merged.items())]


def validate_species_mass_balance(resolution: MaterialSpeciesResolution, *, tolerance: float = 1e-9) -> None:
    if not resolution.ok:
        return
    if not math.isfinite(resolution.mass_balance_after):
        resolution.errors.append("material_resolution.invalid_fraction")
        return
    if abs(resolution.mass_balance_before - resolution.mass_balance_after) > tolerance * max(1.0, abs(resolution.mass_balance_before)):
        resolution.errors.append("material_resolution.mass_not_conserved")
    keys = [entry.name for entry in resolution.species]
    if len(keys) != len(set(keys)):
        resolution.errors.append("material_resolution.duplicate_species_unmerged")


def resolve_compound_component(
    component: Any,
    *,
    material_id: str,
    role: str,
) -> MaterialSpeciesResolution:
    """Resolve one typed source compound into canonical species."""
    formula = _component_value(component, "formula", "")
    fraction = _component_value(component, "fraction")
    basis = _basis_for_component(component)
    policy = _component_value(component, "isotope_policy")
    overrides = _component_value(component, "isotope_overrides", {}) or {}
    result = MaterialSpeciesResolution(material_id=material_id)
    result.source_compound_components = [{
        "formula": formula, "fraction": fraction, "fraction_basis": basis,
        "isotope_policy": policy, "isotope_overrides": overrides,
        "source_note": _component_value(component, "source_note"),
        "assumptions": _component_value(component, "assumptions", []) or [],
    }]
    result.fraction_basis = basis
    if basis not in {"weight_frac", "atom_frac"}:
        result.errors.append("materials.compound_fraction_basis_missing")
        return result
    if policy not in {"natural_elements", "explicit_isotopes", "requires_confirmation"}:
        result.errors.append("materials.compound_isotope_policy_missing")
        return result
    if not _valid_fraction(fraction):
        result.errors.append("material_resolution.invalid_fraction")
        return result
    try:
        parsed = parse_empirical_formula(str(formula))
    except ValueError:
        result.errors.append("materials.unsupported_compound_formula")
        return result
    molecular_weight = sum(_ATOMIC_MASS[symbol] * count for symbol, count in parsed)
    result.parsed_formulas.append({"formula": formula, "stoichiometry": dict(parsed), "molecular_weight": molecular_weight})
    result.isotope_policies.append(policy)
    fissile = {symbol for symbol, _ in parsed} & FISSILE_COMPOUND_ELEMENTS
    if fissile and policy != "explicit_isotopes":
        result.errors.append("materials.fissile_compound_isotope_policy_missing")
        if role == "fuel":
            result.errors.append("materials.fissile_compound_would_erase_enrichment")
        return result
    result.mass_balance_before = float(fraction)
    atom_total = sum(count for _, count in parsed)
    for symbol, count in parsed:
        share = (count * _ATOMIC_MASS[symbol] / molecular_weight) if basis == "weight_frac" else (count / atom_total)
        amount = float(fraction) * share
        isotope_vector = overrides.get(symbol)
        if policy == "explicit_isotopes" and isotope_vector:
            if not isinstance(isotope_vector, dict) or not isotope_vector:
                result.errors.append("materials.compound_conflicts_with_explicit_isotopes")
                continue
            total = sum(float(v) for v in isotope_vector.values() if _valid_fraction(v))
            if total <= 0 or abs(total - 1.0) > 1e-6 and abs(total - 100.0) > 1e-6:
                result.errors.append("materials.compound_conflicts_with_explicit_isotopes")
                continue
            scale = 100.0 if total > 1.0 else 1.0
            for isotope, isotope_fraction in isotope_vector.items():
                if classify_species_name(isotope) != "nuclide" or not isotope.startswith(symbol):
                    result.errors.append("materials.compound_conflicts_with_explicit_isotopes")
                    continue
                _add_species(result.resolved_nuclides, canonical_nuclide_name(isotope), amount * float(isotope_fraction) / scale)
        else:
            _add_species(result.resolved_elements, symbol, amount)
            if policy == "natural_elements":
                result.warnings.append(f"{formula}: {symbol} uses natural element isotopes")
    result.mass_balance_after = sum(result.resolved_elements.values()) + sum(result.resolved_nuclides.values())
    validate_species_mass_balance(result)
    return result


def resolve_material_species(
    *, material_id: str, role: str, composition: dict[str, float] | None,
    composition_basis: str | None, compound_components: Iterable[Any] | None = None,
    strict_composition: bool = False, legacy_compatibility: bool = True,
) -> MaterialSpeciesResolution:
    """Resolve a patch/source material into canonical, renderer-safe species."""
    result = MaterialSpeciesResolution(material_id=material_id, source_composition=dict(composition or {}))
    components = list(compound_components or [])
    bases = {str(_basis_for_component(c)) for c in components}
    if composition:
        bases.add(str(composition_basis))
    bases.discard("None")
    if len(bases) > 1:
        from collections import Counter

        basis_counts = Counter(str(_basis_for_component(c)) for c in components)
        if composition:
            basis_counts.update([str(composition_basis)])
        basis_counts.pop("None", None)
        result.fraction_basis = (
            basis_counts.most_common(1)[0][0] if basis_counts else "atom_frac"
        )
        result.warnings.append("material_resolution.mixed_fraction_basis_fallback")
    else:
        result.fraction_basis = next(iter(bases), composition_basis)
    if components and result.fraction_basis not in {"weight_frac", "atom_frac"}:
        result.fraction_basis = "atom_frac"
        result.warnings.append("material_resolution.mixed_fraction_basis_fallback")

    direct_total = 0.0
    for raw_name, value in (composition or {}).items():
        if not _valid_fraction(value):
            result.errors.append("material_resolution.invalid_fraction")
            continue
        kind = classify_species_name(raw_name)
        if kind == "compound":
            if strict_composition or not legacy_compatibility:
                result.errors.append("materials.compound_in_transport_composition")
                continue
            compat = {
                "formula": raw_name, "fraction": value,
                "fraction_basis": composition_basis,
                "isotope_policy": "natural_elements",
            }
            result.normalization_events.append(f"legacy composition formula {raw_name} moved to compound_components")
            components.append(compat)
            continue
        if kind == "invalid":
            result.errors.append("materials.unresolved_species")
            continue
        direct_total += float(value)
        if kind == "element":
            _add_species(result.resolved_elements, raw_name, float(value))
        else:
            _add_species(result.resolved_nuclides, canonical_nuclide_name(raw_name), float(value))

    # Do not allow a heavy-element compound to silently add a second source of
    # enrichment beside a supplied explicit fuel vector.
    direct_heavy = {name.rstrip("0123456789m") for name in result.resolved_nuclides}
    for component in components:
        one = resolve_compound_component(component, material_id=material_id, role=role)
        result.source_compound_components.extend(one.source_compound_components)
        result.parsed_formulas.extend(one.parsed_formulas)
        result.isotope_policies.extend(one.isotope_policies)
        result.warnings.extend(one.warnings)
        result.errors.extend(one.errors)
        if one.parsed_formulas:
            compound_elements = set(one.parsed_formulas[0]["stoichiometry"])
            if direct_heavy & compound_elements & FISSILE_COMPOUND_ELEMENTS:
                result.errors.append("materials.compound_conflicts_with_explicit_isotopes")
        for name, value in one.resolved_elements.items():
            _add_species(result.resolved_elements, name, value)
        for name, value in one.resolved_nuclides.items():
            _add_species(result.resolved_nuclides, name, value)
    result.mass_balance_before = direct_total + sum(float(_component_value(c, "fraction", 0) or 0) for c in components)
    result.mass_balance_after = sum(result.resolved_elements.values()) + sum(result.resolved_nuclides.values())
    validate_species_mass_balance(result)
    return result


def build_material_species_report(resolutions: Iterable[MaterialSpeciesResolution]) -> dict[str, Any]:
    return {"materials": [resolution.report() for resolution in resolutions]}


def _available_nuclides(cross_sections_path: str) -> set[str]:
    root = ElementTree.parse(cross_sections_path).getroot()
    available: set[str] = set()
    for library in root.findall(".//library"):
        available.update((library.get("materials") or "").split())
    return {canonical_nuclide_name(name) for name in available}


def preflight_cross_sections(
    resolutions: Iterable[MaterialSpeciesResolution], cross_sections_path: str,
) -> list[dict[str, Any]]:
    """Check emitted species before OpenMC/MPI is started."""
    available = _available_nuclides(cross_sections_path)
    errors: list[dict[str, Any]] = []
    for resolution in resolutions:
        for entry in resolution.species:
            if entry.kind == "nuclide" and entry.name not in available:
                errors.append({"code": "runtime.nuclide_not_in_cross_sections", "material_id": resolution.material_id, "species_name": entry.name, "species_kind": entry.kind, "cross_sections_path": cross_sections_path, "suggested_patch_type": "materials"})
            elif entry.kind == "element" and not any(re.match(rf"^{re.escape(entry.name)}\d+", name) for name in available):
                errors.append({"code": "runtime.element_not_supported_by_cross_sections", "material_id": resolution.material_id, "species_name": entry.name, "species_kind": entry.kind, "cross_sections_path": cross_sections_path, "suggested_patch_type": "materials"})
        for code in resolution.errors:
            errors.append({"code": "runtime.material_species_unresolved", "material_id": resolution.material_id, "species_name": None, "species_kind": None, "cross_sections_path": cross_sections_path, "suggested_patch_type": "materials", "resolution_error": code})
    return errors


def preflight_plan_material_species(plan: Any, cross_sections_path: str) -> list[dict[str, Any]]:
    """Resolve the active plan's material entries and preflight them for runtime."""
    model = getattr(plan, "complex_model", None)
    materials = getattr(model, "materials", []) if model is not None else []
    resolutions: list[MaterialSpeciesResolution] = []
    for material in materials:
        composition: dict[str, float] = {}
        for component in getattr(material, "composition", []) or []:
            composition[component.name] = composition.get(component.name, 0.0) + component.percent
        basis_value = getattr(material, "composition_basis", "atom_frac")
        basis = getattr(basis_value, "value", basis_value)
        basis_map = {
            "weight_fraction": "weight_frac", "atom_fraction": "atom_frac",
            "atom_density_barn_cm": "atom_density_barn_cm",
        }
        resolutions.append(resolve_material_species(
            material_id=getattr(material, "id", getattr(material, "name", "unknown")),
            role="fuel" if "fuel" in str(getattr(material, "id", "")).lower() else "unknown",
            composition=composition, composition_basis=basis_map.get(str(basis), str(basis)),
            strict_composition=True, legacy_compatibility=False,
        ))
    return preflight_cross_sections(resolutions, cross_sections_path)


__all__ = [
    "ELEMENT_SYMBOLS", "FISSILE_COMPOUND_ELEMENTS", "SpeciesEntry", "MaterialSpeciesResolution",
    "canonical_nuclide_name", "classify_species_name", "parse_empirical_formula",
    "resolve_compound_component", "resolve_material_species", "merge_duplicate_species",
    "validate_species_mass_balance", "build_material_species_report", "preflight_cross_sections",
    "preflight_plan_material_species",
]
