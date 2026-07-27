"""Material-id alias resolution for patch validation and assembly."""

from __future__ import annotations

import re

from openmc_agent.schemas import AgentBaseModel


class MaterialResolutionResult(AgentBaseModel):
    ok: bool
    original_id: str
    resolved_id: str | None = None
    reason: str | None = None
    issue_code: str | None = None


_DEFAULT_ALIASES: dict[str, str] = {
    "grid_zircaloy4": "zircaloy4",
    "grid_zircaloy_4": "zircaloy4",
    "spacer_zircaloy4": "zircaloy4",
    "zircaloy-4": "zircaloy4",
    "inconel-718": "inconel718",
    "ss-304": "ss304",
    "stainless_steel_304": "ss304",
}


def _normalize_material_id(material_id: str) -> str:
    mid = material_id.strip().lower()
    mid = mid.replace("-", "_")
    mid = re.sub(r"[^a-z0-9_]+", "_", mid)
    mid = re.sub(r"_+", "_", mid).strip("_")
    return mid


def _compact_material_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.strip().lower())


def _material_alias_candidates(value: str) -> set[str]:
    """Return separator-insensitive lexical aliases for a material label.

    The candidates are intentionally lexical only: this function never maps one
    material family to another.  Ambiguous candidates are discarded by
    :func:`infer_material_aliases`.
    """
    tokens = re.findall(r"[a-z0-9]+", value.lower())
    candidates = {
        _normalize_material_id(value),
        _compact_material_key(value),
    }
    for start in range(len(tokens)):
        for end in range(start + 1, min(len(tokens), start + 3) + 1):
            window = tokens[start:end]
            joined = "".join(window)
            underscored = "_".join(window)
            if joined:
                candidates.add(joined)
            if underscored:
                candidates.add(underscored)
    return {c for c in candidates if c}


def _known_lookup(known_material_ids: set[str]) -> dict[str, str]:
    return {_normalize_material_id(mid): mid for mid in known_material_ids}


def infer_material_aliases(
    material_summaries: list[dict[str, object]],
    known_material_ids: set[str] | list[str],
) -> dict[str, str]:
    """Infer unambiguous aliases from accepted material IDs and names.

    This supports safe cases such as a generated ``zircaloy4`` reference when
    there is exactly one accepted material named ``Zircaloy-4``.  If two
    accepted materials could match the same alias, the alias is omitted so
    callers fail closed instead of guessing.
    """
    known = set(known_material_ids)
    buckets: dict[str, set[str]] = {}
    for summary in material_summaries:
        mid = summary.get("material_id")
        if not isinstance(mid, str) or mid not in known:
            continue
        labels: list[str] = [mid]
        for key in ("name", "display_name", "source_label"):
            value = summary.get(key)
            if isinstance(value, str) and value:
                labels.append(value)
        aliases = summary.get("aliases")
        if isinstance(aliases, list):
            labels.extend(value for value in aliases if isinstance(value, str))
        for label in labels:
            for candidate in _material_alias_candidates(label):
                buckets.setdefault(candidate, set()).add(mid)
                buckets.setdefault(_normalize_material_id(candidate), set()).add(mid)

    inferred: dict[str, str] = {}
    for alias, targets in buckets.items():
        if len(targets) == 1:
            target = next(iter(targets))
            inferred[alias] = target
            inferred[_normalize_material_id(alias)] = target
    return inferred


def resolve_material_id(
    material_id: str,
    known_material_ids: set[str],
    aliases: dict[str, str] | None = None,
) -> MaterialResolutionResult:
    """Resolve a material id against known ids and generic aliases."""
    if material_id in known_material_ids:
        return MaterialResolutionResult(
            ok=True,
            original_id=material_id,
            resolved_id=material_id,
            reason="material id already exists",
        )

    lookup = _known_lookup(known_material_ids)
    normalized = _normalize_material_id(material_id)
    if normalized in lookup:
        return MaterialResolutionResult(
            ok=True,
            original_id=material_id,
            resolved_id=lookup[normalized],
            reason="material id normalized to known id",
            issue_code="patch.axial_overlays.material_alias_resolved",
        )

    alias_map: dict[str, str] = {
        _normalize_material_id(k): v for k, v in _DEFAULT_ALIASES.items()
    }
    if aliases:
        alias_map.update({_normalize_material_id(k): v for k, v in aliases.items()})

    alias_target = alias_map.get(normalized)
    if alias_target:
        if alias_target in known_material_ids:
            resolved = alias_target
        else:
            resolved = lookup.get(_normalize_material_id(alias_target))
            if resolved is None:
                chained_target = alias_map.get(_normalize_material_id(alias_target))
                if chained_target in known_material_ids:
                    resolved = chained_target
                elif chained_target is not None:
                    resolved = lookup.get(_normalize_material_id(chained_target))
        if resolved is not None:
            return MaterialResolutionResult(
                ok=True,
                original_id=material_id,
                resolved_id=resolved,
                reason=f"material alias resolved to {resolved!r}",
                issue_code="patch.axial_overlays.material_alias_resolved",
            )

    variant_matches = sorted(
        known_id for known_id in known_material_ids
        if _normalize_material_id(known_id).startswith(f"{normalized}_")
        or normalized.startswith(f"{_normalize_material_id(known_id)}_")
    )
    if len(variant_matches) == 1:
        return MaterialResolutionResult(
            ok=True,
            original_id=material_id,
            resolved_id=variant_matches[0],
            reason=f"material id variant resolved to {variant_matches[0]!r}",
            issue_code="patch.axial_overlays.material_alias_resolved",
        )

    return MaterialResolutionResult(
        ok=False,
        original_id=material_id,
        resolved_id=None,
        reason=f"material id {material_id!r} is not defined",
        issue_code="patch.axial_overlays.material_missing",
    )


__all__ = [
    "MaterialResolutionResult",
    "infer_material_aliases",
    "resolve_material_id",
]
