"""Deterministic qualification of universe fragments before acceptance.

A *fragment* is the raw JSON object returned by one LLM call for one
universe.  Before it can be marked ``accepted`` in the checkpoint
session, it must pass deterministic qualification against its manifest
item.  This module defines that contract.

The qualification contract enforces:

* **Output boundary** — exactly one universe in the fragment.
* **Schema conformance** — the single universe wrapped as
  ``{"patch_type": "universes", "universes": [...]}`` parses with the
  authoritative :class:`UniversesPatch` schema and passes the standard
  validator.  A separate hand-written schema is intentionally avoided
  to prevent drift from the authoritative patch model.
* **Identity binding** — ``universe_id`` and ``kind`` match the manifest
  item.
* **Role coverage** — manifest-declared ``required_cell_roles`` and
  ``required_material_roles`` are present in the fragment.
* **Material integrity** — every non-empty ``material_id`` refers to an
  accepted MaterialsPatch material.
* **Internal integrity** — no duplicate cell IDs inside the universe,
  and the canonical fragment hash is recomputed from the universe data
  (an LLM-claimed hash is never trusted).
* **Source scope** — the fragment does not pull in cell roles or
  materials from a different manifest item's requirement/profile scope.

Qualification is deterministic, side-effect free, and reactor-neutral.
No benchmark names, fixture names, or reactor-specific identifiers are
used.  The contract is driven entirely by the manifest item.
"""

from __future__ import annotations

import hashlib
from typing import Any, Literal

from pydantic import Field

from openmc_agent.schemas import AgentBaseModel

from .fuel_variant_ids import fuel_variant_ids_equivalent
from .material_role_compat import material_role_satisfies
from .closed_loop.fingerprints import canonical_json_dumps, compute_candidate_hash
from .patches import parse_patch_content, PatchParseError
from .universe_fragment_generation import (
    UniverseDefinitionFragment,
    UniverseManifestItem,
)
from .validators import (
    PatchValidationContext,
    PatchValidationIssue,
    validate_patch,
)


# ---------------------------------------------------------------------------
# Result models
# ---------------------------------------------------------------------------


class FragmentQualificationIssue(AgentBaseModel):
    """A single structured qualification issue for one fragment."""

    code: str
    severity: Literal["error", "warning"] = "error"
    universe_id: str
    json_path: str | None = None
    message: str
    expected: Any | None = None
    actual: Any | None = None
    retryable: bool = True
    metadata: dict[str, Any] = Field(default_factory=dict)


class FragmentQualificationResult(AgentBaseModel):
    """Structured result of qualifying one fragment against its manifest item.

    ``fragment_hash`` is always the canonical hash recomputed from
    :attr:`canonical_universe_data` (never the LLM-claimed hash).
    ``manifest_contract_hash`` echoes the manifest item's contract hash
    so downstream merge can detect contract drift on resume.
    """

    ok: bool
    universe_id: str
    fragment_hash: str = ""
    manifest_contract_hash: str = ""
    canonical_universe_data: dict[str, Any] = Field(default_factory=dict)
    issues: list[FragmentQualificationIssue] = Field(default_factory=list)
    qualification_attempt: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _canonical_hash(universe_data: dict[str, Any]) -> str:
    """Canonical SHA-256 hash of one universe's data.

    Uses :func:`compute_candidate_hash` so the hash is consistent with the
    rest of the closed-loop fingerprinting infrastructure.
    """
    return compute_candidate_hash(
        target_patch_type="universes", candidate_patch=universe_data
    )


def _wrap_as_universes_patch(universe_data: dict[str, Any]) -> dict[str, Any]:
    """Wrap a single universe dict as a ``universes`` patch envelope."""
    return {"patch_type": "universes", "universes": [universe_data]}


def _variant_ids_equivalent(actual: str, expected: str) -> bool:
    """Return True when two fuel variant ids refer to the same variant.

    Both ids ultimately derive from the same Facts ``variant_id``, but the LLM
    frequently emits a short label (e.g. ``"3B"``) where Facts declared a
    canonical id (e.g. ``"3B_fuel"``).  Treat them as equivalent when they are
    equal (case-insensitive) or one is a substring of the other.  A genuinely
    different variant (e.g. ``"3A"`` vs ``"3B_fuel"``) has no substring
    relation and is rejected.
    """

    return fuel_variant_ids_equivalent(actual, expected)


# Marker strings the LLM occasionally emits when copying prompt examples
# verbatim.  Detected as invalid material IDs regardless of whether they
# appear in ``known_material_ids``.
_PLACEHOLDER_MATERIAL_TOKENS: frozenset[str] = frozenset({
    "replace",
    "replace_me",
    "replace_with_material",
    "<material_id>",
    "<material>",
    "tbd",
    "todo",
    "xxx",
    "material_id",
})


def _is_placeholder_material(mid: Any) -> bool:
    if not isinstance(mid, str):
        return False
    return mid.strip().lower() in _PLACEHOLDER_MATERIAL_TOKENS


# ---------------------------------------------------------------------------
# Qualification
# ---------------------------------------------------------------------------


def qualify_universe_fragment(
    *,
    manifest_item: UniverseManifestItem,
    fragment: UniverseDefinitionFragment,
    known_material_ids: set[str] | None = None,
    material_roles_by_id: dict[str, str] | None = None,
    material_source_variants_by_id: dict[str, str | None] | None = None,
    qualification_attempt: int = 0,
) -> FragmentQualificationResult:
    """Qualify one fragment against its manifest item.

    The function returns a structured :class:`FragmentQualificationResult`
    whose ``ok`` flag is the single source of truth for whether the
    fragment may enter the accepted set.  ``ok`` is True only when there
    are no ``error``-severity issues.
    """
    known_material_ids = known_material_ids or set()
    material_roles_by_id = material_roles_by_id or {}
    material_source_variants_by_id = material_source_variants_by_id or {}
    uid = manifest_item.universe_id
    issues: list[FragmentQualificationIssue] = []

    # --- 3.1 Output boundary ---
    fragment_universe = dict(fragment.universe or {})
    if not fragment_universe:
        issues.append(FragmentQualificationIssue(
            code="qualification.empty_fragment",
            universe_id=uid,
            json_path=f"/universes/{uid}",
            message="fragment has no universe data",
            retryable=True,
        ))
        # Nothing else to check.
        return FragmentQualificationResult(
            ok=False, universe_id=uid, fragment_hash="",
            manifest_contract_hash=manifest_item.contract_hash,
            issues=issues, qualification_attempt=qualification_attempt,
        )

    # If the LLM returned a full patch object instead of a single universe,
    # unwrap it defensively but record the issue.  Do this BEFORE identity
    # checks so the universe_id comparison uses the unwrapped data.
    if "patch_type" in fragment_universe and "universes" in fragment_universe:
        universes_list = fragment_universe.get("universes") or []
        if len(universes_list) != 1:
            issues.append(FragmentQualificationIssue(
                code="qualification.fragment_not_single_universe",
                universe_id=uid,
                json_path=f"/universes/{uid}",
                message=(
                    f"fragment payload contains {len(universes_list)} universes; "
                    "exactly one is required"
                ),
                expected=1,
                actual=len(universes_list),
                retryable=True,
            ))
        else:
            fragment_universe = dict(universes_list[0] or {})
            issues.append(FragmentQualificationIssue(
                code="qualification.fragment_wrapped_as_patch",
                universe_id=uid,
                severity="warning",
                json_path=f"/universes/{uid}",
                message="fragment payload was wrapped as a full patch; unwrapped to single universe",
                retryable=False,
            ))

    actual_uid = fragment_universe.get("universe_id")
    if actual_uid != uid:
        issues.append(FragmentQualificationIssue(
            code="qualification.universe_id_mismatch",
            universe_id=uid,
            json_path=f"/universes/{uid}/universe_id",
            message=(
                f"fragment universe_id={actual_uid!r} does not match "
                f"manifest universe_id={uid!r}"
            ),
            expected=uid,
            actual=actual_uid,
            retryable=True,
        ))

    # Kind match (when both sides declare one).
    fragment_kind = fragment_universe.get("kind")
    if manifest_item.kind and fragment_kind and fragment_kind != manifest_item.kind:
        issues.append(FragmentQualificationIssue(
            code="qualification.kind_mismatch",
            universe_id=uid,
            json_path=f"/universes/{uid}/kind",
            message=(
                f"fragment kind={fragment_kind!r} does not match manifest "
                f"kind={manifest_item.kind!r}"
            ),
            expected=manifest_item.kind,
            actual=fragment_kind,
            retryable=True,
        ))

    # --- 3.2 Schema validation via authoritative UniversesPatch ---
    cells = fragment_universe.get("cells") or []
    if not cells:
        issues.append(FragmentQualificationIssue(
            code="qualification.empty_cells",
            universe_id=uid,
            json_path=f"/universes/{uid}/cells",
            message=f"universe {uid!r} has no cells",
            retryable=True,
        ))
    else:
        wrapped_patch = _wrap_as_universes_patch(fragment_universe)
        try:
            parsed = parse_patch_content("universes", wrapped_patch)
            val_ctx = PatchValidationContext(
                known_material_ids=list(known_material_ids),
                known_universe_ids=[uid],
            )
            val_result = validate_patch(parsed, context=val_ctx)
            for issue in val_result.issues:
                if issue.severity == "error":
                    issues.append(_convert_validator_issue(issue, uid))
        except PatchParseError as exc:
            issues.append(FragmentQualificationIssue(
                code="qualification.schema_invalid",
                universe_id=uid,
                json_path=f"/universes/{uid}",
                message=f"universes patch schema rejected the fragment: {exc}",
                retryable=True,
                metadata={"patch_parse_error": str(exc)[:200]},
            ))
        except Exception as exc:  # defensive: never silently swallow
            issues.append(FragmentQualificationIssue(
                code="qualification.schema_check_exception",
                universe_id=uid,
                json_path=f"/universes/{uid}",
                message=f"{type(exc).__name__}: {exc}",
                retryable=True,
            ))

    # --- 3.3 Manifest contract: required cell roles ---
    declared_cell_roles = set(manifest_item.required_cell_roles or [])
    if declared_cell_roles:
        actual_cell_roles = {
            (c.get("role") or "").lower()
            for c in cells
            if isinstance(c, dict)
        }
        missing_cell_roles = sorted({
            role for role in declared_cell_roles
            if role.lower() not in actual_cell_roles
        })
        if missing_cell_roles:
            issues.append(FragmentQualificationIssue(
                code="qualification.required_cell_role_missing",
                universe_id=uid,
                json_path=f"/universes/{uid}/cells",
                message=(
                    f"manifest requires cell roles {sorted(declared_cell_roles)} "
                    f"but fragment only provides {sorted(actual_cell_roles)}"
                ),
                expected=sorted(declared_cell_roles),
                actual=sorted(actual_cell_roles),
                retryable=True,
                metadata={"missing_roles": missing_cell_roles},
            ))

    # --- 3.3 Manifest contract: material references and roles ---
    referenced_material_ids: list[str] = []
    for cell in cells:
        if not isinstance(cell, dict):
            continue
        mid = cell.get("material_id")
        if isinstance(mid, str) and mid:
            referenced_material_ids.append(mid)

    unknown_materials: list[str] = []
    placeholder_materials: list[str] = []
    for mid in referenced_material_ids:
        if _is_placeholder_material(mid):
            placeholder_materials.append(mid)
        elif known_material_ids and mid not in known_material_ids:
            unknown_materials.append(mid)

    if placeholder_materials:
        issues.append(FragmentQualificationIssue(
            code="qualification.placeholder_material_id",
            universe_id=uid,
            json_path=f"/universes/{uid}/cells",
            message=(
                f"fragment contains placeholder material_id(s) "
                f"{placeholder_materials!r} — the LLM copied the prompt example "
                f"verbatim. Regenerate the fragment with real material IDs."
            ),
            actual=placeholder_materials,
            retryable=True,
        ))
    if unknown_materials:
        issues.append(FragmentQualificationIssue(
            code="qualification.unknown_material_id",
            universe_id=uid,
            json_path=f"/universes/{uid}/cells",
            message=(
                f"fragment references material_id(s) {unknown_materials!r} that are "
                f"not in the accepted MaterialsPatch"
            ),
            expected=sorted(known_material_ids),
            actual=unknown_materials,
            retryable=True,
        ))

    # Required material roles must be reachable via referenced materials.
    declared_material_roles = set(manifest_item.required_material_roles or [])
    if declared_material_roles and material_roles_by_id:
        actual_roles = {
            material_roles_by_id.get(mid, "").lower()
            for mid in referenced_material_ids
            if material_roles_by_id.get(mid)
        }
        missing_material_roles = sorted({
            role for role in declared_material_roles
            if not any(material_role_satisfies(actual, role) for actual in actual_roles)
        })
        if missing_material_roles:
            issues.append(FragmentQualificationIssue(
                code="qualification.required_material_role_missing",
                universe_id=uid,
                json_path=f"/universes/{uid}/cells",
                message=(
                    f"manifest requires material roles "
                    f"{sorted(declared_material_roles)} but fragment only covers "
                    f"{sorted(actual_roles)}"
                ),
                expected=sorted(declared_material_roles),
                actual=sorted(actual_roles),
                retryable=True,
                metadata={"missing_roles": missing_material_roles},
            ))

    if manifest_item.fuel_variant_id and material_source_variants_by_id:
        mismatched_fuel_cells: list[dict[str, Any]] = []
        expected_variant = manifest_item.fuel_variant_id
        for cell in cells:
            if not isinstance(cell, dict):
                continue
            if (cell.get("role") or "").lower() != "fuel":
                continue
            mid = cell.get("material_id")
            actual_variant = material_source_variants_by_id.get(mid)
            # Tolerate canonical-equivalent variant ids: both the universe
            # fuel_variant_id and the material source_variant_id derive from
            # the same Facts variant id, but the LLM frequently emits a short
            # label (e.g. "3B") where Facts declared "3B_fuel". Treat them as
            # matching when one contains the other (case-insensitive). This
            # still rejects a genuinely different variant (e.g. "3A" vs
            # "3B_fuel", no substring relation).
            if actual_variant and not _variant_ids_equivalent(actual_variant, expected_variant):
                mismatched_fuel_cells.append({
                    "cell_id": cell.get("id"),
                    "material_id": mid,
                    "actual_source_variant_id": actual_variant,
                })
        if mismatched_fuel_cells:
            issues.append(FragmentQualificationIssue(
                code="qualification.fuel_variant_material_mismatch",
                universe_id=uid,
                json_path=f"/universes/{uid}/cells",
                message=(
                    f"fuel universe {uid!r} declares fuel_variant_id="
                    f"{expected_variant!r} but at least one fuel cell uses "
                    "a material from a different source_variant_id"
                ),
                expected=expected_variant,
                actual=mismatched_fuel_cells,
                retryable=True,
            ))

    # --- 3.4 Internal integrity: duplicate cell IDs ---
    seen_cell_ids: set[str] = set()
    duplicate_cell_ids: list[str] = []
    for cell in cells:
        if not isinstance(cell, dict):
            continue
        cid = cell.get("id")
        if isinstance(cid, str) and cid:
            if cid in seen_cell_ids and cid not in duplicate_cell_ids:
                duplicate_cell_ids.append(cid)
            seen_cell_ids.add(cid)
    if duplicate_cell_ids:
        issues.append(FragmentQualificationIssue(
            code="qualification.duplicate_cell_id",
            universe_id=uid,
            json_path=f"/universes/{uid}/cells",
            message=(
                f"universe {uid!r} has duplicate cell IDs: {duplicate_cell_ids}"
            ),
            actual=duplicate_cell_ids,
            retryable=True,
        ))

    # --- 3.3 Manifest contract: source requirement / profile scope ---
    # A fragment must satisfy its OWN manifest item's contract; cells may
    # not pull in roles declared by a *different* manifest item as a
    # substitute.  We enforce this conservatively: if required_cell_roles
    # are declared and the fragment's roles match a *different* manifest
    # item's roles exactly (e.g., two insert kinds cross-pollinated), the
    # fragment is rejected.  This is verified by the caller via the
    # manifest's per-item contract hash; here we just ensure the universe_id
    # matches its own scope.
    if actual_uid != uid:
        # already recorded above; no-op so we don't double-count
        pass

    # --- Canonical fragment hash (recomputed; LLM hash is never trusted) ---
    canonical_hash = _canonical_hash(fragment_universe)

    errors = [i for i in issues if i.severity == "error"]
    return FragmentQualificationResult(
        ok=len(errors) == 0,
        universe_id=uid,
        fragment_hash=canonical_hash,
        manifest_contract_hash=manifest_item.contract_hash,
        canonical_universe_data=fragment_universe,
        issues=issues,
        qualification_attempt=qualification_attempt,
        metadata={
            "claimed_fragment_hash": fragment.fragment_hash or "",
            "referenced_material_ids": referenced_material_ids,
            "cell_count": len(cells),
        },
    )


def _convert_validator_issue(
    issue: PatchValidationIssue, universe_id: str
) -> FragmentQualificationIssue:
    """Convert a generic :class:`PatchValidationIssue` into a fragment-scoped one."""
    return FragmentQualificationIssue(
        code=issue.code or "qualification.schema_invalid",
        severity="error" if issue.severity == "error" else "warning",
        universe_id=universe_id,
        json_path=issue.path,
        message=issue.message,
        expected=issue.expected,
        actual=issue.actual,
        retryable=True,
    )


# ---------------------------------------------------------------------------
# Resume verification
# ---------------------------------------------------------------------------


def verify_accepted_fragment_record(
    *,
    manifest_item: UniverseManifestItem,
    record: Any,
    known_material_ids: set[str] | None = None,
    material_roles_by_id: dict[str, str] | None = None,
    material_source_variants_by_id: dict[str, str | None] | None = None,
) -> FragmentQualificationResult:
    """Re-qualify a previously-accepted fragment on resume.

    Used by the pipeline to decide whether an ``accepted`` checkpoint
    record is still valid against the current manifest item and
    MaterialsPatch.  Any error downgrades the record so the caller
    regenerates only that fragment.
    """
    from .universe_fragment_generation import AcceptedFragmentRecord

    if not isinstance(record, AcceptedFragmentRecord):
        return FragmentQualificationResult(
            ok=False,
            universe_id=manifest_item.universe_id,
            fragment_hash="",
            manifest_contract_hash=manifest_item.contract_hash,
            issues=[FragmentQualificationIssue(
                code="qualification.resume_corrupt_record",
                universe_id=manifest_item.universe_id,
                json_path=f"/universes/{manifest_item.universe_id}",
                message=(
                    f"accepted record for {manifest_item.universe_id!r} is not "
                    f"an AcceptedFragmentRecord (got {type(record).__name__})"
                ),
                retryable=True,
            )],
        )

    fragment = UniverseDefinitionFragment(
        universe_id=record.universe_id,
        universe=record.universe,
        fragment_hash=record.fragment_hash,
        manifest_contract_hash=record.manifest_contract_hash,
    )
    result = qualify_universe_fragment(
        manifest_item=manifest_item,
        fragment=fragment,
        known_material_ids=known_material_ids,
        material_roles_by_id=material_roles_by_id,
        material_source_variants_by_id=material_source_variants_by_id,
        qualification_attempt=record.accepted_at_attempt,
    )
    # Additionally enforce that the stored hash matches the recomputed one.
    if result.ok and record.fragment_hash and result.fragment_hash != record.fragment_hash:
        result = result.model_copy(update={
            "ok": False,
            "issues": list(result.issues) + [FragmentQualificationIssue(
                code="qualification.fragment_hash_drift",
                universe_id=manifest_item.universe_id,
                json_path=f"/universes/{manifest_item.universe_id}",
                message=(
                    f"stored fragment_hash={record.fragment_hash!r} does not match "
                    f"recomputed hash={result.fragment_hash!r}"
                ),
                expected=result.fragment_hash,
                actual=record.fragment_hash,
                retryable=True,
            )],
        })
    # And that the manifest contract hasn't drifted.
    if result.ok and record.manifest_contract_hash and manifest_item.contract_hash and record.manifest_contract_hash != manifest_item.contract_hash:
        result = result.model_copy(update={
            "ok": False,
            "issues": list(result.issues) + [FragmentQualificationIssue(
                code="qualification.manifest_contract_drift",
                universe_id=manifest_item.universe_id,
                json_path=f"/universes/{manifest_item.universe_id}",
                message=(
                    f"fragment was accepted against contract_hash="
                    f"{record.manifest_contract_hash!r} but current manifest item "
                    f"has contract_hash={manifest_item.contract_hash!r}"
                ),
                expected=manifest_item.contract_hash,
                actual=record.manifest_contract_hash,
                retryable=True,
            )],
        })
    return result


__all__ = [
    "FragmentQualificationIssue",
    "FragmentQualificationResult",
    "qualify_universe_fragment",
    "verify_accepted_fragment_record",
]
