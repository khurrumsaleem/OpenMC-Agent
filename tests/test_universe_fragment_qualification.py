"""Tests for deterministic fragment qualification (Part 3 of Step 4B-1).

The qualification contract is the single source of truth for whether a
fragment may enter the accepted set.  These tests cover:

- normal fragment accepted
- empty universes
- two universes in one fragment
- wrong universe ID
- wrong kind
- schema invalid
- duplicate cell IDs
- missing required cell role
- unknown material ID
- missing required material role
- placeholder material (REPLACE) — the run_004 root cause
- canonical fragment hash stability
- source requirement / profile scope mismatch
- cross-requirement role splicing must not satisfy another requirement
"""

from __future__ import annotations

from openmc_agent.plan_builder.universe_fragment_generation import (
    UniverseDefinitionFragment,
    UniverseManifestItem,
)
from openmc_agent.plan_builder.universe_fragment_qualification import (
    FragmentQualificationResult,
    qualify_universe_fragment,
    verify_accepted_fragment_record,
)
from openmc_agent.plan_builder.universe_fragment_generation import (
    AcceptedFragmentRecord,
)


def _manifest_item(
    *,
    universe_id: str = "u_fuel",
    kind: str = "fuel_pin",
    required_cell_roles: list[str] | None = None,
    required_material_roles: list[str] | None = None,
    required_material_ids: list[str] | None = None,
    protected_through_path_roles: list[str] | None = None,
    source_requirement_ids: list[str] | None = None,
    fuel_variant_id: str | None = None,
) -> UniverseManifestItem:
    item = UniverseManifestItem(
        universe_id=universe_id,
        kind=kind,
        required_cell_roles=required_cell_roles or [],
        required_material_roles=required_material_roles or [],
        required_material_ids=required_material_ids or [],
        protected_through_path_roles=protected_through_path_roles or [],
        source_requirement_ids=source_requirement_ids or [f"req:{universe_id}"],
        fuel_variant_id=fuel_variant_id,
    )
    item.recompute_contract_hash()
    return item


def _fragment(universe_id: str, universe: dict) -> UniverseDefinitionFragment:
    return UniverseDefinitionFragment(universe_id=universe_id, universe=universe)


def _fuel_universe(uid: str = "u_fuel", material_id: str = "m_fuel") -> dict:
    return {
        "universe_id": uid,
        "kind": "fuel_pin",
        "cells": [
            {
                "id": "c1", "role": "fuel", "material_id": material_id,
                "region_kind": "cylinder", "r_min_cm": 0.0, "r_max_cm": 0.4,
            },
        ],
    }


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_qualify_accepts_normal_fragment():
    item = _manifest_item(required_cell_roles=["fuel"], required_material_roles=["fuel"])
    frag = _fragment("u_fuel", _fuel_universe())
    result = qualify_universe_fragment(
        manifest_item=item, fragment=frag,
        known_material_ids={"m_fuel"}, material_roles_by_id={"m_fuel": "fuel"},
    )
    assert isinstance(result, FragmentQualificationResult)
    assert result.ok is True
    assert result.fragment_hash
    assert result.manifest_contract_hash == item.contract_hash
    assert result.canonical_universe_data["universe_id"] == "u_fuel"


def test_qualify_rejects_fuel_material_from_wrong_variant():
    item = _manifest_item(
        universe_id="u_fuel_v2",
        required_cell_roles=["fuel"],
        required_material_roles=["fuel"],
        fuel_variant_id="v2",
    )
    frag = _fragment(
        "u_fuel_v2",
        _fuel_universe(uid="u_fuel_v2", material_id="m_fuel_v1"),
    )
    result = qualify_universe_fragment(
        manifest_item=item,
        fragment=frag,
        known_material_ids={"m_fuel_v1", "m_fuel_v2"},
        material_roles_by_id={"m_fuel_v1": "fuel", "m_fuel_v2": "fuel"},
        material_source_variants_by_id={"m_fuel_v1": "v1", "m_fuel_v2": "v2"},
    )
    assert result.ok is False
    assert [issue.code for issue in result.issues] == [
        "qualification.fuel_variant_material_mismatch"
    ]
    assert result.issues[0].expected == "v2"


def test_qualify_accepts_canonical_equivalent_fuel_variant_label():
    """The LLM often emits a short fuel variant label (e.g. '3B') where Facts
    declared the canonical id (e.g. '3B_fuel'); both derive from the same
    Facts variant, so the qualification must treat them as matching when one
    contains the other. A genuinely different variant is still rejected."""
    item = _manifest_item(
        universe_id="u_fuel_3b",
        required_cell_roles=["fuel"],
        required_material_roles=["fuel"],
        fuel_variant_id="3B_fuel",
    )
    frag = _fragment(
        "u_fuel_3b",
        _fuel_universe(uid="u_fuel_3b", material_id="m_fuel_3b"),
    )
    result = qualify_universe_fragment(
        manifest_item=item,
        fragment=frag,
        known_material_ids={"m_fuel_3b"},
        material_roles_by_id={"m_fuel_3b": "fuel"},
        # Short label '3B' vs canonical '3B_fuel' -> equivalent.
        material_source_variants_by_id={"m_fuel_3b": "3B"},
    )
    assert result.ok is True
    assert not any(
        issue.code == "qualification.fuel_variant_material_mismatch"
        for issue in result.issues
    )


def test_qualify_accepts_state_prefixed_fuel_variant_label():
    item = _manifest_item(
        universe_id="u_fuel_3b",
        required_cell_roles=["fuel"],
        required_material_roles=["fuel"],
        fuel_variant_id="3B_fuel",
    )
    frag = _fragment(
        "u_fuel_3b",
        _fuel_universe(uid="u_fuel_3b", material_id="m_fuel_3b"),
    )

    result = qualify_universe_fragment(
        manifest_item=item,
        fragment=frag,
        known_material_ids={"m_fuel_3b"},
        material_roles_by_id={"m_fuel_3b": "fuel"},
        material_source_variants_by_id={"m_fuel_3b": "state_3b"},
    )

    assert result.ok is True
    assert not any(
        issue.code == "qualification.fuel_variant_material_mismatch"
        for issue in result.issues
    )


def test_qualify_rejects_different_state_prefixed_fuel_variant_label():
    item = _manifest_item(
        universe_id="u_fuel_3b",
        required_cell_roles=["fuel"],
        required_material_roles=["fuel"],
        fuel_variant_id="3B_fuel",
    )
    frag = _fragment(
        "u_fuel_3b",
        _fuel_universe(uid="u_fuel_3b", material_id="m_fuel_3b"),
    )

    result = qualify_universe_fragment(
        manifest_item=item,
        fragment=frag,
        known_material_ids={"m_fuel_3b"},
        material_roles_by_id={"m_fuel_3b": "fuel"},
        material_source_variants_by_id={"m_fuel_3b": "state_3a"},
    )

    assert result.ok is False
    assert any(
        issue.code == "qualification.fuel_variant_material_mismatch"
        for issue in result.issues
    )


def test_qualify_accepts_absorber_material_for_poison_requirement():
    item = _manifest_item(
        universe_id="u_pyrex",
        kind="custom",
        required_cell_roles=["poison"],
        required_material_roles=["poison"],
    )
    frag = _fragment(
        "u_pyrex",
        {
            "universe_id": "u_pyrex",
            "kind": "custom",
            "cells": [
                {
                    "id": "poison",
                    "role": "poison",
                    "material_id": "pyrex",
                    "region_kind": "cylinder",
                },
            ],
        },
    )

    result = qualify_universe_fragment(
        manifest_item=item,
        fragment=frag,
        known_material_ids={"pyrex"},
        material_roles_by_id={"pyrex": "absorber"},
    )

    assert result.ok is True
    assert not any(
        issue.code == "qualification.required_material_role_missing"
        for issue in result.issues
    )


def test_qualify_fragment_hash_is_canonical_and_stable():
    item = _manifest_item()
    frag = _fragment("u_fuel", _fuel_universe())
    r1 = qualify_universe_fragment(manifest_item=item, fragment=frag, known_material_ids={"m_fuel"})
    r2 = qualify_universe_fragment(manifest_item=item, fragment=frag, known_material_ids={"m_fuel"})
    assert r1.fragment_hash == r2.fragment_hash
    # Hash is recomputed; an LLM-claimed hash is ignored.
    frag_with_claimed_hash = UniverseDefinitionFragment(
        universe_id="u_fuel", universe=_fuel_universe(),
        fragment_hash="llm_claimed_bogus_hash",
    )
    r3 = qualify_universe_fragment(manifest_item=item, fragment=frag_with_claimed_hash, known_material_ids={"m_fuel"})
    assert r3.fragment_hash == r1.fragment_hash
    assert r3.metadata["claimed_fragment_hash"] == "llm_claimed_bogus_hash"


# ---------------------------------------------------------------------------
# Output boundary
# ---------------------------------------------------------------------------


def test_qualify_rejects_empty_universes():
    item = _manifest_item()
    frag = _fragment("u_fuel", {})
    result = qualify_universe_fragment(manifest_item=item, fragment=frag)
    assert result.ok is False
    assert any(i.code == "qualification.empty_fragment" for i in result.issues)


def test_qualify_rejects_two_universes_in_one_fragment():
    item = _manifest_item()
    universe_payload = {
        "patch_type": "universes",
        "universes": [_fuel_universe(), _fuel_universe("u_other")],
    }
    frag = _fragment("u_fuel", universe_payload)
    result = qualify_universe_fragment(manifest_item=item, fragment=frag)
    assert result.ok is False
    assert any(i.code == "qualification.fragment_not_single_universe" for i in result.issues)


def test_qualify_unwraps_full_patch_with_single_universe():
    """Some LLMs wrap the single universe as a full patch; we unwrap with a warning."""
    item = _manifest_item()
    universe_payload = {
        "patch_type": "universes",
        "universes": [_fuel_universe()],
    }
    frag = _fragment("u_fuel", universe_payload)
    result = qualify_universe_fragment(
        manifest_item=item, fragment=frag, known_material_ids={"m_fuel"},
    )
    assert result.ok is True
    # A warning should be present recording the unwrap.
    assert any(i.code == "qualification.fragment_wrapped_as_patch" for i in result.issues)


def test_qualify_rejects_wrong_universe_id():
    item = _manifest_item(universe_id="u_fuel")
    frag = _fragment("u_fuel", _fuel_universe(uid="u_wrong"))
    result = qualify_universe_fragment(manifest_item=item, fragment=frag)
    assert result.ok is False
    assert any(i.code == "qualification.universe_id_mismatch" for i in result.issues)


def test_qualify_rejects_wrong_kind():
    item = _manifest_item(universe_id="u_fuel", kind="fuel_pin")
    universe = _fuel_universe()
    universe["kind"] = "guide_tube"
    frag = _fragment("u_fuel", universe)
    result = qualify_universe_fragment(manifest_item=item, fragment=frag, known_material_ids={"m_fuel"})
    assert result.ok is False
    assert any(i.code == "qualification.kind_mismatch" for i in result.issues)


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


def test_qualify_rejects_schema_invalid():
    item = _manifest_item()
    # Cell missing a required field (id) → schema error.
    universe = {
        "universe_id": "u_fuel", "kind": "fuel_pin",
        "cells": [{"role": "fuel", "material_id": "m_fuel", "region_kind": "cylinder"}],
    }
    frag = _fragment("u_fuel", universe)
    result = qualify_universe_fragment(
        manifest_item=item, fragment=frag, known_material_ids={"m_fuel"},
    )
    assert result.ok is False
    assert any(i.code.startswith("qualification.schema") for i in result.issues)


def test_qualify_rejects_empty_cells():
    item = _manifest_item()
    universe = {"universe_id": "u_fuel", "kind": "fuel_pin", "cells": []}
    frag = _fragment("u_fuel", universe)
    result = qualify_universe_fragment(
        manifest_item=item, fragment=frag, known_material_ids={"m_fuel"},
    )
    assert result.ok is False
    assert any(i.code == "qualification.empty_cells" for i in result.issues)


def test_qualify_rejects_duplicate_cell_ids():
    item = _manifest_item()
    universe = {
        "universe_id": "u_fuel", "kind": "fuel_pin",
        "cells": [
            {"id": "c1", "role": "fuel", "material_id": "m_fuel", "region_kind": "cylinder", "r_min_cm": 0.0, "r_max_cm": 0.4},
            {"id": "c1", "role": "gap", "material_id": "m_fuel", "region_kind": "annulus", "r_min_cm": 0.4, "r_max_cm": 0.5},
        ],
    }
    frag = _fragment("u_fuel", universe)
    result = qualify_universe_fragment(
        manifest_item=item, fragment=frag, known_material_ids={"m_fuel"},
    )
    assert result.ok is False
    assert any(i.code == "qualification.duplicate_cell_id" for i in result.issues)


# ---------------------------------------------------------------------------
# Manifest contract
# ---------------------------------------------------------------------------


def test_qualify_rejects_missing_required_cell_role():
    item = _manifest_item(required_cell_roles=["fuel", "gap"])
    frag = _fragment("u_fuel", _fuel_universe())  # only fuel
    result = qualify_universe_fragment(
        manifest_item=item, fragment=frag, known_material_ids={"m_fuel"},
    )
    assert result.ok is False
    assert any(i.code == "qualification.required_cell_role_missing" for i in result.issues)


def test_qualify_rejects_unknown_material_id():
    item = _manifest_item()
    universe = _fuel_universe(material_id="m_nonexistent")
    frag = _fragment("u_fuel", universe)
    result = qualify_universe_fragment(
        manifest_item=item, fragment=frag, known_material_ids={"m_fuel"},
    )
    assert result.ok is False
    assert any(i.code == "qualification.unknown_material_id" for i in result.issues)


def test_qualify_rejects_placeholder_material_id():
    """The run_004 root cause: LLM copies 'REPLACE' verbatim from the prompt template."""
    item = _manifest_item()
    universe = _fuel_universe(material_id="REPLACE")
    frag = _fragment("u_fuel", universe)
    result = qualify_universe_fragment(
        manifest_item=item, fragment=frag, known_material_ids={"m_fuel", "REPLACE"},
    )
    assert result.ok is False
    assert any(i.code == "qualification.placeholder_material_id" for i in result.issues)
    # Even if "REPLACE" were (impossibly) in known_material_ids, the placeholder check still fires.


def test_qualify_rejects_missing_required_material_role():
    item = _manifest_item(required_material_roles=["poison"])
    frag = _fragment("u_fuel", _fuel_universe())  # m_fuel has role 'fuel', not 'poison'
    result = qualify_universe_fragment(
        manifest_item=item, fragment=frag,
        known_material_ids={"m_fuel"}, material_roles_by_id={"m_fuel": "fuel"},
    )
    assert result.ok is False
    assert any(i.code == "qualification.required_material_role_missing" for i in result.issues)


def test_qualify_protected_through_path_roles_do_not_inject_material_ids():
    """Protected-through-path roles are a manifest declaration; they do not
    silently inject unknown material IDs into the fragment."""
    item = _manifest_item(
        protected_through_path_roles=["fuel", "cladding"],
        required_cell_roles=["fuel", "cladding"],
    )
    universe = {
        "universe_id": "u_fuel", "kind": "fuel_pin",
        "cells": [
            {"id": "c1", "role": "fuel", "material_id": "m_fuel", "region_kind": "cylinder", "r_min_cm": 0.0, "r_max_cm": 0.4},
            {"id": "c2", "role": "cladding", "material_id": "m_clad", "region_kind": "annulus", "r_min_cm": 0.4, "r_max_cm": 0.5},
        ],
    }
    frag = _fragment("u_fuel", universe)
    # m_clad is unknown → must be rejected.
    result = qualify_universe_fragment(
        manifest_item=item, fragment=frag, known_material_ids={"m_fuel"},
    )
    assert result.ok is False
    assert any(i.code == "qualification.unknown_material_id" for i in result.issues)


# ---------------------------------------------------------------------------
# Source scope
# ---------------------------------------------------------------------------


def test_qualify_does_not_cross_splice_requirements():
    """A fragment must satisfy its OWN manifest item; roles from a sibling
    manifest item cannot be aggregated to satisfy this one."""
    item_a = _manifest_item(
        universe_id="u_a", kind="fuel_pin",
        required_cell_roles=["fuel"],
        source_requirement_ids=["req:a"],
    )
    item_b = _manifest_item(
        universe_id="u_b", kind="control_rod",
        required_cell_roles=["absorber"],
        source_requirement_ids=["req:b"],
    )
    # A fuel-role fragment cannot satisfy the control_rod requirement.
    fuel_fragment = _fragment("u_b", _fuel_universe(uid="u_b"))
    result = qualify_universe_fragment(
        manifest_item=item_b, fragment=fuel_fragment, known_material_ids={"m_fuel"},
    )
    assert result.ok is False
    assert any(i.code == "qualification.required_cell_role_missing" for i in result.issues)

    # And vice-versa: a control_rod fragment cannot satisfy the fuel_pin requirement.
    absorber_fragment = _fragment("u_a", {
        "universe_id": "u_a", "kind": "control_rod",
        "cells": [
            {"id": "c1", "role": "absorber", "material_id": "m_abs", "region_kind": "cylinder", "r_min_cm": 0.0, "r_max_cm": 0.4},
        ],
    })
    result2 = qualify_universe_fragment(
        manifest_item=item_a, fragment=absorber_fragment, known_material_ids={"m_abs"},
    )
    assert result2.ok is False


def test_qualify_contract_hash_independent_of_order():
    """The contract hash is computed per-item; manifest order changes do not
    affect an item's hash."""
    item = _manifest_item(universe_id="u_x", required_cell_roles=["fuel"])
    h1 = item.contract_hash
    # Same contract, different sequence:
    item2 = UniverseManifestItem(
        universe_id="u_x", kind="fuel_pin",
        required_cell_roles=["fuel"],
        source_requirement_ids=["req:u_x"],
    )
    item2.recompute_contract_hash()
    assert item2.contract_hash == h1


def test_qualify_contract_hash_changes_when_contract_changes():
    item_a = _manifest_item(universe_id="u_x", required_cell_roles=["fuel"])
    item_b = _manifest_item(universe_id="u_x", required_cell_roles=["fuel", "cladding"])
    assert item_a.contract_hash != item_b.contract_hash


# ---------------------------------------------------------------------------
# Resume verification
# ---------------------------------------------------------------------------


def test_resume_record_ok_when_consistent():
    item = _manifest_item(required_cell_roles=["fuel"])
    record = AcceptedFragmentRecord(
        universe_id="u_fuel",
        universe=_fuel_universe(),
        # Computed at acceptance time:
        fragment_hash=qualify_universe_fragment(
            manifest_item=item,
            fragment=_fragment("u_fuel", _fuel_universe()),
            known_material_ids={"m_fuel"},
        ).fragment_hash,
        manifest_contract_hash=item.contract_hash,
        qualification_status="passed",
        accepted_at_attempt=0,
    )
    result = verify_accepted_fragment_record(
        manifest_item=item, record=record, known_material_ids={"m_fuel"},
        material_roles_by_id={"m_fuel": "fuel"},
    )
    assert result.ok is True


def test_resume_record_rejects_drifted_hash():
    item = _manifest_item()
    real_hash = qualify_universe_fragment(
        manifest_item=item,
        fragment=_fragment("u_fuel", _fuel_universe()),
        known_material_ids={"m_fuel"},
    ).fragment_hash
    record = AcceptedFragmentRecord(
        universe_id="u_fuel",
        universe=_fuel_universe(),
        fragment_hash="stale_hash_that_does_not_match",
        manifest_contract_hash=item.contract_hash,
        qualification_status="passed",
        accepted_at_attempt=0,
    )
    result = verify_accepted_fragment_record(
        manifest_item=item, record=record, known_material_ids={"m_fuel"},
    )
    assert result.ok is False
    assert any(i.code == "qualification.fragment_hash_drift" for i in result.issues)


def test_resume_record_rejects_drifted_contract_hash():
    """A fragment that still passes qualification against the new manifest
    item but was accepted against a *different* contract hash must be
    downgraded so it is regenerated against the new contract."""
    # Use source_requirement_ids as the differentiator: it does not affect
    # qualification behavior but does affect the contract hash.
    item_old = _manifest_item(source_requirement_ids=["req:old"])
    item_new = _manifest_item(source_requirement_ids=["req:new"])
    assert item_old.contract_hash != item_new.contract_hash

    real_hash = qualify_universe_fragment(
        manifest_item=item_old,
        fragment=_fragment("u_fuel", _fuel_universe()),
        known_material_ids={"m_fuel"},
    ).fragment_hash
    record = AcceptedFragmentRecord(
        universe_id="u_fuel",
        universe=_fuel_universe(),
        fragment_hash=real_hash,
        manifest_contract_hash=item_old.contract_hash,
        qualification_status="passed",
        accepted_at_attempt=0,
    )
    result = verify_accepted_fragment_record(
        manifest_item=item_new, record=record, known_material_ids={"m_fuel"},
    )
    assert result.ok is False
    assert any(i.code == "qualification.manifest_contract_drift" for i in result.issues)


def test_resume_record_rejects_corrupt_payload():
    item = _manifest_item()
    result = verify_accepted_fragment_record(
        manifest_item=item, record="not_a_record_instance",  # type: ignore[arg-type]
        known_material_ids={"m_fuel"},
    )
    assert result.ok is False
    assert any(i.code == "qualification.resume_corrupt_record" for i in result.issues)
