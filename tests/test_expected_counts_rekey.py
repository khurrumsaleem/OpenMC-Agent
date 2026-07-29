"""Tests for re-keying role-keyed expected_counts to universe-id keys.

The downstream lattice pin-count cross-check compares ``expected_counts``
against the ``universe_pattern`` directly (keyed by universe id).  When the
LLM keys ``expected_counts`` by universe kind/role (e.g. ``"fuel_pin"``) but
the pattern uses a concrete universe id (e.g. ``"fuel_variant_3B"``), a
correct pattern would spuriously mismatch.  The assembler re-keys
unambiguous kind→universe-id mappings so the cross-check is consistent.
"""

from __future__ import annotations

from openmc_agent.plan_builder.assembler import _rekey_expected_counts_to_universe_ids


def test_rekey_role_to_universe_id_unambiguous():
    kind_by_id = {
        "fuel_variant_3B": "fuel_pin",
        "guide_tube": "guide_tube",
        "fiss_chamber": "instrument_tube",
    }
    out = _rekey_expected_counts_to_universe_ids(
        {"fuel_pin": 264, "guide_tube": 24, "instrument_tube": 1},
        kind_by_id,
    )
    assert out == {"fuel_variant_3B": 264, "guide_tube": 24, "fiss_chamber": 1}


def test_rekey_preserves_already_universe_id_keyed_counts():
    kind_by_id = {"fuel_variant_3B": "fuel_pin"}
    out = _rekey_expected_counts_to_universe_ids({"fuel_variant_3B": 289}, kind_by_id)
    assert out == {"fuel_variant_3B": 289}


def test_rekey_keeps_role_key_when_kind_is_ambiguous():
    kind_by_id = {"fuel_a": "fuel_pin", "fuel_b": "fuel_pin"}
    out = _rekey_expected_counts_to_universe_ids({"fuel_pin": 200}, kind_by_id)
    assert out == {"fuel_pin": 200}


def test_rekey_preserves_unknown_keys():
    kind_by_id = {"fuel_variant_3B": "fuel_pin"}
    out = _rekey_expected_counts_to_universe_ids({"water_cell": 100}, kind_by_id)
    assert out == {"water_cell": 100}


def test_rekey_noop_without_kind_map():
    assert _rekey_expected_counts_to_universe_ids({"fuel_pin": 289}, {}) == {"fuel_pin": 289}
