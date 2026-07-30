"""Shared pin-map universe mapping helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable


@dataclass(frozen=True)
class PinMapUniverseFallback:
    """A degraded mapping substituted for a missing pin-path universe."""

    kind: str
    universe_id: str
    substitute_kind: str


def build_kind_to_universe_map(
    universes: Iterable[Any] | None,
    pin_map: Any,
) -> tuple[dict[str, str], list[PinMapUniverseFallback]]:
    """Map universe kind to id, with observable water-cell degraded fallback.

    If a pin map declares guide/instrument tube coordinates but the universe
    catalog only provides a water cell, use the water cell instead of letting
    those coordinates fall back to fuel.  This preserves renderability while
    exposing the geometry caveat to callers.
    """

    kind_map: dict[str, str] = {}
    for univ in universes or []:
        kind = getattr(univ, "kind", None)
        uid = getattr(univ, "universe_id", None)
        if kind and uid:
            kind_map[str(kind)] = str(uid)

    default_uid = getattr(pin_map, "default_universe_id", None)
    if default_uid:
        kind_map.setdefault("fuel_pin", str(default_uid))

    fallbacks: list[PinMapUniverseFallback] = []
    water_uid = kind_map.get("water_cell")
    if not water_uid:
        return kind_map, fallbacks

    for kind, attr in (
        ("guide_tube", "guide_tube_coords"),
        ("instrument_tube", "instrument_tube_coords"),
    ):
        if kind in kind_map:
            continue
        if getattr(pin_map, attr, None):
            kind_map[kind] = water_uid
            fallbacks.append(PinMapUniverseFallback(
                kind=kind,
                universe_id=water_uid,
                substitute_kind="water_cell",
            ))

    return kind_map, fallbacks


__all__ = ["PinMapUniverseFallback", "build_kind_to_universe_map"]
