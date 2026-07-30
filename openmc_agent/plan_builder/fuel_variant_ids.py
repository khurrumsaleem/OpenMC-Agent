"""Reactor-neutral fuel variant identifier matching helpers."""

from __future__ import annotations

import re


_NON_DISTINGUISHING_VARIANT_TOKENS = {
    "fuel",
    "pin",
    "source",
    "state",
    "variant",
}


def fuel_variant_tokens(value: str) -> frozenset[str]:
    """Return the distinguishing tokens in a fuel variant label.

    LLMs commonly preserve the meaningful state token while changing generic
    adornments, for example ``state_3b`` vs ``3B_fuel``.  The generic words are
    not variant identity; alphanumeric tokens such as ``3a``/``3b`` are.
    """

    tokens = {
        item
        for item in re.split(r"[^a-z0-9]+", value.lower())
        if item and item not in _NON_DISTINGUISHING_VARIANT_TOKENS
    }
    return frozenset(tokens)


def fuel_variant_ids_equivalent(actual: str | None, expected: str | None) -> bool:
    """Return True when two fuel variant ids identify the same source variant."""

    a = (actual or "").strip().lower()
    b = (expected or "").strip().lower()
    if not a or not b:
        return False
    if a == b:
        return True
    if a in b or b in a:
        return True
    a_tokens = fuel_variant_tokens(a)
    return bool(a_tokens) and a_tokens == fuel_variant_tokens(b)
