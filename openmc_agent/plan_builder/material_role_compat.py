"""Material role compatibility helpers for patch generation contracts."""

from __future__ import annotations


_POISON_ABSORBER_ROLES = {"poison", "absorber"}


def material_role_satisfies(actual_role: str | None, required_role: str | None) -> bool:
    """Return True when an accepted material role satisfies a required role."""

    actual = str(actual_role or "").strip().lower()
    required = str(required_role or "").strip().lower()
    if not actual or not required:
        return False
    if actual == required:
        return True
    return actual in _POISON_ABSORBER_ROLES and required in _POISON_ABSORBER_ROLES
