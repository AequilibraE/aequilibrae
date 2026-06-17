"""Access-based mode-rule engine.

Per plan §1.3 rule 2 we removed the YAML allow-list of OSM ``highway`` values.
Mode assignment is now driven exclusively by source-side access semantics
(OSM ``access``, ``motor_vehicle``, ``bicycle``, ``foot``, ``vehicle``,
``oneway:<mode>``, ``service``, ``junction``; Overture ``access_restrictions``,
``subtype``, ``subclass_rules``).

Each ``ModeRule`` is a small predicate that takes a raw tag dict and returns
``True`` if the link should be flagged as allowing the corresponding mode.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping


@dataclass(frozen=True)
class ModeRule:
    """Predicate-based rule for assigning a single AequilibraE mode."""

    mode_name: str          # e.g. "car"
    mode_code: str          # one-letter mode_id, e.g. "c"
    predicate: Callable[[Mapping], bool]

    def applies(self, tags: Mapping) -> bool:
        try:
            return bool(self.predicate(tags))
        except Exception:  # pragma: no cover
            return False


def compute_modes_string(tags: Mapping, rules: list[ModeRule]) -> str:
    """Concatenate the mode codes whose rules match the given tags."""
    return "".join(sorted({r.mode_code for r in rules if r.applies(tags)}))


def filter_by_modes(modes_string: str, requested_codes: set[str]) -> str:
    """Trim a modes string down to the requested mode codes.

    Returns the intersection of the link's available modes with the user's
    requested modes. An empty string means the link should be dropped.
    """
    return "".join(sorted(set(modes_string) & requested_codes))
