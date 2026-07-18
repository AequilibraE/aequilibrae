"""Access-based mode-rule engine.

Mode assignment is driven exclusively by source-side access semantics
(OSM ``access``, ``motor_vehicle``, ``bicycle``, ``foot``, ``vehicle``,
``oneway:<mode>``, ``service``, ``junction``; Overture ``access_restrictions``,
``subtype``, ``subclass_rules``).

Each ``ModeRule`` is a small predicate that takes a raw tag dict and returns
``True`` if the link should be flagged as allowing the corresponding mode.
"""

from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

from aequilibrae.project.network.importer.exceptions import ImporterError

# AequilibraE mode name -> one-letter mode_id, matching the default modes table.
MODE_CODE = {
    "car": "c",
    "transit": "t",
    "bicycle": "b",
    "walk": "w",
}

DEFAULT_MODES = tuple(MODE_CODE)


def requested_mode_codes(modes: Sequence[str]) -> set:
    """Translate requested mode names into mode codes, rejecting an empty result."""
    codes = {MODE_CODE[m] for m in modes if m in MODE_CODE}
    if not codes:
        raise ImporterError(f"None of the requested modes {modes!r} match the configured modes {sorted(MODE_CODE)}")
    return codes


@dataclass(frozen=True)
class ModeRule:
    """Predicate-based rule for assigning a single AequilibraE mode."""

    mode_name: str  # e.g. "car"
    mode_code: str  # one-letter mode_id, e.g. "c"
    predicate: Callable[[Mapping], bool]

    def applies(self, tags: Mapping) -> bool:
        return bool(self.predicate(tags))


def compute_modes_string(tags: Mapping, rules: list[ModeRule]) -> str:
    """Concatenate the mode codes whose rules match the given tags."""
    return "".join(sorted({r.mode_code for r in rules if r.applies(tags)}))


def filter_by_modes(modes_string: str, requested_codes: set[str]) -> str:
    """Trim a modes string down to the requested mode codes."""
    return "".join(sorted(set(modes_string) & requested_codes))
