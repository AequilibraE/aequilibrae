"""Shared mode codes and filtering helpers."""

from typing import Sequence

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


def filter_by_modes(modes_string: str, requested_codes: set[str]) -> str:
    """Trim a modes string down to the requested mode codes."""
    return "".join(sorted(set(modes_string) & requested_codes))
