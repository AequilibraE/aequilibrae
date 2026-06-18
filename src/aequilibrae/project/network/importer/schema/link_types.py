"""Deterministic, uncapped link-type allocator.

Replaces the legacy 51-cap behaviour. For each distinct ``link_type`` string
encountered during an import we allocate a single-character ``link_type_id``
(used by the AequilibraE ``link_types`` table).

Allocation strategy:
  1. lower-case first letter of the link type
  2. upper-case first letter
  3. next free ASCII letter / digit
"""

import logging
import string
from dataclasses import dataclass
from typing import Iterable

logger = logging.getLogger(__name__)


_FALLBACK_ALPHABET = string.ascii_lowercase + string.ascii_uppercase + string.digits


@dataclass
class LinkTypeAllocator:
    """Allocates ``link_type_id`` codes for new link types."""

    existing: dict

    def __post_init__(self):
        self._used_ids: set = set(self.existing.values())

    def allocate(self, link_type: str) -> str:
        if link_type in self.existing:
            return self.existing[link_type]
        if not link_type:
            link_type = "empty"
        normalised = link_type.strip().lower()

        candidates: list = [normalised[0], normalised[0].upper()]
        candidates.extend(_FALLBACK_ALPHABET)

        for candidate in candidates:
            if candidate not in self._used_ids:
                self._used_ids.add(candidate)
                self.existing[link_type] = candidate
                return candidate

        raise RuntimeError(
            "Exhausted the single-character link_type_id alphabet. "
            "This should not happen in practice; please report a bug."
        )

    def assign_many(self, link_types: Iterable[str]) -> dict:
        out = {}
        for lt in link_types:
            out[lt] = self.allocate(lt)
        return out
