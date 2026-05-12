import logging
import string
from dataclasses import dataclass
from typing import Iterable

logger = logging.getLogger(__name__)

_FALLBACK_ALPHABET = string.ascii_lowercase + string.ascii_uppercase


@dataclass
class LinkTypeAllocator:
    """Allocates ``link_type_id`` codes for new link types."""

    existing: dict

    def __post_init__(self):
        self._used_ids: set = set(self.existing.values())

    @staticmethod
    def count_free_slots(existing: dict) -> int:
        """How many new single-character ids are still available given ``existing``."""
        used = set(existing.values())
        return sum(1 for c in _FALLBACK_ALPHABET if c not in used)

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

        raise RuntimeError("Exhausted the single-character alphabet. Reduce the number of link types in your model.")

    def assign_many(self, link_types: Iterable[str]) -> dict:
        out = {}
        for lt in link_types:
            out[lt] = self.allocate(lt)
        return out
