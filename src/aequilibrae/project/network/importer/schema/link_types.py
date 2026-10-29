import string
from dataclasses import dataclass

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
        return len(set(_FALLBACK_ALPHABET) - set(existing.values()))

    def allocate(self, link_type: str) -> str:
        if link_type in self.existing:
            return self.existing[link_type]
        if not link_type:
            link_type = "empty"
        first = link_type.strip().lower()[0]
        for candidate in (first, first.upper(), *_FALLBACK_ALPHABET):
            if candidate not in self._used_ids:
                self._used_ids.add(candidate)
                self.existing[link_type] = candidate
                return candidate

        raise RuntimeError("Exhausted the single-character alphabet. Reduce the number of link types in your model.")
