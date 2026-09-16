"""LinkTypeAllocator capacity tests (single-character link_type_id constraint)."""

import pytest

from aequilibrae.project.network.importer.schema.link_types import LinkTypeAllocator


def test_count_free_slots_excludes_reserved_ids():
    existing = {"centroid_connector": "z", "default": "y"}
    assert LinkTypeAllocator.count_free_slots(existing) == 50


def test_allocator_assigns_unique_single_char_ids_until_exhaustion():
    existing = {"centroid_connector": "z", "default": "y"}
    alloc = LinkTypeAllocator(existing=dict(existing))

    assigned = [alloc.allocate(f"type_{i}") for i in range(50)]
    assert len(set(assigned)) == 50
    assert all(len(code) == 1 for code in assigned)


def test_allocator_raises_after_exhausting_alphabet():
    existing = {"centroid_connector": "z", "default": "y"}
    alloc = LinkTypeAllocator(existing=dict(existing))
    for i in range(50):
        alloc.allocate(f"type_{i}")
    with pytest.raises(RuntimeError, match="alphabet"):
        alloc.allocate("one_too_many")
