"""Tests for source/simplifier registries and string-name resolution."""

import pytest

from aequilibrae.project.network.importer import SOURCES, SIMPLIFIERS
from aequilibrae.project.network.importer.exceptions import SourceResolutionError
from aequilibrae.project.network.importer.sources.base import resolve_source
from aequilibrae.project.network.importer.simplifiers.base import resolve_simplifier


def test_all_six_sources_registered():
    expected = {
        "osm-overpass",
        "osm-pbf",
        "overture-cloud",
        "geodataframe",
        "file",
        "gmns",
    }
    assert expected.issubset(set(SOURCES.keys()))


def test_two_simplifiers_registered():
    assert {"osmnx", "neatnet"}.issubset(set(SIMPLIFIERS.keys()))


def test_resolve_unknown_source_raises_with_available_list():
    with pytest.raises(SourceResolutionError, match="osm-overpass"):
        resolve_source("definitely-not-a-source")


def test_resolve_unknown_simplifier_raises_with_available_list():
    with pytest.raises(SourceResolutionError, match="osmnx"):
        resolve_simplifier("definitely-not-a-simplifier")


def test_resolve_simplifier_false_returns_none():
    assert resolve_simplifier(False) is None


def test_resolve_simplifier_true_returns_osmnx():
    s = resolve_simplifier(True)
    assert s is not None
    assert s.name == "osmnx"
