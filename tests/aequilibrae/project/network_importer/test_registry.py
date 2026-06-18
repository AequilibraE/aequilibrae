import pytest

from aequilibrae.project.network.importer import SOURCES, SIMPLIFIERS
from aequilibrae.project.network.importer.exceptions import SourceResolutionError
from aequilibrae.project.network.importer.sources.base import resolve_source
from aequilibrae.project.network.importer.simplifiers.base import resolve_simplifier


def test_supported_sources_are_explicit():
    assert set(SOURCES.keys()) == {"osm-overpass", "osm-pbf", "overture-cloud"}


def test_supported_simplifiers_are_explicit():
    assert set(SIMPLIFIERS.keys()) == {"osmnx"}


def test_resolve_unknown_source_raises_with_available_list():
    with pytest.raises(SourceResolutionError, match="osm-overpass"):
        resolve_source("definitely-not-a-source")


def test_resolve_unknown_simplifier_raises_with_available_list():
    with pytest.raises(SourceResolutionError, match="osmnx"):
        resolve_simplifier("definitely-not-a-simplifier")


def test_resolve_simplifier_false_returns_none():
    assert resolve_simplifier(False) is None


def test_resolve_simplifier_true_returns_osmnx():
    simplifier = resolve_simplifier(True)
    assert simplifier is not None
    assert simplifier.name == "osmnx"
