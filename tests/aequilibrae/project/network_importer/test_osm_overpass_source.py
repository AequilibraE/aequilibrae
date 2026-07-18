"""Overpass-source tests with a mocked osmnx fetch (no network access)."""

import json

import networkx as nx
import pytest
from shapely.geometry import LineString, box

from aequilibrae.project.network.importer.download_cache import DownloadCache
from aequilibrae.project.network.importer.exceptions import ImporterError
from aequilibrae.project.network.importer.sources.osm.impl import _configure_osmnx, acquire_overpass

osmnx = pytest.importorskip("osmnx")


def _fake_graph():
    g = nx.MultiDiGraph(crs="EPSG:4326")
    g.add_node(1, x=0.0, y=0.0)
    g.add_node(2, x=0.001, y=0.0)
    g.add_node(3, x=0.001, y=0.001)
    g.add_edge(
        1,
        2,
        key=0,
        osmid=100,
        highway="residential",
        oneway=False,
        length=111.0,
        geometry=LineString([(0.0, 0.0), (0.001, 0.0)]),
    )
    g.add_edge(
        2,
        3,
        key=0,
        osmid=101,
        highway="primary",
        oneway=True,
        maxspeed="50",
        length=111.0,
        geometry=LineString([(0.001, 0.0), (0.001, 0.001)]),
    )
    return g


@pytest.fixture
def cache(tmp_path):
    return DownloadCache(project_base_path=tmp_path, source_name="osm-overpass", tag="test")


def test_acquire_overpass_builds_staged_network_and_cache(monkeypatch, cache):
    monkeypatch.setattr(osmnx, "graph_from_polygon", lambda area, **kw: _fake_graph())

    net = acquire_overpass(modes=("car",), download_cache=cache, model_area=box(-0.001, -0.001, 0.002, 0.002))
    net.validate()
    assert len(net.links) == 2
    assert net.source_meta["source"] == "osm"
    assert net.source_meta["backend"] == "osmnx-overpass"
    assert net.source_meta["source_url"].startswith("overpass:bbox=")
    assert set(net.links["link_type"]) == {"residential", "primary"}
    assert sorted(net.links["direction"]) == [0, 1]
    assert net.links.loc[net.links["link_type"] == "primary", "speed_ab"].iloc[0] == 50.0

    assert (cache.folder / "osm.parquet").exists()
    manifest = json.loads((cache.folder / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["n_nodes"] == 3
    assert manifest["n_edges"] == 2
    assert manifest["sha256"]["osm.parquet"]


def test_acquire_overpass_by_place_name(monkeypatch, cache):
    captured = {}

    def fake_place(place, **kw):
        captured["place"] = place
        return _fake_graph()

    monkeypatch.setattr(osmnx, "graph_from_place", fake_place)
    net = acquire_overpass(modes=("car",), download_cache=cache, place_name="Testville")
    assert captured["place"] == "Testville"
    assert net.source_meta["source_url"] == "overpass:place=Testville"


def test_acquire_overpass_requires_exactly_one_selector(cache):
    with pytest.raises(ImporterError, match="exactly one"):
        acquire_overpass(modes=("car",), download_cache=cache)
    with pytest.raises(ImporterError, match="exactly one"):
        acquire_overpass(modes=("car",), download_cache=cache, model_area=box(0, 0, 1, 1), place_name="x")


def test_acquire_overpass_wraps_insufficient_response(monkeypatch, cache):
    from osmnx._errors import InsufficientResponseError

    def boom(place, **kw):
        raise InsufficientResponseError("nothing here")

    monkeypatch.setattr(osmnx, "graph_from_place", boom)
    with pytest.raises(ImporterError, match="empty or partial response"):
        acquire_overpass(modes=("car",), download_cache=cache, place_name="Nowhere")


def test_acquire_overpass_wraps_request_failure(monkeypatch, cache):
    from requests.exceptions import ConnectionError as RequestsConnectionError

    def boom(place, **kw):
        raise RequestsConnectionError("no route to host")

    monkeypatch.setattr(osmnx, "graph_from_place", boom)
    with pytest.raises(ImporterError, match="request failed"):
        acquire_overpass(modes=("car",), download_cache=cache, place_name="Nowhere")


def test_acquire_overpass_empty_graph_raises(monkeypatch, cache):
    monkeypatch.setattr(osmnx, "graph_from_place", lambda place, **kw: nx.MultiDiGraph(crs="EPSG:4326"))
    with pytest.raises(ImporterError, match="no edges"):
        acquire_overpass(modes=("car",), download_cache=cache, place_name="Empty")


def test_configure_osmnx_applies_project_parameters(monkeypatch):
    # Pin the current global values so monkeypatch restores them on exit.
    for attr in ("overpass_url", "nominatim_url", "timeout", "http_accept_language"):
        monkeypatch.setattr(osmnx.settings, attr, getattr(osmnx.settings, attr))

    _configure_osmnx(osmnx)
    assert osmnx.settings.overpass_url.endswith("/interpreter")
    assert "nominatim" in osmnx.settings.nominatim_url
    assert osmnx.settings.timeout == 540
    assert osmnx.settings.http_accept_language == "en"
