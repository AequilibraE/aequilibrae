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


def _pin_settings(monkeypatch):
    """Snapshot the osmnx globals we mutate so monkeypatch restores them."""
    for attr in ("overpass_url", "nominatim_url", "requests_timeout", "http_accept_language",
                 "overpass_rate_limit"):
        monkeypatch.setattr(osmnx.settings, attr, getattr(osmnx.settings, attr))


def test_configure_osmnx_applies_project_parameters(monkeypatch):
    _pin_settings(monkeypatch)

    _configure_osmnx(osmnx)
    assert "nominatim" in osmnx.settings.nominatim_url
    # osmnx 2.x reads ``requests_timeout``; ``timeout`` is a dead attribute.
    assert osmnx.settings.requests_timeout == 540
    assert osmnx.settings.http_accept_language == "en"


def test_overpass_url_is_a_base_url_not_the_interpreter_endpoint(monkeypatch):
    """osmnx appends '/interpreter' itself; appending it here too 403s every request."""
    import aequilibrae.parameters as params_mod

    _pin_settings(monkeypatch)

    custom = "http://192.168.0.10:12345/api"

    class _StubParameters:
        # Trailing slash included: it must be normalised away, not doubled up.
        parameters = {"osm": {"overpass_endpoint": custom + "/"}}

    monkeypatch.setattr(params_mod, "Parameters", _StubParameters)

    _configure_osmnx(osmnx)
    assert osmnx.settings.overpass_url == custom
    assert not osmnx.settings.overpass_url.endswith("/interpreter")
    # What osmnx will actually request must contain exactly one '/interpreter'.
    effective = osmnx.settings.overpass_url.rstrip("/") + "/interpreter"
    assert effective.count("/interpreter") == 1


def test_overpass_rate_limit_is_configurable(monkeypatch):
    """Self-hosted servers with an unlimited rate limit need client-side limiting off."""
    import aequilibrae.parameters as params_mod

    _pin_settings(monkeypatch)
    monkeypatch.setattr(osmnx.settings, "overpass_rate_limit", True)

    class _StubParameters:
        parameters = {"osm": {"overpass_rate_limit": False}}

    monkeypatch.setattr(params_mod, "Parameters", _StubParameters)
    _configure_osmnx(osmnx)
    assert osmnx.settings.overpass_rate_limit is False


def _reciprocal_graph():
    """Two-way street (reciprocal pair) + a oneway=-1 street (reversed only)."""
    g = nx.MultiDiGraph(crs="EPSG:4326")
    for nid, (x, y) in {1: (0.0, 0.0), 2: (0.001, 0.0), 3: (0.002, 0.0)}.items():
        g.add_node(nid, x=x, y=y)
    fwd = LineString([(0.0, 0.0), (0.001, 0.0)])
    g.add_edge(1, 2, key=0, osmid=100, highway="residential", oneway=False, reversed=False,
               length=111.0, geometry=fwd)
    g.add_edge(2, 1, key=0, osmid=100, highway="residential", oneway=False, reversed=True,
               length=111.0, geometry=LineString(list(fwd.coords)[::-1]))
    # An ``oneway=-1`` way: osmnx normalises it to oneway=True and emits a single
    # already-reversed edge whose geometry runs in the direction of travel.
    g.add_edge(3, 2, key=0, osmid=200, highway="primary", oneway=True, reversed=True,
               length=111.0, geometry=LineString([(0.002, 0.0), (0.001, 0.0)]))
    return g


def test_reciprocal_two_way_edges_are_collapsed(monkeypatch, cache):
    monkeypatch.setattr(osmnx, "graph_from_polygon", lambda area, **kw: _reciprocal_graph())
    net = acquire_overpass(modes=("car",), download_cache=cache, model_area=box(-0.01, -0.01, 0.01, 0.01))
    net.validate()

    # One row for the two-way street, one for the oneway=-1 street.
    assert len(net.links) == 2
    assert sorted(net.links["source_id"].astype(str)) == ["100", "200"]

    # No (a,b)/(b,a) pair may survive: that would double capacity and length.
    pairs = {(int(a), int(b)) for a, b in zip(net.links["a_node"], net.links["b_node"], strict=True)}
    assert not any((b, a) in pairs for a, b in pairs)


_MISSING_NODES = "Some edges missing nodes, possibly due to input data clipping issue."


def _raise(message):
    def fail(*args, **kwargs):
        raise ValueError(message)

    return fail


def _zero_length_graph():
    """A graph whose only edge has coincident endpoints, so the staged link has zero length."""
    g = nx.MultiDiGraph(crs="EPSG:4326")
    g.add_node(1, x=0.0, y=0.0)
    g.add_node(2, x=0.0, y=0.0)
    g.add_edge(1, 2, key=0, osmid=200, highway="residential", oneway=False,
               length=0.0, geometry=LineString([(0.0, 0.0), (0.0, 0.0)]))
    return g


@pytest.fixture
def bbox_attempts(monkeypatch):
    """Records the areas the bbox fallback tries, letting the first attempt succeed."""
    attempts = []

    def fake_polygon(area, **kw):
        attempts.append(area)
        return _fake_graph()

    monkeypatch.setattr(osmnx, "geocode", lambda place: (0.0005, 0.0005))
    monkeypatch.setattr(osmnx, "graph_from_polygon", fake_polygon)
    return attempts


def test_place_query_falls_back_to_bbox_on_osmnx_error(monkeypatch, cache, bbox_attempts):
    """Test that a place query failing with a missing-nodes error is retried as a bbox query."""
    monkeypatch.setattr(osmnx, "graph_from_place", _raise(_MISSING_NODES))

    net = acquire_overpass(modes=("car",), download_cache=cache, place_name="Bigcity")

    assert len(bbox_attempts) == 1
    assert len(net.links) == 2


def test_place_query_falls_back_to_bbox_on_validation_failure(monkeypatch, cache, bbox_attempts):
    """Test that a staged network failing validation is retried as a bbox query."""
    monkeypatch.setattr(osmnx, "graph_from_place", lambda place, **kw: _zero_length_graph())

    net = acquire_overpass(modes=("car",), download_cache=cache, place_name="Zerotown")

    assert len(bbox_attempts) == 1
    assert len(net.links) == 2


def test_explicit_model_area_is_never_replaced_by_a_smaller_box(monkeypatch, cache):
    """Test that a caller-supplied model_area fails outright rather than falling back to a bbox."""
    attempts = []

    def fail(area, **kw):
        attempts.append(area)
        raise ValueError(_MISSING_NODES)

    monkeypatch.setattr(osmnx, "graph_from_polygon", fail)

    with pytest.raises(ValueError, match="missing nodes"):
        acquire_overpass(modes=("car",), download_cache=cache, model_area=box(-0.001, -0.001, 0.002, 0.002))
    assert len(attempts) == 1


def test_place_query_gives_up_once_every_bbox_fails(monkeypatch, cache):
    """Test that the last failure is raised after every fallback box has been tried."""
    monkeypatch.setattr(osmnx, "geocode", lambda place: (0.0005, 0.0005))
    monkeypatch.setattr(osmnx, "graph_from_place", _raise(_MISSING_NODES))
    monkeypatch.setattr(osmnx, "graph_from_polygon", _raise(_MISSING_NODES))

    with pytest.raises(ValueError, match="missing nodes"):
        acquire_overpass(modes=("car",), download_cache=cache, place_name="Nowheresville")
