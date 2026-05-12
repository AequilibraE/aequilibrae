import geopandas as gpd
import pytest
from shapely.geometry import LineString, Point

from aequilibrae.project.network.importer.exceptions import ImporterError
from aequilibrae.project.network.importer.importer import (
    OPTIONAL_SOURCE_META_KEYS,
    REQUIRED_SOURCE_META_KEYS,
    _normalize_importer_columns,
    _normalize_source_meta,
)
from aequilibrae.project.network.importer.staged_network import StagedNetwork


def _minimal_net(source_meta=None):
    nodes = gpd.GeoDataFrame(
        {"node_id": [100000, 100001], "geometry": [Point(0, 0), Point(0, 1)], "modes": ["c", "c"]},
        geometry="geometry",
        crs="EPSG:4326",
    )
    links = gpd.GeoDataFrame(
        {
            "link_id": [1],
            "a_node": [100000],
            "b_node": [100001],
            "direction": [0],
            "modes": ["c"],
            "link_type": ["residential"],
            "distance": [111000.0],
            "geometry": [LineString([(0, 0), (0, 1)])],
        },
        geometry="geometry",
        crs="EPSG:4326",
    )
    return StagedNetwork(nodes=nodes, links=links, source_meta=source_meta or {})


def test_normalize_importer_columns_adds_expected_optional_fields():
    net = _minimal_net(
        {
            "source": "osm",
            "backend": "pyrosm",
            "source_url": "test.osm.pbf",
            "fetched_at": "2026-06-22T00:00:00+00:00",
        }
    )

    _normalize_importer_columns(net)

    assert {"source_id"}.issubset(net.nodes.columns)
    assert {"name", "speed_ab", "speed_ba", "lanes_ab", "lanes_ba", "source_id"}.issubset(net.links.columns)


def test_normalize_source_meta_requires_core_keys():
    net = _minimal_net({"source": "osm"})

    with pytest.raises(ImporterError, match="source_meta missing required keys"):
        _normalize_source_meta(net)


def test_normalize_source_meta_adds_release_key():
    net = _minimal_net(
        {
            "source": "osm",
            "backend": "pyrosm",
            "source_url": "test.osm.pbf",
            "fetched_at": "2026-06-22T00:00:00+00:00",
        }
    )

    _normalize_source_meta(net)

    assert net.source_meta["release"] == ""


def test_source_meta_key_split_is_explicit_and_documented():
    # 'release' is OPTIONAL (unversioned sources like raw OSM); everything else
    # is REQUIRED. This guards the contract against accidental mutation.
    assert "release" in OPTIONAL_SOURCE_META_KEYS
    assert "release" not in REQUIRED_SOURCE_META_KEYS
    assert set(REQUIRED_SOURCE_META_KEYS) == {"source", "backend", "source_url", "fetched_at"}


def test_missing_release_is_allowed_but_missing_other_keys_rejected():
    # Missing only 'release' must pass.
    ok = _minimal_net(
        {
            "source": "osm",
            "backend": "pyrosm",
            "source_url": "test.osm.pbf",
            "fetched_at": "2026-06-22T00:00:00+00:00",
        }
    )
    _normalize_source_meta(ok)

    # Missing any required key must raise, naming the missing key.
    for required in REQUIRED_SOURCE_META_KEYS:
        meta = {
            "source": "osm",
            "backend": "pyrosm",
            "source_url": "test.osm.pbf",
            "fetched_at": "2026-06-22T00:00:00+00:00",
            "release": "2026-01",
        }
        del meta[required]
        with pytest.raises(ImporterError, match="source_meta missing required keys"):
            _normalize_source_meta(_minimal_net(meta))


