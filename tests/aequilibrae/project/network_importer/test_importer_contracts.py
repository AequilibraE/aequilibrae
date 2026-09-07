import geopandas as gpd
import pytest
from shapely.geometry import LineString, Point

from aequilibrae.project.network.importer.exceptions import ImporterError
from aequilibrae.project.network.importer.importer import (
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
    assert ok.source_meta["release"] == ""

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



def test_run_forwards_consolidate_tolerance_to_any_simplifier(empty_project):
    from aequilibrae.project.network.importer.importer import NetworkImporter

    class _Source:
        name = "fake-source"

        def acquire(self, *, modes, download_cache):
            return _minimal_net(
                {
                    "source": "osm",
                    "backend": "pyrosm",
                    "source_url": "test.osm.pbf",
                    "fetched_at": "2026-06-22T00:00:00+00:00",
                }
            )

    class _Simplifier:
        name = "capturing"

        def __init__(self):
            self.kwargs = None

        def simplify(self, net, **kwargs):
            self.kwargs = kwargs
            return net

    simplifier = _Simplifier()
    NetworkImporter(empty_project).run(_Source(), simplify=simplifier, consolidate_tolerance=17.5)

    assert simplifier.kwargs == {"consolidate_tolerance": 17.5}
