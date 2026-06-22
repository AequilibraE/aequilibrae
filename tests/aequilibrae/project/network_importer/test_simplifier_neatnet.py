import geopandas as gpd
import json
import pytest
from shapely.geometry import LineString
from shapely.geometry import Point

from aequilibrae.project.network.importer.simplifiers.impl_neatnet import (
    _DUAL_CARRIAGEWAY_WARNING,
    _transfer_attributes,
    run_neatnet_simplify,
)
from aequilibrae.project.network.importer.staged_network import StagedNetwork


def test_transfer_attributes_collapses_opposing_carriageways_to_coarse_bidirectional_values():
    simplified = gpd.GeoDataFrame(
        {"geometry": [LineString([(0.0, 0.0), (0.0, 0.002)])]},
        geometry="geometry",
        crs="EPSG:4326",
    )
    original = gpd.GeoDataFrame(
        {
            "direction": [1, -1],
            "modes": ["c", "c"],
            "link_type": ["primary", "primary"],
            "name": ["Main St", "Main St"],
            "speed_ab": [60.0, None],
            "speed_ba": [None, 45.0],
            "lanes_ab": [3, None],
            "lanes_ba": [None, 2],
            "source_id": ["fwd", "bwd"],
            "geometry": [
                LineString([(-0.00005, 0.0), (-0.00005, 0.002)]),
                LineString([(0.00005, 0.0), (0.00005, 0.002)]),
            ],
        },
        geometry="geometry",
        crs="EPSG:4326",
    )

    _transfer_attributes(simplified, original)

    row = simplified.iloc[0]
    assert row["direction"] == 0
    assert row["modes"] == "c"
    assert row["link_type"] == "primary"
    assert row["name"] == "Main St"
    assert row["speed_ab"] == 60.0
    assert row["speed_ba"] == 45.0
    assert row["lanes_ab"] == 3
    assert row["lanes_ba"] == 2
    assert row["source_id"] == "fwd"
    assert row["_source_refs"] == ["fwd::ab", "bwd::ba"]
    payload = json.loads(row["source_ids"])
    assert set(payload) == {"fwd", "bwd"}


def test_run_neatnet_simplify_warns_about_dual_carriageway_collapse(monkeypatch):
    pytest.importorskip("neatnet")
    net = StagedNetwork(
        nodes=gpd.GeoDataFrame(
            {
                "node_id": [100000, 100001],
                "geometry": [Point(0.0, 0.0), Point(0.0, 0.001)],
                "modes": ["c", "c"],
                "source_id": ["n0", "n1"],
            },
            geometry="geometry",
            crs="EPSG:4326",
        ),
        links=gpd.GeoDataFrame(
            {
                "link_id": [1],
                "a_node": [100000],
                "b_node": [100001],
                "direction": [1],
                "modes": ["c"],
                "link_type": ["primary"],
                "distance": [100.0],
                "geometry": [LineString([(0.0, 0.0), (0.0, 0.001)])],
                "name": ["Main St"],
                "speed_ab": [40.0],
                "speed_ba": [None],
                "lanes_ab": [1],
                "lanes_ba": [None],
                "source_id": ["s1"],
            },
            geometry="geometry",
            crs="EPSG:4326",
        ),
        source_meta={
            "source": "osm",
            "backend": "pyrosm",
            "source_url": "test.osm.pbf",
            "fetched_at": "2026-06-22T00:00:00+00:00",
            "release": "",
        },
    )

    import neatnet

    monkeypatch.setattr(neatnet, "neatify", lambda geom_only, **kwargs: geom_only)

    with pytest.warns(UserWarning, match="collapse parallel one-way carriageways"):
        out = run_neatnet_simplify(net)
    assert _DUAL_CARRIAGEWAY_WARNING.startswith("neatnet simplification may collapse")
    assert len(out.links) == 1


