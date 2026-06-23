import geopandas as gpd
import json
import pytest
from shapely.geometry import LineString
from shapely.geometry import Point

from aequilibrae.project.network.importer.simplifiers.impl_neatnet import (
    _DUAL_CARRIAGEWAY_WARNING,
    _assign_and_unmerge_suspicious_nodes,
    _classify_orientation_fast,
    _dekink_endpoints_local,
    _find_suspicious_nodes,
    _geometry_summary,
    _link_type_compatible,
    _select_branch_to_unmerge,
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
    assert payload["schema_version"] == 1
    assert set(payload["sources"]) == {"fwd", "bwd"}


def test_link_type_compatibility_blocks_cross_family_transfer():
    # Same type and same functional family are compatible.
    assert _link_type_compatible("primary", "primary") is True
    assert _link_type_compatible("primary", "residential") is True
    # A footway/cycleway must not donate to a road.
    assert _link_type_compatible("primary", "footway") is False
    assert _link_type_compatible("primary", "cycleway") is False
    # Unknown types stay permissive so we never drop the only candidate.
    assert _link_type_compatible("primary", "some_custom_type") is True


def test_orientation_fast_classifies_straight_links_but_not_curved_ones():
    straight = LineString([(0.0, 0.0), (0.0, 0.002)])
    straight_rev = LineString([(0.0, 0.002), (0.0, 0.0)])
    curved = LineString([(0.0, 0.0), (0.001, 0.001), (0.0, 0.002)])

    assert _classify_orientation_fast(_geometry_summary(straight), straight, 1.0) is True
    assert _classify_orientation_fast(_geometry_summary(straight), straight_rev, 1.0) is False
    assert _classify_orientation_fast(_geometry_summary(curved), curved, 0.7) is None


def test_dekink_endpoints_preserves_endpoints_and_removes_local_spike():
    geom = LineString([(0.0, 0.0), (0.0002, 0.0), (-0.0001, 0.00005), (0.0, 0.001), (0.0, 0.002)])
    out = _dekink_endpoints_local(geom)
    assert out.coords[0] == geom.coords[0]
    assert out.coords[-1] == geom.coords[-1]
    assert len(out.coords) < len(geom.coords)


def test_suspicious_node_detection_requires_degree_family_mix_and_short_branch():
    edges = gpd.GeoDataFrame(
        {
            "geometry": [
                LineString([(0, 0), (1, 0)]),
                LineString([(0, 0), (-1, 0)]),
                LineString([(0, 0), (0, 1)]),
                LineString([(0, 0), (0.0001, 0.0001)]),
            ],
            "a_node": [100000, 100000, 100000, 100000],
            "b_node": [100001, 100002, 100003, 100004],
            "link_type": ["primary", "primary", "primary", "footway"],
            "name": ["Main", "Main", "Main", None],
        },
        geometry="geometry",
        crs="EPSG:4326",
    )
    assert _find_suspicious_nodes(edges) == [100000]
    assert _select_branch_to_unmerge(edges, 100000) == 3


def test_unmerge_reassigns_selected_branch_and_offsets_it_slightly():
    edges = gpd.GeoDataFrame(
        {
            "geometry": [
                LineString([(0, 0), (1, 0)]),
                LineString([(0, 0), (-1, 0)]),
                LineString([(0, 0), (0, 1)]),
                LineString([(0, 0), (0.0001, 0.0001)]),
            ],
            "link_id": [1, 2, 3, 4],
            "link_type": ["primary", "primary", "primary", "footway"],
            "name": ["Main", "Main", "Main", None],
        },
        geometry="geometry",
        crs="EPSG:4326",
    )
    out, node_coords = _assign_and_unmerge_suspicious_nodes(edges)
    base = int(out.loc[0, "a_node"])
    split = int(out.loc[3, "a_node"])
    assert split != base
    assert int(out.loc[1, "a_node"]) == base
    assert int(out.loc[2, "a_node"]) == base
    assert node_coords[base] != node_coords[split]
    assert list(out.loc[3, "geometry"].coords)[0] == node_coords[split]


def test_transfer_attributes_does_not_bleed_sidewalk_modes_onto_highway():
    # A highway link with a pedestrian sidewalk running parallel within the
    # buffer must NOT inherit the walk mode from the sidewalk.
    simplified = gpd.GeoDataFrame(
        {"geometry": [LineString([(0.0, 0.0), (0.0, 0.002)])]},
        geometry="geometry",
        crs="EPSG:4326",
    )
    original = gpd.GeoDataFrame(
        {
            "direction": [1, 1],
            "modes": ["c", "w"],
            "link_type": ["primary", "footway"],
            "name": ["Main St", "Sidewalk"],
            "speed_ab": [60.0, None],
            "speed_ba": [None, None],
            "lanes_ab": [2, None],
            "lanes_ba": [None, None],
            "source_id": ["road", "walk"],
            "geometry": [
                LineString([(0.0, 0.0), (0.0, 0.002)]),
                LineString([(0.00008, 0.0), (0.00008, 0.002)]),
            ],
        },
        geometry="geometry",
        crs="EPSG:4326",
    )

    _transfer_attributes(simplified, original)

    row = simplified.iloc[0]
    assert "w" not in row["modes"]
    assert row["modes"] == "c"
    assert row["link_type"] == "primary"


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


