import json

import numpy as np
import pandas as pd
import pytest
import shapely

from aequilibrae.transit import Transit
from aequilibrae.transit.transit_graph_builder import TransitGraphBuilder, geodesic_length
from aequilibrae.utils.geo_utils import haversine

CONNECTOR_METHODS = ["nearest_neighbour", "overlapping_regions"]


def build(project, **kwargs):
    kwargs = {"with_walking_edges": False, "blocking_centroid_flows": False, **kwargs}
    return Transit(project).create_graph(**kwargs)


def edges_of(graph, link_type):
    return graph.edges[graph.edges.link_type == link_type]


def connector_pairs(graph, link_type="access_connector"):
    edges = edges_of(graph, link_type)
    return set(zip(edges.a_node, edges.b_node, strict=True))


def project_zones(project) -> pd.DataFrame:
    return pd.DataFrame([(z.zone_id, z.geometry) for z in project.network.zones], columns=["zone_id", "geometry"])


def assert_lines_run_tail_to_head(graph, rows=None):
    edges = graph.edges if rows is None else graph.edges[rows]
    points = graph.vertices.set_index("node_id").geometry
    lines = shapely.from_wkb(edges.geometry.to_numpy())
    starts = shapely.get_coordinates(shapely.get_point(lines, 0))
    ends = shapely.get_coordinates(shapely.get_point(lines, -1))
    tails = shapely.get_coordinates(shapely.from_wkb(points.loc[edges.a_node].to_numpy()))
    heads = shapely.get_coordinates(shapely.from_wkb(points.loc[edges.b_node].to_numpy()))
    np.testing.assert_allclose(starts, tails, atol=1e-9)
    np.testing.assert_allclose(ends, heads, atol=1e-9)


@pytest.mark.parametrize(
    "connector_method,method",
    [
        ("overlapping_regions", "connector project match"),
        ("overlapping_regions", "direct"),
        ("nearest_neighbour", "connector project match"),
        ("nearest_neighbour", "direct"),
    ],
)
def test_create_line_geometry(coquimbo_example, connector_method, method):
    """Test that every edge gets a line from its tail vertex to its head vertex, with either drawing method."""
    data = Transit(coquimbo_example)
    graph = data.create_graph(
        with_outer_stop_transfers=False,
        with_walking_edges=False,
        blocking_centroid_flows=False,
        connector_method=connector_method,
    )
    assert "geometry" not in graph.edges.columns
    graph.create_line_geometry(method=method, graph="c")
    assert "geometry" in graph.edges.columns
    assert graph.edges.geometry.all()
    assert_lines_run_tail_to_head(graph)


def test_connector_project_match_measures_the_matched_line(coquimbo_example):
    """Test that matched connectors take the geodesic length of their line as walking distance and time."""
    graph = build(coquimbo_example, walking_speed=1.25)
    straight = graph.edges.copy()
    graph.create_line_geometry(method="connector project match", graph="c")

    connectors = graph.edges.link_type.isin(["access_connector", "egress_connector"]).to_numpy()
    lengths = [geodesic_length(shapely.from_wkb(line)) for line in graph.edges.geometry[connectors]]
    np.testing.assert_allclose(graph.edges.walk_distance[connectors], lengths)
    np.testing.assert_allclose(graph.edges.walk_time[connectors], graph.edges.walk_distance[connectors] / 1.25)
    # A path through the street network is never shorter than the straight line
    assert np.all(graph.edges.walk_distance[connectors] >= straight.walk_distance[connectors] - 1e-6)
    pd.testing.assert_frame_equal(graph.edges.loc[~connectors, straight.columns], straight[~connectors])


@pytest.mark.parametrize("blocking", [False, True])
@pytest.mark.parametrize("method", CONNECTOR_METHODS)
def test_every_zone_gets_a_connector(coquimbo_example, method, blocking):
    """Test that every zone connects to a stop, and that egress connectors mirror access into destination vertices."""
    graph = build(coquimbo_example, connector_method=method, blocking_centroid_flows=blocking)
    node_type = graph.vertices.set_index("node_id").node_type
    access, egress = edges_of(graph, "access_connector"), edges_of(graph, "egress_connector")

    zones = graph.vertices.node_id[graph.vertices.node_type == ("origin" if blocking else "od")]
    stops = graph.vertices.node_id[graph.vertices.node_type == "stop"]
    assert set(access.a_node) == set(zones)
    assert set(access.b_node) <= set(stops)
    if method == "nearest_neighbour":
        assert set(access.b_node) == set(stops)

    assert len(access) == len(egress)
    assert set(node_type.loc[egress.a_node]) == {"stop"}
    assert set(node_type.loc[egress.b_node]) == {"destination" if blocking else "od"}
    origins = egress.b_node
    if blocking:
        origins = graph.od_node_mapping.set_index("d_node_id").o_node_id.loc[egress.b_node].to_numpy()
    assert set(zip(origins, egress.a_node, strict=True)) == connector_pairs(graph)


def test_connector_method_exception(coquimbo_example):
    """Test that an unknown connector method is rejected."""
    data = Transit(coquimbo_example)
    with pytest.raises(ValueError):
        data.create_graph(
            with_outer_stop_transfers=False,
            with_walking_edges=False,
            blocking_centroid_flows=False,
            connector_method="something not right",
        )


@pytest.mark.parametrize("method", CONNECTOR_METHODS)
def test_distance_upper_bound_filters_connectors(coquimbo_example, method):
    """Test that a finite distance bound keeps only the connectors within it, under either connector method."""
    bound = 400.0
    unbounded = build(coquimbo_example, connector_method=method)
    graph = build(coquimbo_example, connector_method=method, distance_upper_bound=bound)

    access = edges_of(graph, "access_connector")
    assert len(access) > 0
    assert access.walk_distance.max() <= bound
    assert len(access) == len(edges_of(graph, "egress_connector"))

    full = edges_of(unbounded, "access_connector")
    within = set(zip(*full[full.walk_distance <= bound][["a_node", "b_node"]].to_numpy().T, strict=True))
    if method == "nearest_neighbour":
        assert connector_pairs(graph) == within
    else:
        assert connector_pairs(graph) <= within


def test_isolated_zone_does_not_reach_every_stop(coquimbo_example):
    """Test that a single zone under overlapping regions and a distance bound only reaches stops within the bound."""
    bound = 1500.0
    # The zone closest to a stop, alone, has no neighbour to bound its region
    access = edges_of(build(coquimbo_example), "access_connector")
    zone = access.a_node[access.walk_distance.idxmin()]

    builder = TransitGraphBuilder(
        coquimbo_example,
        with_walking_edges=False,
        blocking_centroid_flows=False,
        connector_method="overlapping_regions",
        distance_upper_bound=bound,
    )
    zones = project_zones(coquimbo_example)
    builder.add_zones(zones[zones.zone_id == zone])
    builder.create_graph()

    access = edges_of(builder, "access_connector")
    stops = (builder.vertices.node_type == "stop").sum()
    assert 0 < len(access) < stops
    assert access.walk_distance.max() <= bound


@pytest.mark.parametrize("method", CONNECTOR_METHODS)
def test_max_connectors_per_zone_keeps_the_nearest_stops(coquimbo_example, method):
    """Test that capping connectors per zone keeps each zone's nearest stops among its candidates."""
    full = edges_of(build(coquimbo_example, connector_method=method), "access_connector")
    graph = build(coquimbo_example, connector_method=method, max_connectors_per_zone=2)
    capped = edges_of(graph, "access_connector")

    assert capped.groupby("a_node").size().max() <= 2
    nearest = full.sort_values(["a_node", "walk_distance", "b_node"]).groupby("a_node").head(2)
    assert connector_pairs(graph) == set(zip(nearest.a_node, nearest.b_node, strict=True))
    assert len(edges_of(graph, "egress_connector")) == len(capped)


def test_connectors_are_geodesic(coquimbo_example):
    """Test that connector distances are geodesic and walking times use the builder's walking speed."""
    graph = build(coquimbo_example, walking_speed=1.4)
    points = graph.vertices.set_index("node_id").geometry
    access = edges_of(graph, "access_connector")
    tails = shapely.get_coordinates(shapely.from_wkb(points.loc[access.a_node].to_numpy()))
    heads = shapely.get_coordinates(shapely.from_wkb(points.loc[access.b_node].to_numpy()))
    distance = haversine(tails[:, 0], tails[:, 1], heads[:, 0], heads[:, 1])
    np.testing.assert_allclose(access.walk_distance, distance)
    np.testing.assert_allclose(access.walk_time, distance / 1.4)


def test_unordered_zone_ids_with_gaps(coquimbo_example):
    """Test that zones with gaps in their IDs, added out of order, build a graph in zone order with geometry."""
    zones = project_zones(coquimbo_example)
    zones["zone_id"] = zones.zone_id * 3 + 7
    zones = zones.sample(frac=1, random_state=4).reset_index(drop=True)

    builder = TransitGraphBuilder(coquimbo_example, with_walking_edges=False)
    builder.add_zones(zones)
    builder.create_graph()
    builder.create_line_geometry()

    mapping = builder.od_node_mapping
    assert mapping.taz_id.tolist() == sorted(zones.zone_id)
    assert (mapping.o_node_id == mapping.taz_id).all()

    # Each zone's vertices sit on the centroid of that zone, not of whichever zone shares its row
    points = builder.vertices.set_index("node_id").geometry
    expected = shapely.centroid(shapely.MultiPolygon(zones.set_index("zone_id").geometry.loc[mapping.taz_id[0]]))
    for column in ("o_node_id", "d_node_id"):
        actual = shapely.from_wkb(points.loc[mapping[column][0]])
        assert actual.distance(expected) < 1e-6

    assert_lines_run_tail_to_head(builder)
    graph = builder.to_transit_graph()
    np.testing.assert_array_equal(graph.centroids, sorted(zones.zone_id))


def test_projected_crs_defaults_to_the_stops_utm_zone(coquimbo_example):
    """Test that without a projected CRS the builder uses and records the UTM zone of the stops' centroid."""
    graph = build(coquimbo_example)
    # Coquimbo is at about 71.3 degrees west and 29.9 degrees south
    assert graph.projected_crs == "EPSG:32719"
    assert graph.config["projected_crs"] == "EPSG:32719"

    explicit = build(coquimbo_example, projected_crs="EPSG:3857")
    assert explicit.config["projected_crs"] == "EPSG:3857"


def test_config_rebuilds_the_saved_graph(build_gtfs_project):
    """Test that a saved configuration with non-default settings rebuilds the same graph on loading."""
    data = build_gtfs_project
    settings = {
        "time_margin": 300,
        "walking_speed": 1.3,
        "trip_inclusion": "midpoint",
        "distance_upper_bound": 2000.0,
        "max_connectors_per_zone": 3,
        "connector_method": "overlapping_regions",
    }
    graph = data.create_graph(with_walking_edges=False, **settings)
    data.save_graphs()

    with data.project.db_connection as conn:
        saved = json.loads(conn.execute("SELECT config FROM transit_graph_configs WHERE period_id=1").fetchone()[0])
    assert saved == graph.config
    assert {key: saved[key] for key in settings} == settings

    loaded = TransitGraphBuilder.from_db(data.project, 1)
    rebuilt = TransitGraphBuilder(data.project, **saved)
    rebuilt.create_graph()
    assert loaded.config == graph.config == rebuilt.config
    columns = graph.edges.columns.drop("geometry", errors="ignore")
    pd.testing.assert_frame_equal(loaded.edges[columns], graph.edges[columns])
    pd.testing.assert_frame_equal(rebuilt.edges[columns], graph.edges[columns])


@pytest.mark.parametrize(
    "attribute,value",
    [
        ("alighting_penalty", 0.0),
        ("wait_time_factor", 0.5),
        ("walk_time_factor", 1.2),
        ("access_time_factor", 1.5),
        ("egress_time_factor", 1.5),
        ("uniform_dwell_time", 10.0),
        ("a_tiny_time_duration", 1.0e-6),
    ],
)
def test_cost_attributes_of_the_builder_are_deprecated(coquimbo_example, attribute, value):
    """Test that setting a builder cost attribute that moved to TransitCosting warns."""
    builder = TransitGraphBuilder(coquimbo_example, with_walking_edges=False)
    with pytest.warns(DeprecationWarning, match=attribute):
        setattr(builder, attribute, value)


def test_legacy_cost_attributes_reach_the_graph_costing(coquimbo_example):
    """Test that deprecated cost attributes still become the default costing of the builder's graphs."""
    builder = build(coquimbo_example)
    with pytest.warns(DeprecationWarning):
        builder.wait_time_factor = 0.5
        builder.alighting_penalty = 60.0
        builder.walk_time_factor = 2.0
    costing = builder.to_transit_graph().costing
    assert (costing.headway_fraction, costing.alighting_penalty, costing.walk_weight) == (0.5, 60.0, 2.0)
    assert builder.wait_time_factor == 0.5
    assert build(coquimbo_example).to_transit_graph().costing is None


def test_inner_stop_transfers_are_deprecated(coquimbo_example):
    """Test that asking for inner stop transfers warns and adds no edges, since transfers pass through the stop."""
    with pytest.warns(DeprecationWarning, match="with_inner_stop_transfers"):
        graph = build(coquimbo_example, with_inner_stop_transfers=True)
    assert not graph.edges.link_type.isin(["inner_transfer"]).any()


def test_saving_loading_removing(build_gtfs_project):
    """Test that graphs save, load and remove per period."""
    data = build_gtfs_project
    # create and save
    graph1 = data.create_graph(
        with_outer_stop_transfers=False,
        with_walking_edges=False,
        blocking_centroid_flows=False,
        connector_method="nearest_neighbour",
    )
    # reloading transit graph
    data.save_graphs()
    data.load()
    graph2 = data.graphs[1]
    pd.testing.assert_frame_equal(graph1.edges, graph2.edges)
    pd.testing.assert_frame_equal(graph1.vertices, graph2.vertices)
    assert graph1.config == graph2.config
    # cannot override existing graph
    with pytest.raises(ValueError):
        data.save_graphs()
    # removing transit graph
    data.save_graphs(force=True)
    data.remove_graphs([1])

    with data.project.transit_connection as pt_con:
        links = pt_con.execute("SELECT link_id FROM links LIMIT 1;").fetchall()
        nodes = pt_con.execute("SELECT node_id FROM nodes LIMIT 1;").fetchall()

    assert links == []
    assert nodes == []

    with pytest.raises(ValueError):
        data.load([1])

    # save multiple transit graph
    graph = data.graphs[1]
    for i in range(10, 13):
        data.project.network.periods.insert(period_id=i, period_start=0, period_end=0, period_description="")
        graph.period_id = i
        graph.save()

    with data.project.transit_connection as pt_con:
        for i in range(10, 13):
            links = pt_con.execute("SELECT link_id FROM links WHERE period_id=? LIMIT 1;", (i,))
            nodes = pt_con.execute("SELECT node_id FROM nodes WHERE period_id=? LIMIT 1;", (i,))
            assert len(links.fetchall()) == 1
            assert len(nodes.fetchall()) == 1

    data.load([10, 11, 12])
    assert list(data.graphs.keys()) == [1, 10, 11, 12]
    # remove multiple transit graph
    data.remove_graphs([10, 11, 12], unload=True)
    data.load()
    assert list(data.graphs.keys()) == [1]
