"""Transit assignment through TransitAssignment and TransitClass: costing, skims, provenance and strategies."""

import json

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import MultiPolygon, box

from aequilibrae.matrix import AequilibraeMatrix
from aequilibrae.paths import NewTransitGraph, TransitAssignment, TransitClass, TransitCosting, TransitGraph
from aequilibrae.transit import Transit
from aequilibrae.transit.transit_graph_builder import TransitGraphBuilder, config_hash

from ..transit.synthetic_gtfs import SERVICE_DATE, add_stop, add_timetable, base_tables, write_feed
from ..transit.test_spiess_florian_reference import AM_PEAK, LAT, LINES, PAPER_COSTING, STOPS, reference_builder

# Every predefined skim, so each one is checked for unreachable pairs
ALL_SKIMS = ["trav_time", "boardings", "alightings", "transfers", "pt_probability", "in_vehicle_trav_time"]
ALL_SKIMS += ["waiting_time", "first_waiting_time", "transfer_waiting_time", "access_trav_time", "egress_trav_time"]


@pytest.fixture
def sf_project(empty_project, tmp_path):
    """The Spiess & Florian (1989) network, imported from a synthetic feed, with a zone at each of A and B."""
    tables = base_tables("SF")
    for stop, lon in STOPS.items():
        add_stop(tables, stop, LAT, lon)
    for route, (stops, run_minutes, headway) in LINES.items():
        add_timetable(tables, route, stops, run_minutes, headway, first=5 * 3600, last=11 * 3600)
    feed = write_feed(tmp_path / "spiess_florian.zip", tables)

    for zone_id, stop in ((1, "A"), (2, "B")):
        empty_project.network.zones.insert(zone_id=zone_id, geometry=zone_box(STOPS[stop]))
    with empty_project.db_connection as conn:
        conn.execute(
            "INSERT INTO periods (period_id, period_start, period_end, period_description) VALUES (?, ?, ?, ?)",
            [AM_PEAK, 7 * 3600, 8 * 3600, "AM peak"],
        )

    gtfs = empty_project.scenario.create_transit_database().new_gtfs_builder(agency="SF", file_path=str(feed))
    gtfs.load_date(SERVICE_DATE)
    gtfs.save_to_disk()
    return empty_project


def zone_box(lon):
    return MultiPolygon([box(lon - 0.002, LAT - 0.002, lon + 0.002, LAT + 0.002)])


@pytest.fixture
def coquimbo_builder(coquimbo_example):
    return Transit(coquimbo_example).create_graph(with_walking_edges=False)


def demand_matrix(graph, values=None, name="pt"):
    matrix = AequilibraeMatrix(index=graph.centroids, index_name="zones")
    matrix[name] = np.ones((graph.num_zones, graph.num_zones)) if values is None else np.asarray(values, float)
    return matrix


def assign(classes, skims=("trav_time",), cores=None, project=None):
    assignment = TransitAssignment(project=project)
    for transit_class in classes:
        assignment.add_class(transit_class)
    assignment.set_time_field("trav_time")
    assignment.set_frequency_field("freq")
    if skims:
        assignment.set_skimming_fields(list(skims))
    assignment.set_algorithm("os")
    if cores is not None:
        assignment.set_cores(cores)
    assignment.execute()
    return assignment


def test_save_skims_saves_one_matrix_file_per_class_with_provenance(coquimbo_example, coquimbo_builder):
    """Test that save_skims saves a matrix file per class whose metadata records the graph and costing."""
    graph = coquimbo_builder.to_transit_graph()
    costing = TransitCosting(wait_weight=2.0)
    pt = TransitClass("pt", graph, demand_matrix(graph), costing=costing)
    assignment = assign([pt], skims=["trav_time", "boardings"], project=coquimbo_example)

    assert assignment.save_skims("am_skims", project=coquimbo_example) == ["am_skims_pt.omx"]

    with coquimbo_example.matrices.get_matrix("am_skims_pt.omx") as saved:
        assert set(saved) == {"trav_time", "boardings"}
        np.testing.assert_array_equal(saved.index, graph.centroids)
        np.testing.assert_array_equal(saved["trav_time"], pt.results.skims["trav_time"])
        provenance = json.loads(saved.metadata["provenance"])

    assert provenance["scenario"] == "root"
    assert provenance["period_id"] == 1
    assert provenance["graph_config_hash"] == config_hash(coquimbo_builder.config)
    assert provenance["costing"] == costing.to_dict()

    with pytest.raises(FileExistsError):
        assignment.save_skims("am_skims", project=coquimbo_example)


def test_save_skims_skips_classes_without_skims(coquimbo_example, coquimbo_builder):
    """Test that save_skims writes nothing for an assignment that computed no skims."""
    graph = coquimbo_builder.to_transit_graph()
    assignment = assign([TransitClass("pt", graph, demand_matrix(graph))], skims=None, project=coquimbo_example)
    assert assignment.save_skims("no_skims", project=coquimbo_example) == []
    assert not coquimbo_example.matrices.file_exists("no_skims_pt.omx")


def test_save_results_records_provenance_and_period(coquimbo_example, coquimbo_builder):
    """Test that saved loads carry their period and a procedure report with each class's provenance."""
    graph = coquimbo_builder.to_transit_graph()
    pt = TransitClass("pt", graph, demand_matrix(graph))
    assignment = assign([pt], project=coquimbo_example)

    results = assignment.results()
    assert (results.period_id == 1).all()
    assert results.pt_volume.sum() > 0

    assignment.save_results("pt_loads", keep_zero_flows=False, project=coquimbo_example)
    saved = coquimbo_example.results.get_results("pt_loads")
    assert len(saved) == (results.pt_volume > 0).sum()
    assert (saved.period_id == 1).all()

    record = coquimbo_example.results.list().set_index("table_name").loc["pt_loads"]
    assert record.scenario == "root"
    provenance = json.loads(record.procedure_report)["setup"]["Classes"]["pt"]["provenance"]
    assert provenance["period_id"] == 1
    assert provenance["graph_config_hash"] == config_hash(coquimbo_builder.config)
    assert provenance["costing"] == TransitCosting().to_dict()


def test_strategy_lists_the_edges_of_one_trip(sf_project):
    """Test that a zone pair's strategy has the edges, volumes and splits of assigning one trip between them."""
    builder = reference_builder(sf_project)
    graph = builder.to_transit_graph()
    pt = TransitClass("pt", graph, demand_matrix(graph, [[0, 1], [0, 0]]), costing=PAPER_COSTING)
    assign([pt], skims=None)
    loads = pt.results.get_load_results().volume

    strategy = pt.strategy(1, 2)
    assert not isinstance(strategy, gpd.GeoDataFrame)
    assert set(strategy.link_id) == set(loads.index[loads > 1e-12])
    np.testing.assert_allclose(strategy.set_index("link_id").volume, loads.loc[strategy.link_id])
    np.testing.assert_allclose(strategy.groupby("a_node").split.sum(), 1.0)

    # Half the trips board each of lines 1 and 2 at A
    boardings = strategy[strategy.link_type == "boarding"].set_index("line_id").split
    assert boardings.filter(like="1_").item() == pytest.approx(0.5)
    assert boardings.filter(like="2_").iloc[0] == pytest.approx(0.5)


def test_strategy_is_a_geodataframe_when_edges_have_geometry(sf_project):
    """Test that a strategy of a graph with edge geometry is a GeoDataFrame in EPSG:4326, drawn tail to head."""
    builder = reference_builder(sf_project)
    builder.create_line_geometry()
    graph = builder.to_transit_graph()
    pt = TransitClass("pt", graph, demand_matrix(graph), costing=PAPER_COSTING)

    strategy = pt.strategy(1, 2)
    assert isinstance(strategy, gpd.GeoDataFrame)
    assert strategy.crs.to_epsg() == 4326
    points = gpd.GeoSeries.from_wkb(builder.vertices.geometry.to_numpy(), index=builder.vertices.node_id.to_numpy())
    starts = [line.coords[0] for line in strategy.geometry]
    assert starts == [point.coords[0] for point in points.loc[strategy.a_node]]


def test_strategy_rejects_zones_outside_the_graph(sf_project):
    """Test that asking for the strategy of a zone the graph does not have raises."""
    graph = reference_builder(sf_project).to_transit_graph()
    pt = TransitClass("pt", graph, demand_matrix(graph), costing=PAPER_COSTING)
    with pytest.raises(ValueError, match="zone 3"):
        pt.strategy(1, 3)


def test_classes_sharing_a_graph_keep_their_own_costing(sf_project):
    """Test that two classes on one graph assign as each would alone, though their costings differ."""
    # The alighting penalty makes transferring at Y cost more than waiting for line 1 alone saves
    paper = TransitCosting(headway_fraction=0.5, alighting_penalty=0.0)
    penalised = TransitCosting(alighting_penalty=480.0)
    skims = ["trav_time", "waiting_time", "boardings"]

    def alone(costing):
        graph = reference_builder(sf_project).to_transit_graph()
        transit_class = TransitClass("pt", graph, demand_matrix(graph, [[0, 1], [0, 0]]), costing=costing)
        assign([transit_class], skims=skims)
        return transit_class.results

    graph = reference_builder(sf_project).to_transit_graph()
    first = TransitClass("first", graph, demand_matrix(graph, [[0, 1], [0, 0]]), costing=paper)
    second = TransitClass("second", graph, demand_matrix(graph, [[0, 1], [0, 0]]), costing=penalised)
    assign([first, second], skims=skims)

    for shared, costing in ((first.results, paper), (second.results, penalised)):
        expected = alone(costing)
        np.testing.assert_array_equal(shared.link_loads, expected.link_loads)
        for name in skims:
            np.testing.assert_array_equal(shared.skims[name], expected.skims[name])
    assert not np.array_equal(first.results.link_loads, second.results.link_loads)


def test_zones_added_out_of_order_load_at_their_own_vertices(sf_project):
    """Test that zones given out of zone_id order still load demand, and skim, at their own vertices."""
    builder = TransitGraphBuilder(sf_project, period_id=AM_PEAK, with_walking_edges=False)

    # Zone 20 is at B and zone 10 at A, listed with B first
    zones = pd.DataFrame({"zone_id": [20, 10], "geometry": [zone_box(STOPS["B"]), zone_box(STOPS["A"])]})
    builder.add_zones(zones)
    builder.create_graph()
    builder.create_line_geometry()
    graph = builder.to_transit_graph()
    np.testing.assert_array_equal(graph.centroids, [10, 20])

    pt = TransitClass("pt", graph, demand_matrix(graph, [[0, 1], [0, 0]]), costing=PAPER_COSTING)
    assign([pt], skims=["trav_time"])

    assert pt.results.unassigned_demand == 0.0
    assert pt.results.skims["trav_time"][0, 1] == pytest.approx(27.75 * 60, abs=0.01)
    assert np.isinf(pt.results.skims["trav_time"][1, 0])

    access = builder.edges[builder.edges.link_type == "access_connector"].set_index("link_id")
    loads = pt.results.get_load_results().volume
    node_type = builder.vertices.set_index("node_id").taz_id
    assert loads.loc[access.index[node_type.loc[access.a_node].to_numpy() == 10]].sum() == pytest.approx(1.0)


def test_unreachable_pairs_are_reported_and_skim_as_infinite(sf_project):
    """Test that demand no strategy serves is reported by zone pair, and its skims are inf rather than zero."""
    graph = reference_builder(sf_project).to_transit_graph()
    pt = TransitClass("pt", graph, demand_matrix(graph, [[0, 2], [3, 0]]), costing=PAPER_COSTING)
    assign([pt], skims=ALL_SKIMS)

    assert pt.results.unassigned_demand == pytest.approx(3.0)
    assert pt.results.unassigned_od.to_dict("records") == [{"origin": 2, "destination": 1, "demand": 3.0}]
    for name in ALL_SKIMS:
        assert np.isinf(pt.results.skims[name][1, 0]), name
        assert np.isfinite(pt.results.skims[name][0, 1]), name


def test_costs_and_waits_add_up_to_the_cost_of_loaded_demand(sf_project):
    """Test that edge costs times volumes plus weighted waits equal loaded demand times its expected cost."""
    graph = reference_builder(sf_project).to_transit_graph()
    costing = TransitCosting(headway_fraction=0.5, wait_weight=1.7, alighting_penalty=60.0)
    demand = np.array([[0.0, 2.0], [3.0, 0.0]])
    pt = TransitClass("pt", graph, demand_matrix(graph, demand), costing=costing)
    assign([pt], skims=["trav_time"])

    cost, _ = pt.edge_costs()
    loaded = np.maximum(cost, graph.context.cost_floor) @ pt.results.link_loads + pt.results.weighted_waiting_time
    skim = pt.results.skims["trav_time"]
    reachable = np.isfinite(skim)
    assert loaded == pytest.approx((demand[reachable] * skim[reachable]).sum(), rel=1e-12)


def test_costing_rejects_negative_and_non_finite_values():
    """Test that a costing with a negative or non-finite parameter cannot be created."""
    for value in (-1.0, np.nan, np.inf):
        with pytest.raises(ValueError, match="finite and nonnegative"):
            TransitCosting(headway_fraction=value)
        with pytest.raises(ValueError, match="finite and nonnegative"):
            TransitCosting(alighting_penalty=value)


def test_costing_rejects_unknown_link_types(sf_project):
    """Test that costing edges of a link type the costing does not know raises instead of costing them zero."""
    edges = reference_builder(sf_project).edges.copy()
    edges.loc[0, "link_type"] = "teleport"
    with pytest.raises(ValueError, match="teleport"):
        TransitCosting().edge_costs(edges)


def test_legacy_builder_cost_attributes_reach_the_class(sf_project):
    """Test that the deprecated builder cost attributes warn and still set the costing a class defaults to."""
    builder = reference_builder(sf_project)
    with pytest.warns(DeprecationWarning, match="alighting_penalty"):
        builder.alighting_penalty = 0.0
    with pytest.warns(DeprecationWarning, match="headway_fraction"):
        builder.wait_time_factor = 0.5
    with pytest.warns(DeprecationWarning, match="no effect"):
        builder.uniform_dwell_time = 30

    graph = builder.to_transit_graph()
    legacy = TransitClass("pt", graph, demand_matrix(graph, [[0, 1], [0, 0]]))
    assert legacy.costing == PAPER_COSTING
    assign([legacy], skims=["trav_time"])
    assert legacy.results.skims["trav_time"][0, 1] == pytest.approx(27.75 * 60, abs=0.01)


def test_new_transit_graph_alias_warns(sf_project):
    """Test that the deprecated NewTransitGraph name still builds a TransitGraph, with a warning."""
    builder = reference_builder(sf_project)
    with pytest.warns(DeprecationWarning, match="NewTransitGraph"):
        graph = NewTransitGraph(
            network=builder.edges,
            centroids=builder.od_node_mapping.o_node_id.to_numpy(),
            od_node_mapping=builder.od_node_mapping,
        )
    assert isinstance(graph, TransitGraph)


def test_time_field_accepts_zero_costs_but_not_negative_ones(sf_project):
    """Test that set_time_field allows free edges, which costings produce, and rejects negative costs."""
    graph = reference_builder(sf_project).to_transit_graph()
    pt = TransitClass("pt", graph, demand_matrix(graph), costing=PAPER_COSTING)
    assert (graph.graph.trav_time == 0).any()

    assignment = TransitAssignment()
    assignment.add_class(pt)
    assignment.set_time_field("trav_time")

    graph.graph["negative"] = graph.graph.trav_time
    graph.graph.loc[0, "negative"] = -1.0
    with pytest.raises(ValueError, match="negative"):
        assignment.set_time_field("negative")


def test_assignment_is_deterministic_for_a_thread_count(coquimbo_builder):
    """Test that a thread count gives bitwise-identical results, and thread counts agree within 1e-12."""
    graph = coquimbo_builder.to_transit_graph()
    skims = ["trav_time", "waiting_time", "boardings"]

    def run(cores):
        pt = TransitClass("pt", graph, demand_matrix(graph))
        assign([pt], skims=skims, cores=cores)
        return pt.results.link_loads.copy(), {name: pt.results.skims[name].copy() for name in skims}

    by_cores = {cores: run(cores) for cores in (1, 2, 4)}
    loads, skimmed = run(2)
    np.testing.assert_array_equal(loads, by_cores[2][0])
    for name in skims:
        np.testing.assert_array_equal(skimmed[name], by_cores[2][1][name])

    for cores in (2, 4):
        np.testing.assert_allclose(by_cores[cores][0], by_cores[1][0], rtol=1e-12, atol=1e-12)
        for name in skims:
            np.testing.assert_allclose(by_cores[cores][1][name], by_cores[1][1][name], rtol=1e-12)
