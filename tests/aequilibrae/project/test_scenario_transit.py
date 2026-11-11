"""Transit data belongs to the scenario a ``Transit`` or graph builder was created for, not the active one."""

import pandas as pd
import pytest

from aequilibrae.transit import Transit, TransitGraphBuilder

COQUIMBO_DATE = "2016-04-13"


def trip_count(scenario) -> int:
    with scenario.transit_connection as conn:
        return conn.execute("SELECT COUNT(*) FROM trips").fetchone()[0]


@pytest.fixture
def root_transit(scenario_example) -> Transit:
    """Root scenario transit gateway, created while root is active, then left with the coquimbo scenario active."""
    scenario_example.use_scenario("root")
    transit = scenario_example.scenario.create_transit_database()
    scenario_example.use_scenario("coquimbo")
    return transit


@pytest.fixture
def root_with_feed(scenario_example, root_transit) -> Transit:
    """Root scenario transit gateway holding the Coquimbo feed, imported while the coquimbo scenario is active."""
    feed = scenario_example.root_scenario.base_path / "scenarios" / "coquimbo" / "gtfs_coquimbo.zip"
    gtfs = root_transit.new_gtfs_builder(agency="LISERCO", file_path=str(feed))
    gtfs.load_date(COQUIMBO_DATE)
    gtfs.save_to_disk()
    return root_transit


def test_transit_reads_its_own_scenario_after_another_is_activated(scenario_example, root_transit):
    """Test that a Transit made for root reads root's empty feed tables while the coquimbo scenario is active."""
    assert scenario_example.scenario.name == "coquimbo"
    assert root_transit.scenario is scenario_example.root_scenario
    assert root_transit.get_table("stops").empty
    assert not scenario_example.transit.get_table("stops").empty


def test_gtfs_import_lands_in_the_scenario_of_its_transit(scenario_example, root_with_feed):
    """Test that a feed imported through root's Transit while coquimbo is active is saved to root only."""
    assert trip_count(scenario_example.root_scenario) == trip_count(scenario_example.scenario)

    # Importing again into root leaves the active scenario alone
    before = trip_count(scenario_example.scenario)
    feed = scenario_example.root_scenario.base_path / "scenarios" / "coquimbo" / "gtfs_coquimbo.zip"
    gtfs = root_with_feed.new_gtfs_builder(agency="LISANCO", file_path=str(feed))
    gtfs.load_date(COQUIMBO_DATE)
    gtfs.save_to_disk()
    assert trip_count(scenario_example.scenario) == before
    assert trip_count(scenario_example.root_scenario) == 2 * before


def test_transit_made_from_a_project_binds_to_its_active_scenario(scenario_example, root_transit):
    """Test that Transit(project) keeps the scenario active when it was made, whichever is activated later."""
    transit = Transit(scenario_example)
    assert transit.scenario is scenario_example.scenario
    assert transit.scenario.name == "coquimbo"
    assert not transit.get_table("stops").empty


def test_transit_graphs_belong_to_each_instance(scenario_example, root_transit):
    """Test that the graphs one Transit builds are not seen by another Transit."""
    coquimbo = scenario_example.transit
    coquimbo.create_graph(with_walking_edges=False)
    assert list(coquimbo.graphs) == [1]
    assert root_transit.graphs == {}
    assert Transit(scenario_example).graphs == {}


def test_graph_builder_binds_to_the_scenario_active_at_construction(scenario_example, root_with_feed):
    """Test that a builder made while root is active builds root's graph after coquimbo is activated."""
    scenario_example.use_scenario("root")
    builder = TransitGraphBuilder(scenario_example, with_walking_edges=False)
    root_zones = len(scenario_example.network.zones)
    scenario_example.use_scenario("coquimbo")
    assert root_zones != len(scenario_example.network.zones)

    builder.create_graph()
    assert builder.scenario is scenario_example.root_scenario
    assert len(builder.od_node_mapping) == root_zones


def test_saved_graphs_stay_in_their_scenario(scenario_example, root_with_feed):
    """Test that a graph saved through root's Transit while coquimbo is active is stored in root only."""
    root_with_feed.create_graph(with_walking_edges=False)
    root_with_feed.save_graphs()

    with scenario_example.root_scenario.transit_connection as conn:
        assert conn.execute("SELECT COUNT(*) FROM links").fetchone()[0] > 0
    with scenario_example.scenario.transit_connection as conn:
        assert conn.execute("SELECT COUNT(*) FROM links").fetchone()[0] == 0
    with scenario_example.scenario.db_connection as conn:
        assert conn.execute("SELECT COUNT(*) FROM transit_graph_configs").fetchone()[0] == 0

    loaded = Transit(scenario_example.root_scenario)
    loaded.load()
    assert list(loaded.graphs) == [1]
    pd.testing.assert_frame_equal(
        loaded.graphs[1].edges.drop(columns="geometry").sort_values("link_id").reset_index(drop=True),
        root_with_feed.graphs[1].edges.drop(columns="geometry").sort_values("link_id").reset_index(drop=True),
    )


def test_build_pt_preload_takes_a_period_id(scenario_example):
    """Test that a preload for a period equals the preload for its window, and that both cannot be given at once."""
    scenario_example.use_scenario("coquimbo")
    scenario_example.network.periods.insert(
        period_id=2, period_start=7 * 3600, period_end=8 * 3600, period_description="AM peak"
    )
    transit = scenario_example.transit

    by_period = transit.build_pt_preload(period_id=2)
    by_window = transit.build_pt_preload(7 * 3600, 8 * 3600)
    assert not by_period.empty
    pd.testing.assert_frame_equal(by_period, by_window)

    with pytest.raises(ValueError, match="not both"):
        transit.build_pt_preload(7 * 3600, 8 * 3600, period_id=2)
    with pytest.raises(ValueError, match="period_id, or both start and end"):
        transit.build_pt_preload(7 * 3600)
    with pytest.raises(ValueError, match="does not exist"):
        transit.build_pt_preload(period_id=99)
