import numpy as np
import pytest

from aequilibrae import TrafficAssignment, TrafficClass, NetworkData, Graph, Project, AequilibraeMatrix
from aequilibrae.paths.vdf import bpr
from aequilibrae.transit import Transit


@pytest.fixture(scope="function")
def transit(coquimbo_example: Project):
    return Transit(coquimbo_example)


@pytest.fixture(scope="function")
def graph(coquimbo_example: Project):
    network = coquimbo_example.network.build(fields=["distance"], modes=["c"])
    network.links["capacity"] = 500.0
    network.links["travel_time"] = network.links.distance / 50
    g = network.build_graph(mode="c", cost="travel_time", skimming=["travel_time"], block_centroid_flows=False)
    return g


@pytest.fixture(scope="function")
def demand(graph):
    n_zones = graph.centroids.shape[0]
    matrix = AequilibraeMatrix(index=graph.centroids)
    matrix["car"] = np.full((n_zones, n_zones), 5.0)  # 5 trips per OD cause a little congestion

    yield matrix


def _assignment(graph: Graph, demand: AequilibraeMatrix, preload=None) -> TrafficAssignment:
    links = graph.network.assign(capacity=500.0, travel_time=graph.network.distance / 50)
    network = NetworkData(links, centroids=graph.centroids, modes=["c"])
    traffic = TrafficClass(
        name="car", mode="c", cost="travel_time", demand=demand, skimming=["travel_time"], block_centroid_flows=False
    )
    return TrafficAssignment(
        network=network,
        classes=[traffic],
        vdf=bpr,
        algorithm="msa",
        time_field="travel_time",
        capacity_field="capacity",
        name_mapping={"alpha": 0.15, "beta": 4.0},
        preload=preload,
    )


def hr_to_sec(e):
    return int(e * 60 * 60)


def calc_preload(transit: Transit, start, end):
    return transit.build_pt_preload(hr_to_sec(start), hr_to_sec(end), inclusion_cond="start")


def test_building_pt_preload(graph: Graph, demand: AequilibraeMatrix, transit: Transit):
    """
    Check that building pt preload works correctly for a basic example from
    the coquimbo network.
    """
    preloads = [calc_preload(transit, start, end) for start, end in [(7, 8), (6.5, 8.5), (5, 10)]]

    # Check preloads increase in size as time period increases
    assert preloads[0]["preload"].sum() == 12484
    assert preloads[1]["preload"].sum() == 21264
    assert preloads[2]["preload"].sum() == 39696

    # the preload returned should be only for links which have transit routes on them
    assert len(preloads[0]) < len(graph.graph)

    assignment = _assignment(graph, demand, preload=preloads[0])
    assignment.execute(max_iterations=1, cores=1)

    # After adding the preload to the assignment object it should be expanded to cover ALL links
    assert len(assignment.assignment.preload) == len(graph.graph)


def test_run(graph: Graph, demand: AequilibraeMatrix, transit: Transit):
    """Tests a full run through of pt preloading."""

    preload = calc_preload(transit, 7, 8)

    # Run non-preloaded assignment and get results
    without_assig = _assignment(graph, demand)
    without_assig.execute(max_iterations=1, cores=1)
    without_res = without_assig.results()

    # Run preloaded assignment and get results
    with_assig = _assignment(graph, demand, preload=preload)
    with_assig.execute(max_iterations=1, cores=1)
    with_res = with_assig.results()

    # Check that average delay increases (ie the preload has reduced speeds)
    assert with_res["Delay_factor_AB"].mean() > without_res["Delay_factor_AB"].mean()
