import pytest

from aequilibrae import TrafficAssignment, TrafficClass, Graph, Project, AequilibraeMatrix
from aequilibrae.transit import Transit
from aequilibrae.utils.create_example import create_example


@pytest.fixture
def project(tmp_path):
    proj = create_example(str(tmp_path / "test_traffic_assignment"), from_model="coquimbo")
    proj.network.build_graphs()
    proj.activate()
    yield proj
    proj.close()


@pytest.fixture
def transit(project: Project):
    return Transit(project)


@pytest.fixture
def graph(project: Project):
    g = project.network.graphs["c"]
    g.set_skimming(["travel_time"])
    g.set_graph("travel_time")
    g.set_blocked_centroid_flows(False)
    g.graph["capacity"] = 500
    g.graph["travel_time"] = g.graph["distance"] / 50
    return g


@pytest.fixture
def demand(graph):
    n_zones = graph.centroids.shape[0]
    matrix = AequilibraeMatrix()
    matrix.create_empty(zones=n_zones, matrix_names=["car"])
    matrix.index = graph.centroids

    matrix.matrices[:, :, 0] = 5  # 5 trips from each OD should cause a little bit of congestion
    matrix.computational_view("car")

    yield matrix

    matrix.close()


def _assignment(
    graph: Graph,
    demand: AequilibraeMatrix,
) -> TrafficAssignment:

    # Create assignment and set parameters
    assignment = TrafficAssignment()
    assignment.set_classes([TrafficClass("car", graph, demand)])

    # # Note: preload has to be added before we set the assignment algorithm
    # if preload is not None:
    #     assignment.add_preload(preload)

    assignment.set_vdf("BPR")
    assignment.set_vdf_parameters({"alpha": 0.15, "beta": 4.0})
    assignment.set_capacity_field("capacity")
    assignment.set_time_field("travel_time")
    assignment.max_iter = 1  # AON assignment
    assignment.set_algorithm("msa")

    return assignment


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

    assignment = _assignment(graph, demand)
    assignment.add_preload(preloads[0])

    # After adding the preload to the assignment object it should be expanded to cover ALL links
    assert len(assignment.preloads) == len(graph.graph)


def test_run(graph: Graph, demand: AequilibraeMatrix, transit: Transit):
    """Tests a full run through of pt preloading."""

    preload = calc_preload(transit, 7, 8)

    # Run non-preloaded assignment and get results
    without_assig = _assignment(graph, demand)
    without_assig.execute()
    without_res = without_assig.results()

    # Run preloaded assignment and get results
    with_assig = _assignment(graph, demand)
    with_assig.add_preload(preload)
    with_assig.execute()
    with_res = with_assig.results()

    # Check that average delay increases (ie the preload has reduced speeds)
    assert with_res["Delay_factor_AB"].mean() > without_res["Delay_factor_AB"].mean()
