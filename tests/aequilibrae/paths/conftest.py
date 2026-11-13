import pytest

from aequilibrae.paths import RouteChoice


@pytest.fixture(scope="function")
def route_choice_setup(sioux_falls_single_class):
    graph = sioux_falls_single_class.network.build_graph(
        mode="c",
        cost="distance",
        fields=["distance", "free_flow_time"],
        block_centroid_flows=False,
        compact=True,
    )

    mat = sioux_falls_single_class.matrices.get_matrix("demand.omx", subset=["matrix"])

    choice = RouteChoice(graph)
    choice.set_cores(1)
    return {"project": sioux_falls_single_class, "graph": graph, "mat": mat, "rc": choice}
