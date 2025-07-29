import pandas as pd
import pytest

from aequilibrae.transit import Transit


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
    data = Transit(coquimbo_example)
    coquimbo_example.network.build_graphs()
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


def test_connector_methods(coquimbo_example):
    data = Transit(coquimbo_example)
    # nearest_neighbour
    graph = data.create_graph(
        with_outer_stop_transfers=False,
        with_walking_edges=False,
        blocking_centroid_flows=False,
        connector_method="nearest_neighbour",
    )
    nn_count = len(graph.edges[graph.edges.link_type == "access_connector"])
    assert nn_count == len(graph.edges[graph.edges.link_type == "egress_connector"])
    assert nn_count == len(graph.vertices[graph.vertices.node_type == "stop"])
    # overlapping_regions
    graph = data.create_graph(
        with_outer_stop_transfers=False,
        with_walking_edges=False,
        blocking_centroid_flows=False,
        connector_method="overlapping_regions",
    )
    assert nn_count <= len(graph.edges[graph.edges.link_type == "access_connector"])
    assert len(graph.edges[graph.edges.link_type == "access_connector"]) == len(
        graph.edges[graph.edges.link_type == "egress_connector"]
    )


def test_connector_method_exception(coquimbo_example):
    data = Transit(coquimbo_example)
    with pytest.raises(ValueError):
        data.create_graph(
            with_outer_stop_transfers=False,
            with_walking_edges=False,
            blocking_centroid_flows=False,
            connector_method="something not right",
        )


def test_connector_method_without_missing(coquimbo_example):
    data = Transit(coquimbo_example)
    # nearest_neighbour
    graph = data.create_graph(
        with_outer_stop_transfers=False,
        with_walking_edges=False,
        blocking_centroid_flows=False,
        connector_method="nearest_neighbour",
    )
    nn_count = len(graph.edges[graph.edges.link_type == "access_connector"])
    assert nn_count == len(graph.edges[graph.edges.link_type == "egress_connector"])
    assert nn_count == len(graph.vertices[graph.vertices.node_type == "stop"])
    # overlapping_regions
    graph = data.create_graph(
        with_outer_stop_transfers=False,
        with_walking_edges=False,
        blocking_centroid_flows=False,
        connector_method="overlapping_regions",
    )
    assert nn_count <= len(graph.edges[graph.edges.link_type == "access_connector"])
    assert len(graph.edges[graph.edges.link_type == "access_connector"]) == len(
        graph.edges[graph.edges.link_type == "egress_connector"]
    )


def test_saving_loading_removing(coquimbo_example):
    data = Transit(coquimbo_example)
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
        data.periods.new_period(i, 0, 0).save()
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
