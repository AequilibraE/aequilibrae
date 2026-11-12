"""Route and search-option validation, including masked imported routes."""

import numpy as np
import pandas as pd
import pytest
from aequilibrae.paths.cython.route_choice_set import RouteChoiceSet
from aequilibrae.utils.cython.bridge import Bridge

from aequilibrae.paths import Network, RouteChoice

from .route_choice_helpers import (
    chain_graph,
    configured_choice,
    coordinates_for,
    diamond,
    mapped_network,
    single_od_demand,
)


@pytest.fixture(params=["psl", "assignment", "assignment-with-psl"])
def import_routes(request):
    """Exercise the three public entry points without repeating validation assertions."""

    def execute(choice, supplied):
        if request.param == "psl":
            return choice.recompute_psl(supplied)
        return choice.execute_from_pandas(supplied, recompute_psl=request.param == "assignment-with-psl")

    return execute


@pytest.mark.parametrize("nodes", [[], ["1", "2"], [("1", "2")], [1]])
def test_invalid_prepare_nodes(route_choice_setup, nodes):
    with pytest.raises(ValueError, match="empty|not valid types"):
        route_choice_setup["rc"].prepare(nodes)


@pytest.mark.parametrize(
    "algorithm,options,error,match",
    [
        ("unknown", {}, AttributeError, "algorithm not available"),
        ("bfsle", {"unknown": 1}, ValueError, "Invalid parameter"),
        ("bfsle", {"disutility_cutoff_constant": 0}, ValueError, "Both must be set or unset"),
    ],
)
def test_invalid_generation_configuration(route_choice_setup, algorithm, options, error, match):
    with pytest.raises(error, match=match):
        route_choice_setup["rc"].set_choice_set_generation(algorithm, **options)


@pytest.mark.parametrize("penalty", [-1.0, 0.0, 0.99, np.nan, np.inf])
def test_invalid_penalty_rejected_at_both_entry_points(penalty):
    graph = diamond()
    options = {"heuristic_scale": 0.0, "penalty": penalty, "max_routes": 1}

    with pytest.raises(ValueError, match="penalty.*finite and >= 1"):
        RouteChoice(graph).set_choice_set_generation("bfsle", **options)

    with Bridge() as bridge, pytest.raises(ValueError, match="penalty.*finite and >= 1"):
        RouteChoiceSet(graph).run(10, 40, bridge=bridge, **options)


@pytest.mark.parametrize(
    "options,match",
    [
        ({"a_star": True}, "explicit heuristic_scale"),
        ({"a_star": True, "heuristic_scale": -1.0}, "finite and nonnegative"),
        ({"a_star": True, "heuristic_scale": np.inf}, "finite and nonnegative"),
        ({"a_star": True, "heuristic_scale": np.nan}, "finite and nonnegative"),
        ({"heuristic_scale": -1.0}, "finite and nonnegative"),
        ({"heuristic": "unknown"}, "heuristic must be one of"),
    ],
)
def test_invalid_search_options_rejected_at_both_entry_points(options, match):
    graph = diamond()

    with pytest.raises(ValueError, match=match):
        RouteChoice(graph).set_choice_set_generation("bfsle", max_routes=1, **options)

    with Bridge() as bridge, pytest.raises(ValueError, match=match):
        RouteChoiceSet(graph).run(10, 40, max_routes=1, bridge=bridge, **options)


@pytest.mark.parametrize(
    "problem,match",
    [
        ("missing", "every graph node ID"),
        ("node", "every graph node ID"),
        ("duplicate", "node IDs must be unique"),
        ("nonfinite", "coordinates must be finite"),
    ],
)
def test_invalid_coordinates(problem, match):
    graph = diamond()
    coordinates = coordinates_for(graph)

    if problem == "missing":
        coordinates = None
    elif problem == "node":
        coordinates = coordinates.drop(20)
    elif problem == "duplicate":
        coordinates = pd.concat([coordinates, coordinates.iloc[:1]])
    else:
        coordinates.loc[20, "x"] = np.nan

    choice = RouteChoice(graph, coordinates=coordinates)
    choice.set_choice_set_generation("bfsle", max_routes=1, a_star=True, heuristic_scale=0.0)
    with pytest.raises(ValueError, match=match):
        choice.execute_single(10, 40)


@pytest.mark.parametrize(
    "origin,destination,route,penalty,reason",
    [
        (20, 40, [71, 12], 0.5, "starts at the wrong node"),
        (10, 30, [71, 12], 0.5, "ends at the wrong node"),
        (10, 40, [71, 12], np.inf, "prohibited turn"),
        (10, 40, [71, 24], 0.5, "disconnected links"),
        (10, 40, [71, 24], None, "disconnected links"),
        (10, 40, [], 0.5, "empty route"),
    ],
)
def test_invalid_routes_are_rejected_even_when_masked(import_routes, origin, destination, route, penalty, reason):
    choice = configured_choice(diamond(penalty))
    choice.set_demand(single_od_demand(origin, destination))
    supplied = pd.DataFrame(
        {
            "origin id": [origin],
            "destination id": [destination],
            "route set": [route],
            "route directions": [[1] * len(route)],
            "cost": [999.0],
            "probability": [1.0],
            "mask": [False],
        },
        index=[9],
    )
    before = supplied.copy(deep=True)

    with pytest.raises(ValueError, match=f"row 9:.*{reason}"):
        import_routes(choice, supplied)

    pd.testing.assert_frame_equal(supplied, before)


@pytest.mark.parametrize("route", [[11], [11, 13], [12, 11, 13], [11, 12]])
def test_partial_expansions_are_rejected(import_routes, route):
    graph = chain_graph()
    assert graph.num_links == 1
    choice = configured_choice(graph)
    choice.set_demand(single_od_demand(10, 40))
    supplied = pd.DataFrame(
        {
            "origin id": [10],
            "destination id": [40],
            "route set": [route],
            "route directions": [[1] * len(route)],
            "probability": [1.0],
            "mask": [False],
        }
    )
    with pytest.raises(ValueError, match="complete ordered graph-link expansions"):
        import_routes(choice, supplied)


def test_imported_link_absent_from_compact_graph_is_rejected():
    links = pd.DataFrame(
        {
            "link_id": [71, 12, 99],
            "a_node": [10, 20, 20],
            "b_node": [20, 40, 50],
            "direction": [0, 0, 0],
            "time": [1.0, 1.0, 1.0],
        }
    )
    graph = Network(links, centroids=[10, 40]).build_graph(
        "c", cost="time", compact=True, remove_dead_ends=True, block_centroid_flows=False
    )
    # The through route survives compression; only the spur to node 50 is removed.
    assert graph.mapping.network_link_to_graph[graph.mapping.network.link_index(99, 1)] == -1
    supplied = pd.DataFrame(
        {
            "origin id": [10],
            "destination id": [40],
            "route set": [[71, 99, 99, 12]],
            "route directions": [[1, 1, -1, 1]],
        }
    )
    with pytest.raises(ValueError, match="absent from the compressed graph"):
        RouteChoiceSet(graph).assign_from_df(supplied, single_od_demand(10, 40), select_links={}, recompute_psl=True)


@pytest.mark.parametrize("link,direction", [(999, 1), (71, -1)])
def test_missing_directed_links_are_errors_even_when_masked(link, direction):
    supplied = pd.DataFrame(
        {
            "origin id": [10],
            "destination id": [40],
            "route set": [[link]],
            "route directions": [[direction]],
            "mask": [False],
        }
    )
    with pytest.raises(ValueError, match="absent from the graph"):
        RouteChoice(diamond()).recompute_psl(supplied)


def test_infinite_link_cost_is_rejected_without_a_turn_ban(import_routes):
    graph = diamond()
    costs = np.ones(graph.mapping.network.link_count)
    costs[graph.mapping.network.link_ids == 71] = np.inf
    graph.set_costs(costs)
    choice = configured_choice(graph)
    choice.set_demand(single_od_demand(10, 40))
    supplied = pd.DataFrame(
        {
            "origin id": [10],
            "destination id": [40],
            "route set": [[71, 12]],
            "route directions": [[1, 1]],
            "probability": [1.0],
        }
    )
    with pytest.raises(ValueError, match="infinite route cost"):
        import_routes(choice, supplied)


def test_import_rejects_intermediate_centroid(import_routes):
    links = pd.DataFrame(
        {
            "link_id": [11, 12, 13],
            "a_node": [10, 20, 10],
            "b_node": [20, 40, 40],
            "direction": [1, 1, 1],
            "time": [1.0, 1.0, 3.0],
        }
    )
    graph = Network(links, centroids=[10, 20, 40]).build_graph(
        "c", cost="time", compact=True, block_centroid_flows=True
    )
    assert not graph.has_turn_restrictions  # Without explicit turns, block the centroid node.
    choice = configured_choice(graph)
    choice.set_demand(single_od_demand(10, 40))
    supplied = pd.DataFrame(
        {
            "origin id": [10],
            "destination id": [40],
            "route set": [[11, 12]],
            "route directions": [[1, 1]],
            "probability": [1.0],
        }
    )
    with pytest.raises(ValueError, match="blocked centroid"):
        import_routes(choice, supplied)


@pytest.mark.parametrize("link,direction", [(999, 0), (75, 0), (66, 0), (0, 2)])
def test_missing_selected_links_raise_consistently(link, direction):
    graph = mapped_network().build_graph("c", "time", compact=True, remove_dead_ends=True)
    with pytest.raises(ValueError):
        graph.selected_links({"selected": [(link, direction)]})
    with pytest.raises(ValueError):
        configured_choice(graph).set_select_links({"selected": [[(link, direction)]]})


@pytest.mark.parametrize("reverse", ["absent", "removed"])
def test_direction_zero_selects_only_available_graph_directions(reverse):
    network = Network(
        pd.DataFrame(
            {
                "link_id": [11, 12],
                "a_node": [10, 20],
                "b_node": [20, 30],
                "direction": [0 if reverse == "removed" else 1, 1],
                "time": 1.0,
            }
        ),
        centroids=[10, 30],
    )

    graph = network.build_graph("c", "time", compact=True)
    if reverse == "removed":
        assert graph.mapping.network_link_to_graph[network.mapping.link_index(11, -1)] == -1

    expected = graph.mapping.graph_link(11, 1)
    np.testing.assert_array_equal(graph.selected_links({"selected": [(11, 0)]})["selected"], [expected])
    with pytest.raises(ValueError, match="absent from the graph"):
        graph.selected_links({"selected": [(11, -1)]})

    choice = configured_choice(graph)
    with pytest.warns(UserWarning, match="OR alternatives"):
        choice.set_select_links({"selected": [(11, 0)]})

    choice.set_demand(single_od_demand(10, 30, flow=2.0))
    choice.execute()
    np.testing.assert_allclose(choice.get_select_link_loading_results()["flow_selected_tot"], [2.0, 2.0])


@pytest.mark.parametrize(
    "ids,directions,match",
    [
        ([11], [], "one entry per link ID"),
        ([11], [0], "directions must be"),
        ([11], [True], "integers"),
        ([True], [1], "integers"),
        ([-11], [1], "nonnegative"),
        ([2**64], [1], "integers"),
        ([11.0], [1], "integers"),
        ([11], [1.0], "integers"),
        ([[11]], [[1]], "one-dimensional"),
    ],
)
def test_invalid_route_identity_is_rejected(ids, directions, match):
    supplied = pd.DataFrame(
        {"origin id": [10], "destination id": [40], "route set": [ids], "route directions": [directions]}
    )
    with pytest.raises(ValueError, match=match):
        configured_choice(chain_graph()).recompute_psl(supplied)


@pytest.mark.parametrize("column", ["origin id", "destination id", "route set", "route directions", "probability"])
def test_required_import_columns(column):
    choice = configured_choice(chain_graph())
    choice.set_demand(single_od_demand(10, 40))
    supplied = pd.DataFrame(
        {
            "origin id": [10],
            "destination id": [40],
            "route set": [[11, 12, 13]],
            "route directions": [[1, 1, 1]],
            "probability": [1.0],
        }
    ).drop(columns=column)

    with pytest.raises(ValueError, match=f"missing required column '{column}'"):
        choice.execute_from_pandas(supplied)


@pytest.mark.parametrize("routes,directions", [(1, [1]), ([11], 1)])
def test_route_arrays_must_be_lists_or_numpy_arrays(routes, directions):
    supplied = pd.DataFrame(
        {"origin id": [10], "destination id": [40], "route set": [routes], "route directions": [directions]}
    )

    with pytest.raises(TypeError, match="must be a list or Numpy array"):
        configured_choice(chain_graph()).recompute_psl(supplied)
