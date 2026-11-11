"""
Tests of the hyperpath module.

py.test aequilibrae/tests/paths/test_hyperpath.py

Bell's network construction by François Pacull
https://aetperf.github.io/2023/05/10/Hyperpath-routing-in-the-context-of-transit-assignment.html
"""

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from aequilibrae.matrix.cython.dense_matrix import DenseMatrix
from aequilibrae.matrix.cython.sparse_matrix import SparseMatrix
from aequilibrae.paths.cython.public_transport import HyperpathGenerating


def create_vertices(n):
    x = np.linspace(0, 1, n)
    y = np.linspace(0, 1, n)
    xv, yv = np.meshgrid(x, y, indexing="xy")
    vertices = pd.DataFrame()
    vertices["x"] = xv.ravel()
    vertices["y"] = yv.ravel()
    return vertices


def create_edges(n, seed):
    m = 2 * n * (n - 1)
    tail = np.zeros(m, dtype=np.uint32)
    head = np.zeros(m, dtype=np.uint32)
    k = 0
    for i in range(n - 1):
        for j in range(n):
            tail[k] = i + j * n
            head[k] = i + 1 + j * n
            k += 1
            tail[k] = j + i * n
            head[k] = j + (i + 1) * n
            k += 1

    edges = pd.DataFrame()
    edges["tail"] = tail
    edges["head"] = head

    rng = np.random.default_rng(seed=seed)
    edges["trav_time"] = rng.uniform(0.0, 1.0, m)
    edges["delay_base"] = rng.uniform(0.0, 1.0, m)
    return edges


# Stop vertices of the Spiess & Florian network
STOP_A, STOP_X, STOP_Y, STOP_B = 0, 1, 2, 3


def create_SF_network(dwell_time=0.0):
    """
    Example network from Spiess, H. and Florian, M. (1989).
    Optimal strategies: A new assignment model for transit networks.
    Transportation Research Part B 23(2), 83-102.

    Transfers alight to the stop vertex and board again, as ``TransitGraphBuilder`` builds them. This network has 16
    vertices and 20 edges. Frequencies are the lines' own, so the paper's waits of half a headway need a headway
    fraction of 0.5.
    """
    freq = {line: 1.0 / (60.0 * headway) for line, headway in {1: 12, 2: 12, 3: 30, 4: 6}.items()}

    # Vertices 0 to 3 are stops A, X, Y and B. Then, per line, a boarding vertex for each segment and an alighting
    # vertex where it ends:
    # line 1: boarding A 4, alighting B 5
    # line 2: boarding A 6, alighting X 7, boarding X 8, alighting Y 9
    # line 3: boarding X 10, alighting Y 11, boarding Y 12, alighting B 13
    # line 4: boarding Y 14, alighting B 15
    rows = [
        # tail, head, link type, travel time, frequency, volume of the published strategy
        (STOP_A, 4, "boarding", 0.0, freq[1], 1 / 2),
        (STOP_A, 6, "boarding", 0.0, freq[2], 1 / 2),
        (STOP_X, 8, "boarding", 0.0, freq[2], 0.0),
        (STOP_X, 10, "boarding", 0.0, freq[3], 0.0),
        (STOP_Y, 12, "boarding", 0.0, freq[3], 1 / 12),
        (STOP_Y, 14, "boarding", 0.0, freq[4], 5 / 12),
        (4, 5, "on-board", 25.0 * 60.0, np.inf, 1 / 2),
        (6, 7, "on-board", 7.0 * 60.0, np.inf, 1 / 2),
        (8, 9, "on-board", 6.0 * 60.0, np.inf, 1 / 2),
        (10, 11, "on-board", 4.0 * 60.0, np.inf, 0.0),
        (12, 13, "on-board", 4.0 * 60.0, np.inf, 1 / 12),
        (14, 15, "on-board", 10.0 * 60.0, np.inf, 5 / 12),
        (7, 8, "dwell", dwell_time, np.inf, 1 / 2),
        (11, 12, "dwell", dwell_time, np.inf, 0.0),
        (5, STOP_B, "alighting", 0.0, np.inf, 1 / 2),
        (7, STOP_X, "alighting", 0.0, np.inf, 0.0),
        (9, STOP_Y, "alighting", 0.0, np.inf, 1 / 2),
        (11, STOP_Y, "alighting", 0.0, np.inf, 0.0),
        (13, STOP_B, "alighting", 0.0, np.inf, 1 / 12),
        (15, STOP_B, "alighting", 0.0, np.inf, 5 / 12),
    ]
    edges = pd.DataFrame(rows, columns=["tail", "head", "link_type", "trav_time", "freq", "volume_ref"])
    demand = pd.DataFrame({"origin_vertex_id": [STOP_A], "destination_vertex_id": [STOP_B], "demand": [1.0]})

    return edges, demand


# Labels towards B, with half-headway waits
SF_LABELS = np.array(
    [1665.0, 1144.2857142857142, 690.0, 0.0, 1500.0, 0.0, 1470.0, 1050.0]
    + [1050.0, 690.0, 480.0, 240.0, 240.0, 0.0, 600.0, 0.0]
)


@pytest.fixture(scope="function")
def network_setup(request):
    """
    Fixture for creating network data based on parameters.

    Parameters are passed using indirect parametrization:
    @pytest.mark.parametrize("network_setup", [
        {"network": "bell", "n": 10, "alpha": 10.0, "dwell_time": 0.0, "seed": 124}
    ], indirect=True)
    """
    network_type = request.param.get("network", "bell")
    n = request.param.get("n", 10)
    alpha = request.param.get("alpha", 10.0)
    dwell_time = request.param.get("dwell_time", 0.0)
    seed = request.param.get("seed", 124)

    result = {}

    if network_type == "bell":
        result["vertices"] = create_vertices(n)
        result["edges"] = create_edges(n, seed=seed)

        delay_base = result["edges"].delay_base.to_numpy(copy=True)
        indices = np.where(delay_base == 0.0)
        delay_base[indices] = 1.0
        result["edges"]["delay_base"] = delay_base
        freq_base = 1.0 / delay_base
        freq_base[indices] = np.inf
        result["edges"]["freq_base"] = freq_base

        if alpha == 0.0:
            result["edges"]["freq"] = np.inf
        else:
            result["edges"]["freq"] = result["edges"].freq_base / alpha

        result["demand"] = pd.DataFrame(
            {
                "orig_vert_idx": result["vertices"].index[:10],
                "dest_vert_idx": np.flip(result["vertices"].index[-10:]),
                "demand": np.full(10, 1),
            }
        )
        result["centroids"] = result["vertices"].index.to_numpy()
        result["all_nodes"] = result["vertices"].index.to_numpy()

    elif network_type == "SF":
        result["edges"], result["demand"] = create_SF_network(dwell_time=dwell_time)
        result["centroids"] = np.array([STOP_A, STOP_B])
        result["all_nodes"] = np.unique(np.hstack((result["edges"]["head"].values, result["edges"]["tail"].values)))
    else:
        raise KeyError(f'Unknown network type "{network_type}"')

    result["nodes_to_indices"] = np.full(result["all_nodes"].max() + 1, -1, dtype="int64")
    result["nodes_to_indices"][result["all_nodes"]] = np.arange(len(result["all_nodes"]))

    return result


def pack_demand(centroids, origins, destinations, values, matrix_type=DenseMatrix):
    """Pack trips between zone vertices into transposed demand, as ``HyperpathGenerating.assign`` takes it."""
    positions = {node: zone for zone, node in enumerate(centroids)}
    matrix = np.zeros((len(centroids), len(centroids)))
    np.add.at(
        matrix,
        ([positions[node] for node in origins], [positions[node] for node in destinations]),
        np.asarray(values, dtype=np.float64),
    )
    source = sparse.csr_array(matrix) if matrix_type is SparseMatrix else matrix
    return matrix_type.from_dict({"demand": source}, transpose=True)


@pytest.fixture(params=[DenseMatrix, SparseMatrix])
def packed_demand(network_setup, request):
    demand = network_setup["demand"]
    origin, destination = demand.columns[:2]
    return pack_demand(network_setup["centroids"], demand[origin], demand[destination], demand["demand"], request.param)


def sf_hyperpath(network_setup, skim_cols=None, **kwargs):
    kwargs.setdefault("headway_fraction", 0.5)
    return HyperpathGenerating(
        network_setup["edges"],
        skim_cols=skim_cols,
        o_vert_ids=network_setup["centroids"],
        d_vert_ids=network_setup["centroids"],
        nodes_to_indices=network_setup["nodes_to_indices"],
        **kwargs,
    )


def assign_sf_demand(hp, network_setup, threads=None):
    demand = network_setup["demand"]
    trips = (demand["origin_vertex_id"], demand["destination_vertex_id"], demand["demand"])
    hp.assign(pack_demand(network_setup["centroids"], *trips), threads=threads)


def skim_results(hp, network_setup):
    """The skims of the last assignment, by zone ID."""
    return hp.skim_matrix.to_matrix(network_setup["centroids"])


@pytest.mark.parametrize("network_setup", [{"network": "bell", "n": 10, "alpha": 10.0}], indirect=True)
def test_bell_assign_parallel_agreement(network_setup, packed_demand):
    """Test that Bell's network assigns the same volumes with one, two and four threads."""
    hp = HyperpathGenerating(
        network_setup["edges"],
        o_vert_ids=network_setup["centroids"],
        d_vert_ids=network_setup["centroids"],
        nodes_to_indices=network_setup["nodes_to_indices"],
    )

    results = []
    for threads in [1, 2, 4]:
        hp.assign(packed_demand, threads=threads)
        results.append(hp._edges.volume.to_numpy(copy=True))

    for result in results[1:]:
        np.testing.assert_allclose(result, results[0], rtol=1e-12, atol=1e-15)


@pytest.mark.parametrize("network_setup", [{"network": "bell", "n": 10, "alpha": 10.0}], indirect=True)
def test_bell_assign_is_bitwise_repeatable(network_setup, packed_demand):
    """Test that repeating an assignment with the same thread count gives bitwise-identical volumes and skims."""
    hp = HyperpathGenerating(
        network_setup["edges"],
        skim_cols=["trav_time", "waiting_time"],
        o_vert_ids=network_setup["centroids"],
        d_vert_ids=network_setup["centroids"],
        nodes_to_indices=network_setup["nodes_to_indices"],
    )

    hp.assign(packed_demand, threads=3)
    volumes = hp._edges.volume.to_numpy(copy=True)
    skims = skim_results(hp, network_setup)["waiting_time"].copy()
    for _ in range(3):
        hp.assign(packed_demand, threads=3)
        np.testing.assert_array_equal(hp._edges.volume.to_numpy(), volumes)
        np.testing.assert_array_equal(skim_results(hp, network_setup)["waiting_time"], skims)


@pytest.mark.parametrize("network_setup", [{"network": "SF"}], indirect=True)
def test_SF_run_01(network_setup):
    """Test that one trip from A to B loads the published strategy and labels every vertex."""
    hp = sf_hyperpath(network_setup)
    hp.run(origin=STOP_A, destination=STOP_B, volume=1.0)

    np.testing.assert_allclose(
        network_setup["edges"]["volume_ref"].values, hp._edges["volume"].values, rtol=1e-05, atol=1e-08
    )
    np.testing.assert_allclose(hp.u_i_vec, SF_LABELS, rtol=1e-08, atol=1e-06)


@pytest.mark.parametrize("network_setup", [{"network": "SF"}], indirect=True)
def test_run_returns_the_destination_labels_when_skimming(network_setup):
    """Test that run keeps the labels towards its own destination, not those of the other skimmed zones."""
    hp = sf_hyperpath(network_setup, skim_cols=["trav_time"])
    hp.run(origin=STOP_A, destination=STOP_B, volume=1.0)
    np.testing.assert_allclose(hp.u_i_vec, SF_LABELS, rtol=1e-08, atol=1e-06)


@pytest.mark.parametrize("network_setup", [{"network": "SF"}], indirect=True)
def test_SF_single_destination(network_setup, packed_demand):
    """Test that packed demand towards a single destination loads the published strategy and keeps its labels."""
    hp = sf_hyperpath(network_setup)
    hp.assign(packed_demand, threads=1)

    np.testing.assert_allclose(
        network_setup["edges"]["volume_ref"].values, hp._edges["volume"].values, rtol=1e-05, atol=1e-08
    )
    np.testing.assert_allclose(hp.u_i_vec, SF_LABELS, rtol=1e-08, atol=1e-06)


@pytest.mark.parametrize("network_setup", [{"network": "SF"}], indirect=True)
def test_SF_assign_01(network_setup, packed_demand):
    """Test that assigning the packed demand loads the published strategy."""
    hp = sf_hyperpath(network_setup)
    hp.assign(packed_demand)

    np.testing.assert_allclose(
        network_setup["edges"]["volume_ref"].values, hp._edges["volume"].values, rtol=1e-05, atol=1e-08
    )


@pytest.mark.parametrize(
    "network_setup,skim_col",
    [
        ({"network": "SF", "dwell_time": 10.0}, "boardings"),
        ({"network": "SF", "dwell_time": 10.0}, "in_vehicle_trav_time"),
        ({"network": "SF", "dwell_time": 10.0}, "egress_trav_time"),
        ({"network": "SF", "dwell_time": 10.0}, "access_trav_time"),
    ],
    indirect=["network_setup"],
)
def test_skimming_cols(network_setup, skim_col):
    """Test that a predefined skim derived from link types becomes an edge column."""
    hp = sf_hyperpath(network_setup, skim_cols=[skim_col])
    assert skim_col in hp._edges.columns, "requested column missing from edges"


@pytest.mark.parametrize("network_setup", [{"network": "SF", "dwell_time": 10.0}], indirect=True)
def test_waiting_time_skimming(network_setup):
    """Test that waiting time comes from the kernel, without derived edge columns or changes to the user's edges."""
    columns = list(network_setup["edges"].columns)
    hp = sf_hyperpath(network_setup, skim_cols=["waiting_time"])

    assert "waiting_time" not in hp._edges.columns, "waiting time isn't a real column"
    for col in ["boardings", "in_vehicle_trav_time", "egress_trav_time", "access_trav_time"]:
        assert col not in hp._edges.columns, "waiting time needs no other skimming columns"
    assert list(network_setup["edges"].columns) == columns, "don't modify the users dataframe"


@pytest.mark.parametrize("network_setup", [{"network": "SF"}], indirect=True)
def test_SF_skimming_01(network_setup, packed_demand):
    """Test that the strategy from A to B skims as published, and that B cannot reach A."""
    skims = ["trav_time", "boardings", "transfers", "pt_probability", "in_vehicle_trav_time"]
    skims += ["waiting_time", "first_waiting_time", "transfer_waiting_time"]
    hp = sf_hyperpath(network_setup, skim_cols=skims)
    hp.assign(packed_demand)
    assert isinstance(hp.skim_matrix, DenseMatrix)
    mats = skim_results(hp, network_setup)

    def pair(value):
        return np.array([[0.0, value], [np.inf, 0.0]])

    np.testing.assert_allclose(mats["trav_time"], pair(27.75 * 60.0), atol=1e-6)  # travel time from paper
    np.testing.assert_allclose(mats["boardings"], pair(1.5))
    np.testing.assert_allclose(mats["transfers"], pair(0.5))
    np.testing.assert_allclose(mats["pt_probability"], pair(1.0))
    np.testing.assert_allclose(mats["in_vehicle_trav_time"], pair(23.5 * 60.0), atol=1e-6)
    np.testing.assert_allclose(mats["waiting_time"], pair(4.25 * 60.0))
    np.testing.assert_allclose(mats["first_waiting_time"], pair(3.0 * 60.0))
    np.testing.assert_allclose(mats["transfer_waiting_time"], pair(1.25 * 60.0))


@pytest.mark.parametrize("network_setup", [{"network": "SF"}], indirect=True)
def test_SF_skimming_02(network_setup, packed_demand):
    """Test that every strategy alights as often as it boards."""
    hp = sf_hyperpath(network_setup, skim_cols=["boardings", "alightings"])
    hp.assign(packed_demand)
    mats = skim_results(hp, network_setup)

    np.testing.assert_allclose(mats["boardings"], mats["alightings"])


@pytest.mark.parametrize("network_setup", [{"network": "SF"}], indirect=True)
def test_wait_weight_changes_labels_but_not_reported_waits(network_setup):
    """Test that the wait weight scales perceived waits in the labels, while reported waits stay physical."""
    hp = sf_hyperpath(network_setup, skim_cols=["trav_time", "waiting_time", "in_vehicle_trav_time"], wait_weight=2.0)
    assign_sf_demand(hp, network_setup)
    mats = skim_results(hp, network_setup)

    # The strategy is the same, since every attractive set still beats the alternatives
    np.testing.assert_allclose(
        network_setup["edges"]["volume_ref"].values, hp._edges["volume"].values, rtol=1e-05, atol=1e-08
    )
    assert mats["waiting_time"][0, 1] == pytest.approx(4.25 * 60.0)
    assert mats["trav_time"][0, 1] == pytest.approx(23.5 * 60.0 + 2 * 4.25 * 60.0)


@pytest.mark.parametrize("network_setup", [{"network": "SF"}], indirect=True)
def test_costs_and_waits_add_up_to_expected_cost(network_setup):
    """Test that edge costs times volumes plus kappa-weighted waits equal demand times origin labels."""
    hp = sf_hyperpath(network_setup, skim_cols=["trav_time"], wait_weight=1.7)
    assign_sf_demand(hp, network_setup)

    volumes = hp._edges.volume.to_numpy()
    costs = np.maximum(network_setup["edges"].trav_time.to_numpy(), hp.cost_floor)
    loaded = costs @ volumes + hp.weighted_waiting_time
    assert loaded == pytest.approx(skim_results(hp, network_setup)["trav_time"][0, 1], rel=1e-12)


@pytest.mark.parametrize("network_setup", [{"network": "SF"}], indirect=True)
def test_flow_is_conserved_at_every_vertex(network_setup):
    """Test that flow into every vertex other than the origin and destination equals the flow out."""
    hp = sf_hyperpath(network_setup)
    assign_sf_demand(hp, network_setup)

    edges = network_setup["edges"]
    volume = hp._edges.volume.to_numpy()
    inflow = np.bincount(edges["head"], weights=volume, minlength=16)
    outflow = np.bincount(edges["tail"], weights=volume, minlength=16)
    inner = np.setdiff1d(np.arange(16), [STOP_A, STOP_B])
    np.testing.assert_allclose(inflow[inner], outflow[inner], atol=1e-12)
    assert outflow[STOP_A] == pytest.approx(1.0)
    assert inflow[STOP_B] == pytest.approx(1.0)


@pytest.mark.parametrize("network_setup", [{"network": "SF"}], indirect=True)
def test_unreachable_demand_is_reported_and_not_loaded(network_setup):
    """Test that demand from B to A, which no strategy serves, is reported as unassigned."""
    hp = sf_hyperpath(network_setup)
    hp.assign(pack_demand(network_setup["centroids"], [STOP_A, STOP_B], [STOP_B, STOP_A], [1.0, 3.0]))

    assert hp.unassigned_demand == pytest.approx(3.0)
    assert hp.unassigned_od.to_dict("records") == [{"origin": STOP_B, "destination": STOP_A, "demand": 3.0}]
    np.testing.assert_allclose(network_setup["edges"]["volume_ref"].values, hp._edges["volume"].values, atol=1e-12)


@pytest.mark.parametrize("network_setup", [{"network": "SF"}], indirect=True)
def test_an_edge_without_a_wait_takes_all_the_flow(network_setup):
    """Test that a walk from A to Y that beats waiting at A leaves the lines boarded at A with exactly zero flow."""
    # Boarding at A expects 1665 s, and walking to Y takes 900 s plus the 690 s expected from there
    walk = pd.DataFrame([(STOP_A, STOP_Y, "walking", 900.0, np.inf, 1.0)], columns=network_setup["edges"].columns)
    network_setup["edges"] = pd.concat([network_setup["edges"], walk], ignore_index=True)
    hp = sf_hyperpath(network_setup, skim_cols=["trav_time", "waiting_time"])
    assign_sf_demand(hp, network_setup)

    edges = network_setup["edges"]
    volume = hp._edges.volume.to_numpy()
    boarding_at_a = ((edges.tail == STOP_A) & (edges.link_type == "boarding")).to_numpy()
    assert np.all(volume[boarding_at_a] == 0.0)
    assert volume[-1] == 1.0
    mats = skim_results(hp, network_setup)
    assert mats["trav_time"][0, 1] == pytest.approx(900.0 + 690.0)
    assert mats["waiting_time"][0, 1] == pytest.approx(0.5 / (1 / 1800 + 1 / 360))


@pytest.mark.parametrize("network_setup", [{"network": "SF"}], indirect=True)
def test_update_costs_matches_a_new_graph(network_setup):
    """Test that new costs, frequencies and waits given to an existing graph assign as a graph built with them."""
    hp = sf_hyperpath(network_setup, skim_cols=["trav_time"])
    assign_sf_demand(hp, network_setup)

    edges = network_setup["edges"].assign(
        trav_time=network_setup["edges"].trav_time * 1.5, freq=network_setup["edges"].freq * 2
    )
    hp.update_costs(edges.trav_time.to_numpy(), edges.freq.to_numpy(), headway_fraction=1.0)
    assign_sf_demand(hp, network_setup)

    network_setup["edges"] = edges
    fresh = sf_hyperpath(network_setup, skim_cols=["trav_time"], headway_fraction=1.0)
    assign_sf_demand(fresh, network_setup)
    np.testing.assert_array_equal(hp._edges.volume.to_numpy(), fresh._edges.volume.to_numpy())
    np.testing.assert_array_equal(
        skim_results(hp, network_setup)["trav_time"], skim_results(fresh, network_setup)["trav_time"]
    )


@pytest.mark.parametrize("network_setup", [{"network": "SF"}], indirect=True)
def test_negative_costs_are_rejected(network_setup):
    """Test that a negative edge cost raises instead of breaking label setting."""
    hp = sf_hyperpath(network_setup)
    costs = network_setup["edges"].trav_time.to_numpy(copy=True)
    costs[6] = -1.0
    with pytest.raises(ValueError, match="nonnegative"):
        hp.update_costs(costs)


@pytest.mark.parametrize("network_setup", [{"network": "SF"}], indirect=True)
def test_strategy_lists_edges_splits_and_volumes(network_setup):
    """Test that the strategy of one trip lists the edges it may use with their splits, leaving volumes alone."""
    hp = sf_hyperpath(network_setup)
    strategy = hp.strategy(STOP_A, STOP_B)

    reference = network_setup["edges"]
    used = reference.index[reference.volume_ref > 0]
    assert list(strategy.index) == list(used)
    np.testing.assert_allclose(strategy.volume, reference.volume_ref[used])

    # Half board each line at A, and at Y one in six of the line 2 riders take line 3
    splits = strategy.set_index(["tail", "head"]).split
    assert splits[STOP_A, 4] == pytest.approx(0.5)
    assert splits[STOP_Y, 12] == pytest.approx(1 / 6)
    assert splits[STOP_Y, 14] == pytest.approx(5 / 6)
    assert "volume" not in hp._edges.columns


@pytest.mark.parametrize("network_setup", [{"network": "SF"}], indirect=True)
def test_assign_rejects_demand_that_does_not_match_the_zones(network_setup):
    """Test that demand with more than one field or another number of zones raises instead of being assigned."""
    hp = sf_hyperpath(network_setup)
    two_fields = DenseMatrix.from_dict({"a": np.ones((2, 2)), "b": np.ones((2, 2))}, transpose=True)
    three_zones = DenseMatrix.from_dict({"demand": np.ones((3, 3))}, transpose=True)
    for demand in (two_fields, three_zones):
        with pytest.raises(ValueError):
            hp.assign(demand)
    with pytest.raises(TypeError):
        hp.assign(np.ones((2, 2)))
