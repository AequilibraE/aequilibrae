import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_array

from aequilibrae.matrix import AequilibraeMatrix, AequilibraeSparseMatrix
from aequilibrae.paths import TransitAssignment, TransitClass
from aequilibrae.paths.graph import TransitGraph


@pytest.fixture
def transit_graph():
    links = pd.DataFrame(
        {
            "link_id": [1, 2],
            "a_node": [10, 20],
            "b_node": [20, 30],
            "direction": [1, 1],
            "trav_time": [1.0, 2.0],
            "freq": [np.inf, np.inf],
            "distance": [3.0, 4.0],
        }
    )
    return TransitGraph(
        network=links,
        centroids=np.array([10, 30]),
        time_field="trav_time",
        frequency_field="freq",
        a_node_field="b_node",
        b_node_field="a_node",
        od_node_mapping=pd.DataFrame({"zone_id": [10, 30], "node_id": [10, 30]}),
        skimming_fields=["distance"],
    )


@pytest.fixture(params=[AequilibraeMatrix, AequilibraeSparseMatrix], ids=["dense", "sparse"])
def transit_matrix(request, transit_graph):
    matrix = request.param(index=transit_graph.centroids)
    values = np.array([[0.0, 5.0], [0.0, 0.0]])
    matrix["pt"] = csr_array(values) if isinstance(matrix, AequilibraeSparseMatrix) else values
    return matrix


def test_transit_assignment_dense_and_sparse_results_match(transit_graph, transit_matrix):
    reference = AequilibraeMatrix(index=transit_graph.centroids)
    reference["pt"] = np.array([[0.0, 5.0], [0.0, 0.0]])
    loads, skims = [], []
    for matrix in (reference, transit_matrix):
        configured = TransitClass(name="pt", graph=transit_graph, matrix=matrix)
        assignment = TransitAssignment()
        assignment.set_classes([configured])
        assignment.set_cores(1)
        assignment.set_algorithm("os")
        assignment.execute(log_specification=False)
        loads.append(assignment.results().copy())
        skims.append({name: values.copy() for name, values in configured.results.skims.items()})

    pd.testing.assert_frame_equal(loads[0], loads[1])
    np.testing.assert_allclose(loads[1].loc[[1, 2], "pt_volume"], 5.0)
    assert skims[0].keys() == skims[1].keys() == {"distance"}
    np.testing.assert_allclose(skims[0]["distance"], skims[1]["distance"])
    assert transit_matrix.totals() == {"pt": 5.0}
