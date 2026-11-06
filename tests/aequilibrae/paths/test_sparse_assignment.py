"""Public assignment accepts sparse demand and agrees with dense demand."""

import numpy as np
from scipy import sparse

from aequilibrae.matrix import from_dict

from .test_assignment_integration import assignment_for, diamond, matrix_for


def test_sparse_assignment_matches_dense_results():
    graph = diamond(turn=True)
    dense = matrix_for(graph)
    sparse_demand = from_dict({name: sparse.csr_array(values) for name, values in dense.items()}, dense.index)
    dense_assignment, dense_traffic = assignment_for(graph, dense)
    sparse_assignment, sparse_traffic = assignment_for(graph, sparse_demand)

    dense_assignment.execute()
    sparse_assignment.execute()

    np.testing.assert_allclose(sparse_traffic.results.link_loads, dense_traffic.results.link_loads)
    np.testing.assert_allclose(sparse_assignment.assignment.total_flow, dense_assignment.assignment.total_flow)
    for name in dense_traffic.results.skims.matrices:
        np.testing.assert_allclose(
            sparse_traffic.results.skims.matrices[name], dense_traffic.results.skims.matrices[name]
        )
