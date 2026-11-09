"""Public assignment accepts sparse demand and agrees with dense demand."""

import numpy as np
import pytest
from scipy import sparse

from aequilibrae import TrafficClass
from aequilibrae.matrix import from_dict

from .test_assignment_integration import assignment_for, diamond, matrix_for


@pytest.mark.parametrize("format", ["csr", "csc", "coo", "bsr", "lil", "dok"])
def test_sparse_assignment_matches_dense_results(format):
    graph = diamond(turn=True)
    dense = matrix_for(graph)
    sparse_demand = from_dict(
        {name: sparse.csr_array(values).asformat(format) for name, values in dense.items()}, dense.index
    )
    dense_assignment, dense_traffic = assignment_for(graph, dense)
    sparse_assignment, sparse_traffic = assignment_for(graph, sparse_demand)

    dense_assignment.execute(cores=1)
    sparse_assignment.execute(cores=1)

    dense_results = dense_assignment.class_results[dense_traffic.name]
    sparse_results = sparse_assignment.class_results[sparse_traffic.name]
    np.testing.assert_allclose(sparse_results.link_loads, dense_results.link_loads)
    np.testing.assert_allclose(sparse_assignment.assignment.total_flow, dense_assignment.assignment.total_flow)
    assert sparse_traffic.info == dense_traffic.info
    for name in dense_results.skims.matrices:
        np.testing.assert_allclose(sparse_results.skims.matrices[name], dense_results.skims.matrices[name])
    for name, matrices in dense_results.select_link_od.matrices.items():
        for field, values in matrices.items():
            np.testing.assert_allclose(sparse_results.select_link_od.matrices[name][field].toarray(), values.toarray())
    for name, values in dense.items():
        np.testing.assert_array_equal(sparse_demand[name].toarray(), values)
        assert sparse_demand[name].format == format


@pytest.mark.parametrize("format", ["csr", "coo", "lil", "dok"])
@pytest.mark.parametrize("bad", [-1.0, np.nan, np.inf, -np.inf])
def test_invalid_sparse_demand_rejected(format, bad):
    demand = from_dict({"flow": sparse.csr_array([[0.0, bad], [0.0, 0.0]]).asformat(format)}, [10, 40])
    with pytest.raises(ValueError, match="finite and nonnegative"):
        TrafficClass(mode="c", cost="time", demand=demand)


@pytest.mark.parametrize("dtype", [np.int32, np.complex128])
def test_sparse_demand_requires_float64(dtype):
    demand = from_dict({"flow": sparse.csr_array([[0, 1], [0, 0]], dtype=dtype)}, [10, 40])
    with pytest.raises(TypeError, match="float64"):
        TrafficClass(mode="c", cost="time", demand=demand)
