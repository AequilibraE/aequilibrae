"""Pack assignment demand and prepare its fixed origins and targets."""

import numpy as np
from libc.stddef cimport size_t
from libcpp cimport bool as cpp_bool

from aequilibrae.matrix.aequilibrae_matrix import AequilibraeMatrix, AequilibraeSparseMatrix
from aequilibrae.matrix.cython.dense_matrix cimport DenseMatrix
from aequilibrae.matrix.cython.sparse_matrix cimport SparseMatrix
from aequilibrae.matrix.cython.matrix_base cimport _MatrixBase
from aequilibrae.utils.cython.array_allocations cimport array


def assignment_demand(matrix, centroids, *, transpose=False):
    """Copy named matrices into fixed, packed dense or sparse demand, optionally transposed."""
    if not isinstance(matrix, (AequilibraeMatrix, AequilibraeSparseMatrix)):
        raise TypeError("Demand must be an AequilibraeMatrix or AequilibraeSparseMatrix")
    if not np.array_equal(matrix.index, centroids):
        raise ValueError("Matrix and graph must have the same centroid order")
    if not matrix or not len(centroids) or matrix.shape != (len(centroids), len(centroids)):
        raise ValueError("Demand must have shape (zones, zones, classes)")

    if isinstance(matrix, AequilibraeSparseMatrix):
        packed = SparseMatrix.from_matrix(matrix, dtype=np.float64, transpose=transpose)
    else:
        packed = DenseMatrix.from_matrix(matrix, dtype=np.float64, transpose=transpose)
    validate_demand(packed, len(centroids))
    return packed


def validate_demand(demand, nodes):
    """Check demand dimensions, dtype and values, and return its shape."""
    if not isinstance(demand, (DenseMatrix, SparseMatrix)):
        raise TypeError("demand must be a DenseMatrix or SparseMatrix")
    if demand.dtype != np.dtype(np.float64):
        raise TypeError("demand must have dtype float64")

    shape = (demand.origin_count, demand.destination_count, demand.field_count)
    if shape[0] != shape[1] or not 1 <= shape[0] <= nodes or shape[2] < 1:
        raise ValueError("demand must have shape (zones, zones, classes), zones <= node_count, classes >= 1")

    values = np.asarray((<_MatrixBase>demand)._values)
    if not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError("Demand values must be finite and nonnegative")

    return shape


def choose_origins(origins, _MatrixBase demand, skimming: bool):
    """Choose origins once; each OD output row must have just one writer."""
    cdef SparseMatrix packed
    zones = demand.origin_count

    if origins is None:
        if skimming:
            return np.arange(zones, dtype=np.uintp)
        if isinstance(demand, SparseMatrix):
            packed = demand
            active = np.any(np.asarray(packed._values) != 0, axis=1)
            return np.flatnonzero([
                np.any(active[packed.row_index[row]:packed.row_index[row + 1]])
                for row in range(zones)
            ]).astype(np.uintp)
        return np.flatnonzero(np.any(np.asarray(demand._values) != 0, axis=(1, 2))).astype(np.uintp)

    indices = np.asarray(origins)
    if indices.ndim != 1 or (indices.size and indices.dtype.kind not in "iu"):
        raise ValueError("origins must be centroid indices")

    indices = np.sort(indices)
    if np.any(indices < 0) or np.any(indices >= zones):
        raise ValueError("origins must be centroid indices")
    if np.any(indices[1:] == indices[:-1]):
        raise ValueError("origins must be unique (workers write disjoint OD rows)")

    return np.asarray(indices, dtype=np.uintp)


# FIXME: We should be using the index of the sparse matrix to determine which
# OD pairs we need to run assignment for. This should just be the col_index for
# a particular origin. We shouldn't explode this for a sparse matrix.
def make_destination_masks(_MatrixBase demand, nodes: int, skimming: bool):
    """Prepare fixed targets, sharing one all-centroid mask when skimming."""
    cdef cpp_bool[:, ::1] masks
    cdef size_t origin, destination, entry
    cdef SparseMatrix packed

    zones = demand.origin_count
    mask_rows = 1 if skimming else zones
    masks = array[cpp_bool]((mask_rows, nodes), True, False)

    if skimming:
        for destination in range(zones):
            masks[0, destination] = True
    elif isinstance(demand, SparseMatrix):
        packed = demand
        active = np.any(np.asarray(packed._values) != 0, axis=1)
        for origin in range(zones):
            for entry in range(packed.row_index[origin], packed.row_index[origin + 1]):
                masks[origin, packed.col_index[entry]] = active[entry]
    else:
        destinations = np.any(np.asarray(demand._values) != 0, axis=2)
        for origin in range(zones):
            for destination in range(zones):
                masks[origin, destination] = destinations[origin, destination]

    counts = np.count_nonzero(np.asarray(masks), axis=1).astype(np.uintp)
    counts.flags.writeable = False
    return masks, counts
