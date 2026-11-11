"""Packed demand types and rows shared by assignment and loading bindings."""

from aequilibrae.matrix.cython.dense_matrix cimport DenseMatrix
from aequilibrae.matrix.cython.sparse_matrix cimport SparseMatrix
from aequilibrae.matrix.cython.views cimport (
    CppDenseMatrix, CppSparseMatrix, CppDenseMatrixOriginView, CppSparseMatrixOriginView,
)


ctypedef CppDenseMatrixOriginView[const double] DenseDemandRow
ctypedef CppSparseMatrixOriginView[const double] SparseDemandRow


ctypedef fused DemandMatrix:
    DenseMatrix
    SparseMatrix


ctypedef fused DemandMatrixView:
    CppDenseMatrix[double]
    CppSparseMatrix[double]


ctypedef fused DemandRowView:
    DenseDemandRow
    SparseDemandRow
