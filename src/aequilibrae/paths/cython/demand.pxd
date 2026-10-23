"""Demand rows shared by the assignment driver and loading bindings."""

from aequilibrae.matrix.cython.views cimport CppDenseMatrixOriginView, CppSparseMatrixOriginView


ctypedef CppDenseMatrixOriginView[const double] DenseDemandRow
ctypedef CppSparseMatrixOriginView[const double] SparseDemandRow


ctypedef fused DemandRowView:
    DenseDemandRow
    SparseDemandRow
