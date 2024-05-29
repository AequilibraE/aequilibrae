from libcpp.vector cimport vector

cdef class Sparse:
    pass

cdef class COO(Sparse):
    cdef:
        vector[size_t] *row
        vector[size_t] *col
        vector[double] *data
        readonly object shape

    cdef void append(COO self, size_t i, size_t j, double v) noexcept nogil
