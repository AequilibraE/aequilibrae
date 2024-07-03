from libcpp.vector cimport vector

cdef class Sparse:
    pass

cdef struct COO_struct:
    vector[size_t] *row
    vector[size_t] *col
    vector[double] *data

cdef class COO(Sparse):
    cdef:
        vector[size_t] *row
        vector[size_t] *col
        vector[double] *data
        readonly object shape

    cdef void append(COO self, size_t i, size_t j, double v) noexcept nogil

    @staticmethod
    cdef void init_struct(COO_struct &struct) noexcept nogil

    @staticmethod
    cdef object from_struct(COO_struct &struct)

    @staticmethod
    cdef void struct_append(COO_struct &struct, size_t i, size_t j, double v) noexcept nogil
