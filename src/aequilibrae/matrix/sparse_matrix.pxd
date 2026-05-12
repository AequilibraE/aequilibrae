from libcpp.vector cimport vector
from libcpp.memory cimport unique_ptr, make_unique
from libcpp cimport bool

cdef class Sparse:
    pass

cdef struct COO_f64_struct:
    unique_ptr[vector[size_t]] row
    unique_ptr[vector[size_t]] col
    unique_ptr[vector[double]] f64_data

cdef struct COO_f32_struct:
    unique_ptr[vector[size_t]] row
    unique_ptr[vector[size_t]] col
    unique_ptr[vector[float]] f32_data

cdef class COO(Sparse):
    cdef:
        unique_ptr[vector[size_t]] row
        unique_ptr[vector[size_t]] col
        bool f64
        unique_ptr[vector[double]] f64_data
        unique_ptr[vector[float]] f32_data
        public object shape

    cdef void f64_append(COO self, size_t i, size_t j, double v) noexcept nogil
    cdef void f32_append(COO self, size_t i, size_t j, float v) noexcept nogil

    @staticmethod
    cdef void init_f64_struct(COO_f64_struct &struct) noexcept nogil

    @staticmethod
    cdef void init_f32_struct(COO_f32_struct &struct) noexcept nogil

    @staticmethod
    cdef object from_f64_struct(COO_f64_struct &struct)

    @staticmethod
    cdef object from_f32_struct(COO_f32_struct &struct)

    @staticmethod
    cdef void f64_struct_append(COO_f64_struct &struct, size_t i, size_t j, double v) noexcept nogil

    @staticmethod
    cdef void f32_struct_append(COO_f32_struct &struct, size_t i, size_t j, float v) noexcept nogil
