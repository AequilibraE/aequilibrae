from libc.stddef cimport size_t
from aequilibrae.paths.cython.search_results cimport CppSearchResults
from aequilibrae.paths.cython.aon_workspace cimport CppAoNWorkspace


cdef extern from "network_loading.hpp" namespace "aequilibrae::paths::cpp::mvp" nogil:
    void cpp_network_loading "aequilibrae::paths::cpp::mvp::network_loading"[T](
        const CppSearchResults &results, size_t destination_count,
        const T *demand, size_t class_count,
        CppAoNWorkspace[T] &workspace, T *link_loads) noexcept
