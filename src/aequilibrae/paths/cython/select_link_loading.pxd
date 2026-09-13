from libc.stddef cimport size_t
from libcpp cimport bool as cpp_bool
from aequilibrae.paths.cython.search_results cimport CppSearchResults
from aequilibrae.paths.cython.aon_workspace cimport CppAoNWorkspace


cdef extern from "select_link_loading.hpp" namespace "aequilibrae::paths::cpp::mvp" nogil:
    void cpp_select_link_loading "aequilibrae::paths::cpp::mvp::select_link_loading"[T](
        const CppSearchResults &results, size_t destination_count,
        const T *demand, size_t class_count, const cpp_bool *selected_links,
        CppAoNWorkspace[T] &workspace, T *od, T *link_loads) noexcept
