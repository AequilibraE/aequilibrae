from aequilibrae.paths.cython.search_results cimport SearchResults, CppSearchResults
from aequilibrae.paths.cython.skimming_context cimport SkimmingContext, CppSkimmingContext
from aequilibrae.paths.cython.workspaces cimport SkimmingWorkspace, CppSkimmingWorkspace
from aequilibrae.paths.cython.outputs cimport SkimmingOutputs, CppSkimmingOriginView


cdef extern from "skimming.hpp" namespace "aequilibrae::paths::cpp::mvp" nogil:
    void cpp_skimming "aequilibrae::paths::cpp::mvp::skimming"[T](
        const CppSearchResults &results, const CppSkimmingContext[T] &context,
        CppSkimmingWorkspace[T] workspace, CppSkimmingOriginView[T] output) noexcept
