from aequilibrae.paths.cython.search_results cimport SearchResults, CppSearchResults
from aequilibrae.paths.cython.context cimport SkimmingContext, CppSkimmingContext
from aequilibrae.paths.cython.workspaces cimport SkimmingWorkspace, CppSkimmingWorkspace
from aequilibrae.paths.cython.outputs cimport SkimmingOutputs, CppSkimmingOriginView


cdef extern from "skimming.hpp" namespace "aequilibrae::paths::cpp::routing" nogil:
    void cpp_skimming "aequilibrae::paths::cpp::routing::skimming"[T](
        const CppSearchResults &results,
        const CppSkimmingContext[T] &context,
        const CppSkimmingWorkspace[T] &workspace,
        const CppSkimmingOriginView[T] &output,
    ) noexcept
