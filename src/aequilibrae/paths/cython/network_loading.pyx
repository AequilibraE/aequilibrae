"""Operations on search paths, with explicit query, scratch and output owners."""

from libcpp.vector cimport vector


def network_loading(SearchResults results not None, LoadingQuery query not None,
                    LoadingWorkspace workspace not None, LoadingOutputs output not None):
    """Accumulate this origin's demand without changing paths or allocating buffers.

    Rows represent physical nodes 0..destination_count-1. Only finalized,
    non-intrazonal paths are loaded. Output is not cleared: reset it once before
    processing the origins of an iteration. Scratch is replaced on every call.
    """
    if query.destination_count > results.node_count:
        raise ValueError("demand destination_count exceeds results node_count")
    if workspace.state_count != results.state_count:
        raise ValueError("workspace state_count does not match results")
    if output.link_count != results.link_count:
        raise ValueError("output link_count does not match results")
    if workspace.class_count != query.class_count or output.class_count != query.class_count:
        raise ValueError("query, workspace and output class_count must match")

    with nogil:
        cpp_network_loading[double](results.read_view(), query.view(), workspace.view(), output.view())
    return output


def reduce_loading_outputs(workers, LoadingOutputs output not None):
    """Replace output with the sum of completed workers, leaving them unchanged."""
    cdef LoadingOutputs worker
    cdef vector[CppLoadingOutputs[double]] views
    # Keep owners alive even if workers is a generator. Only the small view
    # table is built here; the numeric output was allocated by its caller.
    owners = tuple(workers)
    for worker in owners:
        if worker is None:
            raise TypeError("workers must contain LoadingOutputs")
        if worker is output:
            raise ValueError("reduction output must be distinct from its workers")
        if worker.link_count != output.link_count or worker.class_count != output.class_count:
            raise ValueError("worker and output dimensions must match")
        views.push_back(worker.view())

    with nogil:
        cpp_reduce_loading_outputs[double](views.data(), views.size(), output.view())

    return output
