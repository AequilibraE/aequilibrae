"""Selected loading with independent inputs, scratch and optional outputs."""

import operator
from libcpp.vector cimport vector


def select_link_loading(
    SearchResults results not None,
    LoadingQuery query not None,
    SelectLinkContext context not None,
    SelectLinkWorkspace selection_workspace not None,
    LoadingWorkspace loading_workspace=None,
    SelectLinkLoadingOutputs loading_output=None,
    SelectLinkODOutputs od_output=None,
    *,
    origin_row=0,
):
    """Accumulate selected full-path loads and/or replace one selected OD row.

    OD-only calls need no loading workspace. The output row is independent of
    the physical search origin. Missing terminals and intrazonal demand do not
    contribute. Validation completes before any scratch or output is written.
    Returns the two supplied outputs, either of which may be None.
    """
    cdef CppLoadingWorkspace[double] loading
    cdef CppSelectLinkLoadingOutputsView[double] loads
    cdef CppSelectLinkODOriginView[double] od
    cdef size_t row

    if context.link_count != results.link_count:
        raise ValueError("selection link_count does not match results")
    if query.destination_count > results.node_count:
        raise ValueError("demand destination_count exceeds results node_count")
    if selection_workspace.state_count != results.state_count:
        raise ValueError("selection workspace state_count does not match results")

    if loading_workspace is not None:
        if loading_workspace.state_count != results.state_count:
            raise ValueError("loading workspace state_count does not match results")
        if loading_workspace.class_count != query.class_count:
            raise ValueError("loading workspace class_count does not match demand")
        loading = loading_workspace.view()

    if loading_output is not None:
        if loading_workspace is None:
            raise ValueError("link loads require a loading workspace")
        if loading_output.link_count != results.link_count:
            raise ValueError("loading output link_count does not match results")
        if loading_output.class_count != query.class_count:
            raise ValueError("loading output class_count does not match demand")
        if loading_output.set_names != context.set_names:
            raise ValueError("loading output selection names and order must match context")
        loads = loading_output.view()

    if od_output is not None:
        origin_row = operator.index(origin_row)
        if not 0 <= origin_row < od_output.origin_count:
            raise ValueError("origin_row is outside the output row range")
        row = origin_row
        if od_output.destination_count != query.destination_count:
            raise ValueError("OD output destination_count does not match demand")
        if od_output.class_count != query.class_count:
            raise ValueError("OD output class_count does not match demand")
        if od_output.set_names != context.set_names:
            raise ValueError("OD output selection names and order must match context")
        od = od_output.view().origin(row)

    with nogil:
        cpp_select_link_loading[double](
            results.read_view(), query.view(), context.view(),
            selection_workspace.view(), loading, loads, od,
        )

    return loading_output, od_output


def reduce_select_link_loading_outputs(workers, SelectLinkLoadingOutputs output not None):
    """Replace a distinct accumulator with the sum of completed worker loads.

    OD output is not involved. Workers remain unchanged; an empty worker list
    resets the target to zero. Names, order and dimensions must all match.
    """
    cdef SelectLinkLoadingOutputs worker
    cdef vector[CppSelectLinkLoadingOutputsView[double]] views

    owners = tuple(workers)
    for worker in owners:
        if worker is None:
            raise TypeError("workers must contain SelectLinkLoadingOutputs")
        if worker is output:
            raise ValueError("reduction output must be distinct from its workers")
        if (worker.link_count != output.link_count or worker.class_count != output.class_count
                or worker.set_names != output.set_names):
            raise ValueError("worker and output dimensions and ordered selection names must match")

        views.push_back(worker.view())

    with nogil:
        cpp_reduce_select_link_loading_outputs[double](views.data(), views.size(), output.view())

    return output
