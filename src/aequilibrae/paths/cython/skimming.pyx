"""Skim finalized paths using independent inputs, scratch and output owners."""

import operator

from libc.stddef cimport size_t


def skimming(
    SearchResults results not None,
    SkimmingContext context not None,
    SkimmingWorkspace workspace,
    SkimmingOutputs output not None,
    *,
    origin_row=0,
):
    """Replace output.skims[origin_row], without changing the search or other rows.

    Destinations are physical nodes 0..destination_count-1. The output row is
    independent of the search origin. Missing paths produce infinity; finalized
    intrazonal paths produce zero. Supply None for workspace when every field
    projects a stored label. Concurrent calls need separate scratch and must
    write different rows. Validation completes before any scratch or output write.
    """
    cdef CppSkimmingWorkspace[double] scratch
    cdef size_t row

    origin_row = operator.index(origin_row)
    if not 0 <= origin_row < output.origin_count:
        raise ValueError("origin_row is outside the output row range")
    row = origin_row

    # Matching sizes alone is not enough: a different field order would write
    # valid values into the wrong named matrices.
    if context.link_count != results.link_count:
        raise ValueError("skim link_count does not match results")
    if output.destination_count > results.node_count:
        raise ValueError("output destination_count exceeds results node_count")
    if output.field_names != context.field_names:
        raise ValueError("output field names and order must match skimming context")

    # Label-only calls read search results directly and need no state sums.
    if workspace is None:
        if context.additive_field_count:
            raise ValueError("additive skim fields require a workspace")
    else:
        if workspace.state_count != results.state_count:
            raise ValueError("workspace state_count does not match results")
        if workspace.field_count != context.additive_field_count:
            raise ValueError("workspace field_count must match additive_field_count")

        scratch = workspace.view()

    with nogil:
        cpp_skimming[double](
            results.read_view(), context.view(), scratch, output.view().origin(row)
        )

    return output
