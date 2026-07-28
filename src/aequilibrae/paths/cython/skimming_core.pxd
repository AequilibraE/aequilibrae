cimport cython

cpdef void skim_multiple_fields(long origin,
                                long nodes,
                                long zones,
                                long skims,
                                double[:, :] node_skims,
                                long long[:] pred,
                                long long[:] conn,
                                double[:, :] graph_costs,
                                long long[:] reached_first,
                                long found,
                                double[:, :] final_skims) noexcept nogil

cpdef void _copy_skims(
    double[:, :] skim_matrix,
    double[:, :] final_skim_matrix
) noexcept nogil

cpdef void skim_single_path(long origin,
                            long nodes,
                            long skims,
                            double[:, :] node_skims,
                            long long[:] pred,
                            long long[:] conn,
                            double[:, :] graph_costs,
                            long long[:] reached_first,
                            long found) noexcept nogil