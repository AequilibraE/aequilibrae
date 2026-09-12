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
cpdef void skim_single_path_with_turn_penalties(long origin,
                                                long nodes,
                                                long skims,
                                                double[:, :] node_skims,
                                                long long[:] pred,
                                                long long[:] conn,
                                                double[:, :] graph_costs,
                                                long long[:] reached_first,
                                                long found,
                                                double [:] node_turn_penalties,
                                                const long long [:] penalty_indices) noexcept nogil

cpdef int skim_arc_based_paths(long long origin,
                                long long dest_count,
                                long long skims,
                                double[:, :] node_skims,
                                const long long[:] arc_pred,
                                const long long[:] connectors,
                                const double[:, :] graph_costs,
                                const double[:] arc_turn_penalties,
                                const long long[:] penalty_indices,
                                double[:, :] arc_skims_memo,
                                long long[:] arc_visited,
                                long long[:] arc_stack,
                                long long run_id) noexcept nogil
