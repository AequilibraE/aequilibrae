from libc.stdint cimport int64_t, uint32_t

cpdef void save_path_file(
    long origin_index,
    long num_links,
    long zones,
    long long[:] pred,
    long long[:] conn,
    str path_file,
    str index_file,
    bint write_feather,
    const long long[:] arc_pred=*,
    const uint32_t[:] mapping_idx=*,
    const int64_t[:] mapping_data=*,
) noexcept

