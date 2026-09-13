#pragma once

#include <cstddef>

namespace aequilibrae::paths::cpp::mvp {

// Borrowed search inputs. A null mask requests a full search. Otherwise the
// caller supplies one flag per physical node and its nonzero count, prepared
// together so the search need not scan or change the mask.
struct SearchQuery {
  std::size_t node_count = 0;
  std::size_t origin = 0;
  const bool *target_mask = nullptr;
  std::size_t target_count = 0;
};

} // namespace aequilibrae::paths::cpp::mvp
