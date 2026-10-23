#pragma once

#include "context.hpp"
#include "outputs.hpp"
#include "search_query.hpp"
#include "search_results.hpp"
#include "workspaces.hpp"

namespace aequilibrae::paths::cpp::routing {

// The driver prepares these queries once because demand and targets are fixed.
template <typename DemandRow> struct AoNOrigin {
  SearchQuery search;
  DemandRow loading;
};

// These views only group existing components for the assignment driver.
// Operation kernels still accept their own small views, not these groups.
struct AoNInputs {
  SkimmingContext<double> skimming;
  SelectLinkContext selection;
};

struct AoNWorkerView {
  MutableSearchResults search;
  AoNWorkspace<double> workspace;
  LoadingOutputs<double> loading;
  SelectLinkLoadingOutputsView<double> selected_loading;
  // Unlike the borrowed buffers, this scalar lives in the worker table itself.
  // Origin calls take that table entry by reference so additions persist.
  double turn_cost_total = 0;
  double unassigned_demand = 0;

  void reset() noexcept {
    loading.reset();
    selected_loading.reset();
    workspace.selected_od.reset();
    turn_cost_total = 0;
    unassigned_demand = 0;
    // Search and operation kernels replace their own scratch when called.
  }
};

} // namespace aequilibrae::paths::cpp::routing
