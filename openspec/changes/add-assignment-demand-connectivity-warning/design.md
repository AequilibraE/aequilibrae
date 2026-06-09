## Context

Traffic assignment currently validates matrix/graph centroid compatibility and graph fields, but it does not summarize
how much positive demand cannot be loaded because the selected mode graph lacks a usable path. Imported multimodal
projects can legitimately contain zones that are irrelevant for one mode, but they can also contain inconsistent OD
demand and connectors that should be surfaced before users trust assignment results.

## Goals / Non-Goals

**Goals:**

- Warn assignment users when positive class demand is attached to centroids absent from the selected mode graph.
- Warn assignment users when positive OD demand has no path through the selected mode graph.
- Include aggregate demand totals, demand percentages, affected OD-pair counts, and bounded samples.
- Keep default assignment execution non-blocking.

**Non-Goals:**

- Do not change VISUM, OSM, GMNS, or manual network import behavior.
- Do not add a new project database table.
- Do not make assignments fail by default.
- Do not solve disconnected networks automatically by creating connectors or synthetic paths.

## Decisions

1. **Validate at assignment time**

   The check belongs to `TrafficAssignment`/`TrafficClass`, because reachability depends on the selected graph mode,
   matrix core, centroid blocking option, and current graph preparation. Importers cannot know which demand core and
   mode will be assigned.

2. **Warn by default, report structured details**

   The default behavior should log warnings and expose structured validation details for programmatic access. This avoids
   breaking existing partial-network workflows while making lost demand visible.

3. **Separate missing-centroid demand from unreachable OD demand**

   Missing graph centroids usually point to mode-disabled connectors or absent zones. Unreachable OD pairs indicate graph
   disconnection, directionality, blocked centroid-through-flow behavior, or cost-field filtering. Reporting them
   separately makes diagnostics actionable.

4. **Bound expensive diagnostics**

   The implementation should aggregate totals for all positive demand, but cap detailed OD samples. Large dense matrices
   should not produce massive warning strings or result objects.

## Risks / Trade-offs

- **Large matrices can make full reachability checks expensive** -> Reuse graph/path connectivity primitives where
  possible, skip zero-demand cells, and cap samples.
- **Warnings may surprise users with intentionally partial assignments** -> Keep execution non-blocking and report
  fractions so users can judge severity.
- **Multiple classes and cores need clear attribution** -> Report results per traffic class and matrix core/computational
  view.
- **Centroid-through-flow settings affect reachability** -> Run validation against the graph as configured for the
  assignment, including connector and centroid blocking behavior.
