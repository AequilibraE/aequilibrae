# Transit modelling capabilities: gap analysis and plan

Status: revision 3 after two review rounds (2026-10-27); executable once the decisions in §5.10 are
closed. Branch `pedro/transit_capabilities`.

This note lists the public-transport (PT) modelling capabilities that established commercial and
open-source packages offer, checks that against what AequilibraE actually does in code today, and
rates how much work each capability would take and how much it would change the data model. It ends
with cross-cutting design decisions and a four-phase execution plan.

Legend used throughout:

- **Effort**: S (≤ 2 person-weeks), M (2–6), L (1.5–3 months), XL (> 3 months, research-grade).
- **Data model impact**:
  - **None**: graph-time or result-time only.
  - **Additive**: new optional tables or columns, delivered through a migration.
  - **Structural**: changes the meaning of existing tables, or adds a new dimension (e.g. time-sliced
    demand, results per vehicle trip).

---

## 1. Baseline: what AequilibraE does today (and current bugs)

**Data** (`public_transport.sqlite`, GTFS-derived):

- `agencies`, `stops` (with `parent_station`, `transit_fare_zone`), `routes` (one row per *pattern*,
  with `pce`, `seated_capacity`, `total_capacity`), `trips`, `trips_schedule` (arrival/departure per
  stop).
- `route_links`, and `pattern_mapping` (pattern → road `link`/`dir` sequence, from map matching).
- `fare_attributes`, `fare_rules` and `fare_zones` (GTFS Fares v1, stored but never used).
- `stop_connectors`: the table exists, but no Python code reads or writes it.
- Persisted Spiess–Florian (SF) graphs, one per `period_id`, in `nodes`/`links`. Their builder config
  is in `transit_graph_configs` and periods in `periods`, both in `project_database.sqlite`, out of
  reach of any foreign key, trigger or view in `public_transport.sqlite`.
- **Bugs/gaps**:
  - **No editing API or integrity triggers**: unlike `project.network` (which uses `ProjectTable`
    CRUD and relational triggers), `Transit` only exposes read-only `get_table`, GTFS import, graph
    build and preload; `transit_triggers.sql` only updates `route_links.distance`.
  - **Schema type inconsistencies**: `stops.stop_id` is `TEXT PRIMARY KEY`, while
    `route_links.from_stop`/`to_stop` are `INTEGER`, and `route_links.sql` omits the comma between
    its foreign keys. Neither breaks `PRAGMA foreign_keys = ON` today: SQLite accepts table
    constraints without separating commas, and its affinity rules let `INTEGER` child values match
    `TEXT` parent keys (migration 006 runs `PRAGMA foreign_key_check` cleanly). Both are fragile and
    worth normalising. `stop_connectors` uses `INTEGER` IDs and integer times, lacks `distance` and
    `access_mode`, and bakes in `penalty_cost`.
  - **`Scenario` isolation bug**: `Scenario` passes the `Project` into `Transit` (`# FIXME: Hack` in
    `scenario.py`), and `Transit`, `TransitGraphBuilder` and `GTFSRouteSystemBuilder` (through
    `get_active_project()`) resolve every connection through `project.scenario`. Each scenario's
    `Transit` therefore reads and writes whichever scenario is *active*, not its own, and
    `Transit.graphs` is a class-level dict shared by every instance, so a graph built under one
    scenario can be saved into another.
  - **Extended `route_type` fallback**: `Transit.default_capacities` and `default_pces` only map
    standard GTFS `route_type` codes (`0–5, 11, 12`); extended HVT codes (`100–1700`, e.g. `109`
    suburban rail, `400` metro) fall back to `"other"` (`[30, 60]`). Those defaults are class
    attributes on `Transit`, not entries in `parameters.yml` next to the road defaults.
  - **Feed and agency are conflated**: `agencies` holds one row per *feed import* (`agency` is the
    user's label; `feed_date` is empty or the zip file name), not one row per GTFS agency, and
    `routes.agency_id` and `stops.agency_id` point at that import row. GTFS `agency.txt` and
    `feed_info.txt` are never read, so operators, their time zones and the feed version are lost,
    and a multi-agency zip collapses onto one row.
  - **Encoded keys from process-global counters**: import derives every key arithmetically
    (`route_id` and `stop_id` offset by `AGENCY_MULTIPLIER × agency_id`, `pattern_id` from
    `route_id`, `trip_id = pattern_id + n`) from class-level `Constants` counters that persist across
    imports and projects in one Python process, while `ProjectTable` allocates `MAX(key) + 1` and
    takes a single-column key. An editor mixing the two collides (a pattern created at `MAX + 1`
    lies inside the previous pattern's trip-ID block), and `route_links`, `pattern_mapping` and
    `trips_schedule` have composite keys.
  - **Leftover reference tables**: `public_transport.sqlite` carries its own `zones` (never written)
    and `modes` (seeded, never read); `TransitGraphBuilder` reads zones from `project.network.zones`.
    The transit `link_types` and `node_types` are not copies of project tables: they hold the SF
    edge and vertex vocabularies, `create_additional_db_fields` writes `link_types` on every graph
    save, and `links.link_type` is a foreign key into it. The real drift risk is `stops.zone_id`,
    which GTFS import sets from `project.network.zones` and nothing updates afterwards.
  - **Service day vs clock day**: GTFS times past `24:00:00` are stored in `trips_schedule` as values
    above `86400`, but the default period is fixed at `[0, 86400]` (the
    `aequilibrae_default_period_update` trigger rejects any change to it) and the graph builder's
    row filter (`departure >= start AND arrival <= end`) drops stop events after `end`, so the
    default whole-day graph loses every post-midnight service. `build_pt_preload` takes raw
    `start`/`end` seconds with no link to `periods`. No convention says whether periods may exceed
    `86400` or wrap.

**Import/export**:

- GTFS import for one service date per feed, with `frequencies.txt` code (currently broken).
- Bus map matching onto the road network (`split_links_at_stops`, `RouteMapMatcher`), and
  per-route-type capacities and PCEs. Per-mode maximum speeds (`set_maximum_speeds`) are accepted
  but no longer applied (below).
- `transfers.txt` and `pathways.txt` are **not** read.
- GTFS export code (`RouteSystem`, `gtfs_writer`), which cannot run (below).
- **Bugs/gaps**:
  - **Missing GTFS columns**: in `column_order.py`, `routes.txt:agency_id`,
    `fare_attributes.txt:agency_id`, `trips.txt:block_id`, `stop_times.txt:pickup_type` (and
    `drop_off_type`), `frequencies.txt:exact_times`, `stops.txt:location_type` and both
    `shape_dist_traveled` fields are commented out or omitted, and `agency.txt` and `feed_info.txt`
    are never read. Multi-agency single-zip feeds collapse onto one agency, pick-up/set-down
    restrictions on express services are lost, through-routed trips (`block_id`) cannot be
    identified, stations are dropped (only stops served by `stop_times` are saved, so
    `parent_station` points at missing rows), and stops are cut out of looped shapes by
    nearest-point projection, which falls back to straight lines.
  - **`frequencies.txt` import is broken**: `gtfs_loader.py` pops `trip` (string) from `stop_times`
    into `trip_replacements[trip]`, then checks `if trip.trip_id in trip_replacements` (integer
    `-1`), so every frequency-based trip is skipped (`< 2 stops`). A second `frequencies.txt` window
    for the same trip fails, because the template was already popped and the `trip not in
    self.stop_times` check raises. Template times are not rebased to `start_seconds`, and
    `new_trip_str` is formatted before `+= shift`, so every copy in a window gets the same key and
    only the last survives.
  - **Scheduled dwell times wiped on import**: `__load_stop_times` overwrites both `arrival_time` and
    `departure_time` with `max(arrival_time, departure_time)`, erasing all GTFS station dwell and
    layover times in `trips_schedule`.
  - **Stop-time de-confliction regressed (import aborts on ties)**: since the pandas 3 port,
    `__deconflict_stop_times` computes de-conflicted, speed-capped `times` but writes back the raw
    `arrival_time` and asserts strictly increasing times, so any trip with two consecutive stops at
    the same time (routine in minute-resolution timetables) stops the import with a bare
    `AssertionError`, and `set_maximum_speeds` corrections are never applied.
  - **Loop/lasso stop-pair deletion**: `__load_trips_table` runs `drop_duplicates(["seqkey"])` on
    `(next_stop, stop)` pairs, silently deleting repeated stop-to-stop pairs on double-loop or
    out-and-back deviation trips, and hashes `"".join(stop_times.stop.values)` without a delimiter.
  - **Fares v1 import/export bugs**: `parse_csv` rewrites every empty numeric field as `0`, so empty
    `transfers` (*unlimited* transfers) becomes `0` (*no* transfers) and empty `transfer_duration`
    (*no limit*) becomes `0` before `Fare.populate`, which also turns `price = 0` into `None`.
    `FareRule` casts `contains` to `int`, and `write_fares` casts `origin_id` and `destination_id`,
    crashing on alphanumeric `zone_id`s.
  - **GTFS export cannot run**: `RouteSystem` is not exported by `aequilibrae.transit` and nothing
    calls it. It was ported from another schema: every reader in `route_system_reader/` queries
    tables that do not exist in `public_transport.sqlite` (`transit_agencies`, `transit_stops`,
    `transit_routes`, `transit_patterns`, `transit_trips`, `Transit_Pattern_Links`), so
    `load_route_system()` fails on its first query, and `write_trips` and `write_fares` query
    `Transit_Agencies`, `Transit_Fare_Attributes` and `Transit_Fare_Rules`. The writers also
    hard-code a foreign `agency_url`, leave `agency_timezone` empty, blank every `parent_station`,
    write internal numeric IDs and invent one seven-day service per pattern. No test exercises it;
    it needs a rewrite, not a repair.
  - **Import does not scale**: `gtfs_loader.py` builds a Python object per stop, route, trip and
    pattern and runs a `groupby` loop per trip, and `RouteMapMatcher` computes a full shortest-path
    tree (`early_exit=False`) for each intermediate stop of every matched pattern, so metro-scale
    feeds (hundreds of thousands of trips, millions of `stop_times` rows) take hours to import.
  - **Only buses and trolleybuses can be map matched**: `mode_correspondence.py` maps only
    `route_type` `3` and `11` to the road mode `t`, so `map_match()` drops every other type with a
    warning whatever `route_types` it is given, and extended-type buses (`700–716`, `800`) and
    street-running trams get no `pattern_mapping` rows. Rail, tram and ferry patterns keep their
    `shapes.txt` (or stop-to-stop) geometry. Anything that needs physical links (`A5` run times,
    `C1` link loads, road preload) exists for route types `3` and `11` only.

**Graph** (`TransitGraphBuilder`):

- Builds a classic SF graph: stop, boarding, alighting and OD vertices; on-board, dwell, boarding,
  alighting, connector, inner/outer transfer and walking edges.
- Connectors are **Euclidean** (nearest-neighbour or overlapping-regions) at a fixed walking speed.
  The road network is only used optionally (`connector project match`) to redraw connector geometry
  and recompute connector `trav_time` from the matched path length, still measured in
  `projected_crs` on the graph passed in (the car graph in the gallery, so walkers obey one-way
  rules).
- **Bugs/gaps**:
  - **Inverted `a_node`/`b_node` (`head`/`tail`) and non-contiguous `zone_id` crash**:
    `TransitGraphBuilder` sets `a_node = head` and `b_node = tail` (the opposite of road `links`,
    `links.sql` docs, and `paths/cpp/`). As a result, `create_line_geometry` draws every edge
    backwards and `__connector_project_match` routes shortest paths backwards on one-way links; in
    addition, `vertices.at[row.a_node - 1, "geometry"]` crashes or miswires geometries whenever
    `zone_id` has gaps, does not start at `1`, or is not stored in ascending order.
  - **Physical time and generalised cost are conflated in `trav_time`**: `alighting_penalty` (default
    `480 s`), `0.5 * uniform_dwell_time`, and the `access`/`egress`/`walk` perception factors are
    baked directly into `edges["trav_time"]`. Those cost attributes are also commented out of
    `__config_attrs`, so a reloaded graph neither separates physical time from cost nor records how
    it was costed.
  - **Headways are not true period averages, and can collapse to zero wait**:
    `_add_mean_headway_to_segments` sorts by `trip_id` before `arrival`, so consecutive differences
    are negative whenever trip IDs are not in time order. It then runs `.groupby("pattern_id").min()`
    without `trip_id`, which picks the **single smallest difference across all stops and trips**
    (making the subsequent `.mean()` a no-op). When that minimum is zero or negative,
    `_create_line_segments` sets `freq = inf`, so the whole pattern gets **zero wait**. (The
    `pattern_id = row.seq` typo is mostly harmless, because a change in `seq` already resets the trip
    index except when the previous pattern's filtered rows end at `seq = 0`.) It also does not
    distinguish scheduled headway (what riders see while a line runs) from the effective period trip
    rate $N_{\text{trips}} / T_{\text{period}}$ (what SF waits and capacity use, Phase 1).
  - **Stop-event period filtering is inconsistent**: `_create_line_segments` keeps *every* segment
    of a pattern once any stop event of any of its trips falls inside `[start, end]` (its join
    ignores `seq`), so topology is not truncated, but a trip that only touches the window (e.g.
    reaches its last stop at 07:01) pulls its pattern into the period. Headways and run times then
    come from in-window stop events only, and segments no in-window trip covers fall back to
    whole-day mean run times.
  - **No street-level inter-stop transfers**: `outer_transfer` and `walking` edges are only built
    where `parent_station` is non-empty. With `blocking_centroid_flows=True`, surface transfers
    between cross-street or opposite-kerb stops without a shared `parent_station` (most GTFS bus
    stops) are impossible.
  - **Parallel transfer-edge topology**: enabling `inner_transfer` or `outer_transfer` connects
    `alighting → boarding` directly with finite frequency alongside the infinite-frequency `alighting
    → stop` edge, distorting SF node splits and bypassing `boarding` edges (undercounting
    `boardings` skims and splitting flows needed by fares, capacities and stop summaries).
  - **Loop/self-intersecting patterns in `TransitGraphBuilder` (unverified)**: once `gtfs_loader.py`
    stops dropping repeated stop pairs, patterns visiting a `stop_id` twice still lack a graph-build
    regression test (`_create_dwell_edges` merges on `line_seg_idx`, so a test should confirm graph
    wiring).
  - **Web Mercator connector distances**: `projected_crs` defaults to `EPSG:3857` and
    `_create_connector_edges` measures connectors with a `KDTree` in that CRS, while outer-transfer
    and walking edges use haversine. Mercator stretches distances by $1 / \cos(\text{latitude})$, so
    connector walk times and `distance_upper_bound` run about 15% long in Coquimbo (30° S) and 50%
    or more at 50° latitude, and disagree with the walking edges in the same graph. `get_srid`
    returns `4326` unconditionally, so there is no project-level projected CRS to fall back on.
  - **Connector generation defects**: `nearest_neighbour` links each *stop* to its nearest zone
    centroid, so a zone that is no stop's nearest centroid gets no connector and becomes
    unreachable (and zero-cost under today's skims). A finite `distance_upper_bound` crashes
    `nearest_neighbour` (`KDTree` returns index `n` for out-of-range stops, and `.iloc` raises),
    and under `overlapping_regions` gives an isolated zone an infinite radius, connecting it to every
    stop. `max_connectors_per_zone > 0` fails under pandas 3, because `groupby("b_node").apply`
    drops `b_node` from the columns. None of these paths is tested.
  - **Dwell counted twice**: stop-to-stop times already include dwell (many feeds set `arrival =
    departure`, and the importer currently merges the two), yet every intermediate stop also gets a
    `uniform_dwell_time` (`30 s`) `dwell` edge, plus `15 s` on each boarding and alighting edge, so
    in-vehicle time grows by about `30 s` per stop passed.
  - **Two transit graph classes and an alias shim**: the legacy `TransitGraph(GraphBase)` in
    `graph.py` (still exported from `aequilibrae.paths`) exists next to `NewTransitGraph`, and
    `paths/public_transport.py` is an import alias marked `FIXME: Remove in 2.0`
    (`paths/graph_building.py` is a road-graph alias, not transit code).

**Assignment** (`TransitAssignment` / `HyperpathGenerating`):

- Uncongested optimal strategies only, in standalone Cython (`public_transport.pyx` / `hyperpath.pyx`)
  and parallel by destination. One all-or-nothing loading of strategies; no equilibrium loop.
- Not yet ported to `paths/cpp/`: reallocates full thread buffers per `assign()` call, initialises
  and frees a `FourAryHeap` plus skim scratch buffers on **every destination**, and has no in-place
  `update_costs` / `update_frequencies` API. `HyperpathGenerating.__init__` copies the edge frame
  and rebuilds the CSC structure, so every cost change rebuilds the object (the heap's O(1)
  `reset_heap` would remove the per-destination allocation without the port).
- Output is raw SF edge volumes only.
- **Assignment & skim bugs**:
  - **Demand misassigned when zones are not stored in `zone_id` order**:
    `OptimalStrategies.execute` indexes `hypergraph._o_vert_ids` and `_d_vert_ids` by raw
    `coo_matrix` row/col offsets. `TransportClassBase` checks `matrix.index` against
    `graph.centroids`, which `sanitise_centroids` sorts, but `od_node_mapping` follows the order of
    `self.zones`: the `zones` table's rowid order, or the frame passed to `add_zones`. When the two
    orders differ, demand from one zone is loaded at another zone's vertices (reproduced on a
    two-zone graph), and skims come back indexed in `od_node_mapping` order.
  - `waiting_time` is computed by subtraction (`trav_time - in_vehicle_trav_time - egress_trav_time -
    access_trav_time`), which sweeps every `alighting_penalty` (`480 s`), boarding/alighting dwell
    half-time, `walking_trav_time` and `transfer_time` into reported wait time.
  - `transfers` only counts `inner_transfer` and `outer_transfer` edges (both `False` by default), so
    default `transfers` skims are all zeros.
  - Unreachable OD pairs (`fmax`) are overwritten with `0.0`, while `skim_matrix` and component
    `skim_i_vec` are zero-initialised and disconnected zones (`nodes_to_indices == -1`) are skipped
    before `compute_SF_in` — making disconnected ODs look like zero-cost trips to mode choice.
  - `HyperpathGenerating.run(o, d, v)` copies `u_i_vec` from thread 0 after it has also processed
    the remaining skim destinations, so with skimming enabled it returns another destination's
    labels (origin label `0.0` instead of `1665 s` on the Spiess & Florian network).
  - **No skim persistence or result provenance**: `TransitAssignment` has `save_results` for SF edge
    volumes only; there is no `save_skims`, and the skimming example's "saving the skimming results"
    step calls `save_results`. Saved result tables carry no `period_id`, graph-config hash or class
    costing, so a result cannot be traced back to the graph and parameters that produced it.

**Road coupling**: `Transit.build_pt_preload` turns scheduled trips into a PCE preload on road links
(through `pattern_mapping`). Nothing flows back from road to transit: bus run times ignore road
congestion, and unmapped patterns silently contribute zero preload.

**Reusable machinery elsewhere in AequilibraE**:

- `ProjectTable` (`SpatialProjectTable` / `NonSpatialProjectTable`) transactional table CRUD and
  trigger patterns on `project.network`.
- C++ routing/loading infrastructure in `paths/cpp/` (`FourAryHeap`, `workspaces.hpp`,
  `select_link_loading.hpp`), driven from Python by `AssignmentInputs`
  (`paths/assignment_context.py`), whose `update_costs` the SF port should mirror.
- Road route choice: choice-set generation, path-size logit and select-link.
- State-augmented / contracted non-additive pricing design (`pedro/toll_treatment`).
- Scenarios, periods, a results database, and the matrix API.

---

## 2. Capabilities found in practice: assignment, crowding and fares

This section and the next summarise what established commercial and open-source packages offer:
frequency-based and timetable-based commercial suites, an agent-based simulator, and open Python
toolkits. They are based on public documentation, product knowledge and source code. Packages are
not named; §4 rates how widely each capability is offered.

### 2.1 Assignment and route choice

- **Strategy-based (frequency-based) assignment**: optimal strategies is the common core. Extended
  variants add:
  - State-augmented networks ("journey levels") for transfer-dependent fares, mode-sequence rules
    and park-and-ride-once.
  - Logit choice at origin connectors and auxiliary (walk) nodes.
  - Flow distribution at stops by frequency *and* travel time, and passenger information levels.
  - Per-line and per-node boarding penalties, wait-time factors, headway fractions, and perception
    weights per mode.
- **Line-level headway-based route choice** (`B4`): route search on line headways and run times (no
  vehicle timetable), with a choice model at stops.
- **Path enumeration and evaluation**:
  - Multi-route choice sets, built by enumeration with spread factors or by pruning (re-running the
    shortest path with each route or mode combination removed).
  - Logit or nested logit over paths, which handles non-additive fares naturally and nests under
    mode choice. Some packages add incremental (pivot) logit against a reference scenario.
- **Timetable-based assignment**:
  - Connection search by branch & bound, shortest path, RAPTOR or connection scan, with Pareto
    filtering on departure and arrival time.
  - Preselection of dominated connections, then a choice model across connections with an
    independence correction.
  - Departure-time or adaptation-time choice, penalising the shift from the wished departure time.
- **Other multipath methods**: generalised-cost multipath assignment with headway-based path
  combination, and logit-based stochastic user equilibrium.

### 2.2 Crowding and capacity

- **Congested assignment**: volume-dependent in-vehicle discomfort (crowd curves, seated vs
  standing) solved by an FW/MSA-type outer loop.
- **Capacitated assignment**:
  - Frequency-based: strict capacity through effective headways or stop waiting penalties (MSA),
    giving capacity-feasible flows.
  - Timetable-based: hard restraint (boarding refused when full) or soft restraint (discomfort),
    solved iteratively. Some implementations ignore crowding effects at stops.
  - Route-level capacity constraints and wait-time adjustment for full vehicles.
- **Agent-based simulation**: every vehicle is simulated with seats, standing room and door
  operations, so dwell time depends on boardings and **bunching and denied boarding emerge**.

### 2.3 Fares

- **Fare systems** over sets of lines, with fares summed across independent systems.
- **Ticket types**: free, flat, distance (fare stages or fare points, high/low bands), accumulated,
  zone count, from–to zone matrix, short-distance, time-based and direct-distance.
- **Rules**: per-transport-system supplements; transfer allowances and durations; tap-free transfer
  networks; consecutive-leg logic; ticket selection by priority or by cheapest valid combination.
- **Revenue** calculation and allocation to operators.

---

## 3. Capabilities found in practice: supply, outputs and agent-based simulation

### 3.1 Supply, network preparation and editing

- **Stops and access**: three-level stop hierarchy (stop → stop area → stop point) with transfer
  walk-time matrices; dedicated walk and bike networks linked to stops; multiple connectors per zone
  with a speed curve from short walk legs to long drive legs; pruning of redundant connectors and
  footpaths; clustering of stops and zones; non-transit legs for walk and drive access and
  park-and-ride.
- **GTFS handling**: filtering by area, time, date, service or trip; timetable-to-frequency
  conversion by pattern; pattern and direction builders; map matching of lines and GPS tracks onto
  roads; common-trip aggregation (combined headway $1 / \sum 1/h$).
- **Editing**:
  - Line route and time-profile editing, timetable and graphical-timetable editors.
  - Sketching new lines from stop sequences, and generating departures from headway or interval
    specifications.
  - Database triggers that cascade deletes and keep geometry and lengths current when stops move.
- **Supply dynamics**: PT run times from road link times (transit time functions) and dwell times
  from boardings and alightings.
- **QA**: integrity tests and automated fixes.

### 3.2 Outputs and analysis

- **Loads**: boardings and alightings, line and segment loads, transfers between lines and modes,
  PT volumes on road links.
- **Analyses**: strategy and path analyses, stop-to-stop and traversal matrices, select-link and
  select-line, screenline checkpoints.
- **Skims**: journey and perceived journey time, in-vehicle time, access/egress, origin vs transfer
  wait, number of transfers, fare, service frequency, adaptation time, crowding.
- **Operator indicators**: vehicle-km and vehicle-hours, line costs and revenues, line blocking
  (vehicle scheduling), interchange analysis.
- **Calibration**: OD adjustment to counts, by LP-based pivoting or matrix estimation.
- **Interop**: export to other modelling and spreadsheet formats.

### 3.3 Agent-based and on-demand simulation

- **Choice** through co-evolutionary replanning (route, mode, departure time) rather than
  closed-form assignment.
- **Routing**: RAPTOR with range queries, utility-based transfer penalties and intermodal
  access/egress with feeder modes (bike, car, on-demand) and radius-expanding stop search.
- **Ecosystem**: crowding-aware scoring (load-factor discomfort, denied boarding); distance and zone
  fares; demand-responsive transit with dispatch and rebalancing; evolving paratransit supply;
  GTFS import with mapping onto the road network; a deterministic PT simulation for speed.

---

## 4. Consolidated feature list

**AequilibraE today**: Yes, Partial or No. **In practice**: how widely established commercial and
open-source packages offer the capability: **Standard** (essentially all), **Common** (most) or
**Specialist** (a few).

### A. Supply, editing and network data

| # | Feature | AequilibraE today | In practice | Effort | Data model |
|---|---|---|---|---|---|
| A1 | GTFS `transfers.txt` (stop-, route- and trip-level rules, types `0–5`) and `pathways.txt`, stations and entrances stored as `location_type` rows, min transfer times, stop hierarchy, and street inter-stop footpaths independent of `parent_station` | Partial (parent stations only) | Standard | M | **Structural** (`transfers`, `pathways` and `stop_footpaths` tables are additive, but `stops` gains station and entrance rows that consumers must filter by `location_type`) |
| A2 | Network-based access/egress: walk on the road graph (pre-skimmed `stop_connectors` or embedded walk links), multiple connectors with distance bounds, bike/drive access legs | Partial (Euclidean) | Standard | M | Structural on unused `stop_connectors` (`stop_id` `TEXT`, `zone_id`, `access_mode`, `distance`) |
| A3 | Vehicle types with seated/total capacity and `standing_area_m2` per trip/pattern | Partial (per pattern, no standing area) | Common | S | Additive (`vehicle_types`, `trips.vehicle_type`) |
| A4 | PT editing suite (`transit.edit` + SQLite triggers): CRUD on agencies/stops/routes/patterns/trips, pattern duplicate/flip/extend/re-map-match, headway/schedule editing, scenario variants | No | Standard | L | Additive (see §5.1; database-allocated keys replace encoded IDs for new rows) |
| A5 | Bus run times from road congestion (TTFs) and dwell time from boardings/alightings | No | Common | M | Additive (`route_link_seq` and `stop_fraction` on `pattern_mapping`, Phase 2; TTF definitions, Phase 4.2) |
| A6 | Multi-feed/multi-agency (`agency.txt`, `routes.agency_id`), feed provenance (`feed_info.txt`), multi-date import, GTFS export rewritten against the real schema; GTFS Fares v2 | Partial (one `agencies` row per import; export cannot run) | Standard | M–L | **Structural** (`feeds` table split from `agencies`, Phase 2; `feed_id` on every table holding GTFS IDs; Fares v2 + extensions beside v1, Phase 3) |
| A7 | Stop-level `pickup_type`/`drop_off_type` permissions and through-routing / interlining (`block_id`, GTFS `transfer_type=4` stay-on-seat) | No | Common | S–M | Additive (`route_links` flags, `in_seat_transfers`) |
| A8 | Canonical transit mode hierarchy & extended GTFS `route_type` (`100–1700`) mapping (local bus vs BRT vs rail submodes), driving default capacities, PCEs and map-matching eligibility | Partial (raw `route_type` subset; only types `3` and `11` can be map matched) | Standard | S | Additive (`transit_modes` mapping) |
| A9 | Metro-scale GTFS import: vectorised loader (array-based pattern hashing, bulk inserts) and faster map matching, with import-time and memory targets | Partial (object-per-trip loader; a full shortest-path tree per stop when map matching) | Standard | M–L | None |

**Notes**:

- **A1** fixes two baseline defects at once: respecting explicit GTFS transfer rules, and generating
  walk-network footpaths between nearby stops (e.g. within 400–800 m via `split_links_at_stops`) even
  when `parent_station` is null. Standardise the topology so stop-to-stop footpaths run `stop_A →
  stop_B` (`freq = inf`), reserving direct `alighting → boarding` edges strictly for line-to-line
  timed (`transfer_type=1`) or stay-on-seat (`A7`) rules (an edge cannot forbid anything: stop-level
  `transfer_type=3` removes the footpath, route-level prohibitions need per-line arrival copies of
  the stop). That keeps all normal boardings flowing through `boarding` edges.
- **A2**:
  - Snapping stops to the walk-mode road graph lets us either embed walk links directly in the SF
    graph (auxiliary-transit style, giving walk-all-the-way ODs and transfers with no connector
    count truncation) or pre-skim shortest walk paths into `stop_connectors` (needed by RAPTOR in
    `B7` and smaller for SF). Supporting both from the same snapped walk graph keeps SF and RAPTOR
    consistent. Redesign `stop_connectors` in Phase 2 (`TEXT` `stop_id`, `zone_id`, `access_mode`,
    `distance`, physical `trav_time`, no `penalty_cost`).
  - Note that multi-connector zones still assign AoN to one stop under standard SF unless paired with
    connector logit (`B4`).
- **A3**: adding `standing_area_m2` (or crush standing density in pax/$\text{m}^2$) alongside
  `seated_capacity` and `total_capacity` is needed to apply standard UK WebTAG / Australian ATAP
  crowding curves directly in `B5` and `C5`.
- **A4** combines `ProjectTable`-backed gateways on `project.transit`, add/delete and pattern
  geometry/map-matching operations, cascade and consistency triggers in `transit_triggers.sql`, and
  §5.1 service windows.
- **A5** closes the road↔transit loop. `pattern_mapping` already uses `seq` for road-link order
  (`PRIMARY KEY(pattern_id, "seq")`) and does not record which stop-to-stop segment each road link
  belongs to; adding `route_link_seq` at map-matching time (Phase 2) feeds the iterative speed/dwell
  feedback algorithm (Phase 4.2). Stops usually sit mid-link (`RouteMapMatcher` splits links at
  stops, then merges the halves back into one `pattern_mapping` row), so a road link that hosts a
  stop belongs to two segments: keep one row per road link, with the `route_link_seq` of the segment
  it starts in and a nullable `stop_fraction` giving where the stop sits along it, so `A5` splits the
  link's time between the two segments while `build_pt_preload` and the plugin's layer are
  untouched. Only map-matched modes get road run times; rail, tram and ferry keep their
  schedule-derived `route_links` run times.
- **A6**: `agencies` is today an import-batch table: each import gets one row and its encoded IDs
  keep feeds apart, but GTFS `agency.txt` and `feed_info.txt` are never read. Add a `feeds` table
  (feed path, `feed_info` version and validity dates, description, import timestamp), real GTFS
  agencies in `agencies`, and `feed_id` on `routes`, `stops`, `trips` and the fare, transfer and
  pathway tables (nullable for project-level objects such as modelled fares). The migration copies
  each `agencies` row into `feeds` with the same ID and keeps it as that feed's single agency, so
  existing `routes.agency_id` and `stops.agency_id` stay valid and no parent table is rebuilt. Part
  1 (feeds, import and the exporter, Phase 2) and part 2 (Fares v2 data model and export, Phase 3)
  are `M` each; the fare calculator is `B12`.
- **A7** prevents illegal local trips on pick-up-only/set-down-only express stops (must be part of
  pattern identity so differing permissions split `pattern_id`) and removes false transfer/wait
  penalties on through-routed rail and bus lines via zero-penalty `alighting → boarding` dwell edges.
- **A8** stops extended GTFS rail/metro feeds defaulting to 60-passenger bus capacities and gives
  `B1` user-meaningful modes (e.g. separating BRT from local bus when both have `route_type = 3`).
  Default capacities and PCEs move from class attributes on `Transit` to `parameters.yml`, next to
  the road defaults.
- **A9** replaces the object-per-trip loader with array-based pattern hashing and bulk inserts, and
  stops `RouteMapMatcher` computing a full shortest-path tree per intermediate stop. It lands with
  the multi-feed lifecycle in Phase 2 because both rewrite the same import path, reads the GTFS
  fields the schema could not store before (§1), and carries the import targets from §5.8.

### B. Assignment and route choice

| # | Feature | AequilibraE today | In practice | Effort | Data model |
|---|---|---|---|---|---|
| B1 | Full generalised cost per `TransitClass`: weights per component and mode/line, boarding penalties per line/stop, transfer penalties, linear wait-time factors, class-specific VOT | Partial (baked into `trav_time`) | Standard | M–L | **Structural** for persisted graphs (physical `links` columns and default costing land in Phase 1; see §5.4) |
| B2 | Additive fares (flat per boarding, per network/boarding area, linear per-km) in cost and skims | No | Standard | S | None (compiled from Phase 3 fare tables) |
| B3 | Journey levels, pattern chords & destination-thread OD zonal pricing: transfer-dependent fares, intra-line distance/station-pair fares, mode-sequence rules | No | Common | L–XL | None (graph-time vertex and edge expansion; needs the §5.3 cost overlay) |
| B4 | Extended SF choice: logit at origin connectors (`S`, ships with `A2`) and at destination-side and interior nodes, RTPI / frequency-and-time stop split, regular-headway wait models | No | Common | M–L | None (SF kernel extension) |
| B5 | **Congested transit assignment**: in-vehicle crowding discomfort (seated vs standing / pax per $\text{m}^2$) and optional station/walkway VDFs | No | Common | L | Additive (segment/walkway capacity & standing area on graph) |
| B6 | **Capacitated assignment**: strict capacity with on-board (`dwell`) priority via effective frequencies or fail-to-board | No | Common | L | Same as B5 |
| B7 | **Schedule-based routing** (rRAPTOR / McRAPTOR): Pareto connection search and range-query skims by departure window | No | Common | L–XL | Additive (optional time-sliced skim outputs) |
| B8 | **Schedule-based assignment**: connection choice, departure-time / adaptation choice, then trip capacity | No | Common | XL | **Structural** (time-sliced demand, results per trip/segment) |
| B9 | Path-based multi-route PT choice (enumeration/pruning + (nested) logit, PSL) — exact for per-journey caps and whole-path fares, and for time-window transfers on timetable paths (`B7`) | No | Standard | L | Additive (path tables in results) |
| B10 | Park-and-ride / kiss-and-ride / feeders: (a) matrix-coupled station choice with lot capacity (`M`); (b) multimodal graph (`L`) | No | Standard | M–L | Additive (`pnr_lots` table) |
| B11 | Multi-class transit assignment over a shared physical SF graph with class-specific generalised costs | Partial | Standard | S | None |
| B12 | Fare calculator over complete journeys: legs → price with per-leg and per-operator breakdown (Python reference, then C++) | No | Common | M | None (reads the Phase 3 fare tables) |

**Notes**:

- **B1/B2/B11**:
  - Separate unweighted physical attributes on the graph from class-level generalised cost on
    `TransitClass` (mirroring road `Graph` + `TrafficClass`; see §5.4).
  - Allow per-mode (`A8`), per-line and per-stop overrides, plus a monetary fare channel converted via
    class VOT. Because costing happens at assignment time, multiple classes share one persisted graph.
  - Part 1 (default costing that reproduces today's baked-in values) lands in Phase 1, the rest in
    Phase 3. Non-linear wait-time functions belong to `B4`.
  - Without state a `boarding` edge cannot tell a first boarding from a transfer, so today's
    `alighting_penalty` is charged on every alighting, the last one included. Add an optional
    two-level stop layer (not yet boarded / boarded): duplicate stops, walking edges, boarding edges
    and egress connectors, keep the in-vehicle network shared, and send `alighting` edges to level-1
    stops (requires `blocking_centroid_flows=True`). Transfer penalties then differ from boarding
    penalties; transfers (level-1 boardings), $P_{\text{PT}}$ (level-0 boardings) and first wait
    (waits at level-0 stops) become additive skims; and flat fares with free transfers price exactly
    on level-0 boarding edges. It is `B3`'s unpaid/paid construction, moved forward (Effort S–M, Data
    model None; timing in §5.10).
  - Wait weights and headway fractions must be set per node (stop) or per class: scaling one line's
    `freq` changes the frequency-share split at the stop.
- **B3**:
  - Replicating vertices across 3–6 journey levels handles boarding-count fares, flat transfer
    discounts and mode-sequence rules with the SF kernel unchanged; `B5`/`B6`/`C1` must sum edge
    volumes across all level-copies of each physical segment (same rule as `non_additive_tolls` in
    `toll_modeling.rst`, branch `pedro/toll_treatment`).
  - **Scope boundary**: journey levels do *not* scale to per-journey **fare caps** (combinatorial
    remaining-cap balance), which belong in `B9`, or to **clock-limited transfers**
    (`transfer_duration`, e.g. 90-min expiry), which need timetable times (`B7`/`B8`, or `B9` on
    timetable paths). Naive origin-zone replication also blows up ($\times N_{\text{zones}}$) for
    OD zonal matrices — **however**, because SF runs one destination $d$ per pass, an OD zonal fare
    depending only on `(boarding_zone, dest_zone(d))` can be priced per destination thread with
    **zero vertex replication**, under these conditions:
    - The fare is charged on **access connectors** (zone → first stop), with
      `blocking_centroid_flows=True` and walk-all-the-way ODs routed outside `access_connector` edges
      (otherwise walk-only trips pay the transit fare). Charging "on first boarding" is a 2-level
      state (`unpaid → paid`), which avoids both walk-only and access-walk edge cases. It duplicates
      only the pedestrian layer (stops, walking, boarding and egress edges), with the in-vehicle
      network shared, so it costs well under $2\times$ graph size; levels beyond two (boarding counts)
      must also copy the in-vehicle network per level.
    - All egress stops of $d$ share one fare zone.
    - Access-connector pricing breaks when riders walk from the access stop to a stop in another fare
      zone before boarding, or walk at the egress end to a stop in another zone.
    - The kernel needs **per-destination edge costs** (a cost overlay or callback per destination),
      which is an API change rather than zero work.
  - Single-line distance-stage or station-pair fares can use **boarding-to-alighting pattern chords**
    (`stop_u_board → stop_v_alight`). Chords grow quadratically in stop count (fine for rail, heavy
    for long bus lines), and chord flows must be projected back onto covered physical segments — with
    chords spanning $(u < m < v)$ treated as continuing `dwell` priority at stop $m$ for `B5` and
    `B6`.
- **B4**:
  - **Connector logit** is required for `A2` to spread zone demand across multiple walk connectors
    (standard SF treats `freq = inf` connectors as deterministic AoN). Origin-connector logit can be
    applied directly at zone origins after the backward pass (origins are never interior nodes when
    `blocking_centroid_flows=True`), with the origin label replaced by the logit expected cost; it is
    `S` and ships with `A2` in Phase 3. Destination-side logit is logit at interior `alighting` and
    stop vertices: it needs an efficient-edge rule (only edges with $u_j < u_i$) to keep strategies
    acyclic and the label order valid, and stays in `B4` (`M`).
  - **Stop split & headways**: scaling `freq` by `1 / wait_time_factor` (e.g. `0.5` for clockface
    service) gets single-line wait right but overstates common-line wait savings and still splits
    riders purely by $f_a$ regardless of in-vehicle time differences. Adding RTPI / countdown
    information splits (Gentile et al. 2005; passenger information levels) handles modern
    app-informed riders and express-vs-stopper corridors properly.
- **B5/B6** cannot simply reuse road `LinearApproximation` (FW/CFW/BFW):
  - Even for separable fixed-frequency crowding (`B5`), Spiess & Florian's (1989, §5) convex
    objective is $\sum_a \int_0^{v_a} c_a(x)\,dx + \sum_d \sum_i w_i^d$, with waits kept per
    destination ($v_a^d \le f_a w_i^d$), so any exact line search and gap must include each strategy
    direction's total waiting time $W = \sum_d \sum_i w_i^d$. $W$ is linear in the step, so it only
    adds the constant $W_y - W_x$ to the line-search derivative, the hook road
    `LinearApproximation` already uses for fixed and turn costs; what cannot be reused is its road
    subproblem and per-class result handling.
  - Once **seated-vs-standing priority** (continuing `dwell` riders keep seats before `boarding`
    riders; Schmöcker et al. 2011), **boarding-dependent dwell times (`A5`)**, or **effective
    frequencies (`B6`, Cepeda–Cominetti–Florian 2006)** are active, the Jacobian is **asymmetric**
    and no convex Beckmann objective exists — invalidating exact line search and CFW/BFW Hessian
    conjugacy. Plan on **MSA, MSWA or self-regulated averaging** and the **Marcotte–Nguyen / CCF
    variational-inequality gap**:
    $$\text{Gap}(v) = \frac{\sum_a c_a(v)\,v_a + \sum_d \sum_i w_i^d(v) - \sum_{od} d_{od}\,u_{od}(v)}{\sum_a c_a(v)\,v_a + \sum_d \sum_i w_i^d(v)}, \qquad w_i^d(v) = \max_{a \in A_i^+} \frac{v_a^d}{f_a(v)},$$
    which is non-negative and zero only at equilibrium (Marcotte & Nguyen 1998; CCF 2006). The
    maximum is per destination, so averaged flows on finite-frequency edges must be kept per
    destination across iterations (about 10 GB dense at 4,000 zones × 300,000 boarding edges);
    aggregate $v_a$ is not enough. With `B4` logit active the VI gap stays positive, so also report
    the fixed-point gap $\lVert y - v \rVert / \lVert v \rVert$.
  - **On-board priority in B6**: residual capacity available to `boarding` passengers on segment $k$
    is $C_k - v_{\text{prio},k}$ (using period trip count $N_{\text{trips}} \times
    \text{total\_capacity}$, in the period units of the demand), never $C_k -
    v_{\text{on-board},k}$. Priority flow is `dwell` plus `in_seat_transfers` plus chords passing the
    stop, summed over journey-level copies; timed-transfer edges count as boardings. Clamp the
    residual at zero while averaging. CCF's existence result assumes every OD keeps an uncapacitated
    alternative (e.g. walking): add one or report denied demand, because effective frequencies
    otherwise clip to `MIN_FREQ` and over-capacity ODs show finite costs near $10^{20}$ s. Codina &
    Rosell (2017) give an MSA heuristic for the same congested, capacitated strategy model.
  - **Station/walkway crowding**: because station walkways and transfers are ordinary SF edges, BPR
    VDFs on `walking`/`transfer`/`access` edges fit into `B5`'s outer loop at minimal extra cost.
- **B7**:
  - Commit to **rRAPTOR / McRAPTOR** (Delling et al. 2015) in `paths/cpp/` as the primary C++
    schedule router: its rounds give the arrival-time × transfers Pareto set by construction, and
    rRAPTOR reuses labels across departures, which suits origin-parallel range skims (profile CSA
    can also keep per-transfer labels; the choice is about range skims, not representability).
    One-hop footpath relaxation is exact only for transitively closed footpaths, so `A1`'s
    `stop_footpaths` must be closed within the walk radius and `A2`'s pre-skimmed `stop_connectors`
    must reach every stop within it. `B8` choice sets need relaxed dominance (journeys within a
    tolerance of the best), because a strict Pareto set is too narrow for logit choice.
  - **Memory discipline**: materialising one dense skim core per time slice across a metropolitan
    zoning system ($4{,}000^2 \times 8 \text{ skims} \times 24 \text{ slices} \times 8\text{ B}
    \approx 25\text{ GB}$ per period) should be opt-in. By default, integrate range queries over the
    period window inside C++ to output summary cores (mean perceived time, adaptation time,
    50th/90th percentile time, viable departures, unreachable share).
- **B8** is the only item that needs a desired-departure profile per period (an OD × departure-slice
  matrix only if the demand model supplies one) and results keyed by vehicle trip. Do it after `B7`
  has proved the engine.
- **B9**:
  - Route/mode-breaker pruning (re-running the shortest path with each route or mode combination
    removed) maps onto AequilibraE's existing choice-set and PSL code if paths are expressed as legs
    `(line, board_stop, alight_stop)`.
  - It prices **per-journey caps** and other whole-path fares exactly without journey-level state
    explosion, and **time-window transfers** exactly only when paths carry timetable times (`B7`);
    on headway-based paths elapsed time includes random waits. Daily or weekly **fare caps** depend
    on the rider's other trips, so no single-trip assignment prices them exactly: use an expected
    per-trip fare, or tour-level demand. Common lines survive if legs are
    `(board_stop, line bundle, alight_stop)` with combined-headway waits inside fare-homogeneous
    bundles; the existing BFSLE bans single links, so line breakers need group bans.
- **B10** — ship **B10a (matrix-coupled station choice, Effort M)** before **B10b**:
  - Skim congested auto $(o \to k)$ from `TrafficAssignment` and transit $(k \to d)$ from
    `TransitAssignment`, evaluate discrete lot choice $P(k \mid o, d)$ with Sinkhorn/dual shadow
    prices for lot capacity, and split PnR demand into auto leg matrix $D_{ok}^{\text{auto}}$ (loaded
    in `TrafficAssignment`) and PT leg matrix $D_{kd}^{\text{PT}}$ (loaded in `TransitAssignment`).
    Lots enter both graphs as pseudo-zones (road destinations, transit origins with walk
    connectors), so both matrix indices grow to zones plus lots.
  - Unlike a monolithic car+transit SF graph (`B10b`), this needs **no `B3` dependency**, keeps car
    use at the home end, and loads PnR cars onto congested station access roads. Using the same lot
    for the PM return as for the AM outbound trip still needs tour- or PA-based demand, or deriving
    the PM lot choice by transposing the AM choice. KnR and bike/DRT feeders use the same split
    without lot shadow prices.

### C. Outputs and analysis

| # | Feature | AequilibraE today | In practice | Effort | Data model |
|---|---|---|---|---|---|
| C1 | Stop/line boardings & alightings, segment load profiles, max-load point, line/mode transfer matrices, pax-km/hours, **passenger loads on physical road/rail links**, screenlines & GEH/%RMSE | Partial (SF edge volumes only) | Standard | S–M | None (derived from edges + `pattern_mapping` / `route_links`; transfer matrices accumulate in the kernel; road links need `route_link_seq`) |
| C2 | Select-link/line/stop and traversal (stop-to-stop) matrices for strategies | No | Standard | M | None |
| C3 | Operator indicators: vehicle-km/hours, peak fleet size, cost, revenue | No | Common | S–M | Additive (cost parameters on vehicle types) |
| C4 | Accessibility / isochrones / travel-time cubes, after `B7` | No | Common | S | None |
| C5 | Mode-choice & appraisal-ready skims: perceived cost, true first vs transfer wait, uncrowded vs crowded IVT, fare, true transfers/boardings, `inf` on disconnected ODs, logsums | Partial (wait/transfer/unreachable bugs) | Standard | S–M | None |
| C6 | Transit OD estimation against boardings/line/screenline counts (gradient method: one residual skim and one extra loading per iteration) | No | Standard | M | None |
| C7 | Supply & GTFS QA diagnostics: negative segment times, speed outliers, orphan stops/zones, loop checks, unmapped `pattern_mapping` routes | No | Common | S | None |
| C8 | Single-OD strategy extraction: hyperpath edges with node-split probabilities and expected volumes for one OD pair, as a GeoDataFrame for QGIS | Partial (`HyperpathGenerating.run` loads one OD; no export) | Standard | S | None |

**Notes**:

- **C1** is the highest value-to-effort reporting item in this document. Projecting `on-board`
  passenger loads onto physical `pattern_mapping` road links and `route_links` geometries also gives
  QGIS immediate corridor bandwidth maps without overlapping pattern linestrings. Rail, tram and ferry
  loads project onto `route_links` (stop-to-stop) geometry; road links carry loads only for
  map-matched modes. Once transfers run `alighting → stop → boarding` (Phase 1), line-to-line
  transfer matrices cannot be derived from aggregate edge volumes, because the boarding split at a
  stop differs by destination: accumulate $\sum_d v^d_{X \to s}\, p^d_{Y \mid s}$ per stop in the
  loading pass. Projection onto road links needs `pattern_mapping.route_link_seq` (Phase 2).
- **C2** cannot mirror `select_link_loading.hpp`, which marks single-predecessor shortest-path
  trees. Strategies split flow at nodes, so per destination and selection set it needs a backward
  label $P_i^d(S)$, the probability of traversing a selected edge (an "any" accumulator in the label
  pass; select-link OD flows are $d_{od} P_o^d(S)$), plus a forward marked/unmarked propagation in
  the loading pass for select-link edge volumes. Board-to-alight stop matrices per leg are cheap,
  because the alight-or-stay choice at an `alighting` vertex is deterministic for a given
  destination.
- **C5**:
  - Accumulate expected waiting time directly inside the SF backward pass rather than subtracting
    from `trav_time`: at node $i$ it is $\alpha_i / F_i$, where $F_i$ is the combined *physical*
    frequency of the attractive set and $\alpha_i$ the headway fraction (`0.5` regular, `1.0`
    random), not multiplied by the class's wait-time weight (today `wait_time_factor` acts as
    $\alpha$, defaulting to a full headway). A stop is not intrinsically a first-wait or a
    transfer-wait stop: one destination's strategy reaches the same stop vertex from access
    connectors and from `alighting` edges, so first wait needs a reset-at-boarding accumulator or the
    two-level stop layer (`B1`); transfer wait is total minus first wait. Compute expected transfers
    as $\mathbb{E}[\text{boardings}] - P_{\text{PT}}$, counting timed-transfer edges as boardings and
    excluding `in_seat_transfers`, where $P_{\text{PT}}$ is the strategy probability of boarding at
    least one vehicle (a non-additive "any boarding" accumulator, worth reporting as a skim), so
    walk-all-the-way ODs (`boardings = 0`) never report `-1` transfers. Spiess & Florian (1989)
    references ($\alpha = 0.5$): first wait 3.0 min, transfer wait 1.25 min, in-vehicle 23.5 min,
    boardings 1.5, transfers 0.5, total 27.75 min.
  - **Component-skim discontinuity warning**: in deterministic SF, expected *generalised cost* $u_i$
    is continuous when a marginal line enters or leaves the attractive set, but individual component
    skims (`in_vehicle_time`, `waiting_time`, `fare`, `boardings`) **jump discontinuously**, causing
    noise in mode-choice feedback and Rule-of-a-Half appraisal. Passing consistent generalised cost
    (or smoothing line inclusion via `B4`) avoids this.
  - **Phasing**: Phase 1 computes true first and transfer waits, transfers and `inf` in the kernel;
    Phase 3 adds perceived cost and fare skims; crowded IVT arrives with `B5` (Phase 4.2) and logsums
    with `B4` (Phase 4.1).
- **C7** catches real-world GTFS and editing defects upfront (e.g. `arrival < prev_departure` in
  `_compute_segment_travel_time`, whose negative `trav_time` is silently clamped to `1e-12` because
  no caller passes `check_edges=True`, and silent `pattern_mapping` misses that zero out bus
  preload).
- **C8** is the calibration tool practitioners reach for first: it answers "why does this zone pair
  ride line X?" and lets every Phase 1 repair be checked visually. It needs no new pass:
  `HyperpathGenerating.run(o, d, 1.0)` already loads one OD, the edges with $v_a > \varepsilon$ are
  its strategy and $v_a / v_i$ its node splits. It needs `run()`'s label bug fixed (§1) and the
  edge-orientation flip for its GeoDataFrame, and lands before `C2`.

### D. Agent-based and on-demand

| # | Feature | Effort | Recommendation |
|---|---|---|---|
| D1 | DRT / on-demand dispatch, minibus evolution | XL | Out of scope for core; interoperate instead |
| D2 | Within-day vehicle simulation (bunching, denied boarding) | XL | Out of scope; B6 gives static capacity effects |
| D3 | Export of agent-based simulation transit schedules and vehicle definitions (network-mapped schedules, vehicle types) | M | After the Phase 2 GTFS exporter; needs persisted stop-to-link references and pseudo-links for modes that are not map matched; defer import |
| D4 | Line blocking / vehicle scheduling (MIP or min-cost flow over trips) | L | Out of scope (operations planning); `C3`'s peak fleet estimate covers modelling needs |

---

## 5. Cross-cutting design decisions (decide before coding)

### 5.1 How service plans are represented (affects SF, A4, A5, B5–B8, C3 and `build_pt_preload`)

Today every graph is computed from GTFS trips (`trips_schedule`) over a period. Planning scenarios
need to say "line X every 6 min in the AM peak", and **A5** needs to update bus segment run times
iteratively from road congestion. Three options:

1. **Synthesise fake trips into `trips_schedule` as sole source of truth.**
   - Single table for frequency and schedule engines, but bloats SQLite, makes QGIS/hand edits
     awkward, and **breaks down under A5**: every road↔transit iteration would rewrite cumulative
     timestamps across tens of thousands of `trips_schedule` rows, pushing trips across period
     boundaries and jittering period trip counts mid-equilibrium.
2. **Add an ad-hoc `pattern_services` override table** disconnected from `trips_schedule`.
   - Cheap for frequency models, but creates two conflicting sources of truth unless strictly gated.
3. **GTFS-`frequencies.txt`-style service windows + segment run-time profiles (recommended).**
   - Store `pattern_headways` (`pattern_id`, `start_s`, `end_s`, `headway_s`, `first_departure_s`,
     `exact_times`), one row per service window with GTFS `frequencies.txt` semantics, and
     `pattern_segment_profiles` (`pattern_id`, `start_s`, `end_s`, `seq`, `run_time_s`,
     `dwell_time_s`), where `seq` is the `route_links.seq` segment index and `dwell_time_s` is the
     dwell at that segment's `from_stop`. Both are keyed on time windows, not `period_id`: `periods`
     lives in the project database, where it can be renumbered or redefined without any foreign key
     or trigger reaching `public_transport.sqlite`. Period headways, trip counts and run times are
     derived at graph build by intersecting windows with the period.
   - A pattern is `schedule` in a window when it has trips in `trips_schedule` there and `headway`
     when it has `pattern_headways` rows (from GTFS `frequencies.txt` or `A4` sketching), never both.
     On import, `frequencies.txt` windows with `exact_times = 1` are expanded into `trips_schedule`
     (the expansion Phase 1 repairs), and windows with `exact_times = 0` are stored as
     `pattern_headways` rows rather than as fabricated trips. Schedule-derived means are computed at
     graph build, not stored, so edits to `trips_schedule` cannot leave stale copies.
   - **Authority rule**: frequency engines (SF, `B5`/`B6`) read `trips_schedule` means for
     `schedule` patterns, overridden by any stored profile rows, and the profiles for `headway`
     patterns. Schedule engines (`B7`/`B8`) read `trips_schedule`, optionally shifted by `A5`
     deltas. `A5` writes congested run and dwell times to a separate overlay keyed by run and
     period, never into the base profiles, so every road → transit loop starts from the same supply
     and `trips_schedule` is never mutated.
   - **Headway expansion for schedule engines**: `headway` patterns are expanded lazily in C++ from
     `first_departure_s` (GTFS `frequencies.txt` `start_time`, or a user-set offset for sketched
     lines). Without a defined offset, transfer timings between headway-driven lines would be
     arbitrary.
   - Relies on the Phase 1 `Scenario.transit` fix, so each scenario owns an isolated
     `public_transport.sqlite`.
   - **Service-day convention**: `trips_schedule`, `pattern_headways` and `periods` all count seconds
     from the start of the GTFS service day, which may exceed `86400` (a `24:00–27:00` late-night
     period is valid). Graph builders and `build_pt_preload` filter on that axis and never wrap.
     Importing a further service date of a feed (`A6`), for late-night periods that span two
     service days, merges into that feed's stops and patterns, offsets its trips by
     `86400 × day_index` and records `service_date` on each trip, so GTFS trip IDs stay unique per
     (`feed_id`, `service_date`) and export can write each trip's date.
   - `TransitGraphBuilder` and `build_pt_preload` today read `trips_schedule` only; both must read
     `pattern_headways` and the segment profiles as well, or headway-specified and edited patterns
     reach neither SF graphs nor road preloads.

### 5.2 Strategy-based vs path-based core

Strategies (SF) are AequilibraE's strength: they are fast, exact, and represent common lines.
Path-based choice (`B9`) is what path-enumeration packages use for capped/time-window fares and
for nesting under mode choice. Recommendation: keep SF as the core and extend it (`B1–B6`, `B3` for
sequence and OD-zonal fares). Add path-based choice as a **separate engine** that reuses road
route-choice code, rather than forcing one paradigm to do both.

### 5.3 C++ engine consolidation: SF kernel port and rRAPTOR

- **Port SF to `paths/cpp/hyperpath.hpp` in Phase 1 (`Effort: M`; about 700 lines of Cython in
  `hyperpath.pyx` with skims, parallelism and tests, plus the 450-line `HyperpathGenerating` wrapper
  in `public_transport.pyx`)**: move `hyperpath.pyx` into `paths/cpp/` with persistent per-thread
  workspaces (matching `workspaces.hpp`) and zero-copy `update_costs(c_a, f_a)`, so `B5`/`B6`
  iterations stop rebuilding `HyperpathGenerating` (a pandas copy and a CSC conversion per cost
  change). Per-destination heap allocation does not need the port (`FourAryHeap` already has an O(1)
  `reset_heap`), and the per-destination `qsort` can go: attractive edges leave the heap in
  nondecreasing $u_j + c_a$, so loading can walk the recorded extraction order backwards. Fix the
  kernel API in the port so later items do not reopen it:
  - a per-node wait constant $\kappa_i = \alpha_i \omega_i$ (headway fraction × class wait weight)
    in place of the hard-coded `beta = 1.0`, with a per-skim-column version so reported waits use
    $\alpha_i$ alone;
  - skim operators beyond addition: "any" (for $P_{\text{PT}}$ and select-link probabilities) and
    "reset at boarding" (for first wait);
  - a sparse per-destination cost overlay (`B3` zonal pricing);
  - outputs: unassigned demand per OD, total waiting time $W$ per class (`B5`/`B6` line search and
    gap), an optional per-destination store of flows on finite-frequency edges (CCF gap), and the
    attractive set and splits for `C8`;
  - per-destination hooks: a per-stop line-to-line accumulator (`C1` transfer matrices) and forward
    marked-flow propagation (`C2`) in the loading pass, and a node split rule (frequency share, or
    logit at origin connectors and interior nodes for `B4`);
  - exact infinite frequency (a flag, not `1e20`), so a node with an attractive walk or dwell edge
    gives finite-frequency edges exactly zero share;
  - a check that costed $c_a > 0$, since negative mode bonuses break label setting and strategy
    acyclicity;
  - bitwise-identical results for a fixed thread count, whatever the scheduling (static destination
    blocks, fixed-order reduction).
- **Schedule engine (`B7`/`B8`)**: implement **rRAPTOR / McRAPTOR** in `paths/cpp/raptor.hpp`,
  parallel over origins × departure windows, with in-kernel period aggregation for summary skims and
  opt-in per-slice matrix output.

### 5.4 Separate physical graph persistence from class generalised cost

Do **not** bake perception weights, boarding/alighting penalties or fares into persisted
`links.trav_time` or `transit_graph_configs`. Instead:

- `transit_graph_configs` persists **topological and physical builder settings** (`period_id`, CRS,
  connector method/bounds, footpath radius, transfer rules).
- `nodes`/`links` persist **unweighted physical attributes** (in-vehicle run time, dwell time, walk
  distance/time, scheduled headway, period trip count, capacity) under new column names, and a
  physical **`segment_id`** (pattern + `seq` for line edges, stop for stop edges). `trav_time` and
  `freq` exist only on the costed in-memory graph, so code reading the old columns fails loudly
  rather than silently using uncosted times. Loads then aggregate correctly once journey levels
  (`B3`) or chords copy edges.
- Replacing persisted `links.trav_time` and `freq` is a **structural** change: transit migration
  007 drops saved graphs and logs their configs so they can be rebuilt (§5.9); the cost parameters
  of existing graphs were never persisted, so a rebuild in place is impossible.
- **Behavioural costing** (component/mode/line weights, boarding and transfer penalties, wait
  factors, fare schedules, VOT) lives on `TransitClass`, is evaluated into `c_a` and `f_a` at
  assignment time, and is recorded in `results_database` procedure reports — matching road `Graph` +
  `TrafficClass`.

### 5.5 Compatibility, migrations and documentation

- **QAequilibraE**: the QGIS plugin vendors a copy of this library (1.7.0 today), and its transit UI
  and processing algorithms call `Transit.new_gtfs_builder`, `Transit.create_graph(**kwargs)`,
  `Transit.save_graphs`, `TransitGraphBuilder.from_db`, `create_line_geometry`,
  `to_transit_graph`, `TransitAssignment` (with `set_time_field("trav_time")` and
  `set_frequency_field("freq")`) and `TransitClass`. Flipping `a_node`/`b_node`, redefining
  `links.trav_time`, renaming builder keyword arguments and moving cost parameters onto
  `TransitClass` all break it. The schema is API too: the plugin loads `stops`, `routes`,
  `pattern_mapping` and `links` as editable `transit_*` SpatiaLite layers, queries `agencies`,
  `routes` (with its capacity columns), `stops`, `route_links`, `trips` and `trips_schedule` with
  raw SQL, clears a fixed list of eleven transit tables (including `stop_connectors` and the Fares
  v1 tables) before re-import, and sets `GTFSRouteSystemBuilder.gtfs_data.agency` attributes
  directly, so the `feeds` split, the `stop_connectors` replacement, the new `pattern_mapping`
  columns, the fare changes and the `A9` loader rewrite are plugin breaks too. Each phase lists its
  public API and schema changes; keep deprecation shims (old keyword accepted with a
  `DeprecationWarning`) for one minor release where feasible. Ownership and plugin releases are in
  §5.9.
- **Migrations carry triggers too**: existing projects and the shipped example zips (Coquimbo)
  receive new or changed triggers only through numbered migrations (transit, or network for
  `periods` and `transit_graph_configs`), never from the `tables/` or `triggers/` definitions alone.
  Every trigger change, in any phase, ships as a migration, and the example zips are rebuilt (or
  migrated on load) in the same change. Migrations run in one transaction with
  `foreign_keys = ON`, which cannot be switched off inside it; `stops.agency_id` is an immediate
  foreign key and `links.stop_id` and `links.link_type` are `RESTRICT`, so a migration cannot drop
  and recreate `agencies`, `stops` or `link_types` while child rows exist. Prefer new tables and
  `ADD COLUMN`, and rebuild children before parents.
- **Documentation is a deliverable per phase**: `transit_graph.rst`, `transit_skimming.rst`,
  `hyperpath_routing.rst` and the three gallery examples describe today's `a_node`/`b_node` order and
  costed `trav_time`, and become wrong after Phase 1. Data-model pages are generated by
  `docs/table_documentation.py` from `table_list.txt`, the `--@` docstrings in `tables/*.sql` and
  the live schema, so every new table needs its SQL file and docstrings; `attributes_documentation`
  rows only feed `FieldEditor` descriptions. Every schema change lands in both `tables/*.sql` (new
  projects and the pages) and a numbered migration (existing projects).

### 5.6 Validation and test strategy

Transit tests total about 1,700 lines: Bell's toy network and the Spiess & Florian (1989) network
for the SF kernel (labels, volumes and skims, built as raw kernel edges in the `inner_transfer`
topology), Coquimbo smoke tests for import, graph build and assignment, and two trigger tests; there
are no synthetic GTFS feeds. Every test builds with `blocking_centroid_flows=False` and
`with_walking_edges=False`, so the builder's defaults are never exercised, and no test checks an
exact result through `TransitGraphBuilder`. Every Phase 1 repair changes results, so reference
fixtures land before the repairs, not after:

- **Analytic references**: the Spiess & Florian (1989) example network rebuilt as a GTFS fixture
  that goes through `TransitGraphBuilder` (default `blocking_centroid_flows=True`, walking edges, a
  parent station), with exact strategy flows, labels and component skims (the existing kernel test
  encodes the boarding-bypass bug: it expects `boardings` `1.0`, not `1.5`); hand-built two-line
  common-lines cases for headway, wait, transfer and unreachable-OD skims; later, small
  Cepeda–Cominetti–Florian (2006) capacitated cases for `B6`.
- **Invariants checked on every assignment**: loaded demand equals the matrix total minus reported
  unreachable demand; flow is conserved at every non-zone vertex (so `on-board` volume on segment
  $k$ equals `dwell` plus `boarding` inflow, and per-line boardings equal alightings); at fixed costs
  $\sum_a c_a v_a + W = \sum_{od} d_{od} u_{od}$, which ties skims to loads; skims are `inf`, never
  `0`, where no strategy exists.
- **GTFS fixture library**: small synthetic feeds, each with an expected `public_transport.sqlite`
  content check, landing with the phase that can store them. Phase 1: loop/lasso routes,
  `frequencies.txt` (single and multiple windows), post-midnight trips, consecutive stops with
  equal times, extended `route_type` codes, empty `transfers` and `transfer_duration`, alphanumeric
  fare zones and Fares v1. Phase 2: a multi-agency single zip, `transfers.txt` types `0–5`,
  `pathways.txt`, stations and entrances (`location_type` `1–4`), `shape_dist_traveled` and
  pick-up/drop-off restrictions, plus a validator-clean export round trip once the Phase 2 exporter
  lands. Phase 3: Fares v2.
- **Persisted-graph determinism**: build → save → load → rebuild gives identical `nodes`/`links`
  (`a_node`/`b_node`, `segment_id`, physical attributes) and assignment results that are bitwise
  identical for a fixed thread count and within $10^{-12}$ relative across thread counts.
- **Tolerance policy**: result-changing repairs come with a before/after table on the CI models and
  fixture feeds, plus a one-line entry each under "Changes in Behaviour" in `aequilibrae2_0.rst`
  (later, the release notes of the 2.x release that carries the change). Fixtures stay small enough
  for CI; anything that needs a metropolitan feed lives in `benchmarking/`.

### 5.7 Results provenance and multi-period orchestration

- **Provenance**: every saved transit result and skim records `period_id`, the graph-config hash
  from `transit_graph_configs`, the `TransitClass` costing specification (§5.4) and the scenario, in
  the `results_database` procedure report and in skim matrix metadata. Add `save_skims` to
  `TransitAssignment` (OMX, mirroring `TrafficAssignment`) and key result tables by period.
- **Multi-period runs**: periods and per-period graphs exist, but nothing runs one set of classes
  over several periods, keeps one demand matrix per period, or aggregates daily loads from period
  loads. Add a thin driver over `TransitAssignment` for that, and define the mapping between road
  assignment periods and transit periods that `A5` (bus run times from congested road speeds) and
  `B10a` (station choice from auto and PT skims) both need.

### 5.8 Performance targets and benchmark models

`benchmarking/transit_performance.py` exists but sets no targets, still uses the pre-2.0 matrix API
(`create_empty`, `computational_view`) and reads a local `lyon` model that is not in the repository;
Phase 1 ports it to the benchmark set below first. `TransitGraphBuilder` is pandas-heavy (1,750
lines), `connector_method="overlapping_regions"` scales with zones × stops, the importer is
object-per-trip, and the SF kernel rebuilds its edge structure on every cost change (§5.3). Fix one
small model (a synthetic multi-line feed with rail, stations, transfers and post-midnight trips,
because Coquimbo is a single bus route with 78 stops and no stations, fares or frequencies) and one
metropolitan feed (multi-agency, several thousand zones, rail plus bus) as the benchmark set, and
attach a target to each phase: import time and peak memory (Phase 2, `A9`), graph build time per
period (Phases 1 and 3), SF assignment and skim time per class (Phase 1 port), and iterations ×
per-iteration time for `B5`/`B6` (Phase 4.2). Production models run on 64 GB workstations; CI
fixtures never size anything where memory matters. Each phase records its baseline in its first
week and writes its numeric target here before work starts. Initial targets, to be confirmed
against those baselines: the C++ SF port is never slower than the Cython kernel on either model;
metropolitan import runs in under a tenth of today's time within 16 GB; graph build per period
with `A1`/`A2` footpaths and connectors stays within twice the Phase 1 build time.

### 5.9 Releases, ownership and rollback

- **Versions**: `meson.build` already says `2.0.0` (last tag `v1.7.2`), and the alias module is
  marked `FIXME: Remove in 2.0`. Phase 1 targets **2.0.0**: every breaking transit change lands
  before 2.0.0 is tagged, documented in a `transit_api_migration.rst` page under the "API Changes"
  toctree of `aequilibrae2_0.rst`, with result changes under its "Changes in Behaviour". Phases 2
  and 3 change schemas only through migrations, keep the columns QAequilibraE reads (or ship its
  update in the same release), and ship as 2.x minors (2a in 2.1, 2b in 2.2, Phase 3 from 2.3);
  Phase 4 sub-phases follow as later minors. Deprecated names survive one minor release, as
  `db_connection_spatial` does until 2.1.
- **QAequilibraE ownership**: each phase names one developer who owns the plugin side. Before an
  AequilibraE release with a transit API or schema change is tagged, that developer has a
  QAequilibraE branch passing the plugin's transit tests (`test_pt_assignment.py`,
  `test_pt_procedures.py`, `test_pt_navigator_dialog.py`, `test_transit_layer_registration.py`)
  against the release candidate. The plugin vendors 1.7.0 and must already absorb the 2.0 matrix
  API in `transit_assignment_procedure.py`, so Phase 1 rides in that same plugin release.
- **Rollback**: downgrades are not supported (`database_upgrades.rst`), so structural migrations
  only drop derived data or copy before rewriting. Transit migration 007 (Phase 1) deletes saved
  transit graphs from `public_transport.sqlite` and their `transit_graph_configs` rows from
  `project_database.sqlite`, logging each config as JSON so the graph can be rebuilt with the same
  settings; supply tables are untouched. The two databases commit one after the other, so
  `Transit.load` treats a config without graph rows as not built. `project.upgrade()` migrates only
  the active scenario, so the upgrade loops over `list_scenarios()`. The release notes tell users to
  copy the project folder before upgrading: that copy is the rollback.

### 5.10 Decisions pending

Each item has a recommended answer; confirm or change it before the phase in brackets starts.

- **Q1 Release gate [Phase 1]**: 2.0.0 waits for Phase 1b. The edge flip and physical `links`
  columns cannot be phased in by deprecation, and Phases 2–3 build on them.
- **Q2 Physical column names [Phase 1]**: persisted `links` carry renamed physical columns;
  `trav_time`/`freq` exist only on the costed in-memory graph.
- **Q3 Saved graphs on upgrade [Phase 1]**: invalidate (drop and log configs), never rebuild: the
  cost parameters of existing graphs were never persisted.
- **Q4 Public names [Phase 1]**: keep `HyperpathGenerating` as the wrapper over `hyperpath.hpp`;
  rename `NewTransitGraph` to `TransitGraph` once the legacy class is gone, with a
  `NewTransitGraph` alias until 2.1.
- **Q5 Distance rule [Phase 1]**: candidate search in a local UTM zone, all reported distances
  geodesic; drop the "project SRID" alternative, which depends on unscheduled `get_srid` work.
- **Q6 Benchmark set [Phase 1]**: name the metropolitan feed (its licence must allow use in
  `benchmarking/`) and build the small multi-line CI model (§5.8).
- **Q7 Scenario upgrades [Phase 1]**: `project.upgrade()` iterates every scenario.
- **Q8 Default headway fraction [Phase 1]**: $\alpha = 0.5$ (regular service) replaces today's
  full-headway default as a `TransitClass` setting; per-stop values arrive with `B4`'s
  regular-headway wait models. The class wait weight $\omega$ is a separate parameter and never
  enters reported waits.
- **Q9 Multi-feed IDs [Phase 2]**: database-allocated surrogate keys, with GTFS IDs unique per
  `feed_id` (trips per `feed_id` and `service_date`, §5.1), as `A4` and the 2b editing API assume;
  retire the multiplier-encoded IDs of `constants.py` and the process-global `Constants` caches.
- **Q10 Capacity authority [Phase 2]**: `vehicle_types` wins; the `routes` capacity and PCE columns
  become trigger-maintained copies until 3.0; `parameters.yml` only seeds defaults at import.
- **Q11 Walk-link tables [Phase 2]**: `stop_connectors` for zone↔stop (`A2`) and a separate
  `stop_footpaths` table for stop↔stop (`A1`, with a `source` of GTFS transfer, pathway or street).
- **Q12 Route-level table [Phase 2]**: add a `lines` table (one row per GTFS route) in 2a and keep
  the per-pattern copies on `routes` refreshed by triggers while QAequilibraE reads them.
- **Q13 `route_link_seq` backfill [Phase 2]**: infer it by projecting stops onto the mapped path;
  leave `NULL` (re-map-match required) where inference fails.
- **Q14 Cross-database integrity [Phase 2]**: foreign keys, triggers and persistent views cannot
  span the two files, so graph invalidation uses a supply version, and references into the project
  database (`pattern_mapping` links, `stops.zone_id`, period windows) are checked in Python and by
  `C7`.
- **Q15 Fares v1 beside v2 [Phase 3]**: keep the v1 tables as imported, add v2 beside them and
  compile both in the calculator; convert v1 to v2 only on request. Warn that pre-2.0 imports stored
  free fares as `NULL` and unlimited transfers as `0`, which no conversion can tell apart.
- **Q16 Ticket selection default [Phase 3]**: cheapest valid combination, with priority order as
  the per-project option.
- **Q17 Two-level stop layer [Phase 3]**: ships with `B1` (rest) in Phase 3; pull it into Phase 1b
  only if the port's "any" and "reset" skim operators prove insufficient for first wait and
  $P_{\text{PT}}$.
- **Q18 Default period [Phase 1]**: the network migration sets it to `[0, 172800]` (service-day
  times up to `48:00:00`), and its period rate uses $T_{\text{period}} = 86400$ s, so whole-day
  frequencies stay daily averages.

---

## 6. Proposed four-phase plan

### Definition of done

A phase or sub-phase is done when:

- every schema change has a `tables/*.sql` definition with `--@` docstrings and a `table_list.txt`
  entry, a numbered migration, an upgrade test from projects of the previous release (one with
  saved transit graphs, one with a scenario) and `attributes_documentation` rows, and the example
  zips are rebuilt or migrated on load (§5.5);
- its §5.6 fixtures, invariants and analytic cases pass in CI, and each result change has its
  before/after row and release-note entry;
- its §5.8 target is met on both benchmark models and recorded;
- its documentation pages and gallery examples build;
- the QAequilibraE branch for the release passes the plugin's transit tests (§5.9).

In addition:

- **Phase 1**: the Spiess & Florian network, built from a synthetic feed through
  `TransitGraphBuilder`, reproduces the published strategy flows and expected times (boardings 1.5,
  transfers 0.5, 27.75 min at $\alpha = 0.5$); default `TransitClass` costing reproduces 1.7
  results on the CI models except for rows in the before/after table; build → save → load → rebuild
  is identical.
- **Phase 2**: every fixture feed round-trips import → export → import with identical tables (fare
  tables from Phase 3, which adds fare export); the metropolitan feed meets its import target; every
  `transit.edit` operation leaves `PRAGMA foreign_key_check` empty and `C7` clean; an edited headway
  pattern reaches the SF graph and `build_pt_preload`.
- **Phase 3**: the fare calculator matches hand-priced fixture journeys for every rule type;
  compiled tier-1 costs equal the calculator on enumerated paths; `C1` loads conserve flow
  (boardings equal alightings per line); a two-period run records full provenance.
- **Phase 4.x**: each algorithm reproduces its analytic case (Cepeda–Cominetti–Florian 2006 for
  `B6`), reaches its stated gap on both benchmark models and meets its §5.8 target.

### Phase 1 — Repair existing issues

Fix defects in what AequilibraE already ships, settle the persisted-graph contract and align the SF
engine with `paths/cpp/`. The import → build → assign → skim workflow keeps its shape, but results
change and the API changes listed below break existing code, so Phase 1 ships in 2.0.0 (§5.9).
After the validation work it runs as two sub-phases. **1a** (no signature changes, but results
change; merged as each repair lands and released in 2.0.0) holds scenario isolation, the CRS,
service-day, period-rule, headway, connector, dwell-edge and loop-test repairs, the zone-order fix,
`inf` skims and guards, `save_skims`, the `run()` fix, `C8` edge IDs and splits, and the GTFS reader
repairs. **1b** (breaking, gates 2.0.0) holds edge orientation, transfer topology, the physical
split with default costing and its provenance record, the kernel port with in-kernel waits and
transfers, the `C8` GeoDataFrame, legacy removal, and compatibility and documentation:

- **Validation first (§5.6)**: before any repair, rebuild the Spiess & Florian network from a
  synthetic feed through `TransitGraphBuilder` and extend its expectations to waits and skims; land
  the assignment invariants, the Phase 1 GTFS fixtures and the persisted-graph determinism test,
  marked `xfail` where today's code is wrong. Each repair removes its `xfail`, so it is checked
  against a known answer rather than against the previous output. Then port the kernel with result
  parity on the existing kernel tests, and only then add the in-kernel wait and transfer skims
  (1b), so they are written once, in C++.
- **Scenario isolation**: bind `Transit`, `GTFSRouteSystemBuilder` and `TransitGraphBuilder` to the
  owning `Scenario` (which already exposes `network`, `db_connection`, `transit_connection` and
  `project_base_path`) instead of the active `Project`, drop `get_active_project()` from
  `GTFSRouteSystemBuilder`, and make `Transit.graphs` an instance attribute. `Transit(project)`
  keeps working by binding to `project.scenario` when constructed.
- **`TransitGraphBuilder` repairs**:
  - Normalise `a_node = tail` and `b_node = head` across `TransitGraphBuilder`, persisted `links`
    and `NewTransitGraph` (matching road `Graph`, `links.sql` docs and `paths/cpp/`), and look up
    vertex geometries by `node_id` rather than `row.a_node - 1` so non-contiguous `zone_id`s and
    `connector project match` on one-way links work.
  - Replace the `EPSG:3857` default: search connector candidates in a local UTM zone (from the stop
    centroid) and measure every reported distance geodesically, so connectors agree with the
    haversine walking edges (§5.10, Q5). Record the CRS actually used in `transit_graph_configs`.
  - Adopt the service-day convention (§5.1): periods may exceed `86400`, a network migration
    redefines the trigger-protected default period as the whole service day (§5.10, Q18), and the
    builder and `build_pt_preload` filter trips on that axis so post-midnight services are no longer
    dropped.
  - Replace `_add_mean_headway_to_segments` rather than patch it (its intended "smallest gap per
    trip, then mean" is biased low): per pattern, store the scheduled headway
    $(t_{\text{last}} - t_{\text{first}}) / (N_{\text{trips}} - 1)$ at the first stop and the period
    rate $N_{\text{trips}} / T_{\text{period}}$, and give SF the period rate, which is consistent with
    period demand and capacity. Merge duplicate trips (zero gaps) for frequency but keep their
    capacity, and never map `headway <= 0` to `freq = inf`.
  - Assign trips to periods by one trip-level rule shared with `build_pt_preload` (its
    `inclusion_cond`: `start`, `end`, `midpoint` or `any`), and compute headway, $N_{\text{trips}}$
    and run times from those complete trips, so boundary stop events stop pulling patterns into a
    period and run times stop falling back to whole-day means; `build_pt_preload` also accepts a
    `period_id` in place of raw seconds.
  - Add a loop-route regression test (a pattern visiting a `stop_id` and stop pair twice) covering
    import, dwell and transfer edges.
  - Clean transfer topology so stop-to-stop transfers run `stop_A → stop_B` (`freq = inf`), keeping
    normal transfer boardings on `boarding` edges. The existing Spiess & Florian (1989) kernel test
    uses `inner_transfer` edges: re-express it in this topology, where its `boardings` skim becomes
    `1.5` (today `1.0`) and `transfers` stays `0.5`.
  - Fix connector generation: guarantee every zone a connector (add zone-side nearest stops to the
    stop-side nearest zone), apply a finite `distance_upper_bound` as a filter in both methods, and
    use `groupby(...).head(n)` for `max_connectors_per_zone`.
  - Build `dwell` edges from scheduled dwell (`departure - arrival`, §5.1 `dwell_time_s`) rather than
    `uniform_dwell_time`, once the importer keeps arrival and departure apart.
  - Separate unweighted physical attributes on the persisted graph from behavioural cost (§5.4).
    Persisted `links` carry the physical attributes under new column names; `trav_time` and `freq`
    exist only on the in-memory graph that `TransitClass` costs, so code reading the old columns
    fails loudly instead of silently using uncosted times. Add a physical `segment_id` to every edge
    and persist all topological builder attributes in `transit_graph_configs`. This (and flipping
    `a_node`/`b_node`) changes persisted `links`, so transit migration 007 drops saved graphs
    (§5.9).
  - **Default costing on `TransitClass` (`B1` part 1, `Effort: S–M`)**: `alighting_penalty`, the
    dwell halves on boarding and alighting edges, `wait_time_factor` and the access/egress/walk
    perception factors move from builder attributes to a `TransitClass` costing specification whose
    defaults reproduce today's values except the headway fraction (§5.10, Q8), so default users see
    only the repairs and the new $\alpha$ in the before/after table. Per-mode/line/stop weights, VOT
    and fares stay in Phase 3.
- **Assignment, skim & validation repairs**:
  - Sort `self.zones` and `od_node_mapping` by `taz_id` (`add_zones`, `_create_od_vertices`,
    `from_db`), return skims in centroid order, and assert in `OptimalStrategies.execute` that
    `od_node_mapping` order matches `graph.centroids`, so the raw `coo_matrix` offsets stay aligned
    with zones.
  - Compute true expected waiting time ($\alpha_i / F_i$ over attractive sets, from physical
    frequencies and the headway fraction $\alpha_i$, unweighted by the wait-time weight), split into
    first and transfer wait, and true transfer counts ($\mathbb{E}[\text{boardings}] -
    P_{\text{PT}}$) inside the SF kernel instead of subtracting from costed `trav_time`, implemented
    once, in the C++ port. Decide the default $\alpha$ (§5.10, Q8): today `wait_time_factor = 1.0`
    makes the expected wait a full headway, twice the usual regular-service assumption.
  - Initialise skim buffers to `inf`/`NaN` (not `0.0`) so skipped disconnected zones and unreached
    component skims do not report `0.0`; guard against negative segment travel times (`arrival <
    prev_departure`) on import and graph build, landing with the import fix that keeps arrival and
    departure apart, which is what exposes them; fix `HyperpathGenerating.run()` so it returns the
    labels of the requested destination.
  - Add `TransitAssignment.save_skims`, and record `period_id`, graph-config hash, scenario and the
    `TransitClass` costing with every saved result and skim (§5.7). This is the only phase that
    implements provenance; later phases only add fields to the costing record.
  - Add `C8` single-OD strategy extraction (`Effort: S`) ahead of the Phase 3 reporting work: it is
    the tool used to check every repair in this phase. Edge IDs and split probabilities ship first
    (1a); the GeoDataFrame export follows the orientation flip (1b).
- **GTFS reader & writer repairs**:
  - Map extended HVT `route_type` codes (`100–1700`) to their base types for default
    capacities/PCEs and in `mode_correspondence.py`, so extended-type buses (`700–716`, `800`) can be
    map matched; `A8` later overrides this one function. Reading `agency_id`, `block_id`,
    `pickup_type`/`drop_off_type`, `exact_times` and `location_type` moves to Phase 2 with the
    columns that store them: enabling them in `column_order.py` alone breaks import
    (`routes.txt:agency_id` overwrites the internal `Route.agency_id` foreign key on every feed,
    `fare_attributes.txt:agency_id` fails a strict six-header `zip`, and `exact_times` breaks the
    four-value unpack in `__load_frequencies`).
  - Fix `gtfs_loader.py`: restore the de-conflicted, speed-capped times in `__deconflict_stop_times`
    and replace its `assert` with a logged repair; repair broken `frequencies.txt` expansion
    (`trip.trip` key, multi-window support, rebasing to `start_seconds`, unique `new_trip_str`),
    which Phase 2 keeps for `exact_times = 1` windows; preserve `arrival_time` vs `departure_time`
    instead of overwriting both with `max(...)`; stop `drop_duplicates(["seqkey"])` from deleting
    repeated stop pairs on loop routes; and delimit `pattern_hash` stop IDs.
  - Fix Fares v1 import so `price = 0`, unlimited transfers (`transfers = ""` → `NULL`), unlimited
    `transfer_duration` (`""` → `NULL`) and string `zone_id`s are stored as published: `parse_csv`
    keeps empty numeric fields as `NULL` instead of `0`, and `Fare` and `FareRule` stop casting.
  - Delete the unreachable `RouteSystem`, `route_system_reader` and `gtfs_writer` code; Phase 2
    writes a new exporter against the real schema.
- **C++ SF kernel port (§5.3, `Effort: M`)**: port `hyperpath.pyx` to `paths/cpp/hyperpath.hpp` with
  persistent per-thread workspaces, zero-copy `update_costs(c_a, f_a)` and the kernel API fixed in
  §5.3, keeping `HyperpathGenerating` as its public wrapper. If it is not ready when 1b's kernel
  work starts, the in-kernel waits and transfers go into the Cython kernel and the port moves to
  the start of Phase 2, so it never blocks 2.0.0.
- **Legacy removal (1b)**: delete the legacy `TransitGraph(GraphBase)` class and the
  `paths/public_transport.py` alias with the graph-contract change rather than with the port, so
  both go in 2.0.0 even if the port slips; rename `NewTransitGraph` to `TransitGraph` with a
  `NewTransitGraph` alias until 2.1 (§5.10, Q4). (`paths/graph_building.py` is a road-graph alias;
  its removal is general 2.0 clean-up.)
- **Result-change management**: these fixes change results for existing users (headways, waits,
  transfers, unreachable pairs, connector distances, dwell, post-midnight services, zone order) and
  follow the §5.6 tolerance policy. The GTFS import repairs reach only feeds imported after the
  upgrade: the release notes tell users to re-import feeds with `frequencies.txt`, loop routes,
  scheduled dwell, tied stop times, free fares or unlimited transfers.
- **Compatibility and documentation (§5.5, §5.9)**: the breaking changes are the persisted
  `links.a_node`/`b_node` order and physical columns; builder cost attributes (`alighting_penalty`,
  `wait_time_factor`, access/egress/walk factors) moving to `TransitClass`, with `wait_time_factor`
  becoming the headway fraction $\alpha$; the `projected_crs` default; the topology behind
  `with_inner_stop_transfers`/`with_outer_stop_transfers`; the meaning of the predefined skims
  `waiting_time`, `transfers`, `inner_transfers` and `outer_transfers`; `inf` for unreachable pairs;
  skims returned in centroid order; and the removal of the legacy `aequilibrae.paths.TransitGraph`,
  the `aequilibrae.paths.public_transport` alias and the already-deprecated `seed`,
  `geometry_noise` and `noise_coef` builder arguments. Document them in
  `transit_api_migration.rst`, keep a `DeprecationWarning` shim for renamed builder keywords until
  2.1, and update `transit_graph.rst`, `transit_skimming.rst`, `hyperpath_routing.rst` and the three
  gallery examples for the new edge orientation and the physical/behavioural split. Drop the
  skimming example's `freq` skim, which sums the `1e20` stand-in for infinite frequency.

### Phase 2 — Settle the supply data model, then add PT editing tools

Land every supply schema change and the importer that fills it first (**Phase 2a**, release 2.1),
then turn `project.transit` from an import-only wrapper into a supply authoring and
scenario-editing toolkit (**Phase 2b**, release 2.2) built on the final schema, so the editing
layer and triggers are not reworked later (`A4`, `A6` part 1, `C7`). 2a holds the schema migrations
with the triggers that keep the `routes` copies of `vehicle_types` and `lines` current (Q10, Q12),
the multi-feed GTFS lifecycle and exporter, service specs in use and the `C7` import checks; 2b
holds the `transit.edit` API (`A4`), the integrity triggers and the `C7` post-edit checks. Editing
builds on AequilibraE's `ProjectTable` layer. New fare tables and fare editing ship with the fare
work in Phase 3; Phase 2 touches fares only to re-point the existing Fares v1 tables at `feeds` and
to keep them consistent under the 2b triggers.

- **Supply schema migrations (first)**:
  - Normalise `route_links.from_stop`/`to_stop` to `TEXT` (`stops.stop_id` and `links.stop_id`
    already are), and add the missing foreign-key comma in `route_links.sql`, so key matching no
    longer relies on SQLite affinity rules.
  - §5.1: `pattern_headways` and `pattern_segment_profiles`, keyed on service windows.
  - `A1`: `transfers` (stop-, route- and trip-level rules), `pathways` and `stop_footpaths`
    (`from_stop_id`, `to_stop_id`, `distance`, physical `trav_time`, source) tables, and
    `stops.location_type` so stations and entrances are kept (graph builds and connectors use only
    `location_type = 0` until the Phase 3 `A1` builder). `A2`: replace `stop_connectors` with
    zone↔stop connectors (`zone_id`, `TEXT` `stop_id`, direction, `access_mode`, `distance`,
    physical `trav_time`, no `penalty_cost`).
  - `A3`: `vehicle_types` table (`seated_capacity`, `total_capacity`, `standing_area_m2`, `pce`,
    boarding/alighting service rates, unit operating costs) linked from `routes`/`trips`. It is the
    capacity authority; `routes.seated_capacity`, `total_capacity` and `pce` become
    trigger-maintained copies until 3.0, because QAequilibraE reads them.
  - `A5` (schema prep): stop-to-stop segment index (`route_link_seq`, distinct from existing
    road-link `seq`) and a nullable `stop_fraction` on `pattern_mapping`, one row per road link as
    today, backfilled for existing projects (§5.10, Q13).
  - `A7`: `pickup_type`/`drop_off_type` on `route_links` and `in_seat_transfers` table (`block_id` /
    `transfer_type=4`).
  - `A8`: `transit_modes` hierarchy table mapping GTFS/HVT `route_type` and route overrides to user
    submodes, driving default capacities, PCEs and map-matching eligibility; the default capacities
    and PCEs move from `Transit` class attributes to `parameters.yml`, next to the road defaults.
  - `A6` (provenance): `feeds` table split from `agencies` by the ID-preserving migration in the
    `A6` note, `feed_id` on every table that holds GTFS identifiers (including `fare_attributes`
    and `fare_zones`, whose `agency_id` points at the import row today), `trips.service_date` (§5.1,
    backfilled from `agencies.service_date`), and real GTFS agencies in `agencies` with
    `agency_timezone` and `agency_url`, which a valid export needs.
  - Route level (§5.10, Q12): a `lines` table (one row per GTFS route: `route_id`, `feed_id`,
    `agency_id`, names, `route_type`, colour, default vehicle type), so route attributes stop being
    repeated on every pattern row of `routes`; pattern rows keep their copies, refreshed by
    triggers, because QAequilibraE reads them.
  - Drop the transit-side `zones` and `modes` tables, which nothing writes or reads (`IF EXISTS`:
    the shipped Coquimbo example has no transit `zones`). Views are not an option: a persistent
    SQLite view cannot reference another database file, and a foreign key cannot target a view.
    Keep `link_types` (the SF edge vocabulary and the target of `links.link_type`) and `node_types`
    (the SF vertex vocabulary).
  - `pnr_lots` (`B10a`) and the TTF definitions (`A5`) are standalone tables with no consumer
    before Phase 4, so they land with their algorithms.
- **`ProjectTable`-backed `transit.edit` API**:
  - Transactional CRUD gateways for `agencies`, `stops`, `routes`/`patterns`, `trips` and the new
    supply tables above. `ProjectTable` takes a single-column key, so `route_links`,
    `pattern_mapping` and `trips_schedule` (composite keys) are replaced as whole sequences per
    pattern or trip, and every new table gets a single-column surrogate key. New rows take
    database-allocated keys; the encoded IDs (`AGENCY_MULTIPLIER`, `ROUTE_ID_MULTIPLIER`, ...) and
    the process-global `Constants` counters are retired (§5.10, Q9), and GTFS identifiers stay in
    `stops.stop`, `lines.route_id` and `trips.trip`, unique per `feed_id` (trips per `feed_id` and
    `service_date`); `routes.route` repeats on every pattern of a route.
  - **Stop & station operations**: add, move, snap to road/walk links, merge/cluster stops, and
    manage the `parent_station` hierarchy.
  - **Pattern operations**: sketch new routes/patterns from stop sequences or road paths without
    GTFS, **duplicate**, **flip/reverse**, **extend/truncate**, insert/remove stops (re-indexing
    `route_links.seq` `0..S-2` and shifting/interpolating `trips_schedule` stop rows `0..S-1` in
    Python), split patterns on differing `pickup_type`/`drop_off_type`, and **re-map-match**
    individual edited patterns onto the road network (`RouteMapMatcher`), writing
    `pattern_mapping.route_link_seq` as it goes.
  - **Service & timetable editing**: create, shift, scale or replace service plans via §5.1 headway
    windows (`pattern_headways` + `pattern_segment_profiles`) and/or trip timetables; every
    `transit.edit` operation emits on the `Transit.transit` signal (`aeq_signal.SIGNAL`: a
    `pyqtSignal` inside QGIS, a `PythonSignal` outside), which QAequilibraE already connects for
    import progress, so the plugin can refresh its layers.
- **SQLite consistency triggers (`transit_triggers.sql`)**: cascade deletes, geometry upkeep and
  cascade ID updates; referential integrity across `stops`, `stop_connectors`, `stop_footpaths`,
  `routes`, `route_links`, `pattern_mapping`, `trips`, `trips_schedule`, the new supply tables and
  the Fares v1 tables; and a supply version that every supply edit bumps and that
  `transit_graph_configs` records at build, so `Transit.load` refuses stale graphs (the configs
  live in the project database, out of reach of transit triggers). Persisted `links.stop_id` loses
  its `RESTRICT` foreign key to `stops`, so derived graphs no longer block stop edits. QAequilibraE
  edits `stops`, `routes`, `pattern_mapping` and `links` directly as SpatiaLite layers, so
  invariants that must survive manual edits are triggers, and Python-only ones (`seq` re-indexing,
  schedule interpolation) are re-checked by `C7`.
- **Multi-feed GTFS lifecycle (`A6` part 1, `A9`)**: multi-feed/multi-date/multi-agency import (a
  further date of a feed merges into its stops and patterns, §5.1), the `A9` vectorised loader with
  the import-time and memory targets from §5.8, reading `agency.txt`, `feed_info.txt`,
  `transfers.txt`, `pathways.txt`, `shape_dist_traveled`, station rows and the fields
  `column_order.py` comments out today (`agency_id`, `block_id`, `pickup_type`/`drop_off_type`,
  `exact_times`, `location_type`) into the new tables, area/time filtering, and a new GTFS exporter
  written against the `public_transport.sqlite` schema (internal IDs mapped back to `stops.stop`,
  `routes.route` and `trips.trip`, `calendar_dates.txt` from each trip's service date, a
  validator-clean `.zip`).
- **Service specs in use (§5.1)**: `TransitGraphBuilder` derives headways, trip counts and run and
  dwell times from `trips_schedule` and `pattern_headways` intersected with the period, overridden
  by `pattern_segment_profiles`, and `build_pt_preload` reads the same sources, so headway-specified
  and edited patterns reach SF graphs and road preloads.
- **Supply & GTFS QA diagnostics (`C7`)**: automated integrity checks run on GTFS import, after
  edits and before graph builds and preloads (negative/zero run times, speed outliers, broken
  `pattern_mapping` sequences, orphan stops, isolated zones), including the references that no
  foreign key or trigger can guard because they point into the project database: `pattern_mapping`
  road links removed by road edits or the network simplifier, and stale `stops.zone_id`. A
  `routes.map_match_status` (`matched`, `failed`, `manual`, `stale`) makes failed matches visible
  and keeps manual fixes from being re-matched.
- **Documentation (§5.5)**: `--@` docstrings and `table_list.txt` entries for every new table,
  `attributes_documentation` rows for `FieldEditor`, regenerated data-model pages, and a
  `transit.edit` page with the pattern and service operations.

### Phase 3 — Network preparation, costing, fares and reporting

Land the network-preparation builders, the `TransitClass` cost model, the fare work and standard
reporting/skimming on top of the existing uncongested SF engine. With Phase 2's supply schema and
this phase's fare schema in place, Phase 4 changes no existing supply or fare table; it only adds
`pnr_lots` (`B10a`), TTF definitions (`A5`), station and walkway capacities (`B5`), result tables
(`B7`, `B9`) and `B8`'s time-sliced demand and trip-load tables. The fare work needs only Phase
2a's `feeds` table, so it can start while Phase 2b is in progress; only fare editing waits for
Phase 2b.

- **Network-preparation builders**:
  - `A1`: populate `stop_footpaths` for both station pathways and walk-network street inter-stop
    footpaths independent of `parent_station`, transitively closed within the walk radius as `B7`
    needs, honouring the `transfers` rules (`transfer_type=3` removes the footpath; route-level
    prohibitions need per-line arrival copies of the stop).
  - `A2`: network-based multimodal access/egress builder snapping stops to walk/bike/drive graphs
    (`split_links_at_stops`) to populate multi-connector `stop_connectors` (and optional embedded
    walk links for SF), with origin-connector logit (`B4` part 1, `Effort: S`) so several connectors
    carry flow at once.
  - `A7`: wire pick-up/drop-off permissions and `in_seat_transfers` into `TransitGraphBuilder`.
- **Costing, multi-class & baseline reporting tooling**:
  - `B1` (rest), `B11`: extend the Phase 1 `TransitClass` costing with per-component and
    per-mode/line weights, per-line/stop boarding penalties, transfer penalties through the
    two-level stop layer (`B1` note; §5.10, Q17), linear wait-time factors, VOT, rider category and
    fare medium, and run multi-class uncongested SF (`B11`) over a shared physical graph. Fares
    enter through the fare work below.
  - `C1`: stop boardings/alightings, line/pattern ridership, segment load profiles, max-load points,
    line/mode transfer matrices (accumulated per destination in the SF kernel, `C1` note),
    pax-km/hours, **passenger-load projection onto physical road/rail links** for QGIS bandwidth
    maps (road links for map-matched modes through `route_link_seq`, `route_links` geometry for
    rail, tram and ferry), screenlines, and GEH/%RMSE helpers.
  - `C3` (basic): scheduled vehicle-km, vehicle-hours, peak fleet estimate and operating cost.
    Farebox revenue comes from the fare work below.
  - `C5`: perceived-cost skims from the `B1` costing, next to the physical skims Phase 1 computes in
    the kernel (first vs transfer wait, IVT, access/egress/walk, boardings/transfers). Fare skims
    come from the fare work below; crowded IVT arrives with `B5` (Phase 4.2) and logsums with `B4`
    (Phase 4.1).
  - **Multi-period driver (§5.7, `Effort: S–M`)**: run one set of classes over several periods,
    keep one demand matrix per period, aggregate daily loads, and define the road↔transit period
    mapping that `A5` and `B10a` use. Provenance (Phase 1) now records the full `B1` costing.
  - **Documentation (§5.5)**: a `TransitClass` costing page, the reporting and skim catalogue, and
    updated gallery examples that save skims through `save_skims`.

- **Interop (`D3`, `Effort: M`)**: export network-mapped schedules and vehicle types for
  agent-based simulation, on top of the Phase 2 GTFS exporter and `A3` vehicle types, with
  pseudo-links for modes that are not map matched.

#### Fares in Phase 3

All fare work lands together, so every later engine (`B3`, `B7`, `B9`) consumes one fare
representation and one set of pricing rules. Fares fall into three tiers, according to what an
engine needs in order to price them exactly:

| Tier | Examples | Representation | Exact in |
|---|---|---|---|
| 1. Additive | Flat fare per boarding, boarding fare per network or boarding area, linear per-km fare | Edge costs (this phase) | Every engine |
| 2. A few states / intra-line | Free or discounted transfers by count, fare integration across networks, boarding-zone × destination-zone fares, intra-line distance/station-pair chords, mode-sequence premiums | Journey levels, pattern chords, or destination-thread zonal pricing (`B3`, Phase 4.3) | SF with `B3` (zonal fares under the conditions in the `B3` note); RAPTOR with labels per state |
| 3. Depends on the whole path | Distance stages accumulated across legs, all-traversed-zone sets, time-limited transfers, per-journey fare caps, cheapest-ticket selection (daily/weekly caps: approximate only) | Fare calculator on complete paths | `B9` (time limits only on timetable paths); `B7`/`B8` where times are known; strategies only as approximate skims |

- **Fare data model (`A6` part 2)**:
  - GTFS Fares v2 tables: `fare_media`, `rider_categories`, `fare_products` (`amount`, `currency`,
    rider category, fare medium), `fare_leg_rules` (network, from/to area, from/to timeframe,
    `rule_priority`), `fare_leg_join_rules`, `fare_transfer_rules` (`transfer_count`,
    `duration_limit`, `duration_limit_type`, `fare_transfer_type`), `areas`/`stop_areas`,
    `networks`/`route_networks` and `timeframes`.
  - Keep the Fares v1 tables as imported and add the v2 tables beside them, each with `feed_id`
    (nullable for project-level modelled fares, which may span feeds) and a single-column surrogate
    key (their natural keys are composite). The fare calculator compiles v1, v2 and the extension
    tables into one in-memory fare system, with a documented reading of v1's multi-leg and
    `contains_id` rules; an explicit v1-to-v2 conversion serves users who want to edit an imported
    v1 system (§5.10, Q15). Export writes back the version that was imported, so v1 feeds
    round-trip losslessly and no migration rewrites existing fare data.
  - Extension tables for what core Fares v2 does not express (it has no distance fields and no
    caps): `fare_distance_stages` (distance bands per network, with optional linear interpolation
    between stages, or high/low band pricing), zone-count / contained-zone-set rules (price by the
    number or exact set of areas traversed, as in GTFS v1 `contains_id`), and `fare_caps` (cap
    amount per time window, rider category and fare medium).
  - Fare editing gateways on Phase 2's `ProjectTable` layer, with triggers keeping products, leg
    rules, transfer rules, areas and stops consistent.
- **Fare calculator (`B12`, independent of any algorithm)**:
  - `price(legs, rider_category, fare_media)`, where each leg is (route/network, boarding stop,
    alighting stop, boarding time, alighting time, in-vehicle distance). It returns the price plus a
    per-leg and per-operator breakdown.
  - Pricing rules: leg matching by network, area and timeframe with `rule_priority`; leg joins;
    transfer rules (counts, duration limits, discount types); distance stages; zone counts; caps;
    prices summed across independent fare systems; and, where several products apply, either
    priority order or the cheapest valid combination, configurable per project.
  - Python reference implementation in this phase; the C++ port lands with `B9` (Phase 4.3), before
    `B7` needs it.
  - Used by tests, revenue, approximate skims, path-based choice (`B9`), a fare criterion in RAPTOR
    (`B7`), and validation of journey levels (`B3`): on enumerated paths, the compiled graph cost
    must equal the calculator's price.
- **Fare classification on graph build (`Effort: S`)**: classify each fare system into a tier,
  report which engine can price it exactly, and warn (never drop rules silently) when a tier-2 or
  tier-3 rule is approximated.
- **Tier-1 compilation into SF (`B2`)**: boarding fares on `boarding` edges (by network and boarding
  area) and linear per-km fares on `on-board` edges, converted to time through each `TransitClass`'s
  VOT for its rider category and fare medium. This gives exact fare-sensitive routing and an exact
  expected-fare skim.
- **Fare skims beyond tier 1 (until Phase 4.3; `Effort: S–M`, built on `C8`'s strategy
  extraction)**: extract probability-weighted top-$k$ paths from the destination strategy (rather
  than winner-take-all greedy edges, which collapse common-line splits), price them with the
  calculator — restricted to active/sampled ODs in Python until the C++ calculator lands — and flag
  the skim as approximate in its metadata.
- **Revenue (`C3`)**: tier 1 exactly, from edge volumes × edge fares; tiers 2–3 from the calculator
  weighted across the top-$k$ strategy branches of non-zero-demand ODs (approximate until
  `B3`/`B9`). Shared-ticket revenue is allocated to operators by boardings, passenger-km or the
  calculator's per-leg breakdown.
- **Fare QA (`C7`)**: routes or stops no leg rule covers, overlapping rules with equal priority,
  areas without stops, currency mismatches, and products referencing missing fare media or rider
  categories.
- **Prerequisite**: the physical `segment_id` on every edge (Phase 1), so fare-driven edge copies
  in Phase 4.3 aggregate onto physical segments.

### Phase 4 — Implement new algorithms (sub-phased by complexity)

With data structures, editing, costing and reporting settled in Phases 1–3, new assignment and
routing algorithms roll out in four complexity tiers:

| Sub-phase | Complexity | Items | Summary & key dependencies |
|---|---|---|---|
| **4.1** | **Moderate** (Effort M to M–L each; single-pass SF extensions & matrix coupling) | `C2`, `B4`, `B10a` | **`C2`** select-link/line/stop & stop-to-stop traversal matrices in `hyperpath.hpp` (backward "any" labels plus forward marked flows, `C2` note); **`B4`** destination-side and interior-node logit (origin-connector logit ships with `A2` in Phase 3), RTPI / frequency-and-time stop splits & regular-headway waits, reporting a fixed-point gap next to the VI gap once logit is active; **`B10a`** matrix-coupled PnR/KnR/feeder station choice (with its `pnr_lots` table and lots as pseudo-zones in both graphs) coupling `TrafficAssignment` auto skims with `TransitAssignment` PT skims over the §5.7 period mapping, and Sinkhorn/dual lot-capacity shadow prices |
| **4.2** | **Medium–High** (Effort M–L each; iterative equilibrium, road → PT hand-off & inverse problems) | `B5`, `B6`, `A5`, `C6` | **`B5`** congested SF with WebTAG/ATAP seated-vs-standing (`dwell`-priority) crowding curves and station/walkway VDFs (walkway capacities added here); **`B6`** strict-capacity SF ($C_k - v_{\text{prio},k}$, with an uncapacitated fallback or reported denied demand) via CCF effective frequencies / fail-to-board, solved by MSA/MSWA against the CCF VI gap with per-destination waits $\max_a v_a^d / f_a(v)$; **`A5`** (after `B5`, reusing its outer loop for the dwell feedback) TTF definitions and bus run times from congested road speeds + boarding-driven dwell times, as a road → transit hand-off plus a transit-internal dwell loop (no road↔transit fixed point while bus stops do not reduce road capacity; map-matched modes only, rail, tram and ferry keep schedule run times); **`C6`** transit ODME against boarding, line and screenline counts by the gradient method (one residual skim and one extra loading per iteration; no `C2` dependency) |
| **4.3** | **High** (Effort L to L–XL each; state expansion & path enumeration) | `B3` (+ `B10b`), `B9` | **`B3`** journey-level state-augmented SF and intra-line pattern chords, compiled from the Phase 3 fare rules and checked against the fare calculator (3–6 levels for transfer rules and mode sequences, copying the in-vehicle network per level beyond two, summing physical segment loads across levels/chords, plus zero-replication destination-thread OD zonal pricing through the §5.3 cost overlay and optional **`B10b`** graph-coupled PnR); **`B9`** path-based PT route choice (route/mode-breaker with group bans + PSL/nested logit over line-bundle legs, giving exact per-journey **fare caps**, and **time-window transfers** on timetable paths, on the C++ port of the `B12` fare calculator) |
| **4.4** | **Very High** (Effort L–XL, `C4` S; new C++ schedule engine & dynamic demand/supply dimensions) | `B7`, `C4`, `B8` | **`B7`** C++ `rRAPTOR`/`McRAPTOR` router in `paths/cpp/raptor.hpp` with Pareto connection search over transitively closed footpaths and in-kernel period-summary / sliced skims; **`C4`** timetable accessibility & isochrones (`S` once `B7` lands); **`B8`** full schedule-based assignment with connection + departure/adaptation-time choice over a desired-departure profile per period, relaxed-dominance choice sets and iterative trip-level capacity restraint (`XL`, structural) |

Phases 1–3 plus Phases 4.1–4.2 cover the frequency-based capabilities that are standard in
commercial practice: strategy-based assignment with congestion and strict capacity, headway-based
route choice through `B4`, transit ODME (`C6`), and supply authoring. Extended strategy-based
assignment also needs journey levels (`B3`, Phase 4.3) on top of `B4` (Phase 4.1). Phases 4.3–4.4
complete the remaining commercial feature set apart from on-demand services, within-day simulation
and line blocking (`D1`, `D2`, `D4`).

---

## Sources

Package documentation and source code reviewed for §2–§3 are not cited, so that packages are not
named.

- GTFS Schedule reference (Fares v2 files and fields): https://gtfs.org/documentation/schedule/reference/
- Spiess, H., & Florian, M. (1989). Optimal strategies: a new assignment model for transit networks.
  *Transportation Research Part B*, 23(2), 83–102.
- Gentile, G., Nguyen, S., & Pallottino, S. (2005). Route choice on transit networks with online
  information at stops. *Transportation Science*, 39(3), 289–297.
- Cepeda, M., Cominetti, R., & Florian, M. (2006). A frequency-based assignment model for congested
  transit networks with strict capacities: characterization and computation of equilibria.
  *Transportation Research Part B*, 40(6), 437–459.
- Schmöcker, J.-D., Fonzone, A., Shimamoto, H., Kurauchi, F., & Bell, M. G. H. (2011).
  Frequency-based transit assignment considering seat capacities. *Transportation Research Part B*,
  45(2), 392–408.
- Delling, D., Pajor, T., & Werneck, R. F. (2015). Round-based public transit routing.
  *Transportation Science*, 49(3), 591–604.
- Codina, E., & Rosell, F. (2017). A heuristic method for a congested capacitated transit
  assignment model with strategies. *Transportation Research Part B*, 106, 293–320.
- Marcotte, P., & Nguyen, S. (1998). Hyperpath formulations of traffic assignment problems. In
  P. Marcotte & S. Nguyen (Eds.), *Equilibrium and Advanced Transportation Modelling*. Kluwer.
- Spiess, H. (1990). A gradient approach for the O-D matrix adjustment problem. Centre de recherche
  sur les transports, Université de Montréal.
- UK Department for Transport. *TAG Unit M3.2: Public Transport Assignment Modelling* (crowding
  curves).
- Australian Transport Assessment and Planning (ATAP) Guidelines. *M1 Public Transport: Parameter
  Values Technical Report* (2021 edition; crowding multipliers).
