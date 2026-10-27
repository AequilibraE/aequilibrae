# Transit modelling capabilities: gap analysis and plan

Status: draft for discussion (2026-10-27). Branch `pedro/transit_capabilities`.

This note lists what practice software, Quetzal and `polaris-studio` offer for public-transport (PT)
modelling, checks that against what AequilibraE actually does in code today, and rates how much work
each capability would take and how much it would change the data model. It ends with cross-cutting
design decisions and a four-phase execution plan.

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
  with `pce`, `seated_capacity`, `total_capacity`), `trips`, `trips_schedule` (exact
  arrival/departure per stop).
- `route_links`, and `pattern_mapping` (pattern → road `link`/`dir` sequence, from map matching).
- `fare_attributes`, `fare_rules` and `fare_zones` (GTFS Fares v1, stored but never used).
- `stop_connectors`: the table exists, but no Python code reads or writes it.
- Persisted Spiess–Florian (SF) graphs, one per `period_id`, in `nodes`/`links`. Their builder config
  is in `transit_graph_configs`; periods are defined in the project's `periods` table.
- **Bugs/gaps**:
  - **No editing API or integrity triggers**: unlike `project.network` (which uses `ProjectTable`
    CRUD and relational triggers), `Transit` only exposes read-only `get_table`, GTFS import, graph
    build and preload; `transit_triggers.sql` only updates `route_links.distance`.
  - **`Scenario` isolation bug**: `Scenario` passes the root `Project` into `Transit` (`# FIXME:
    Hack` in `scenario.py`), so `scenario.transit` reads and writes the root
    `public_transport.sqlite` instead of isolating scenario transit supply.
  - **Extended `route_type` fallback**: `Transit.default_capacities` and `default_pces` only map
    standard GTFS `route_type` codes (`0–5, 11, 12`); extended HVT codes (`100–1700`, e.g. `109`
    suburban rail, `400` metro) fall back to `"other"` (`[30, 60]`).

**Import/export**:

- GTFS import for one service date per feed, with `frequencies.txt` support.
- Bus map matching onto the road network (`split_links_at_stops`, `RouteMapMatcher`), and
  per-route-type capacities, PCEs and max speeds.
- `transfers.txt` and `pathways.txt` are **not** read.
- GTFS export (`gtfs_writer`).
- **Bugs/gaps**: in `column_order.py`, `routes.txt:agency_id`, `fare_attributes.txt:agency_id`,
  `trips.txt:block_id`, `stop_times.txt:pickup_type` (and `drop_off_type`), and
  `stops.txt:location_type` are commented out or omitted. Multi-agency single-zip feeds collapse onto
  one agency, pick-up/set-down restrictions on express services are lost, and through-routed trips
  (`block_id`) cannot be identified.

**Graph** (`TransitGraphBuilder`):

- Builds a classic SF graph: stop, boarding, alighting and OD vertices; on-board, dwell, boarding,
  alighting, connector, inner/outer transfer and walking edges.
- Connectors are **Euclidean** (nearest-neighbour or overlapping-regions) at a fixed walking speed.
  The road network is only used optionally to redraw connector geometry (`connector project match`).
- **Bugs/gaps**:
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
    index.) It also does not distinguish scheduled headway (wait time on partial-period lines) from
    effective period trip rate $N_{\text{trips}} / T_{\text{period}}$ (needed for capacity).
  - **Stop-event period filtering truncates long lines**: `_create_line_segments` and
    `_add_mean_headway_to_segments` filter stop rows (`departure >= start AND arrival <= end`) rather
    than trips, dropping downstream segments of commuter trips that depart inside the period and
    arrive after `end`.
  - **No street-level inter-stop transfers**: `outer_transfer` and `walking` edges are only built
    where `parent_station` is non-empty. With `blocking_centroid_flows=True`, surface transfers
    between cross-street or opposite-kerb stops without a shared `parent_station` (most GTFS bus
    stops) are impossible.
  - **Parallel transfer-edge topology**: enabling `inner_transfer` or `outer_transfer` connects
    `alighting → boarding` directly with finite frequency alongside the infinite-frequency `alighting
    → stop` edge, distorting SF node splits and bypassing `boarding` edges (undercounting
    `boardings` skims and splitting flows needed by fares, capacities and stop summaries).
  - **Loop/self-intersecting patterns (unverified)**: patterns that visit a `stop_id` twice have no
    regression test. A code read found no miswire: `_create_dwell_edges` already merges on
    `line_seg_idx`, and the duplicate rows `_create_inner_stop_transfer_edges` creates are removed by
    `drop_duplicates`. A loop-route test should confirm this before it is treated as a bug.

**Assignment** (`TransitAssignment` / `HyperpathGenerating`):

- Uncongested optimal strategies only, in standalone Cython (`public_transport.pyx` / `hyperpath.pyx`)
  and parallel by destination. One all-or-nothing loading of strategies; no equilibrium loop.
- Not yet ported to `paths/cpp/`: reallocates full thread buffers per `assign()` call, initialises
  and frees a `FourAryHeap` plus skim scratch buffers on **every destination**, and has no in-place
  `update_costs` / `update_frequencies` API.
- Output is raw SF edge volumes only.
- **Skim bugs**:
  - `waiting_time` is computed by subtraction (`trav_time - in_vehicle_trav_time - egress_trav_time -
    access_trav_time`), which sweeps every `alighting_penalty` (`480 s`), boarding/alighting dwell
    half-time, `walking_trav_time` and `transfer_time` into reported wait time.
  - `transfers` only counts `inner_transfer` and `outer_transfer` edges (both `False` by default), so
    default `transfers` skims are all zeros.
  - Unreachable OD pairs (`fmax`) are overwritten with `0.0` instead of `inf`/`NaN`, making
    disconnected ODs look like zero-cost trips to downstream mode choice.

**Road coupling**: `Transit.build_pt_preload` turns scheduled trips into a PCE preload on road links
(through `pattern_mapping`). Nothing flows back from road to transit: bus run times ignore road
congestion, and unmapped patterns silently contribute zero preload.

**Reusable machinery elsewhere in AequilibraE**:

- `ProjectTable` (`SpatialProjectTable` / `NonSpatialProjectTable`) transactional table CRUD and
  trigger patterns on `project.network`.
- C++ routing/loading infrastructure in `paths/cpp/` (`FourAryHeap`, `workspaces.hpp`,
  `select_link_loading.hpp`, `AssignmentInputs`).
- Road route choice: choice-set generation, path-size logit and select-link.
- State-augmented / contracted non-additive pricing design (`pedro/toll_treatment`).
- Scenarios, periods, a results database, and the matrix API.

---

## 2. What the practice software offers

The descriptions below come from public vendor documentation where it was reachable (Visum help is
public), and otherwise from product knowledge. The EMME and TransCAD reference manuals are not
publicly indexed, so treat those rows as indicative.

### 2.1 PTV Visum

- **Three PuT assignments**:
  - *Transport-system-based*: no lines.
  - *Headway-based*: line level, frequency-driven, with passenger information levels and route
    choice at stops.
  - *Timetable-based*: vehicle-journey level. Branch & bound or shortest-path connection search,
    preselection of dominated connections, then a choice model across connections. Includes an
    independence correction and departure/adaptation-time handling.
  - The fare model and capacity restraint can be used in both line-based procedures.
- **Capacity restraint (timetable-based)**:
  - Hard: boarding is refused when the vehicle is full.
  - Soft: in-vehicle discomfort.
  - Solved iteratively. Crowding effects *at stops* are explicitly not modelled.
- **Fare model**:
  - Fare systems (sets of lines) and ticket types: distance (fare points), zone count, from–to zone
    matrix, short-distance, time-based, direct-distance.
  - Ticket priority selection, per-transport-system supplements, and fares summed across fare
    systems.
  - Revenue calculation.
- **Operator model**: line costs and revenues, vehicle-km and vehicle-hours, line blocking (vehicle
  scheduling), interchange/transfer analysis, timetable and graphical-timetable editors, and PuT run
  times computed from road (PrT) link times.
- **Network**: three-level stop hierarchy (stop → stop area → stop point) with transfer walk-time
  matrices; line route and time-profile editing.
- **Skims**: journey time, perceived journey time, in-vehicle time, access/egress, origin vs
  transfer wait, number of transfers, fare, service frequency, adaptation time, crowding.

### 2.2 Bentley OpenPaths (EMME + CUBE, with AGENT and DYNAMEQ)

OpenPaths is Bentley's 2024 umbrella brand for EMME, CUBE, AGENT and DYNAMEQ. Bentley's marketing
lists frequency-based, schedule-based and crowded transit assignment.

**EMME**:

- *Standard* transit assignment (SF optimal strategies).
- *Extended* transit assignment:
  - **Journey levels**: a state-augmented network for transfer-dependent fares, mode-sequence rules
    and PnR-once.
  - Logit choice at origin connectors and auxiliary-transit nodes.
  - Flow distribution at stops by frequency *and* travel time.
  - Per-line and per-node boarding penalties, wait-time factors and headway fractions; perception
    weights per mode.
- *Congested* transit assignment: volume-dependent crowding cost on segments, solved by an
  FW/MSA-type outer loop.
- *Capacitated* transit assignment: MSA with stop waiting penalties or effective headways, giving
  capacity-feasible flows.
- **Surface transit speed** updating: dwell time from boardings/alightings, run time from auto
  congestion through transit time functions (TTFs).
- Strategy and path analyses, traversal and stop-to-stop matrices.

**CUBE (Voyager PT)**:

- Frequency-based **route enumeration + route evaluation**: multi-routing with logit at stops,
  walk-choice and alternative-alighting models, spread factors.
- Fare systems: FREE, FLAT, FROMTO, ACCUMULATE, DISTANCE, HILO, ZONE.
- Iterative **crowd model** with crowd curves on in-vehicle time and wait-time adjustment for full
  vehicles.
- Non-transit leg generation (walk/drive access, PnR).
- Line, transfer-between-lines and select-link outputs.

### 2.3 Caliper TransCAD

- Route-system transit network built on the road line layer.
- **Assignment methods**: all-or-nothing, *Pathfinder* (Caliper's generalised-cost multipath method
  with headway-based path combination), optimal strategies, stochastic user equilibrium, and
  capacity-restrained/UE transit assignment with route-level capacity.
- Complex fares (flat, zonal, mixed, transfer fares).
- PnR and KnR drive access, with lot-capacity redirection in some deployments.
- Stop, route and segment ridership; flow tables; skims.

### 2.4 MATSim (agent-based)

- **Schedule-based simulation of every transit vehicle** in the QSim, sharing road links with cars
  where configured.
  - Vehicle types carry seats, standing room and per-passenger access/egress times, plus door
    operation modes.
  - So dwell time depends on boardings, and **bunching and denied boarding emerge** under strict
    capacity.
- **SwissRailRaptor router** (RAPTOR):
  - One to two orders of magnitude faster than the legacy router.
  - Range queries, utility-based transfer penalties.
  - **Intermodal access/egress** with feeder modes (bike, car, DRT) and radius-expanding stop search.
- **Ecosystem**:
  - pt2matsim (GTFS/HAFAS/OSM → network-mapped schedule).
  - Crowding-aware scoring (load-factor discomfort, denied boarding).
  - PT fare contribs (distance/zone).
  - **DRT/DVRP** (on-demand dispatch, rebalancing) and **minibus** (evolving paratransit supply).
  - A deterministic PT simulation (SBB) for speed.
- Choice happens through co-evolutionary replanning (route, mode, departure time), not
  closed-form assignment.

---

## 3. Related open/Python ecosystems: Quetzal and `polaris-studio`

### 3.1 Quetzal (read from `D:\OL\src\quetzal`, v3.11.2)

Quetzal (SYSTRA) is a pandas/geopandas *step model*. Its tables are `links` (one row per transit
segment per trip/pattern), `nodes`, `zones`, `footpaths`, `zone_to_transit`, `road_links`,
`road_nodes`, `zone_to_road` and `road_to_transit`. They are serialised to HDF, zip or JSON (Quenedi
web editor). It has **no PT capacity or crowding model**: a search for capacity/crowd finds only
road code.

| Area | Capability (module) |
|---|---|
| GTFS | Filter by area/time/date/service/trip; **convert timetable to frequencies** by pattern over a time range; pattern and direction builders (`io/gtfs_reader`) |
| Network prep | Multi-connector **ntlegs** with a short-walk → long-drive speed curve; footpaths; drop redundant connectors and footpaths; cluster stops, zones and nodes; **cast PT lines onto roads**; map-match lines and GPS tracks; **common-trip aggregation** (combined headway 1/Σ1/h); **sketch new lines** (`add_network`); integrity tests and fixes |
| Frequency path search | Dijkstra with headway waits, then **route-breaker / mode-breaker pruning**: re-run with each route or mode combination removed to build a diverse multi-path choice set (`engine/pathfinder.py`) |
| Strategies | Pure-Python optimal strategy finder and assignment, plus strategy aggregation and plots (`optimalmodel.py`) |
| Timetable | Time-expanded graph pathfinder; **Connection Scan Algorithm** (CSA) with Pareto departure/arrival filtering, an on-stop variant and a pickup/drop-off filter; wished-departure-time expansion and **time-shift** disutility (`connectionscanmodel.py`, `timeexpandedmodel.py`, `engine/csa.py`) |
| PnR | Car to PnR node to PT shortcut graph (`parkridemodel.py`) |
| Choice | **Nested logit over mode / PT mode / PT path**, segmented by purpose or class; **incremental (pivot) logit** against a reference scenario; utilities from time components, price and transfers |
| Fares | Route fares with allowed transfers and durations, OD (zone) fares, tap-free transfer networks, consecutive-leg logic, cheapest-breakdown search (`engine/fares.py`, `analysis_pt_fare`) |
| Loading & outputs | Path-based loading; boardings, alightings and transfers per node and link; PT volumes on road links; segmented and time-expanded loading; line loads; earnings (revenue); screenline checkpoints |
| Calibration | **LP-based OD pivoting to counts** (`analysis_linear_solver`) |
| Interop | Export to CUBE (.lin/.net/.mat), EMME, Excel and Quenedi |

What Quetzal mainly adds beyond AequilibraE:

1. Path-based multi-route choice sets with logit, which handle non-additive fares naturally.
2. Timetable routing (CSA).
3. Network-preparation and QA tooling.
4. Scenario sketching.

It has no crowding or capacity, and its engines are pure Python/SciPy.

### 3.2 `polaris-studio` (Argonne POLARIS Python supply toolkit)

AequilibraE's `public_transport.sqlite` table layout (`agencies`, `stops`, `routes` keyed by
`pattern_id`, `route_links`, `pattern_mapping`, `trips`, `trips_schedule`) shares its lineage with
`polaris-studio`, which pairs a PT supply-editing class (`EditSupply`, reached through
`Transit.edit()`) with `QPolaris`. The only release checked here is the 24.3.117 wheel. Newer
releases may do more, so check the current one before relying on anything below.

- **Programmatic editing (24.3.117)**: add stops, routes, patterns (from a stop sequence) and trips;
  suggest and ensure pattern departures from interval specifications; delete routes and patterns;
  set capacities by route type or by route.
- **Consistency triggers (24.3.117)**: cascade deletes (routes → patterns → trips → schedules,
  stops → connectors), protection against deleting stops still in use, and upkeep of link length,
  stop coordinates and geometry when stops move.
- **Walk and bike access networks**: `transit_walk` and `transit_bike` tables link stops to
  dedicated walk and bike networks, which is directly relevant to `A2`.
- **GTFS lifecycle**: GTFS import and export through the same reader/writer lineage as AequilibraE.
- **Not found in 24.3.117**: pattern duplicate, flip/reverse, extend/truncate or stop insertion;
  re-map-matching of single edited patterns; triggers on fare tables.

Porting `polaris-studio`'s editing and trigger design onto AequilibraE 2.0's `ProjectTable`
abstraction gives Phase 2 a partial blueprint. The pattern operations and fare consistency are new
work.

---

## 4. Consolidated feature list

Status columns: ✔ = has it, ~ = partial, blank = no. Columns: **AeQ** = AequilibraE, **Vis** =
Visum, **OP** = OpenPaths (EMME/CUBE), **TC** = TransCAD, **MS** = MATSim, **Qz** = Quetzal.

### A. Supply, editing and network data

| # | Feature | AeQ | Vis | OP | TC | MS | Qz | Effort | Data model |
|---|---|---|---|---|---|---|---|---|---|
| A1 | GTFS `transfers.txt`/`pathways.txt` (`location_type`), min transfer times, stop hierarchy, and street inter-stop footpaths independent of `parent_station` | ~ (parent stations only) | ✔ | ✔ | ✔ | ✔ | ✔ | S–M | Additive (`transfers` table; populate `stop_connectors` for footpaths) |
| A2 | Network-based access/egress: walk on the road graph (pre-skimmed `stop_connectors` or embedded walk links), multiple connectors with distance bounds, bike/drive access legs | ~ (Euclidean) | ✔ | ✔ | ✔ | ✔ | ✔ | M | Additive (`stop_connectors` + access-mode column) |
| A3 | Vehicle types with seated/total capacity and `standing_area_m2` per trip/pattern | ~ (per pattern, no standing area) | ✔ | ✔ | ✔ | ✔ | | S | Additive (`vehicle_types`, `trips.vehicle_type`) |
| A4 | PT editing suite (`polaris-studio`-style `transit.edit` + SQLite triggers): CRUD on agencies/stops/routes/patterns/trips, pattern duplicate/flip/extend/re-map-match, headway/schedule editing, scenario variants | | ✔ | ✔ | ✔ | ~ | ✔ | M–L | **Additive/Structural** (see §5.1) |
| A5 | Bus run times from road congestion (TTFs) and dwell time from boardings/alightings | | ✔ | ✔ | ~ | ✔ (emergent) | | M | Additive (segment index on `pattern_mapping`, TTF definitions) |
| A6 | Multi-feed/multi-agency (`routes.agency_id`), multi-date import, round-trip edited GTFS export; GTFS Fares v2 | ~ | ✔ | ✔ | ✔ | ✔ | ~ | S–M | Additive (Fares v2 and fare extension tables; Phase 3) |
| A7 | Stop-level `pickup_type`/`drop_off_type` permissions and through-routing / interlining (`block_id`, GTFS `transfer_type=4` stay-on-seat) | | ✔ | ✔ | ✔ | ~ | ~ | S | Additive (`route_links` flags, `in_seat_transfers`) |
| A8 | Canonical transit mode hierarchy & extended GTFS `route_type` (`100–1700`) mapping (local bus vs BRT vs rail submodes) | ~ (raw `route_type` subset) | ✔ | ✔ | ✔ | ✔ | ✔ | S | Additive (`transit_modes` mapping) |

**Notes**:

- **A1** fixes two baseline defects at once: respecting explicit GTFS transfer rules, and generating
  walk-network footpaths between nearby stops (e.g. within 400–800 m via `split_links_at_stops`) even
  when `parent_station` is null. Standardise the topology so stop-to-stop footpaths run `stop_A →
  stop_B` (`freq = inf`), reserving direct `alighting → boarding` edges strictly for line-to-line
  timed (`transfer_type=1`), forbidden (`transfer_type=3`), or stay-on-seat (`A7`) rules. That keeps
  all normal boardings flowing through `boarding` edges.
- **A2**:
  - Snapping stops to the walk-mode road graph lets us either embed walk links directly in the SF
    graph (EMME auxiliary-transit style, giving walk-all-the-way ODs and transfers with no connector
    count truncation) or pre-skim shortest walk paths into `stop_connectors` (needed by RAPTOR in
    `B7` and smaller for SF). Supporting both from the same snapped walk graph keeps SF and RAPTOR
    consistent.
  - Note that multi-connector zones still assign AoN to one stop under standard SF unless paired with
    connector logit (`B4`).
- **A3**: adding `standing_area_m2` (or crush standing density in pax/$\text{m}^2$) alongside
  `seated_capacity` and `total_capacity` is needed to apply standard UK WebTAG / Australian ATAP
  crowding curves directly in `B5` and `C5`.
- **A4** takes the add/delete operations and cascade-delete triggers from `polaris-studio` (§3.2)
  and adds what it lacks: `ProjectTable`-backed gateways on `project.transit`, pattern
  geometry/map-matching operations, wider consistency triggers in `transit_triggers.sql`, and §5.1
  period service specs.
- **A5** closes the road↔transit loop. `pattern_mapping` holds road links per pattern but not which
  stop-to-stop segment (`seq`) each belongs to; recording `seq` at map-matching time (Phase 3) feeds
  the iterative speed/dwell feedback algorithm (Phase 4.2).
- **A7** prevents illegal local trips on pick-up-only/set-down-only express stops (must be part of
  pattern identity so differing permissions split `pattern_id`) and removes false transfer/wait
  penalties on through-routed rail and bus lines via zero-penalty `alighting → boarding` dwell edges.
- **A8** stops extended GTFS rail/metro feeds defaulting to 60-passenger bus capacities and gives
  `B1` user-meaningful modes (e.g. separating BRT from local bus when both have `route_type = 3`).

### B. Assignment and route choice

| # | Feature | AeQ | Vis | OP | TC | MS | Qz | Effort | Data model |
|---|---|---|---|---|---|---|---|---|---|
| B1 | Full generalised cost per `TransitClass`: weights per component and mode/line, boarding/transfer penalties per line/stop, wait-time functions, class-specific VOT | ~ (baked into `trav_time`) | ✔ | ✔ | ✔ | ✔ | ✔ | S–M | **Structural** for persisted graphs (`links.trav_time` and `freq` become physical; see §5.4) |
| B2 | Additive fares (flat per boarding, per distance/chord) in cost and skims | | ✔ | ✔ | ✔ | ✔ | ✔ | S | None (compiled from Phase 3 fare tables) |
| B3 | Journey levels / state-augmented graph: transfer-dependent fares, mode-sequence rules; plus destination-thread OD zonal pricing | | ~ | ✔ | ~ | n/a | ~ (path-based) | L | None (graph-time vertex replication) |
| B4 | Extended SF choice: logit at origin/destination connectors, RTPI / frequency-and-time stop split, regular-headway wait models | | ✔ | ✔ | ✔ | n/a | | M | None (SF kernel extension) |
| B5 | **Congested transit assignment**: in-vehicle crowding discomfort (seated vs standing / pax per $\text{m}^2$) and optional station/walkway VDFs | | ✔ | ✔ | ✔ | ✔ | | L | Additive (segment/walkway capacity & standing area on graph) |
| B6 | **Capacitated assignment**: strict capacity with on-board (`dwell`) priority via effective frequencies or fail-to-board | | ✔ | ✔ | ~ | ✔ | | L | Same as B5 |
| B7 | **Schedule-based routing** (rRAPTOR / McRAPTOR): Pareto connection search and range-query skims by departure window | | ✔ | ✔ | ? | ✔ | ✔ (CSA) | L | Additive (optional time-sliced skim outputs) |
| B8 | **Schedule-based assignment**: connection choice, departure-time / adaptation choice, then trip capacity | | ✔ | ✔ | ? | ✔ | ~ | XL | **Structural** (time-sliced demand, results per trip/segment) |
| B9 | Path-based multi-route PT choice (enumeration/pruning + (nested) logit, PSL) — exact vehicle for capped / time-window fares | | ✔ | ✔ (CUBE) | ✔ | ✔ (agents) | ✔ | L | Additive (path tables in results) |
| B10 | Park-and-ride / kiss-and-ride / feeders: (a) matrix-coupled station choice with lot capacity (`M`); (b) multimodal graph (`L`) | | ✔ | ✔ | ✔ | ✔ | ✔ | M–L | Additive (`pnr_lots` table) |
| B11 | Multi-class transit assignment over a shared physical SF graph with class-specific generalised costs | ~ | ✔ | ✔ | ✔ | ✔ | ✔ | S | None |

**Notes**:

- **B1/B2/B11**:
  - Separate unweighted physical attributes on the graph from class-level generalised cost on
    `TransitClass` (mirroring road `Graph` + `TrafficClass`; see §5.4).
  - Allow per-mode (`A8`), per-line and per-stop overrides, plus a monetary fare channel converted via
    class VOT. Because costing happens at assignment time, multiple classes share one persisted graph.
- **B3**:
  - Replicating vertices across 3–6 journey levels handles boarding-count fares, flat transfer
    discounts and mode-sequence rules with the SF kernel unchanged; `B5`/`B6`/`C1` must sum edge
    volumes across all level-copies of each physical segment (same rule as `non_additive_tolls` in
    `toll_modeling.rst`, branch `pedro/toll_treatment`).
  - **Scope boundary**: journey levels do *not* scale to **fare caps** (combinatorial remaining-cap
    balance) or **clock-limited transfers** (`transfer_duration`, e.g. 90-min expiry), which belong
    in `B9` or `B7`/`B8`. Naive origin-zone replication also blows up ($\times N_{\text{zones}}$) for
    OD zonal matrices — **however**, because SF runs one destination $d$ per pass, an OD zonal fare
    depending only on `(boarding_zone, dest_zone(d))` can be priced per destination thread with
    **zero vertex replication**, under these conditions:
    - The fare is charged on **access connectors** (zone → first stop). Charging "on first boarding"
      is itself a state (paid / not yet paid).
    - All egress stops of $d$ share one fare zone.
    - It breaks when riders walk from the access stop to a stop in another fare zone before boarding,
      or walk at the egress end to a stop in another zone.
    - The kernel needs **per-destination edge costs** (a cost overlay or callback per destination),
      which is an API change rather than zero work.
  - Single-line distance or station-pair fares can use **boarding-to-alighting pattern chords**.
    Chords grow with the square of a pattern's stop count (fine for rail, heavy for long bus lines),
    and chord loads must be split back onto physical segments for `C1`, `B5` and `B6`.
- **B4**:
  - **Connector logit** is required for `A2` to spread zone demand across multiple walk connectors
    (standard SF treats `freq = inf` connectors as deterministic AoN). Origin-connector logit can be
    applied directly at zone origins after the backward pass.
  - **Stop split & headways**: scaling `freq` by `1 / wait_time_factor` (e.g. `0.5` for clockface
    service) gets single-line wait right but overstates common-line wait savings and still splits
    riders purely by $f_a$ regardless of in-vehicle time differences. Adding RTPI / countdown
    information splits (Gentile et al. 2005; Visum passenger information levels) handles modern
    app-informed riders and express-vs-stopper corridors properly.
- **B5/B6** cannot simply reuse road `LinearApproximation` (FW/CFW/BFW):
  - Even for separable fixed-frequency crowding (`B5`), Spiess & Florian's (1989, §5) convex
    objective is $\sum_a \int_0^{v_a} c_a(x)\,dx + \sum_i w_i$, so any exact line search and gap must
    include each strategy direction's total waiting time $W = \sum_i w_i$ (which road
    `LinearApproximation` lacks).
  - Once **seated-vs-standing priority** (continuing `dwell` riders keep seats before `boarding`
    riders; Schmöcker et al. 2011), **boarding-dependent dwell times (`A5`)**, or **effective
    frequencies (`B6`, Cepeda–Cominetti–Florian 2006)** are active, the Jacobian is **asymmetric**
    and no convex Beckmann objective exists — invalidating exact line search and CFW/BFW Hessian
    conjugacy. Plan on **MSA / self-regulated averaging (MSWA)** and the **Marcotte–Nguyen / CCF
    variational-inequality gap**:
    $$\text{Gap}(v, w) = \frac{\sum_a c_a(v)\,v_a + \sum_i w_i - \sum_{od} d_{od}\,u_{od}(v)}{\sum_a c_a(v)\,v_a + \sum_i w_i},$$
    tracking strategy-bundle or destination-bush waits because experienced wait $w_i \neq \max_a
    v_a / f_a(v)$ on convex combinations of strategies across iterations.
  - **On-board priority in B6**: residual capacity available to `boarding` passengers on segment $k$
    is $C_k - v_{\text{dwell},k}$ (using period trip count $N_{\text{trips}} \times
    \text{total\_capacity}$), never $C_k - v_{\text{on-board},k}$.
  - **Station/walkway crowding**: because station walkways and transfers are ordinary SF edges, BPR
    VDFs on `walking`/`transfer`/`access` edges fit into `B5`'s outer loop at minimal extra cost.
- **B7**:
  - Commit to **rRAPTOR / McRAPTOR** in `paths/cpp/` as the primary C++ schedule router: RAPTOR's
    transfer-count rounds naturally build Pareto connection sets for `B8`, whereas plain
    earliest-arrival CSA collapses transfer trade-offs.
  - **Memory discipline**: materialising one dense skim core per time slice across a metropolitan
    zoning system ($4{,}000^2 \times 8 \text{ skims} \times 24 \text{ slices} \times 8\text{ B}
    \approx 25\text{ GB}$ per period) should be opt-in. By default, integrate range queries over the
    period window inside C++ to output summary cores (mean perceived time, adaptation time,
    50th/90th percentile time, viable departures, unreachable share).
- **B8** is the only item that forces a new *demand* dimension (OD × departure slice) and results
  keyed by vehicle trip. Do it after `B7` has proved the engine.
- **B9**:
  - Quetzal's route/mode-breaker approach maps onto AequilibraE's existing choice-set and PSL code if
    paths are expressed as legs `(line, board_stop, alight_stop)`.
  - It is the exact way to price **fare caps** and **time-window transfers** without journey-level
    state explosion, at the cost of losing common-lines unless waits are recomputed on combined
    headways of attractive lines.
- **B10** — ship **B10a (matrix-coupled station choice, Effort M)** before **B10b**:
  - Skim congested auto $(o \to k)$ from `TrafficAssignment` and transit $(k \to d)$ from
    `TransitAssignment`, evaluate discrete lot choice $P(k \mid o, d)$ with Sinkhorn/dual shadow
    prices for lot capacity, and split PnR demand into auto leg matrix $D_{ok}^{\text{auto}}$ (loaded
    in `TrafficAssignment`) and PT leg matrix $D_{kd}^{\text{PT}}$ (loaded in `TransitAssignment`).
  - Unlike a monolithic car+transit SF graph (`B10b`), this needs **no `B3` dependency**, keeps car
    use at the home end, and loads PnR cars onto congested station access roads. Using the same lot
    for the PM return as for the AM outbound trip still needs tour- or PA-based demand, or deriving
    the PM lot choice by transposing the AM choice. KnR and bike/DRT feeders use the same split
    without lot shadow prices.

### C. Outputs and analysis

| # | Feature | AeQ | Vis | OP | TC | MS | Qz | Effort | Data model |
|---|---|---|---|---|---|---|---|---|---|
| C1 | Stop/line boardings & alightings, segment load profiles, max-load point, line/mode transfer matrices, pax-km/hours, **passenger loads on physical road/rail links**, screenlines & GEH/%RMSE | ~ (SF edge volumes only) | ✔ | ✔ | ✔ | ✔ | ✔ | S | None (derived from edges + `pattern_mapping` / `route_links`) |
| C2 | Select-link/line/stop and traversal (stop-to-stop) matrices for strategies | | ✔ | ✔ | ✔ | ✔ | ✔ | M | None |
| C3 | Operator indicators: vehicle-km/hours, peak fleet size, cost, revenue; line blocking | | ✔ | ~ | ~ | ✔ | ~ | S–M (blocking: L) | Additive (cost parameters on vehicle types) |
| C4 | Accessibility / isochrones / travel-time cubes | | ✔ | ~ | ✔ | ✔ | ~ | S once B7 exists | None |
| C5 | Mode-choice & appraisal-ready skims: perceived cost, true first vs transfer wait, uncrowded vs crowded IVT, fare, true transfers/boardings, `inf` on disconnected ODs, logsums | ~ (wait/transfer/unreachable bugs) | ✔ | ✔ | ✔ | n/a | ✔ | S–M | None |
| C6 | Transit OD estimation against boardings/line/screenline counts | | ✔ | ✔ | ✔ | n/a | ✔ (LP) | L | None |
| C7 | Supply & GTFS QA diagnostics: negative segment times, speed outliers, orphan stops/zones, loop checks, unmapped `pattern_mapping` routes | | ✔ | ~ | ~ | ✔ | ✔ | S | None |

**Notes**:

- **C1** is the highest value-to-effort reporting item in this document. Projecting `on-board`
  passenger loads onto physical `pattern_mapping` road links and `route_links` geometries also gives
  QGIS immediate corridor bandwidth maps without overlapping pattern linestrings.
- **C2** needs one extra marked-flow propagation per destination in the SF kernel, mirroring
  `select_link_loading.hpp` on the road side.
- **C5**:
  - Accumulate expected waiting time ($1 / f_i$ on attractive boarding bundles, unscaled by
    `wait_time_factor`) directly inside `_SF_in_first_pass_full`, distinguishing first wait (at
    origin-connected stops) from transfer wait, rather than subtracting from `trav_time`.
  - **Component-skim discontinuity warning**: in deterministic SF, expected *generalised cost* $u_i$
    is continuous when a marginal line enters or leaves the attractive set, but individual component
    skims (`in_vehicle_time`, `waiting_time`, `fare`, `boardings`) **jump discontinuously**, causing
    noise in mode-choice feedback and Rule-of-a-Half appraisal. Passing consistent generalised cost
    (or smoothing line inclusion via `B4`) avoids this.
- **C7** catches real-world GTFS and editing defects upfront (e.g. `arrival < prev_departure` in
  `_compute_segment_travel_time`, which currently crashes `_check_edges` on negative `trav_time`, and
  silent `pattern_mapping` misses that zero out bus preload).

### D. Agent-based and on-demand (MATSim territory)

| # | Feature | Effort | Recommendation |
|---|---|---|---|
| D1 | DRT / on-demand dispatch, minibus evolution | XL | Out of scope for core; interoperate instead |
| D2 | Within-day vehicle simulation (bunching, denied boarding) | XL | Out of scope; B6 gives static capacity effects |
| D3 | Export to MATSim `transitSchedule.xml`/`transitVehicles.xml` (and import of pt2matsim output) | S–M | Worth doing alongside GTFS round-trip export |

---

## 5. Cross-cutting design decisions (decide before coding)

### 5.1 How service plans are represented (affects A4, A5, B7, B8, C3)

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
3. **GTFS-`frequencies.txt`-style period service specs + segment run-time profiles (recommended).**
   - Store `pattern_periods` (`pattern_id`, `period_id`, `service_source`, `headway_s`, `n_trips`,
     `first_departure_s`, `exact_times`) and `pattern_segment_profiles` (`pattern_id`, `period_id`,
     `seq`, `run_time_s`, `dwell_time_s`).
   - `service_source` makes each `(pattern_id, period_id)` unambiguously either `schedule`
     (populated from `trips_schedule` on GTFS import) or `headway` (populated from GTFS
     `frequencies.txt` or `A4` `transit.edit` scenario sketching). `exact_times` keeps its GTFS
     meaning (a headway-defined service with exact departure times) and is imported, rather than
     commented out as it is today in `column_order.py`.
   - **Authority rule**: frequency engines (SF, `B5`/`B6`) always read the profiles, which are
     initialised from `trips_schedule` means for `schedule` patterns. Schedule engines (`B7`/`B8`)
     read `trips_schedule` for `schedule` patterns, optionally shifted by the run-time deltas `A5`
     writes into the profiles. `A5` therefore updates `run_time_s` and `dwell_time_s` in place
     without mutating `trips_schedule` or crossing period boundaries.
   - **Headway expansion for schedule engines**: `headway` patterns are expanded lazily in C++ from
     `first_departure_s` (GTFS `frequencies.txt` `start_time`, or a user-set offset for sketched
     lines). Without a defined offset, transfer timings between headway-driven lines would be
     arbitrary.
   - Pair with fixing `Scenario.transit` so each scenario owns an isolated `public_transport.sqlite`.

### 5.2 Strategy-based vs path-based core

Strategies (SF) are AequilibraE's strength: they are fast, exact, and represent common lines.
Path-based choice (`B9`) is what Quetzal and CUBE use for capped/time-window fares and for nesting
under mode choice. Recommendation: keep SF as the core and extend it (`B1–B6`, `B3` for sequence and
OD-zonal fares). Add path-based choice as a **separate engine** that reuses road route-choice code,
rather than forcing one paradigm to do both.

### 5.3 C++ engine consolidation: SF kernel port and rRAPTOR

- **Port SF to `paths/cpp/hyperpath.hpp` in Phase 1 (`Effort: M`; about 700 lines of Cython with
  skims, parallelism and tests)**: move `hyperpath.pyx` into
  `paths/cpp/` with persistent per-thread workspaces (matching `workspaces.hpp`) and zero-copy
  `update_costs(c_a, f_a)`. That removes per-destination heap/buffer allocations, makes 50-iteration
  `B5`/`B6` loops fast, and shares conventions with `select_link_loading.hpp` (`C2`).
- **Schedule engine (`B7`/`B8`)**: implement **rRAPTOR / McRAPTOR** in `paths/cpp/raptor.hpp`,
  parallel over origins × departure windows, with in-kernel period aggregation for summary skims and
  opt-in per-slice matrix output.

### 5.4 Separate physical graph persistence from class generalised cost

Do **not** bake perception weights, boarding/alighting penalties or fares into persisted
`links.trav_time` or `transit_graph_configs`. Instead:

- `transit_graph_configs` persists **topological and physical builder settings** (`period_id`, CRS,
  connector method/bounds, footpath radius, transfer rules).
- `nodes`/`links` persist **unweighted physical attributes** (in-vehicle run time, dwell time, walk
  distance/time, scheduled headway, period trip count, capacity) and a physical **`segment_id`**
  (pattern + `seq` for line edges, stop for stop edges). Loads then aggregate correctly once journey
  levels (`B3`) or chords copy edges.
- Changing what persisted `links.trav_time` and `freq` mean is a **structural** change, so the
  migration that introduces it must invalidate or rebuild saved graphs.
- **Behavioural costing** (component/mode/line weights, boarding and transfer penalties, wait
  factors, fare schedules, VOT) lives on `TransitClass`, is evaluated into `c_a` and `f_a` at
  assignment time, and is recorded in `results_database` procedure reports — matching road `Graph` +
  `TrafficClass`.

---

## 6. Proposed four-phase plan

### Phase 1 — Repair existing issues

Fix defects in what AequilibraE already ships and align the SF engine with `paths/cpp/`, without
changing user workflows:

- **Scenario isolation**: fix `Scenario` (`scenario.py`) so `scenario.transit` connects to the
  scenario's own `public_transport.sqlite` and `project_database.sqlite` rather than the root
  project's databases.
- **`TransitGraphBuilder` repairs**:
  - Fix `_add_mean_headway_to_segments`: sort by `arrival` within `(pattern_id, seq)` so differences
    are never negative, group by `["pattern_id", "trip_id"]` instead of
    `.groupby("pattern_id").min()`, fix the `pattern_id = row.seq` typo, stop `headway <= 0` from
    producing `freq = inf`, and store both scheduled headway and period trip count $N_{\text{trips}}$.
  - Filter trips rather than stop rows across periods (`_create_line_segments`,
    `_add_mean_headway_to_segments`) so long commuter trips are not truncated at `end`.
  - Add a loop-route regression test (a pattern visiting a `stop_id` twice) covering dwell and
    transfer edges; fix only if it fails.
  - Clean transfer topology so stop-to-stop transfers run `stop_A → stop_B` (`freq = inf`), keeping
    normal transfer boardings on `boarding` edges.
  - Separate unweighted physical attributes on the persisted graph from behavioural cost (§5.4),
    add a physical `segment_id` to every edge, and persist all topological builder attributes in
    `transit_graph_configs`. This changes what persisted `links.trav_time` and `freq` mean, so ship
    it with a migration that invalidates or rebuilds saved graphs.
- **Skim & validation repairs**:
  - Compute true expected waiting time ($1 / f_i$ on attractive boarding bundles, unscaled by
    `wait_time_factor`) and true transfer counts inside the SF kernel instead of subtracting from
    costed `trav_time`.
  - Return `inf`/`NaN` (not `0.0`) on disconnected OD pairs; guard against negative segment travel
    times (`arrival < prev_departure`) on import and graph build.
- **GTFS reader repairs**: enable commented-out `agency_id`, `block_id`, `pickup_type`/`drop_off_type`
  and `location_type` fields in `column_order.py`; map extended HVT `route_type` codes (`100–1700`)
  in default capacities/PCEs.
- **C++ SF kernel port (§5.3, `Effort: M`)**: port `hyperpath.pyx` to `paths/cpp/hyperpath.hpp` with
  persistent per-thread workspaces and zero-copy `update_costs(c_a, f_a)`.
- **Result-change management**: these fixes change results for existing users (headways, waits,
  transfers, unreachable pairs). Publish a before/after comparison on the Coquimbo example and a
  changelog entry describing each change.

### Phase 2 — Settle the supply data model, then add PT editing tools

Land every supply schema change first, then turn `project.transit` from an import-only wrapper into
a supply authoring and scenario-editing toolkit (`A4`, `A6` part 1, `C7`, `D3`) built on the final
schema, so the editing layer and triggers are not reworked later. Editing adapts `polaris-studio`'s
add/delete operations and cascade triggers (§3.2) onto AequilibraE's `ProjectTable` layer; the
pattern operations and wider triggers are new work. Fare tables and fare editing ship with the fare
work in Phase 3, so Phase 2 does not touch fares.

- **Supply schema migrations (first)**:
  - §5.1: `pattern_periods` and `pattern_segment_profiles`.
  - `A1`: `transfers` and `pathways` tables.
  - `A3`: `vehicle_types` table (`seated_capacity`, `total_capacity`, `standing_area_m2`, `pce`,
    boarding/alighting service rates, unit operating costs) linked from `routes`/`trips`.
  - `A5` (schema prep): stop-to-stop segment index (`seq`) on `pattern_mapping` plus TTF parameter
    definitions.
  - `A7`: `pickup_type`/`drop_off_type` on `route_links` and `in_seat_transfers` table (`block_id` /
    `transfer_type=4`).
  - `A8`: `transit_modes` hierarchy table mapping GTFS/HVT `route_type` and route overrides to user
    submodes.
  - `B10` (schema prep): `pnr_lots` table (road node, transit stop(s), capacity, parking charge).
- **`ProjectTable`-backed `transit.edit` API**:
  - Transactional CRUD gateways for `agencies`, `stops`, `routes`/`patterns`, `route_links`,
    `pattern_mapping`, `trips`, `trips_schedule` and the new supply tables above.
  - **Stop & station operations**: add, move, snap to road/walk links, merge/cluster stops, and
    manage the `parent_station` hierarchy.
  - **Pattern operations**: sketch new routes/patterns from stop sequences or road paths without
    GTFS (as `polaris-studio` does), plus new operations `polaris-studio` 24.3.117 lacks:
    **duplicate**, **flip/reverse**, **extend/truncate**, insert/remove stops, split patterns on
    differing `pickup_type`/`drop_off_type`, and **re-map-match** individual edited patterns onto the
    road network (`RouteMapMatcher`), writing the `pattern_mapping` segment index as it goes.
  - **Service & timetable editing**: create, shift, scale or replace service plans via §5.1 period
    headways (`pattern_periods` + `pattern_segment_profiles`) and/or trip timetables; hooks for
    `QAequilibraE`.
- **SQLite consistency triggers (`transit_triggers.sql`)**: start from `polaris-studio`'s cascade
  deletes and geometry upkeep, then extend them to cascade updates and protect referential integrity
  across `stops`, `routes`, `route_links`, `pattern_mapping`, `trips`, `trips_schedule` and the new
  supply tables, and invalidate stale persisted SF graphs on supply edits.
- **Multi-feed GTFS lifecycle & interop (`A6` part 1, `D3`)**: multi-feed/multi-date/multi-agency
  import without ID collisions, area/time filtering, clean round-trip `gtfs_writer` export of edited
  networks to valid GTFS `.zip`, and MATSim `transitSchedule.xml`/`transitVehicles.xml` export/import
  (`D3`).
- **Supply & GTFS QA diagnostics (`C7`)**: automated integrity checks run on GTFS import and after
  edits (negative/zero run times, speed outliers, broken `pattern_mapping` sequences, orphan stops,
  isolated zones).

### Phase 3 — Network preparation, costing, fares and reporting

Land the network-preparation builders, the `TransitClass` cost model, the fare work and standard
reporting/skimming on top of the existing uncongested SF engine. With Phase 2's supply schema and
this phase's fare schema in place, Phase 4 needs no further data-model churn (except `B8`'s
time-sliced demand/trip-load tables).

- **Network-preparation builders**:
  - `A1`: populate `stop_connectors` for both station pathways and walk-network street inter-stop
    footpaths independent of `parent_station`, honouring the `transfers` rules.
  - `A2`: network-based multimodal access/egress builder snapping stops to walk/bike/drive graphs
    (`split_links_at_stops`) to populate multi-connector `stop_connectors` (and optional embedded
    walk links for SF).
  - `A7`: wire pick-up/drop-off permissions and `in_seat_transfers` into `TransitGraphBuilder`.
- **Costing, multi-class & baseline reporting tooling**:
  - `B1`, `B11`: `TransitClass` generalised-cost specification (per-component and per-mode/line
    weights, per-line/stop boarding and transfer penalties, wait-time functions, VOT, rider category
    and fare medium) and multi-class uncongested SF execution (`B11`) over a shared physical graph.
    Fares enter through the fare work below.
  - `C1`: stop boardings/alightings, line/pattern ridership, segment load profiles, max-load points,
    transfer matrices, pax-km/hours, **passenger-load projection onto physical road/rail links** for
    QGIS bandwidth maps, screenlines, and GEH/%RMSE helpers.
  - `C3` (basic): scheduled vehicle-km, vehicle-hours, peak fleet estimate and operating cost.
    Farebox revenue comes from the fare work below.
  - `C5`: mode-choice and appraisal-ready unweighted physical skims (perceived cost, first vs
    transfer wait, IVT, access/egress/walk, boardings/transfers). Fare skims come from the fare work
    below.

#### Fares in Phase 3

All fare work lands together, so every later engine (`B3`, `B7`, `B9`) consumes one fare
representation and one set of pricing rules. Fares fall into three tiers, according to what an
engine needs in order to price them exactly:

| Tier | Examples | Representation | Exact in |
|---|---|---|---|
| 1. Additive | Flat fare per boarding, boarding fare per network or boarding area, linear per-km fare | Edge costs (this phase) | Every engine |
| 2. A few states | Free or discounted transfers by count, fare integration across networks, boarding-zone × destination-zone fares, mode-sequence premiums | Journey levels or destination-thread zonal pricing (`B3`, Phase 4.3) | SF with `B3`; RAPTOR with labels per state |
| 3. Depends on the whole path | Distance stages accumulated across legs, time-limited transfers, fare caps, cheapest-ticket selection | Fare calculator on complete paths | `B9`; `B7`/`B8` where times are known; strategies only as approximate skims |

- **Fare data model (`A6` part 2)**:
  - GTFS Fares v2 tables: `fare_media`, `rider_categories`, `fare_products` (`amount`, `currency`,
    rider category, fare medium), `fare_leg_rules` (network, from/to area, from/to timeframe,
    `rule_priority`), `fare_leg_join_rules`, `fare_transfer_rules` (`transfer_count`,
    `duration_limit`, `duration_limit_type`, `fare_transfer_type`), `areas`/`stop_areas`,
    `networks`/`route_networks` and `timeframes`.
  - Translate the existing Fares v1 tables (`fare_attributes`, `fare_rules`, `fare_zones`) into v2,
    on import and through a migration for existing projects, so there is one internal
    representation. Keep v1 output in `gtfs_writer` for GTFS round-trips.
  - Extension tables for what core Fares v2 does not express (it has no distance fields and no
    caps): `fare_distance_stages` (distance bands per network, with optional linear interpolation
    between stages; Visum fare points, CUBE DISTANCE/HILO), zone-count fares (price by the number of
    areas traversed; Visum zone-based, CUBE ZONE), and `fare_caps` (cap amount per time window,
    rider category and fare medium).
  - Fare editing gateways on Phase 2's `ProjectTable` layer, with triggers keeping products, leg
    rules, transfer rules, areas and stops consistent.
- **Fare calculator (independent of any algorithm)**:
  - `price(legs, rider_category, fare_media)`, where each leg is (route/network, boarding stop,
    alighting stop, boarding time, alighting time, in-vehicle distance). It returns the price plus a
    per-leg and per-operator breakdown.
  - Pricing rules: leg matching by network, area and timeframe with `rule_priority`; leg joins;
    transfer rules (counts, duration limits, discount types); distance stages; zone counts; caps;
    prices summed across independent fare systems (as Visum does); and, where several products
    apply, either priority order (Visum) or the cheapest valid combination (Quetzal's cheapest
    breakdown), configurable per project.
  - Python reference implementation first; C++ once `B9`/`B7` need the throughput.
  - Used by tests, revenue, approximate skims, path-based choice (`B9`), a fare criterion in RAPTOR
    (`B7`), and validation of journey levels (`B3`): on enumerated paths, the compiled graph cost
    must equal the calculator's price.
- **Fare classification on graph build**: classify each fare system into a tier, report which
  engine can price it exactly, and warn (never drop rules silently) when a tier-2 or tier-3 rule is
  approximated.
- **Tier-1 compilation into SF (`B2`)**: boarding fares on `boarding` edges (by network and boarding
  area) and linear per-km fares on `on-board` edges, converted to time through each `TransitClass`'s
  VOT for its rider category and fare medium. This gives exact fare-sensitive routing and an exact
  expected-fare skim.
- **Fare skims beyond tier 1 (until Phase 4.3)**: price each OD pair's most likely path through the
  strategy (following the highest-share edge at each node) with the calculator, and flag the skim
  as approximate in its metadata.
- **Revenue (`C3`)**: tier 1 exactly, from edge volumes × edge fares; tiers 2–3 from the calculator
  on the same representative paths (approximate until `B3`/`B9`). Shared-ticket revenue is
  allocated to operators by boardings, passenger-km or the calculator's per-leg breakdown.
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
| **4.1** | **Low–Moderate** (Effort S–M each; single-pass SF extensions & matrix coupling) | `C2`, `B4`, `B10a` | **`C2`** marked-flow select-link/line/stop & stop-to-stop traversal matrices in `hyperpath.hpp`; **`B4`** origin/destination connector logit (completing `A2`), RTPI / frequency-and-time stop splits & regular-headway waits; **`B10a`** matrix-coupled PnR/KnR/feeder station choice coupling `TrafficAssignment` auto skims with `TransitAssignment` PT skims and Sinkhorn/dual lot-capacity shadow prices |
| **4.2** | **Medium–High** (Effort M–L each; iterative equilibrium & road↔PT feedback) | `B5`, `B6`, `A5` | **`B5`** congested SF with WebTAG/ATAP seated-vs-standing (`dwell`-priority) crowding curves and station/walkway VDFs; **`B6`** strict-capacity SF ($C_k - v_{\text{dwell},k}$) via CCF effective frequencies / fail-to-board, solved by MSA/MSWA against the Marcotte–Nguyen VI gap with strategy-bundle waiting times; **`A5`** TTF bus run times from congested road speeds + boarding-driven dwell times in an outer road↔transit loop |
| **4.3** | **High** (Effort L each; state expansion, path enumeration & inverse problems) | `B3` (+ `B10b`), `B9`, `C6` | **`B3`** journey-level state-augmented SF, compiled from the Phase 3 fare rules and checked against the fare calculator (3–6 levels for transfer rules and mode sequences, summing physical segment loads across levels, plus zero-replication destination-thread OD zonal pricing and optional **`B10b`** graph-coupled PnR); **`B9`** path-based PT route choice (route/mode-breaker + PSL/nested logit, giving exact **fare caps** and **time-window transfers**); **`C6`** transit ODME against boarding, line and screenline counts using `C2` |
| **4.4** | **Very High** (Effort L–XL; new C++ schedule engine & dynamic demand/supply dimensions) | `B7`, `C4`, `B8` (+ `C3` blocking) | **`B7`** C++ `rRAPTOR`/`McRAPTOR` router in `paths/cpp/raptor.hpp` with Pareto connection search and in-kernel period-summary / sliced skims; **`C4`** timetable accessibility & isochrones (`S` once `B7` lands); **`B8`** full schedule-based assignment with connection + departure/adaptation-time choice over time-sliced demand and iterative trip-level capacity restraint (`XL`, structural); optional **`C3`** MIP/min-cost-flow line blocking (`L`) |

Phases 1–3 plus Phases 4.1–4.2 bring AequilibraE to parity with EMME's standard, congested and
capacitated transit assignment, Visum's headway-based procedure, and `polaris-studio`'s supply
authoring workflow. Parity with EMME's extended transit assignment needs journey levels (`B3`,
Phase 4.3) on top of `B4` (Phase 4.1). Phases 4.3–4.4 complete the remaining commercial feature set
apart from within-day agent-based simulation.

---

## Sources

- PTV Visum help, timetable-based assignment:
  https://cgi.ptvgroup.com/vision-help/VISUM_2025_ENG/Content/1_Benutzermodell%20OeV/1_6_Fahrplanfeine%20Umlegung.htm
- PTV Visum help, headway-based assignment:
  https://cgi.ptvgroup.com/vision-help/VISUM_2025_ENG/Content/1_Benutzermodell%20OeV/1_6_Taktfeine%20Umlegung.htm
- PTV Visum help, capacity restriction:
  https://cgi.ptvgroup.com/vision-help/VISUM_2024_ENG/Content/1_Benutzermodell%20OeV/1_6_Kapazitaetsbeschraenkung.htm
- PTV Visum help, PuT fare model:
  https://cgi.ptvgroup.com/vision-help/VISUM_2025_ENG/Content/1_Betreibermodell%20OeV/1_7_%C3%96V-Tarifmodell.htm
- PTV Visum public transport planning: https://www.ptvgroup.com/en/products/ptv-visum/public-transport-planning
- Bentley OpenPaths: https://www.bentley.com/software/openpaths/software/mobility-simulation/ and
  https://br-en.virtuosity.com/openpaths
- CUBE Voyager PT (Advanced) course outline: https://citilabs.teachable.com/courses/109625
- EMME-based transit assignment documentation (secondary):
  https://tau-docs.tmr.qld.gov.au/ptm/docs/PTM%20Assignment/model_assignment/ and
  https://tmg.utoronto.ca/doc/2.0/tmgtoolbox2_emme/tools/Assign/AssignTransit.html
- Caliper TransCAD transit assignment: https://www.caliper.com/glossary/what-is-transit-assignment.htm;
  CTPS work program (secondary): https://www.ctps.org/data/calendar/pdfs/2016/MPO_0107_Modeling_Work_Program.pdf
- MATSim SwissRailRaptor: https://jitpack.io/p/simunto/matsim-sbb-extensions;
  pt2matsim: https://github.com/matsim-org/pt2matsim;
  crowding (MUM 2025): https://www.matsim.org/conferences/mum2025/abstracts/MUM25_paper_13.pdf
- Argonne National Laboratory, POLARIS / `polaris-studio` documentation (Public Transport & `transit.edit`):
  https://polaris.taps.anl.gov/polaris/index.html and https://pypi.org/project/polaris-studio/;
  editing API and triggers checked in the `polaris_studio-24.3.117` wheel.
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
- Codina & Rosell (2017), congested capacitated transit assignment with strategies:
  https://ideas.repec.org/a/eee/transb/v106y2017icp293-320.html
- Quetzal source, `D:\OL\src\quetzal` (v3.11.2, commit f0fe274).
