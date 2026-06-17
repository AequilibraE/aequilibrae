# Plan: Replace the custom OSM importer with a pluggable network-acquisition framework

## 1. Goal & scope

Deprecate and remove AequilibraE's hand-rolled OSM importer (Overpass downloader + node/way → links/nodes builder) and replace it with a **layered, source-agnostic framework** in which OpenStreetMap is just one of several network providers. The same framework supports Overture Maps, arbitrary GeoDataFrames, common geospatial files (GeoPackage / GeoJSON / Shapefile / FlatGeobuf), and the existing GMNS path.

The framework adds first-class **network simplification** (dual carriageways, roundabouts, complex junctions, degree-2 nodes) via OSMnx by default and `neatnet` as an opt-in higher-quality alternative.

All new third-party packages live in a single optional extra **`aequilibrae[create]`** (with `[osm]` kept as an alias for one release). Base install footprint is unchanged.

### 1.1 Sources in scope this iteration

| Source | Backend | Notes |
|---|---|---|
| OSM Overpass (place name) | `osmnx.graph_from_place` | first-class |
| OSM Overpass (bbox / polygon) | `osmnx.graph_from_polygon` | first-class |
| OSM `.osm.pbf` (local file) | `pyrosm` | first-class |
| Overture `transportation` (cloud only) | `overturemaps` Python client | first-class, **only backend** |
| Arbitrary `(nodes_gdf, links_gdf)` | `GeoDataFrameSource` | first-class |
| GeoPackage / GeoJSON / Shapefile / FlatGeobuf | `FileSource` (geopandas) | first-class |
| GMNS CSV bundle | `GMNSSource` (wraps existing `GMNSBuilder`) | first-class |
| osm4gmns export | reuses GMNS path (osm4gmns produces GMNS) | first-class via GMNS |
| OSM `.osm` / `.osm.bz2` XML | — | **not implemented** (negligible use; users convert to `.pbf` via `osmium`) |
| OSRM / sumo / visum | — | documented extension points, not implemented |

### 1.2 Non-goals (deferred)

- GTFS / rail / true public transport extraction (today's "transit = bus-capable highway" parity is kept).
- Lane-level / movement-level modelling (osm4gmns territory; interoperable via the GMNS path).
- Auto-creation of centroids/zones from administrative polygons.
- Conflation of multiple sources in one import. Each import is single-source.
- Consumption of Overture turn-restriction / scoped speed-limit rules by AequilibraE's graph builder. They are preserved as JSON for downstream tools; consumption is a follow-up PR.
- **Progress bars / progress callbacks.** The codebase deliberately does not add UI progress affordances. Logging is the only user-facing progress signal.

### 1.3 Five explicit, deliberate design constraints (user-directed)

These are central to the design and not negotiable in the implementation:

1. **No `ALTER TABLE` is ever issued by the importer.** Not for source-specific fields, **not even for framework provenance fields.** The current `OSMBuilder` widens `links` with `ALTER TABLE links ADD COLUMN <field> <type>` for every entry under `parameters.yml::network.links.fields` — gone. The new committer reads the live `links` / `nodes` schema and writes each source attribute to a same-named existing column **if and only if** that column already exists. Everything else (raw source tags, provenance IDs, per-edge merge maps, …) is **JSON-encoded into the existing `other_attributes TEXT` column** which AequilibraE already documents for exactly this purpose.
2. **No link-type filtering at import time.** The current parameter file (`network.osm.modes.<mode>.link_types`) defines per-mode allow-lists of OSM `highway` values — removed. Every link the source surfaces is preserved; nothing is dropped because of its `highway`/`class`/`subclass` tag. Users who want a leaner network filter post-import with plain SQL (`DELETE FROM links WHERE link_type IN ('footway','service',…)`) or by passing a `custom_filter` string straight to OSMnx (Overpass QL) or a predicate to the Overture source.
3. **Mode filtering is preserved and is the only filtering the importer does.** The `modes=("car","transit",…)` argument still gates the import. Mode assignment is computed by a Python `MODE_RULES` engine that consults **only access semantics** (OSM `access`, `motor_vehicle`, `bicycle`, `foot`, `vehicle`, `oneway:<mode>`, `service`, `junction`; Overture `access_restrictions`, `subtype`, `subclass_rules`). A link with `highway=footway` and `modes=("car",)` is dropped because no car mode applies to it, not because `footway` is on a blocklist.
4. **No user choice of projected CRS, no `clean` / `commit` knobs.** Auto-UTM (centroid-based) is the only supported projected CRS for the internal projection step (used by the simplifier and Overture segment substring math). It is documented but not user-configurable. Likewise the importer always cleans (clips to the model area when one is supplied) and always commits — there is no `clean=False` / `commit=False` knob. The whole import is a single atomic operation against the project; users who want to inspect-then-edit-then-write should build a `GeoDataFrameSource` themselves.
5. **No progress bars / progress callbacks.** No `Progress` adapter, no `tqdm`, no `WorkerThread`/`SIGNAL` plumbing inside the new importer. The importer uses the existing `logging` infrastructure only (stage-boundary `logger.info` messages). QAequilibraE will need to adapt to this; flagged as an explicit downstream coordination item.
6. **Raw downloaded data is persisted to the project folder.** Every source that fetches data over the network (OSM Overpass, Overture cloud) writes the raw, untransformed payload to `<project_path>/downloaded_data/` before any parsing or transformation runs. This is mandatory, not configurable. It serves as a local cache, an audit trail, and the bridge from "live" to "reproducible/offline" re-imports. Sources reading from already-local files (`.osm.pbf`, GeoPackage, GeoJSON, GMNS CSVs, user GDFs) write **nothing** to this folder — the user's own files are already the on-disk record. (See §4.6 for the layout.)
7. **No user choice of Overture backend, no `keep_rule_arrays` knob.** Overture imports always use the official `overturemaps` cloud client (`backend="cloud"` is hardcoded) and always preserve the rule arrays (`access_restrictions`, `prohibited_transitions`, `subclass_rules`, `speed_limits`) verbatim in `other_attributes`. The DuckDB and pyarrow-parquet Overture backends are dropped from this iteration; users wanting offline Overture re-imports rely on the raw download saved under `<project>/downloaded_data/` (§4.6).
8. **No OSM XML source.** The `.osm` / `.osm.bz2` path is removed. Users with XML dumps convert them to `.pbf` with `osmium cat in.osm -o out.osm.pbf` (one-line tool; standard in the OSM ecosystem) and import via `pbf_path=`.

---

## 2. Availability of Overture Maps Python tooling (research summary)

Confirmed at planning time:

| Package | What it provides | Verdict |
|---|---|---|
| **`overturemaps`** (official, MIT, v1.0.0 May 2026, PyPI + conda-forge) | CLI + `record_batch_reader(theme, bbox)` returning a streaming `pyarrow.RecordBatchReader`; `geodataframe(theme, bbox)` returning a `geopandas.GeoDataFrame`. Reads directly from Overture's S3 `us-west-2` GeoParquet using bbox row-group statistics, so only bbox-relevant rows are transferred. Resolves the latest release via Overture's STAC catalog. | **The only Overture backend we use** (§1.3 rule 7). Pure-Python, light dep tail. |
| `duckdb` + Overture S3 paths | SQL over `read_parquet('s3://overturemaps-us-west-2/...')`. | Evaluated but **rejected** for this iteration to keep the dep set lean and the API single-pathway. Power users wanting SQL access can construct a `GeoDataFrameSource` from the cached payload. |
| `pyarrow.dataset` on the same paths | Same data, no DuckDB. | Evaluated but **rejected** for the same reason. |
| **`overture2gmns`** (community, beta) | Converts Overture transportation → GMNS CSVs. | Documented as an external workflow; not taken as a dependency. |

Overture's transportation schema (`segment` + `connector`) maps cleanly onto AeQ's `links` + `nodes`:

- `connector.id` (GERS UUID) → goes into `nodes.other_attributes` JSON.
- `segment.connectors[]` (sorted by `at ∈ [0,1]`) → links: split segment at every intermediate connector via `shapely.ops.substring(line, at_i, at_{i+1}, normalized=True)`. First and last connectors become `a_node` / `b_node`.
- `segment.class` ∈ {motorway, trunk, primary, …} → `link_type` (clean closed enumeration, no 51-cap problem).
- `subtype` ∈ {road, rail, water, …} drives mode rules.
- `speed_limits[]`, `access_restrictions[]`, `prohibited_transitions[]`, `subclass_rules[]`: the global entry (no `between`/`when` scoping) drives `direction` and `speed_ab`/`speed_ba`; the full arrays are preserved verbatim in `links.other_attributes` JSON.
- Overture lanes are not in the current public schema → `lanes_ab`/`lanes_ba` left NULL on Overture imports.

---

## 3. Current implementation summary (what we're replacing)

Files under `src/aequilibrae/project/network/`:

- `osm/osm_downloader.py` — `OSMDownloader(WorkerThread)`. Overpass QL `way["highway"]` per sub-polygon, gridded under `max_query_area_size ≈ 2.5 Gm²`, retries 429/504, dedup, returns pandas DFs.
- `osm/osm_builder.py` — `OSMBuilder(WorkerThread)`. Splits ways at shared (intersection) nodes by counting node references, builds `LINESTRING` WKT manually, normalises `highway` → `link_type` (cap of 51 then `aggregate_link_type`), `json_normalize`s tag dicts, maps the mode/link-type matrix from `parameters.yml`, derives `direction` from `oneway`, halves `lanes`/`speed` for bidirectional ways, dumps `links.parquet`/`nodes.parquet` to `<project>/osm_data/`, drops triggers, bulk-inserts to spatialite, re-adds triggers, optionally clips to `model_area`. **Issues `ALTER TABLE links ADD COLUMN <field> <type>`** for every YAML-listed field — the behaviour we are explicitly removing per §1.3 (1).
- `osm/place_getter.py` — Nominatim `/search` wrapper.
- `osm/osm_params.py` — HTTP headers, Overpass `maxsize`.
- `osm/model_area_gridding.py` — polygon subdivision for the clip step.
- `network.py::Network.create_from_osm(model_area, place_name, modes, clean)` — orchestrator.
- `parameters.yml::network.osm.{all_link_types, modes.{car,transit,bicycle,walk}.{link_types, mode_filter, unknown_tags}}` and top-level `osm:` block.

Schema, already in `database_specification/network/tables/`:

- `links.sql` columns: `ogc_fid, link_id, a_node, b_node, direction, distance, modes, link_type, name, fixed_cost_ab, fixed_cost_ba, speed_ab, speed_ba, travel_time_ab, travel_time_ba, capacity_ab, capacity_ba, other_attributes (TEXT)`. `other_attributes` already documented as "Other attributes of the link. Preferably in json format" — the new committer simply starts using it.
- `nodes.sql` columns: `ogc_fid, node_id, is_centroid, modes, link_types, other_attributes (TEXT)`. Same story.
- `about.sql` — key/value (`infoname`, `infovalue`) metadata table with `About.add_info_field()` API for adding new keys. **This is where source-of-import provenance lands** (see §10).

Callers, tests, and docs to migrate:

- `tests/aequilibrae/project/test_network.py::test_create_from_osm`
- `tests/aequilibrae/project/test_osm_downloader.py`
- `tests/aequilibrae/project/test_place_getter.py`
- `docs/source/examples/network_manipulation/plot_create_from_osm.py`
- `src/aequilibrae/project/network/osm/__init__.py::placegetter` re-export.

---

## 4. Architecture

### 4.1 Layered pipeline

```
                       ┌──────────────────────────┐
   user picks one  ──▶ │  Source (acquisition)    │
                       │  - OSMOverpassSource     │  ── saves raw to
                       │  - OSMPbfSource          │     'downloaded_data/'
                       │  - OvertureCloudSource   │  ── saves raw to
                       │  - GeoDataFrameSource    │     'downloaded_data/'
                       │  - FileSource            │
                       │  - GMNSSource            │
                       └────────────┬─────────────┘
                                    │ returns
                                    ▼
                       ┌──────────────────────────┐
                       │  RoutableNetwork (IR)    │
                       │  - nodes_gdf (EPSG:4326) │
                       │  - links_gdf (EPSG:4326) │
                       │  - source_meta           │
                       └────────────┬─────────────┘
                                    │ (always, unless simplify=False)
                                    ▼
                       ┌──────────────────────────┐
                       │  Simplifier              │
                       │  - OSMnxSimplifier (def) │
                       │  - NeatnetSimplifier     │
                       └────────────┬─────────────┘
                                    │
                                    ▼
                       ┌──────────────────────────┐
                       │  SpatialiteWriter        │
                       │  (schema-aware routing,  │
                       │   never ALTERs anything) │
                       └──────────────────────────┘
                                    │
                                    ▼
                       ┌──────────────────────────┐
                       │  about-table provenance  │
                       │  (source, source URL,    │
                       │   release, fetched_at)   │
                       └──────────────────────────┘
```

### 4.2 Module layout

```text
src/aequilibrae/project/network/
  importer/
    __init__.py
    importer.py             # NetworkImporter orchestrator
    ir.py                   # RoutableNetwork dataclass + invariants
    exceptions.py
    db_writer.py            # SpatialiteWriter (schema-aware routing, §4.4)
    about_writer.py         # writes source provenance into the about table (§10)
    schema/
      __init__.py
      modes.py              # MODE_RULES engine (access-based, no link-type allow-lists)
      link_types.py         # deterministic, uncapped link-type allocator
      attributes.py         # _split_attributes / JSON merge for other_attributes
    download_cache.py       # raw-payload writer for OSM Overpass + Overture cloud (§4.6)
    sources/
      __init__.py
      base.py               # Source Protocol + registry
      osm/
        __init__.py
        overpass.py         # OSMOverpassSource (uses osmnx; saves raw JSON to downloaded_data/)
        pbf.py              # OSMPbfSource (uses pyrosm)
        tags_to_ir.py       # raw OSM tags -> IR columns
      overture/
        __init__.py
        cloud.py            # OvertureCloudSource (uses overturemaps; saves raw parquet to downloaded_data/)
        schema_to_ir.py     # segment+connector -> IR
      generic/
        __init__.py
        geodataframe.py     # GeoDataFrameSource
        file.py             # FileSource (geopandas.read_file)
        gmns.py             # GMNSSource (wraps existing GMNSBuilder)
    simplifiers/
      __init__.py
      base.py               # Simplifier Protocol + registry
      osmnx_simplifier.py
      neatnet_simplifier.py
```

No `progress.py`, no `result.py` (no preview/commit handle), no `report.py` — all dropped per §1.3 (4) and §1.3 (5).

### 4.3 `RoutableNetwork` intermediate representation (IR)

```python
@dataclass
class RoutableNetwork:
    nodes: gpd.GeoDataFrame          # required: node_id (int), geometry (Point, 4326), modes (str)
                                     # PLUS arbitrary free-form columns (raw source attrs).
    links: gpd.GeoDataFrame          # required: link_id, a_node, b_node, direction, modes,
                                     # link_type, distance (m), geometry (LineString, 4326)
                                     # PLUS arbitrary free-form columns. May also carry a
                                     # `_source_id` column (string) used internally by the
                                     # simplifier to build per-merged-edge attribute dicts;
                                     # this column is folded into other_attributes by the
                                     # committer and never reaches the DB as a real column.
    crs_geo: str = "EPSG:4326"
    source_meta: dict = field(default_factory=dict)
    # source_meta drives BOTH the per-row other_attributes provenance entries AND the
    # about-table provenance writes; see §10. Keys include: 'source' (e.g. 'osm-pbf'),
    # 'backend' ('pyrosm'), 'source_url' (file path / overpass URL / overture release URI),
    # 'release', 'fetched_at' (ISO timestamp).

    def to_multidigraph(self) -> "networkx.MultiDiGraph": ...
    @classmethod
    def from_multidigraph(cls, G, source_meta=None) -> "RoutableNetwork": ...
    def validate(self) -> None: ...
```

**Schema invariants** (enforced by `validate()`, unit-tested):

- `nodes.node_id` unique, integer, ≥ 10000 (matches today's `node_start`).
- `links.a_node` / `links.b_node` ∈ `nodes.node_id`.
- `links.distance` in metres, `> 0`.
- `links.direction ∈ {-1, 0, 1}`.
- `links.modes` non-empty, built from one-char mode codes (`"cwbt"` etc.).
- Both GDFs in `EPSG:4326` at IR boundaries.

**Lossless free-form columns.** Sources are encouraged to emit every attribute they can extract — OSM tags, Overture properties, user GDF columns — as free-form columns. The IR does not enforce a known column list; the **committer** does the schema-aware routing.

### 4.4 Schema-aware committer (`SpatialiteWriter`) — column routing & `other_attributes`

The committer is **strictly non-schema-modifying**. It issues **no `ALTER TABLE` statements at all** (§1.3 rule 1). It reads the live `links` / `nodes` schema from the spatialite project and routes IR columns based on what already exists.

```python
class SpatialiteWriter:
    PROTECTED = {"ogc_fid", "geometry"}
    JSON_COL  = "other_attributes"

    def __init__(self, conn):
        self.link_cols = _list_columns(conn, "links")
        self.node_cols = _list_columns(conn, "nodes")
        if self.JSON_COL not in self.link_cols or self.JSON_COL not in self.node_cols:
            raise ImporterError(
                "This project's links/nodes tables do not include the 'other_attributes' "
                "column. Recreate the project with the current AequilibraE version, or add "
                "the column manually: ALTER TABLE links ADD COLUMN other_attributes TEXT;"
            )

    def _split_attributes(self, gdf, table_cols):
        known   = [c for c in gdf.columns
                   if c in table_cols and c not in self.PROTECTED and c != self.JSON_COL]
        extras  = [c for c in gdf.columns
                   if c not in table_cols and c not in self.PROTECTED and not c.startswith("_")]
        direct  = gdf[known + ["geometry"]]
        extra_json = gdf[extras].apply(_row_to_json_dropping_nans, axis=1)
        if self.JSON_COL in gdf.columns:
            extra_json = _merge_json(gdf[self.JSON_COL], extra_json)
        return direct, extra_json
```

Routing rules implemented by `_split_attributes` and unit-tested:

1. **Column exists on the target table** → write to that column. Type coercion (text/int/float) is attempted; on failure a single warning is logged and the offending value falls back into the row's JSON (so nothing is silently lost).
2. **Column does not exist** → JSON-encode into `other_attributes`. `None` / `NaN` values are omitted from the JSON object.
3. **`ogc_fid` and `geometry`** are protected (spatialite-managed / written separately).
4. **Columns whose name starts with `_`** are treated as internal IR scratch (e.g. `_source_id` used during simplification to drive attribute reconciliation). They are not written to the DB and not added to `other_attributes` — they vanish at the committer boundary.
5. **The committer never issues `ALTER TABLE`** — neither for source-specific tags nor for framework provenance. If `other_attributes` is missing from the schema (very old project), the committer raises with a clear actionable message rather than silently dropping data.

#### 4.4.1 Provenance in `other_attributes`

Per-row provenance lives in JSON alongside the source tags:

```json
{
  "source_id": "w53435546",
  "source_id_list": {
    "w53435546": {"highway": "residential", "name": "Meteorological Lane", "surface": "gravel"},
    "w53435547": {"highway": "residential", "name": "Meteorological Lane"}
  },
  "highway": "residential",
  "surface": "gravel",
  "bridge": "yes",
  "ref": "A3"
}
```

- For a non-simplified link, only the source's own attribute keys are present, plus `source_id` (string).
- For a **simplified / merged** link, `source_id_list` is a **dict-of-dicts** keyed by the per-source-edge `source_id`, with each inner dict containing the original attribute set for that source edge. This is the structure the user requested: it preserves the full pre-simplification information without flattening or losing the edge-of-origin for any tag. The "primary" `source_id` is set to the first (lowest-`source_id`) merged edge for determinism, and that edge's tag set is also lifted to the top level for convenient SQL access via `json_extract(other_attributes, '$.surface')`.
- Direct simplification (no merge happened — e.g. degree-2 collapse of a single edge chain) produces `source_id_list` with one entry; no special case.

#### 4.4.2 User pathway to typed columns

A user who wants a particular attribute as a typed column simply runs `ALTER TABLE links ADD COLUMN surface TEXT` (manually, **outside** the importer) before the import. The next import then writes `surface` directly to that column. Querying JSON-stored attributes uses spatialite's `json_extract`:

```sql
SELECT link_id, json_extract(other_attributes, '$.surface') AS surface FROM links;
SELECT link_id, json_extract(other_attributes, '$.source_id_list."w53435546".highway')
  FROM links;
```

A future PR can add a `Network.promote_attribute(table, key, sql_type)` helper that lifts a JSON key into a real column and back-fills existing rows — explicitly out of scope here.

### 4.5 `Source` and `Simplifier` Protocols

```python
class Source(Protocol):
    name: ClassVar[str]
    required_extras: ClassVar[tuple[str, ...]]
    def acquire(self, *, modes: tuple[str, ...], download_cache: "DownloadCache") -> RoutableNetwork: ...

class Simplifier(Protocol):
    name: ClassVar[str]
    required_extras: ClassVar[tuple[str, ...]]
    def simplify(self, net: RoutableNetwork, **kwargs) -> RoutableNetwork: ...
```

Two small registries `SOURCES` / `SIMPLIFIERS` keyed by `name`. No `Progress` parameter anywhere. The `download_cache` is the per-import handle described in §4.6; local-file sources receive it but are not required to write anything.

### 4.6 Raw-download cache (`<project>/downloaded_data/`)

Every source that retrieves data over the network writes the raw, untransformed payload to a project-local folder **before any parsing/transformation runs**. Local-file sources do not write anything (the user's file is already the on-disk record).

Layout:

```text
<project_path>/
  downloaded_data/
    osm-overpass/
      2026-06-17T01-12-34Z__nauru/
        query.overpassql         # the exact Overpass QL string sent
        response.json            # raw Overpass JSON response (gz-compressed if > 10 MB)
        manifest.json            # {source, backend, source_url, modes, custom_filter,
                                 #  fetched_at, bbox, place_name, response_bytes, sha256}
    overture-cloud/
      2026-06-17T01-15-02Z__bbox_xmin_ymin_xmax_ymax/
        segments.parquet         # raw RecordBatchReader output for type=segment
        connectors.parquet       # raw RecordBatchReader output for type=connector
        manifest.json            # {source, source_url, release, modes, bbox,
                                 #  fetched_at, segments_rows, connectors_rows, sha256}
```

Rules:

- The subfolder name is `<ISO timestamp>__<short tag>` where the short tag is the place name (slugified) for OSM Overpass place queries, the bbox tuple for OSM Overpass bbox queries, or the bbox tuple for Overture.
- Files are written in the same format the source naturally produces — no re-encoding: Overpass JSON stays JSON; Overture RecordBatchReader output is written as `.parquet` via `pyarrow.parquet.write_table` directly from the in-memory `pyarrow.Table` (no geopandas round-trip). This guarantees a byte-faithful record of what the remote returned.
- `manifest.json` records the request parameters, the response size in bytes, and a SHA-256 of the payload(s). It is human-readable and forms the canonical reproducibility record.
- The folder is created if it does not exist; it is **never auto-cleaned**. Disk-space management is the user's responsibility, documented in the migration guide.
- If the write fails (disk full, permission denied), the import fails fast with a clear error before any parsing runs. There is no fallback to "skip the cache and continue", because the cache is part of the documented contract per §1.3 rule 6.

`DownloadCache` is the small helper class that sources use:

```python
class DownloadCache:
    def __init__(self, project_path: Path, source_name: str, tag: str): ...
    @property
    def folder(self) -> Path: ...
    def write_bytes(self, name: str, payload: bytes) -> Path: ...
    def write_table(self, name: str, table: "pyarrow.Table") -> Path: ...
    def write_manifest(self, manifest: dict) -> Path: ...
```

The `NetworkImporter` instantiates one `DownloadCache` per import and threads it into `Source.acquire(...)`. Local-file sources (`OSMPbfSource`, `FileSource`, `GMNSSource`, `GeoDataFrameSource`) accept the parameter but ignore it.

The folder path is also written to the `about` table on every import (see §10: `network_source_download_cache = "<relative path under project>"`) so downstream tooling can locate the raw payload.

---

## 5. Public API

(Hard break: `create_from_osm` is removed. No deprecation shim by default — see §15 open question 7.)

### 5.1 Generic orchestrator

```python
class Network:
    def import_network(
        self,
        source: Source | str,
        *,
        modes: Sequence[str] = ("car", "transit", "bicycle", "walk"),   # the only filter
        simplify: Simplifier | str | bool = "osmnx",
        consolidate_tolerance: float | None = 10.0,                     # metres in auto-UTM
        **source_kwargs,
    ) -> None:
        """
        Acquire from `source`, optionally simplify, clip to the project's model area
        (always), and commit (always) to the project's links/nodes tables.

        `source` is a Source instance or one of the registered names:
        'osm-overpass', 'osm-pbf', 'overture-cloud',
        'geodataframe', 'file', 'gmns'.

        Returns None: import is atomic, all-or-nothing. To inspect/edit data before
        commit, construct a GeoDataFrameSource yourself from your own pipeline.
        """
```

There is intentionally **no**:

- `link_types` / `allowed_highways` / `denylist` argument (§1.3 rule 2).
- `clean` argument (§1.3 rule 4): the model-area clip is always applied when a model area is supplied through the source kwargs (`model_area=` for OSM/Overture).
- `commit` argument (§1.3 rule 4): the importer always writes to the project.
- `projected_crs` argument (§1.3 rule 4): auto-UTM (by centroid) is the only choice; documented.
- `progress` argument (§1.3 rule 5).

### 5.2 Typed convenience wrappers

```python
    def import_from_osm(
        self, *,
        model_area: Polygon | None = None,
        place_name: str | None = None,
        pbf_path: str | Path | None = None,
        modes: Sequence[str] = ("car", "transit", "bicycle", "walk"),
        custom_filter: str | None = None,           # raw OSMnx/Overpass filter passthrough
        simplify: str | bool = "osmnx",
        consolidate_tolerance: float | None = 10.0,
    ) -> None:
        """Chooses the right OSM source based on which of the three 'where' kwargs is set."""

    def import_from_overture(
        self, *,
        model_area: Polygon,                        # required: Overture is bbox-driven
        release: str | None = None,                 # default: latest via STAC
        modes: Sequence[str] = ("car", "transit", "bicycle", "walk"),
        simplify: str | bool = "osmnx",
        consolidate_tolerance: float | None = 10.0,
    ) -> None:
        """Always uses the official `overturemaps` cloud client; rule arrays always preserved."""

    def import_from_geodataframes(
        self, *,
        nodes: gpd.GeoDataFrame,
        links: gpd.GeoDataFrame,
        crs: str | int | None = None,
        column_mapping: dict[str, str] | None = None,
        simplify: str | bool = False,
    ) -> None: ...

    def import_from_file(
        self, *,
        links_path: str | Path,
        nodes_path: str | Path | None = None,
        layer_links: str | None = None,
        layer_nodes: str | None = None,
        column_mapping: dict[str, str] | None = None,
        simplify: str | bool = False,
    ) -> None: ...
```

All return `None`. All write the project unconditionally. All use auto-UTM internally.

### 5.3 Removed / kept

- **Removed**: `Network.create_from_osm` (raises `AttributeError` with migration message). `aequilibrae.project.network.osm.placegetter`. `parameters.yml::network.osm.*`. `parameters.yml::network.links.fields.*.osm_source` / `osm_behaviour`.
- **Trimmed**: top-level `parameters.yml::osm:` → only endpoint + timeout knobs, forwarded to `osmnx.settings`.
- **Kept**: `Network.create_from_gmns` as a thin alias for `import_network(GMNSSource(...), simplify=False)`. All existing GMNS users keep working unchanged.

---

## 6. Source-specific mapping notes

### 6.1 OSM sources (`osm-overpass`, `osm-pbf`)

The OSM source acquires every way the backend returns and interprets only the routing-relevant tags. **No link-type filtering** (§1.3 rule 2). The `modes` argument is the only filter.

Routing-relevant tag → IR column mapping (typed fields):

| OSM tag(s) | IR column | Notes |
|---|---|---|
| `highway` | `link_type` | Uncapped, deterministic char allocator. Raw value also passed through as the `highway` free-form column. |
| `oneway`, `junction=roundabout` | `direction` | `-1/0/1` |
| `lanes`, `lanes:forward`, `lanes:backward` | `lanes_ab`, `lanes_ba` | Proper forward/backward split (replaces today's divide-by-2 bug) |
| `maxspeed`, `maxspeed:forward`, `maxspeed:backward` | `speed_ab`, `speed_ba` | mph/km/h parsed |
| `name` | `name` | direct |
| computed | `distance` | metres, from auto-UTM projected geometry |
| computed | `modes` | from `MODE_RULES` over access tags only |

All **other** tags (`bridge`, `tunnel`, `surface`, `ref`, `cycleway*`, `busway*`, `sidewalk`, `smoothness`, `hgv`, `maxweight`, …) are passed through to the IR as free-form columns with their **raw OSM key names** (colons `:` replaced with underscores `_` to be SQL-friendly). The committer (§4.4) decides per column: known → its column; unknown → `other_attributes` JSON.

Provenance:

- `_source_id` (IR scratch) ← string-cast OSM way id (first id post-simplification merge).
- After commit: `source_id` and `source_id_list` (the dict-of-dicts described in §4.4.1) appear in `other_attributes`.
- About-table writes: `source = "osm"`, `source_backend = "osmnx" | "pyrosm"`, `source_url = "<overpass URL>" | "<pbf path>" | "<place name>"`, `source_fetched_at = "<ISO ts>"` (see §10).

Raw-download cache (§4.6):

- `OSMOverpassSource` writes the exact Overpass QL `query.overpassql` and the raw `response.json` (gz-compressed if > 10 MB) plus a `manifest.json` to `<project>/downloaded_data/osm-overpass/<ts>__<tag>/`. This happens **before** any parsing — if the parse subsequently fails the user still has the raw payload for inspection / re-import.
- `OSMPbfSource` writes nothing (the user's `.pbf` file is already the on-disk record). The about-table `network_source_download_cache` key is set to `null` for this source.

### 6.2 Overture source (`overture-cloud`)

The Overture importer always uses the official `overturemaps` cloud client (the DuckDB and pyarrow-parquet backends were considered and dropped per §1.3 rule 7). There is no `backend` argument and no `keep_rule_arrays` argument — rule arrays are always preserved.

Pipeline:

1. Fetch `theme=transportation, type=connector` and `type=segment` within `model_area.bounds` using `overturemaps.record_batch_reader(...)` once per type. The two resulting `pyarrow.Table`s are written immediately to `<project>/downloaded_data/overture-cloud/<ts>__<bbox>/` as `connectors.parquet` and `segments.parquet`, with a `manifest.json` capturing release id, bbox, modes, fetched_at, row counts, and SHA-256 of each parquet (§4.6).
2. Build nodes from `connector` (Point geometry, AeQ `node_id` allocated from 10000, `_source_id = connector.id`).
3. Build links from `segment`:
   - Sort `connectors[]` by `at`.
   - For each `(connector_i, connector_{i+1})`, take `shapely.ops.substring(segment.geometry, at_i, at_{i+1}, normalized=True)`. Sub-link `a_node` / `b_node` map to the connectors' AeQ node ids.
   - `link_type` ← `class`. Every value preserved (§1.3 rule 2).
   - `modes` derived from `subtype` + `subclass_rules` + global entries in `access_restrictions[]` via `MODE_RULES`. Only mode-based filtering applies.
   - `direction` derived from global `access_restrictions[]` entries with `access_type="denied"` and a directional `heading`.
   - `speed_ab`/`speed_ba` from `speed_limits[]` global entry (null `between`, null/car mode filter); units parsed.
   - `lanes_ab`/`lanes_ba` left NULL.
   - All other Overture properties (`names.primary`, `road_surface`, `road_flags`, `subclass`, `level_rules`, `routes`, `destinations`, `width_rules`, …) emitted as free-form IR columns. The full `access_restrictions`, `prohibited_transitions`, `subclass_rules`, `speed_limits` arrays are **always** also emitted (JSON-string-encoded values), landing in `other_attributes` after committer routing.

Provenance:

- `_source_id` (IR scratch) ← Overture GERS id (string).
- After commit: same `source_id` / `source_id_list` pattern in `other_attributes`.
- About-table writes: `source = "overture"`, `source_backend = "cloud"`, `source_url = "<release URI>"`, `source_release = "<release tag from STAC or user-pinned>"`, `source_fetched_at = "<ISO ts>"`, `source_download_cache = "downloaded_data/overture-cloud/<ts>__<bbox>"`.

### 6.3 Generic sources (`geodataframe`, `file`)

Pure adapters. Validate user-supplied GDFs against IR invariants (§4.3), apply `column_mapping` to rename columns, reproject to 4326 if needed. **All** user columns (known or unknown) emitted to the IR — the committer routes via §4.4. Simplification defaults to `False`.

About-table writes: `source = "geodataframe"` / `"file"`, `source_url = "<file path>"` (file only), `source_fetched_at = "<ISO ts>"`.

### 6.4 GMNS source

Wraps existing `GMNSBuilder`. `Network.create_from_gmns` preserved as backwards-compatible alias.

About-table writes: `source = "gmns"`, `source_url = "<gmns bundle dir>"`, `source_fetched_at = "<ISO ts>"`.

---

## 7. Simplification layer

Same code path for every source (operates on `RoutableNetwork`, not source-specific structures). OSMnx by default, `neatnet` opt-in. **Auto-UTM only** — no user CRS choice.

```python
def simplify(net, *, mode="osmnx", consolidate_tolerance=10.0):
    if mode is False: return net
    G  = net.to_multidigraph()
    Gp = ox.projection.project_graph(G)                           # auto-UTM by centroid
    if mode in (True, "osmnx"):
        Gs = ox.simplification.simplify_graph(Gp, edge_attrs_differ=("highway", "name"))
        if consolidate_tolerance:
            Gs = ox.simplification.consolidate_intersections(
                Gs, tolerance=consolidate_tolerance, rebuild_graph=True, dead_ends=True,
            )
        Gout = ox.projection.project_graph(Gs, to_crs="EPSG:4326")
        return RoutableNetwork.from_multidigraph(Gout, source_meta=net.source_meta)
    if mode == "neatnet":
        try:
            import neatnet
        except ImportError as e:
            raise OptionalDependencyError(
                "neatnet is required for simplify='neatnet'. "
                "Install via `pip install aequilibrae[create]`."
            ) from e
        edges = ox.convert.graph_to_gdfs(Gp, nodes=False)
        simplified = neatnet.simplify_network(edges)
        return _gdf_to_routable(simplified, crs_back="EPSG:4326", source_meta=net.source_meta)
    raise ValueError(f"Unknown simplify mode: {mode!r}")
```

**Attribute reconciliation across simplification:** the simplifier uses `_source_id` to build the per-merged-edge dict-of-dicts described in §4.4.1. For each merged edge:

- `_source_id` ← lexicographically smallest source id among merged edges (deterministic primary).
- `_source_id_list` ← `{src_id: {tag_key: tag_value, ...}, ...}` covering every merged source edge with its full original attribute set.
- The "primary" edge's tags are also lifted to top-level free-form columns so SQL queries like `WHERE link_type='motorway'` work without `json_extract`.

The committer (§4.4) then routes these into `other_attributes` as the dict-of-dicts.

---

## 8. Dependency packaging (`pyproject.toml`)

```toml
[project.optional-dependencies]
create = [
  "osmnx>=2.0,<3",
  "networkx>=3.0",
  "scikit-learn",        # required by osmnx.consolidate_intersections
  "pyrosm>=0.6.2",       # .osm.pbf reader
  "neatnet>=0.2",        # opt-in simplifier
  "overturemaps>=1.0",   # Overture cloud client (the only Overture backend)
]
# Back-compat alias for one release; deprecated in the next minor.
osm = ["aequilibrae[create]"]
```

Base install footprint is unchanged. `pyarrow`, `geopandas`, `requests`, `shapely`, `pyproj` are already hard deps and they alone are sufficient to run `GeoDataFrameSource`, `FileSource`, and `GMNSSource`. OSM and Overture-cloud paths require the extra.

`duckdb` is **not** added — the DuckDB Overture backend was dropped per §1.3 rule 7.

Lazy imports + `OptionalDependencyError` raised at the source/simplifier `__init__` (or first method call). Error message names the missing package and includes `pip install aequilibrae[create]`.

---

## 9. Parameters file changes (`src/aequilibrae/parameters.yml`)

- **Remove** `network.osm.{all_link_types, modes.*}` (the per-mode link-type allow-lists and tag-filter blocks — §1.3 rule 2).
- **Remove** `network.links.fields.*.osm_source` / `osm_behaviour` keys (the column-widening matrix — §1.3 rule 1). The field definitions themselves are kept; they still describe the DB schema for non-OSM imports.
- **Trim** top-level `osm:` to endpoint and timeout knobs:
  ```yaml
  osm:
    overpass_endpoint: "https://overpass-api.de/api"
    nominatim_endpoint: "https://nominatim.openstreetmap.org/"
    accept_language: "en"
    timeout: 540
  ```
  Forwarded to `osmnx.settings` on first call. `max_query_area_size`, `max_attempts`, `sleeptime` dropped (OSMnx handles tiling/retries internally).
- **Add** top-level `overture:` block:
  ```yaml
  overture:
    release: null                  # null → STAC "latest"
    s3_region: "us-west-2"
    data_root: "s3://overturemaps-us-west-2/release"
    connect_timeout: 10
    request_timeout: 60
  ```
  Forwarded to `overturemaps` on first call.

Audit: in-repo, the removed `network.osm.*` keys are only consumed by `osm_downloader.get_osm_filter` and `osm_builder.__establish_modes_for_all_links` (both deleted). No QAequilibraE references found.

---

## 10. Database side-effects: zero `ALTER TABLE`s, provenance in `about`

**The importer issues no `ALTER TABLE` statements.** No `source`, no `source_id`, no `source_id_list` columns are added to `links` / `nodes`. All such information lives in `other_attributes` JSON (per-row source provenance, §4.4.1) or in the `about` table (whole-import provenance, this section).

The `about` table is the project's existing key/value metadata store. The committer calls `About.add_info_field()` (idempotent: only adds if the key isn't already there) for the following keys on every successful import, then writes the values:

| `about` infoname | Example value |
|---|---|
| `network_source` | `"osm"` / `"overture"` / `"geodataframe"` / `"file"` / `"gmns"` |
| `network_source_backend` | `"osmnx"` / `"pyrosm"` / `"cloud"` |
| `network_source_url` | `"https://overpass-api.de/api"` / `"<pbf path>"` / `"Nauru"` / `"s3://overturemaps-us-west-2/release/2026-05-20.0"` |
| `network_source_release` | Overture release id, else null |
| `network_source_fetched_at` | ISO 8601 timestamp |
| `network_source_modes` | `"car,transit,bicycle,walk"` |
| `network_source_simplify` | `"osmnx"` / `"neatnet"` / `"false"` |
| `network_source_consolidate_tolerance` | `"10.0"` (metres in auto-UTM) |
| `network_source_download_cache` | `"downloaded_data/osm-overpass/2026-06-17T01-12-34Z__nauru"` / `null` for local-file sources |
| `network_source_aequilibrae_version` | current AeQ version string |

These overwrite on each subsequent re-import (the `about` table is a `(UNIQUE infoname)` store; re-imports update in place). They are the canonical record of "where did this network come from". The two existing related fields (`projection`, `model_name`, etc.) are untouched.

**Schema precondition:** the committer raises if `links.other_attributes` or `nodes.other_attributes` doesn't exist — these are required for the new importer and are already part of the `database_specification` schema (so any project created with a current AeQ version has them). For very old projects, the migration guide instructs the user to add the columns manually.

---

## 11. Tests

### 11.1 Delete

- `tests/aequilibrae/project/test_osm_downloader.py`
- `tests/aequilibrae/project/test_place_getter.py`

### 11.2 Modify

- `tests/aequilibrae/project/test_network.py::test_create_from_osm` → renamed `test_import_from_osm`, switched to the new API.

### 11.3 Add — framework

`tests/aequilibrae/project/network_importer/`:

- `test_ir.py` — `RoutableNetwork.validate()` invariants; multidigraph round-trip preserves topology, modes, and free-form columns.
- `test_registry.py` — string-name source/simplifier resolution; unknown name raises with a helpful list.
- `test_committer_routing.py` — **central test for §4.4**:
  - synthetic IR with mixed known + unknown columns,
  - assert known columns land in real columns (`name`, `speed_ab`, …),
  - assert unknown columns land in `other_attributes` JSON, NaNs omitted,
  - **assert no `ALTER TABLE` is issued** (snoop on `sqlite_master.sql` before and after import — must be byte-identical for `links` and `nodes`),
  - assert pre-existing `other_attributes` content from the IR is merged not overwritten,
  - assert columns starting with `_` are stripped (not in DB, not in JSON),
  - assert a manually-added user column (e.g. `surface`) is honoured: `ALTER TABLE links ADD COLUMN surface TEXT` → import → assert `surface` populated, **not** in `other_attributes`.
- `test_committer_no_alter_table.py` — strict test: instrument the spatialite connection to count any `ALTER` statement executed during a full import; expected count is **zero**.
- `test_committer_missing_other_attributes.py` — synthetic project missing the `other_attributes` column → importer raises with the documented actionable message.
- `test_about_provenance.py` — after a successful import, all `network_source_*` keys in `about` table have the expected values, and re-import updates them in place.
- `test_other_attributes_dict_of_dicts.py` — after simplification, `json_extract(other_attributes, '$.source_id_list')` returns the dict-of-dicts described in §4.4.1; top-level keys are lifted from the primary source edge; sub-source attributes are recoverable via `json_extract(..., '$.source_id_list."<id>"."<key>"')`.

### 11.4 Add — mode filtering (no link-type filtering)

- `test_mode_filter_only.py` — load a fixture containing `highway=motorway` + `highway=footway` + `highway=cycleway`.
  - `modes=("car",)` → keeps motorway, drops footway and cycleway (mode rule, not link-type rule).
  - `modes=("walk",)` → keeps footway and motorway-with-foot-yes (rare), drops car-only roads.
  - assert that the `link_type` column on the result still contains every value present in the input (no link-type filtering happened upstream).
  - assert no kwarg named `link_types` / `allowed_highways` / `denylist` / `link_type_filter` is accepted on any `import_*` method (introspect signatures).
- `test_no_progress_or_crs_or_clean_kwargs.py` — introspect signatures of all `import_*` methods to assert the **absence** of `progress`, `projected_crs`, `clean`, `commit` kwargs.

### 11.5 Add — per source

- `test_osm_pbf_source.py` — bundled `tests/data/osm/grand_canyon_tiny.osm.pbf` (~50 KB). Subtests: simplify off / osmnx / neatnet; mode filtering; optional-dep error; assert `downloaded_data/` is **not** written for this local-file source.
- `test_osm_overpass_source.py` — gated by random + `GITHUB_WORKFLOW=Code coverage`, hits `place_name="Vatican City"`. Asserts the raw Overpass JSON + query + manifest land under `<project>/downloaded_data/osm-overpass/<ts>__vatican-city/` and the `about` table's `network_source_download_cache` matches.
- `test_overture_cloud_source.py` — gated like Overpass; tiny bbox. Asserts raw `segments.parquet` + `connectors.parquet` + manifest land under `<project>/downloaded_data/overture-cloud/<ts>__<bbox>/`; asserts no `backend` / `parquet_path` / `keep_rule_arrays` kwargs are accepted by `import_from_overture` (signature introspection); asserts the rule arrays are present in committed `other_attributes` JSON unconditionally.
- `test_overture_cloud_source_unit.py` — same coverage as the live test, but driven from a pre-canned `pyarrow.Table` fixture in `tests/data/overture/transportation_tiny.parquet` so the unit test is deterministic and offline; monkeypatches `overturemaps.record_batch_reader` to return the fixture.
- `test_geodataframe_source.py` — hand-built 5-node / 7-link GDF; validation errors on invariant violations.
- `test_file_source.py` — round-trip the fixture via GeoPackage / GeoJSON / Shapefile.
- `test_gmns_source.py` — uses an existing GMNS fixture in `tests/data/`.
- `test_download_cache.py` — exercises `DownloadCache` directly: folder creation, file write, manifest write, SHA-256 stability, disk-full failure path.

Also extend `test_no_progress_or_crs_or_clean_kwargs.py` (§11.4) to additionally assert that `import_from_overture` rejects `backend`, `parquet_path`, and `keep_rule_arrays`, and that `import_from_osm` rejects `xml_path`.

### 11.6 Add — simplifier

- `test_simplifier_osmnx.py` — exercises both OSM and Overture IRs (same simplifier), asserts node/link reduction and dict-of-dicts attribute reconciliation.
- `test_simplifier_neatnet.py` — `pytest.importorskip("neatnet")`, asserts urban grid collapse.

### 11.7 Coverage

Keep `tool.coverage.report.fail_under = 75`. Verify after the rewrite.

---

## 12. Documentation

- Rewrite `docs/source/examples/network_manipulation/plot_create_from_osm.py` (same filename, new content) using `import_from_osm`.
- Add `plot_create_from_overture.py`: small bbox download from Overture cloud + `links.explore()` map.
- Add `plot_import_custom_geodataframe.py`: build a tiny network in pandas/shapely, push through `import_from_geodataframes`.
- Add `docs/source/modeling_with_aequilibrae/network/importing.rst`:
  - Source matrix (which source for which use case, deps required).
  - Simplifier matrix (osmnx vs neatnet vs false). **Auto-UTM is the only projection** — documented here, not configurable.
  - **Schema mapping reference**, with an explicit section "Where does my OSM/Overture tag end up?" that walks the `_split_attributes` algorithm and shows the dict-of-dicts structure of `source_id_list` after simplification.
  - **Mode-vs-link-type filtering note** (§1.3 rule 2): no in-import link-type allow-lists; here is the post-import SQL pattern.
  - **No-knobs note** (§1.3 rule 4): no `clean` / `commit` / `projected_crs` arguments. The importer is atomic: build a `GeoDataFrameSource` yourself if you need pre-commit inspection.
  - **Provenance note** (§10): how to read `about` for whole-import metadata and `other_attributes` JSON for per-row metadata.
  - **Raw-download cache note** (§4.6): the `<project>/downloaded_data/` layout, what's in `manifest.json`, how to use the cached payload to re-run an import offline, and a one-paragraph "disk-space management is the user's responsibility" warning.
  - **OSM XML note**: not supported; convert with `osmium cat in.osm -o out.osm.pbf` then `pbf_path=`.
  - **Overture backend note**: cloud is the only supported backend; for offline re-imports use the cached `segments.parquet` + `connectors.parquet` in `downloaded_data/` (the docs include a 10-line snippet showing how to construct a `GeoDataFrameSource` from them).
  - Migration table `create_from_osm` → `import_from_osm`, including the parameter-file removal.
  - Extension guide: "How to add a new source" (~40 lines, with an OSRM example skeleton).
- Update `README.md` install hint: `pip install aequilibrae[create]`.
- Add a release-notes entry in `docs/source/release_notes/` covering all five constraints from §1.3.

---

## 13. Performance expectations

| Area | Today (custom OSM) | osmnx Overpass | osmnx + pyrosm pbf | Overture cloud | + osmnx simplify | + neatnet simplify |
|---|---|---|---|---|---|---|
| Vatican City (~0.5 km²) | ~5 s | ~5 s | ~1 s | ~3 s | +0.5 s | +2 s |
| Nauru (~21 km²) | ~30 s | ~25 s | ~3 s | ~10 s | +1 s | +5 s |
| Mid-city (~500 km²) | 5–10 min | 3–8 min | **20–40 s** | 30–90 s | +30 s | +2–5 min |
| US state (~250 000 km²) | hours / fails | hours | **5–15 min** | 5–20 min | +5–10 min | +30–60 min |

Notes:

- Overture cloud transfers ≪ Overpass for the same area because of GeoParquet row-group statistics + bbox push-down — typically 5–20× less network I/O for road data.
- Simplification reduces node count by 40–70 % and edge count by 30–50 % on urban networks (Boeing 2020), speeding every downstream AeQ operation.
- Memory: the OSMnx MultiDiGraph is the peak (~3–4 GB for a 1 M-edge graph); it's dropped immediately after simplification.
- The dict-of-dicts `source_id_list` after simplification typically adds ~200–800 bytes per merged link (n source edges × m tags). For very large urban areas this can dominate the per-row `other_attributes` size; see §16 open question 9 about an optional cap.
- **Raw-download cache disk usage**: a typical Nauru-sized OSM Overpass response is ~5 MB JSON; a mid-city Overpass response is ~50–500 MB. Overture parquet payloads for the same areas are roughly 30–40 % smaller. The cache is **never** auto-pruned; users with many imports should periodically clean `<project>/downloaded_data/` manually. Documented in the migration guide.

---

## 14. Capabilities lost / behaviour changes

User-facing list for the migration guide:

1. **No `ALTER TABLE` at all from the importer.** OSM tags / Overture properties / user GDF columns / source provenance all land in `other_attributes` JSON unless a same-named column already exists. To get a typed column, `ALTER TABLE` it yourself before importing. (§1.3 rule 1, §4.4, §10.)
2. **No link-type allow-lists or denylists in the importer.** Use post-import SQL (`DELETE FROM links WHERE link_type IN (…)`) or pass `custom_filter` (OSM) / a predicate (Overture). (§1.3 rule 2.)
3. **YAML-driven mode/link-type matrix removed.** Mode rules are now Python (`schema/modes.py`), based purely on access semantics.
4. **`aggregate_link_type` 51-cap collapse removed.** Every distinct `highway` / `class` value is preserved.
5. **No `clean` / `commit` / `projected_crs` / `progress` knobs.** The importer is single-purpose, atomic, auto-UTM, log-only. Pre-commit inspection is via building your own `GeoDataFrameSource`. (§1.3 rules 4–5.)
6. **Raw-download cache mandatory.** Every network import writes raw payloads to `<project>/downloaded_data/`. No way to opt out; users with disk constraints prune the folder manually. (§1.3 rule 6, §4.6.)
7. **No `backend` / `parquet_path` / `keep_rule_arrays` for Overture.** Cloud is the only backend; rule arrays are always preserved. (§1.3 rule 7.)
8. **No OSM XML source.** Convert XML to PBF with `osmium cat in.osm -o out.osm.pbf` and use `pbf_path=`. (§1.3 rule 8.)
9. **`Network.create_from_osm` removed.** Replaced by `Network.import_from_osm`. Old call raises `AttributeError`.
10. **`placegetter` public re-export removed.** Use `osmnx.geocoder.geocode_to_gdf` directly.
11. **`max_query_area_size`, `sleeptime`, `max_attempts` knobs removed.** OSMnx tiles and retries internally.
12. **Hand-rolled 429/504 retry replaced** by OSMnx's (different default backoff).
13. **Side-effect `<project>/osm_data/` parquet dump replaced** by the structured raw-download cache in `<project>/downloaded_data/`.
14. **`clean` semantics shift slightly**: the polygon clip now always runs **after** simplification, so a long simplified edge that straddles the polygon edge may be trimmed differently than today.
15. **Lanes/speed divide-by-2 replaced** by proper `:forward` / `:backward` split.
16. **Custom Overpass server still supported** via `Parameters().parameters["osm"]["overpass_endpoint"]` → `osmnx.settings.overpass_url`.
17. **`WorkerThread` / `SIGNAL` events removed.** QAequilibraE's existing progress-bar integration on the OSM importer will stop receiving signals — coordinate with QAequilibraE for a logging-based replacement (or a wrapper there).
18. **Overture turn-restrictions / scoped speed-limits**: preserved verbatim in `other_attributes` JSON but **not** consumed by AeQ's graph builder in this iteration. Follow-up PR.
19. **Overture lane counts**: not in Overture's current public schema → `lanes_ab` / `lanes_ba` NULL on Overture imports.
20. **GERS id stability**: Overture re-imports change AeQ `link_id` / `node_id` numbering; only the `source_id` inside `other_attributes` (GERS) is stable across releases.
21. **`osm_id INTEGER` column not added.** Read `json_extract(other_attributes, '$.source_id')` and cast to integer if needed.

---

## 15. Implementation order (PR-sized chunks)

Each PR is independently reviewable. PRs 1–6 do not break any existing user code (old `create_from_osm` keeps working until PR 7).

1. **PR 1 — Framework scaffolding & deps**
   - `[create]` extra (+ `[osm]` alias) in `pyproject.toml`.
   - `OptionalDependencyError` + `require()` helper in `aequilibrae.utils.optional_dependency`.
   - Empty module skeleton (`importer/`, `sources/`, `simplifiers/`).
   - CI matrix entry `pip install -e .[create,tests]`.

2. **PR 2 — IR, registry, committer, about provenance, download cache** (the heart of the framework)
   - `RoutableNetwork` + `validate()`.
   - `Source` / `Simplifier` protocols + registries.
   - `SpatialiteWriter` with **schema-aware `_split_attributes`** (§4.4). Reuses existing `remove_triggers` / `add_triggers` helpers. **No `ALTER TABLE` anywhere.**
   - `AboutWriter` (§10): wraps `About.add_info_field` + `write_back` for the `network_source_*` keys.
   - `DownloadCache` (§4.6).
   - `GeoDataFrameSource` (simplest source, end-to-end test).
   - Tests: `test_ir.py`, `test_registry.py`, `test_committer_routing.py`, `test_committer_no_alter_table.py`, `test_committer_missing_other_attributes.py`, `test_about_provenance.py`, `test_download_cache.py`, `test_geodataframe_source.py`, `test_no_progress_or_crs_or_clean_kwargs.py`.

3. **PR 3 — OSM sources + tag-to-IR**
   - `osm/tags_to_ir.py` — interprets routing-relevant tags into typed IR columns; passes all other tags through as free-form columns.
   - `MODE_RULES` (Python, access-based, no link-type allow-lists).
   - `OSMOverpassSource` (writes raw to `downloaded_data/osm-overpass/...`), `OSMPbfSource` (writes nothing).
   - `Network.import_from_osm`.
   - Tests with bundled tiny `.pbf`; the no-link-type-filter test (§11.4); the no-`xml_path` signature assertion.

4. **PR 4 — Simplifiers + dict-of-dicts reconciliation**
   - `OSMnxSimplifier`, `NeatnetSimplifier`.
   - Attribute reconciliation building `_source_id_list` (dict-of-dicts) per merged edge.
   - Tests against OSM PBF fixture, including `test_other_attributes_dict_of_dicts.py`.

5. **PR 5 — Overture cloud source**
   - `OvertureCloudSource` (uses `overturemaps`; writes raw `segments.parquet` + `connectors.parquet` + manifest to `downloaded_data/overture-cloud/...`).
   - `overture/schema_to_ir.py` (segment + connector → IR with substring splitting; rule-arrays always passed through as free-form columns).
   - `Network.import_from_overture` (no `backend`, no `parquet_path`, no `keep_rule_arrays` kwargs).
   - Tests with bundled tiny Overture parquet fixture (monkeypatched into `overturemaps.record_batch_reader`) + gated live test.

6. **PR 6 — File source + GMNS adapter**
   - `FileSource`, `GMNSSource`.
   - `Network.import_from_geodataframes`, `Network.import_from_file`.
   - Migrate `create_from_gmns` to internally use the new committer.

7. **PR 7 — Removal & docs**
   - Delete `osm_downloader.py`, `osm_builder.py`, `place_getter.py`, `osm_params.py`, `model_area_gridding.py`.
   - Remove `Network.create_from_osm`.
   - Trim `parameters.yml` per §9.
   - Rewrite example, add Overture + custom-GDF examples, write `importing.rst`, write migration guide.
   - Update `README.md` install hint.

---

## 16. Open questions / things to confirm during implementation

1. **OSMnx 2.x API surface.** Confirm signatures of `ox.simplification.simplify_graph`, `ox.simplification.consolidate_intersections`, `ox.projection.project_graph`, `ox.convert.graph_to_gdfs`. Pin `osmnx>=2.0,<3`.
2. **neatnet API stability.** Pin to a known-good version; thin adapter for future renames.
3. **pyrosm Python 3.13 / 3.14 wheels.** AeQ supports 3.10–3.14. If wheels missing for the newer versions, document Overpass / Overture as fallbacks.
4. **`overturemaps` wheel on Windows.** Pure-Python, depends on pyarrow + boto3 / requests. Verify on Windows CI.
5. **Auto-UTM edge case** for areas crossing UTM zones (e.g. continental imports). OSMnx picks UTM by centroid; for very wide areas this can introduce small distance errors. Default is acceptable for typical city/regional imports; document the limitation; for very large areas users should split the import or accept the small distortion.
6. **Download-cache compression threshold** (currently set at "> 10 MB → gzip" for Overpass JSON). Confirm during PR 3; alternative is "always gzip". Trade-off: smaller disk vs. ability to `cat` / `jq` the file directly.
7. **QAequilibraE coupling**. With `WorkerThread` / `SIGNAL` removed, QAequilibraE's existing progress-bar handler on `create_from_osm` will go silent. Coordinate with QAequilibraE maintainers; possibilities are (a) update QAequilibraE to log-tail, (b) add a thin wrapper in QAequilibraE that runs `import_from_osm` on a background thread and emits Qt signals from there. **No `Progress` adapter in this codebase** per §1.3 rule 5.
8. **Overture release pinning** in tests. Recommendation: frozen file fixture for unit tests, `release=None` (STAC "latest") for the live integration test, default `release=None` in production with documentation on how to pin.
9. **`other_attributes` size budget.** The simplifier's dict-of-dicts can produce 200–800 bytes per row on dense OSM data, occasionally more on heavily-merged edges. Should we cap individual row JSON length (e.g. 64 KB) to protect against pathological cases? Recommendation: log-and-truncate at 64 KB, recording the truncated key list in the import-time log. Decide during PR 4.
10. **Type coercion warnings on committer routing.** Decision: single aggregated warning per import per (column, reason) tuple to avoid log spam on large imports; the dropped values still survive in `other_attributes`. Verify during PR 2.
11. **`source_id` as primary edge convention.** Lexicographically smallest source id is deterministic but arbitrary. Alternative: longest-geometry edge. Recommendation: keep "lexicographically smallest" for simplicity and document; users wanting the longest-geometry primary can post-process from the full `source_id_list`. Confirm during PR 4.
