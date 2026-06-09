## 1. SQLite Source Inventory And Fixtures

- [x] 1.1 Add compact VISUM-like SQLite fixtures covering `NODE`, `LINK`, `CONNECTOR`, `ZONE`, `TSYS`, `MODE`, `LINKTYPE`, `COUNTLOCATION`, and relational geometry tables.
- [x] 1.2 Add fixture cases for directed `LINK` row pairs, asymmetric `TSYSSET`, empty `TSYSSET`, sparse source link numbers, duplicate node coordinates, and zone/node ID collisions.
- [x] 1.3 Add fixture cases for `LINKPOLY`, reverse-direction link geometry, straight source links without `LINKPOLY`, connector geometry, and reconstructable zone surfaces.
- [x] 1.4 Add fixture cases for connector `T0_TSYS(...)`, explicit zero private connector times, missing required tables, and unparseable CRS metadata.
- [x] 1.5 Document the Karlsruhe SQLite export inventory and expected real-data counts in implementation notes.

## 2. Reader And Normalization

- [x] 2.1 Implement VISUM SQLite file validation and required-table discovery.
- [x] 2.2 Implement table readers for `NETWORK`, `TSYS`, `MODE`, `LINKTYPE`, `NODE`, `ZONE`, `LINK`, `CONNECTOR`, and `COUNTLOCATION`.
- [x] 2.3 Parse `NETWORK.PROJECTIONDEFINITION` and transform source coordinates into project storage CRS.
- [x] 2.4 Normalize source transport-system, mode, and link-type definitions into the existing VISUM mapping configuration model.
- [x] 2.5 Normalize directed SQLite `LINK` row pairs into source link records with AB/BA directional attributes.
- [x] 2.6 Normalize SQLite connector rows into source connector records keyed by zone, node, and direction.
- [x] 2.7 Report recognized deferred SQLite source tables without importing public-transport schedule or turn-restriction semantics.

## 3. Geometry Reconstruction

- [x] 3.1 Reconstruct node and zone centroid point geometry from source coordinates.
- [x] 3.2 Reconstruct link geometry from node endpoints and ordered `LINKPOLY` vertices.
- [x] 3.3 Reuse reversed link geometry for opposite source directions when only one source orientation has `LINKPOLY`.
- [x] 3.4 Construct straight source geometry for links and connectors when VISUM SQLite stores no shape vertices.
- [x] 3.5 Reconstruct supported zone multipolygons from `SURFACEID`, `SURFACEITEM`, `FACEITEM`, `EDGE`, `EDGEITEM`, and `POINT`.
- [x] 3.6 Add geometry validity diagnostics for missing endpoints, broken surface references, invalid polygons, and CRS transformation failures.

## 4. Import Implementation

- [x] 4.1 Add the public `Project.network.create_from_visum_sqlite(...)` entry point consistent with the GeoJSON importer API.
- [x] 4.2 Share or factor VISUM diagnostics, source-reference, mode-mapping, ignored-transport-system, and link-type normalization logic with the GeoJSON importer.
- [x] 4.3 Import SQLite nodes while preserving source node IDs and applying duplicate-coordinate offset or strict rejection behavior.
- [x] 4.4 Import SQLite zones and centroids while preserving zone IDs and remapping colliding regular node IDs.
- [x] 4.5 Import collapsed SQLite links with compact AequilibraE link IDs, directional modes, link types, assignment fields, and source metadata.
- [x] 4.6 Import SQLite connectors with compact AequilibraE link IDs, deterministic connector keys, directional modes, assignment fields, and source metadata.
- [x] 4.7 Preserve supported count-location associations to imported links without performing demand adjustment.
- [x] 4.8 Ensure validation failures stop before writes or provide explicit transactional cleanup diagnostics.

## 5. Assignment Field Behavior

- [x] 5.1 Derive SQLite link travel times from source length and speed and derive capacity from the selected source capacity field.
- [x] 5.2 Use connector `T0_TSYS(...)` source values for mapped transport systems instead of GeoJSON missing-field fallback behavior.
- [x] 5.3 Treat explicit zero private connector travel times as positive epsilon costs and report zero-time diagnostics.
- [x] 5.4 Report non-assignment-ready SQLite links or connectors after source-value and epsilon rules are applied.
- [x] 5.5 Verify imported SQLite private-traffic graphs pass `set_time_field(...)` and `set_capacity_field(...)` on compact fixtures.

## 6. Connectivity Validation

- [x] 6.1 Build source-connectivity extraction from SQLite `LINK` and `CONNECTOR` records by mapped AequilibraE mode.
- [x] 6.2 Compare SQLite-imported AequilibraE modal connectivity against SQLite source connectivity.
- [x] 6.3 Compare GeoJSON-imported modal connectivity against equivalent SQLite source connectivity on the Karlsruhe export.
- [x] 6.4 Compare SQLite-imported and GeoJSON-imported modal connectivity for equivalent source exports.
- [x] 6.5 Keep impedance comparison separate from modal-connectivity comparison and report expected differences from fallback or epsilon handling.

## 7. Real Karlsruhe Smoke Checks

- [x] 7.1 Import `C:\Users\Pablo Barceló\Downloads\Karlsruhe\Karlsruhe-sqlite.sqlite3` into a fresh Karlsruhe SQLite example project.
- [x] 7.2 Import `Visum_3_modes.omx` into the regenerated SQLite example project.
- [x] 7.3 Run graph-building and assignment-readiness checks for the car graph.
- [x] 7.4 Run a short traffic-assignment smoke test using the `Car` matrix core.
- [x] 7.5 Record any centroid connectivity warnings, zero connector epsilon diagnostics, and graph comparison differences in implementation notes.

## 8. Documentation And Examples

- [x] 8.1 Add API docstrings for the VISUM SQLite import entry point and any public diagnostics or comparison helpers.
- [x] 8.2 Add Sphinx documentation for required SQLite tables, geometry reconstruction, CRS handling, mapping options, assignment values, zero connector epsilon behavior, and deferred VISUM scope.
- [x] 8.3 Add a compact executable documentation example using local VISUM-like SQLite fixtures.
- [x] 8.4 Document how to use SQLite source connectivity to validate GeoJSON-imported model connectivity.

## 9. Verification

- [x] 9.1 Run focused VISUM SQLite importer unit tests.
- [x] 9.2 Run focused VISUM SQLite graph and assignment-readiness tests.
- [x] 9.3 Run focused GeoJSON/SQLite connectivity comparison tests.
- [x] 9.4 Run matrix import tests if real Karlsruhe smoke uses OMX project import.
- [x] 9.5 Run ruff on touched package and test files.
- [x] 9.6 Run relevant documentation example or doctest checks.
- [x] 9.7 Run `openspec.cmd status --change add-visum-sqlite-import-pipeline` and resolve incomplete artifacts.
