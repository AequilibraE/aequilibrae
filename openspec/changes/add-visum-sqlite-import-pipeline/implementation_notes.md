# Implementation Notes

## Karlsruhe SQLite Inventory

Source file checked: `C:\Users\Pablo Barceló\Downloads\Karlsruhe\Karlsruhe-sqlite.sqlite3`.

The importer currently consumes the private-traffic network tables `NETWORK`, `NODE`, `ZONE`, `TSYS`, `MODE`,
`LINKTYPE`, `LINK`, `CONNECTOR`, `COUNTLOCATION`, `LINKPOLY`, `POINT`, `EDGE`, `EDGEITEM`, `FACEITEM`, and
`SURFACEITEM`. The Karlsruhe export also contains recognized deferred tables for public transport, stop, line-route,
turn, and detector-like workflows that are reported but not imported by this change.

The real SQLite import completed with no errors and these source object counts:

- Nodes: 8,432
- Zones: 726
- Links: 10,902
- Connectors: 2,821

Relevant diagnostics from the real SQLite import:

- `directional-mode-split`: 298
- `sqlite-zero-connector-time`: 1,849
- `empty-transport-systems`: 845
- `non-assignment-ready`: 3,640
- `coincident-node-offset`: 19
- `coincident-centroid-offset`: 108
- `node-id-remapped`: 3

## Directional Mode Split

The first Karlsruhe comparison showed that both the GeoJSON and SQLite paths produced extra modal arcs relative to the
SQLite source graph. The root cause was the shared VISUM importer collapsing asymmetric AB/BA transport-system sets into a
single bidirectional AequilibraE link row with the union of modes. Because AequilibraE stores one modal set per link row,
the fix is to split asymmetric source links and connectors into two one-way rows.

After the split, both imports match the SQLite source graph exactly by mapped mode.

| Mode | Source arcs | SQLite imported arcs | GeoJSON imported arcs | Missing | Extra |
| ---- | ----------- | -------------------- | --------------------- | ------- | ----- |
| `b`  | 19,915      | 19,915               | 19,915                | 0       | 0     |
| `c`  | 18,378      | 18,378               | 18,378                | 0       | 0     |
| `h`  | 18,433      | 18,433               | 18,433                | 0       | 0     |
| `t`  | 12,271      | 12,271               | 12,271                | 0       | 0     |
| `w`  | 3,070       | 3,070                | 3,070                 | 0       | 0     |

SQLite-imported and GeoJSON-imported modal graphs also match each other exactly for the same modes.

## Assignment Smoke

A fresh project was generated at:

`C:\tmp\aequilibrae\karlsruhe-sqlite-example-split-20260608_160406`

The `Visum_3_modes.omx` file was imported into the project with `project.matrices.import_file(...)`, and the `Car` core
was used for an all-or-nothing assignment smoke test.

Assignment smoke result:

- Matrix total: 776,784.884
- Car graph nodes: 8,367
- Car graph zones: 726
- Car graph links: 25,041
- Assignment result rows: 14,021
- Non-zero result rows: 6,569
- `Car_tot` sum: 36,187,327.676

Graph building reported 28 matrix centroids not present in the compressed car graph:
`2000115` through `2000142`. The assignment still executed successfully; this remains a real-model connectivity warning
rather than an importer error.

## Verification Notes

Focused SQLite importer tests and focused SQLite graph-readiness tests pass. GeoJSON validation tests pass in smaller
slices, but the full `test_network_visum_geojson.py` file exits the Python process silently on Windows/Python 3.14 when
it reaches `test_non_positive_assignment_fields_are_diagnostic_warnings` after the preceding tests. The same test passes
in isolation and in tail slices, including with the new SQLite tests. Ruff is clean for touched importer and test files.
