## Context

AequilibraE has a VISUM GeoJSON import pipeline in progress that builds private-traffic networks from exported GIS
layers. The Karlsruhe VISUM SQLite export shows that SQLite contains richer source data than GeoJSON for several
objects, especially connector travel times, VISUM transport systems, link-type defaults, and relational geometry.

The inspected Karlsruhe SQLite export contains explicit `NODE`, `LINK`, `CONNECTOR`, `ZONE`, `TSYS`, `MODE`,
`LINKTYPE`, and `COUNTLOCATION` tables. It also exposes geometry through relational tables: `LINKPOLY` for link shape
vertices and `SURFACEITEM`/`FACEITEM`/`EDGE`/`EDGEITEM`/`POINT` for zone surfaces. The export stores projected
coordinates and a WKT projection definition in `NETWORK.PROJECTIONDEFINITION`.

The SQLite source still has the same structural issues that required engineering in the GeoJSON path:

- sparse VISUM link numbers up to `2,030,064,882`;
- duplicate node coordinate groups;
- regular node IDs colliding with zone IDs;
- connectors identified by `ZONENO`, `NODENO`, and `DIRECTION`, not a connector number;
- transport-system availability encoded by VISUM `TSYSSET` strings.

The same export also shows that explicit zero private connector times exist in SQLite. These zeroes are source data,
not missing fields, and should be treated differently from GeoJSON connector fields that were not exported at all.

## Goals / Non-Goals

**Goals:**

- Add a first-class VISUM SQLite import entry point under the project network API.
- Parse source VISUM object tables and transform them into the same AequilibraE network shape produced by the GeoJSON
  importer where the source topology is equivalent.
- Reconstruct source-faithful geometry for links, connectors, nodes, centroids, and supported zone polygons.
- Use SQLite-native assignment data where available, including connector `T0_TSYS(...)`.
- Preserve source identifiers and diagnostics in the same style as the GeoJSON importer.
- Compare imported graph connectivity by mode against the SQLite source and the existing GeoJSON-derived Karlsruhe
  project so fallback assignment values are not mistaken for modal connectivity changes.

**Non-Goals:**

- Import VISUM public transport schedules, line routes, vehicle journeys, stop timing, or route-system behavior.
- Import or estimate OD matrices from SQLite.
- Implement VISUM turn restrictions or node-control delay modeling.
- Perform count-based OD adjustment, calibration, or validation beyond preserving supported count-location associations.
- Preserve every VISUM source table or user-defined attribute.

## Decisions

### Build the real SQLite importer before graph comparison

The graph comparison needs most of the same parsing as the importer: `LINK`, `CONNECTOR`, `TSYSSET`, directionality,
source IDs, geometry, and CRS. A throwaway comparison-only parser would duplicate the hard part without producing a
reusable feature. The SQLite importer should therefore be implemented first, and graph comparison should be expressed as
tests and validation utilities on top of the imported model and source reader.

Alternative considered: build a one-off SQLite connectivity parser only for checking the GeoJSON fallback. This was
rejected because it would still need to solve VISUM SQLite semantics but would not produce a durable user workflow.

### Reuse VISUM mapping and diagnostics concepts

The SQLite importer should share mode mapping, ignored-transport-system handling, link-type normalization, diagnostics,
source-reference reporting, compact internal link IDs, duplicate-node offset policy, zone/node collision remapping, and
connector key behavior with the GeoJSON importer. SQLite-specific readers should feed normalized source records into
common import logic where feasible.

Alternative considered: implement SQLite import as a fully separate pipeline. This would be faster initially but would
increase drift between VISUM import paths and make graph equivalence harder to reason about.

### Collapse directed VISUM link rows into AequilibraE directional links

The SQLite `LINK` table is direction-expanded: each source `NO` has two rows with opposite `FROMNODENO` and `TONODENO`.
The importer should collapse these rows into one AequilibraE link, storing AB/BA directional attributes and using compact
internal `link_id` values. Links whose two directions have empty `TSYSSET` should be skipped with diagnostics, consistent
with GeoJSON import behavior.

Alternative considered: import each SQLite `LINK` row as a separate one-way AequilibraE link. This would preserve the
source table shape but would not match the existing AequilibraE directional-field model or the GeoJSON importer output.

### Reconstruct VISUM geometry without inventing non-source geometry

The importer should use source geometry when VISUM stores it. For links, it should build geometry from node endpoints
plus ordered `LINKPOLY` vertices when they exist. Reverse-direction rows should reuse the same shape in reverse. Links
without `LINKPOLY` in either direction should use straight node-to-node geometry because that is the geometry available
in the VISUM export. Connectors should use straight centroid-to-node geometry unless a future VISUM export exposes
connector shape vertices. Zone polygons should be reconstructed from `ZONE.SURFACEID` through the surface, face, edge,
edge-item, and point tables when possible.

Alternative considered: prefer GeoJSON files for geometry and SQLite for attributes. This creates a multi-source import
that is harder to reproduce and validate. The SQLite importer should be able to stand on the SQLite export alone.

### Use SQLite assignment values and epsilon for explicit zero connector times

For links, the importer should derive travel time from SQLite `LENGTH` and `V0PRT` and capacity from `CAPPRT` or
documented period-specific capacity fields when selected. For connectors, it should use `T0_TSYS(<transport system>)`
where present. When SQLite explicitly stores a private connector time as zero, the importer should import a positive
epsilon travel time and report an explicit-zero diagnostic. This differs from GeoJSON missing connector times, which use
a fallback speed because the field was not exported.

Alternative considered: apply the GeoJSON fallback speed to SQLite zero connector times. This was rejected because zero
is explicit source data and likely represents instant transfer in the VISUM model.

### Validate connectivity by mode, not by impedance

The primary comparison between SQLite, GeoJSON, and AequilibraE should check modal connectivity: which directed
node-to-node and centroid-to-node arcs exist for each mapped mode. Impedance values should be compared separately because
GeoJSON fallback and SQLite epsilon handling intentionally affect costs without changing whether a mode can use a link
or connector.

Alternative considered: compare full assignment-ready graphs including costs. This would conflate mode availability with
expected differences in connector travel-time source data.

## Risks / Trade-offs

- VISUM relational polygon reconstruction can be subtle when faces contain enclaves or reversed edges -> Add small
  fixtures for surface reconstruction and validate Karlsruhe zone polygon counts and geometry validity.
- CRS is an ESRI-style WKT without an EPSG code -> Use `pyproj.CRS.from_wkt(...)` and transform explicitly, reporting the
  parsed CRS and any transform failure.
- Shared GeoJSON/SQLite import code can become tangled -> Introduce only source-reader abstractions that remove real
  duplication and keep public APIs clear.
- Explicit zero connector times can create very low-cost centroid access -> Use positive epsilon, make it configurable if
  needed, and report affected source connectors.
- VISUM public transport objects are present in the SQLite export -> Recognize and report them as deferred rather than
  partially importing transit semantics.
- Connectivity comparison can hide turn restrictions because AequilibraE import does not model VISUM turns -> Scope
  comparison to link/connector modal graph connectivity and report turn restrictions as deferred.

## Migration Plan

No database migration is expected unless implementation introduces new durable source-mapping or count-location storage.
The importer should populate empty projects or explicitly targeted projects using existing network tables and source
metadata columns/patterns from the GeoJSON importer. Failed imports should stop before writes when validation catches the
problem, or provide explicit cleanup diagnostics if a transaction cannot cover the whole operation.

## Open Questions

- Should the public method accept only a SQLite file path, or should folder-based discovery pick a conventional
  `*.sqlite`/`*.sqlite3` file when present beside VISUM GeoJSON exports?
- Which capacity field should be default for SQLite import when both `CAPPRT` and period-specific fields such as
  `CAP_1H`/`CAP_24H` exist?
- Should zero connector epsilon be fixed globally, configurable per import, or read from project parameters?
