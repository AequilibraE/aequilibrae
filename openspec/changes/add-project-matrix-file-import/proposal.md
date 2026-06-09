## Why

Project users can load OMX files through `AequilibraeMatrix` and can register matrix files that are already inside the project `matrices` folder, but there is no direct project-level endpoint for importing an external matrix file. The VISUM GeoJSON workflow now produces project networks separately from demand, so users need a simple way to bring a matching OMX demand file into the project registry.

## What Changes

- Add a project matrix gateway endpoint for importing an existing external `.omx` or `.aem` file into a project's `matrices` folder.
- Register the imported file in the project `matrices` table and return the resulting `MatrixRecord`.
- Validate file type, source existence, destination/name uniqueness, and matrix readability before registration.
- Preserve existing low-level `AequilibraeMatrix` OMX support and existing `new_record` / `update_database` behavior.

## Capabilities

### New Capabilities

None.

### Modified Capabilities

- `matrix-io`: add project-level matrix file import behavior for supported matrix files.

## Impact

- Affected API: `Project.matrices` gateway.
- Affected code: `aequilibrae/project/data/matrices.py` and focused project matrix tests.
- Affected docs/specs: matrix I/O OpenSpec delta and user-facing project component documentation if the endpoint is exposed publicly.
- Dependencies: no new dependency; existing `openmatrix` support remains the OMX reader.
