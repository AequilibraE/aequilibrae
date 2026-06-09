## Context

`AequilibraeMatrix` already reads and writes OMX files through `openmatrix`, and `Matrices.new_record(...)` can register `.omx` or `.aem` files that already exist in a project's `matrices` folder. `Matrices.update_database()` can also discover unregistered files in that folder. Users with an external demand file currently need to manually copy it into the project folder and then call lower-level registration or discovery methods.

The new endpoint belongs on the existing `Project.matrices` gateway because that gateway owns the project matrix folder, uniqueness checks, record creation, and matrix loading helpers.

## Goals / Non-Goals

**Goals:**

- Provide one public method to import a supported matrix file from outside the project into the project `matrices` folder.
- Validate the source file and destination record before registration.
- Reuse existing matrix loading and record creation behavior for core counting and database writes.
- Keep behavior deterministic for default destination file names and matrix record names.

**Non-Goals:**

- Add a new OMX parser or change `AequilibraeMatrix` OMX semantics.
- Convert imported OMX files to `.aem` by default.
- Match matrix indices to project zones or assignment graphs during import.
- Import VISUM proprietary matrix formats or automate assignment setup.

## Decisions

- Add `Matrices.import_file(path, name=None, file_name=None)` as the project-level endpoint. This keeps the API discoverable alongside `get_matrix`, `new_record`, and `update_database`.
- Copy the source file into the project `matrices` folder and register the copied file. Registering external paths directly would break scenario copying and the existing project folder contract.
- Accept only `.omx` and `.aem` files. This mirrors existing project registry behavior and avoids pretending CSV/trip-list import is the same capability.
- Derive defaults from the source filename. If `name` is omitted, use the destination stem normalized with the same dot/space-to-underscore convention used by `update_database`; if `file_name` is omitted, use the source filename.
- Validate the matrix by loading it before copying and again through `new_record` after copying. This reuses existing core-count behavior and catches unreadable files before committing a registry record.

## Risks / Trade-offs

- Duplicate large files increase disk usage -> importing into the project folder keeps scenarios and project archives self-contained.
- The endpoint does not verify matrix indices against project zones -> assignment compatibility remains the caller's responsibility, consistent with existing matrix registration.
- A failed registration after copying could leave an unregistered copied file -> remove the copied file on registration failure when the copied file was created by this endpoint.
