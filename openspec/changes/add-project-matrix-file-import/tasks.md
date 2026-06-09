## 1. API Implementation

- [x] 1.1 Add `Matrices.import_file(...)` to copy supported external `.omx` and `.aem` files into the project matrix folder.
- [x] 1.2 Reuse existing matrix loading and `new_record(...)` registration to validate cores and create the project record.
- [x] 1.3 Guard against unsupported extensions, missing sources, same-path imports, duplicate names, and duplicate destination file names.

## 2. Tests And Documentation

- [x] 2.1 Add focused project matrix tests for successful OMX import and registry access.
- [x] 2.2 Add focused tests for duplicate/unsupported import failures.
- [x] 2.3 Update project matrix documentation for the new import endpoint.
- [x] 2.4 Run focused matrix project tests and OpenSpec status validation.
