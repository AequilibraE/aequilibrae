## ADDED Requirements

### Requirement: VISUM SQLite import is documented
The system SHALL document the VISUM SQLite import workflow and its relationship to VISUM GeoJSON imports.

#### Scenario: Documenting the SQLite import workflow
- **WHEN** the VISUM SQLite import API is added
- **THEN** user documentation SHALL describe required and optional VISUM SQLite tables, CRS handling, mapping
  configuration, geometry reconstruction, assignment-field derivation, connector zero-time epsilon behavior,
  count-location scope, and deferred public-transport and demand workflows

#### Scenario: Documenting connectivity validation
- **WHEN** VISUM SQLite import documentation is added
- **THEN** the documentation SHALL explain how SQLite source connectivity can be compared with SQLite-imported and
  GeoJSON-imported AequilibraE graph connectivity
- **AND** distinguish modal-connectivity validation from impedance or assignment-result comparison

#### Scenario: Providing an executable SQLite example
- **WHEN** examples are updated for VISUM SQLite import
- **THEN** the documentation SHALL include an executable example or gallery script using compact local VISUM-like SQLite
  fixtures
- **AND** the example SHALL avoid external network downloads
