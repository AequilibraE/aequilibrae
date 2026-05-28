## ADDED Requirements

### Requirement: VISUM GeoJSON import is documented
The system SHALL document the VISUM GeoJSON import workflow and its private-traffic scope.

#### Scenario: Documenting the import workflow
- **WHEN** the VISUM GeoJSON import API is added
- **THEN** user documentation SHALL describe required and optional layers, mapping configuration, CRS handling, assignment-ready field derivation, count-location import, and deferred public-transport and demand workflows

#### Scenario: Providing an executable example
- **WHEN** examples are updated for VISUM GeoJSON import
- **THEN** the documentation SHALL include an executable example or gallery script using compact local VISUM-like fixtures
- **AND** the example SHALL avoid external network downloads
