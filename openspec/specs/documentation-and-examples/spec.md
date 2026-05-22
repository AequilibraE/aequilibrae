# Documentation And Examples Specification

## Purpose

This specification captures the current behavioral contract for documentation, examples, doctests, and documentation build workflows.

## Requirements

### Requirement: Documentation is Sphinx-based

The system SHALL maintain user documentation under the Sphinx documentation tree.

#### Scenario: Building documentation

- **WHEN** documentation is built locally or in CI
- **THEN** the system SHALL use `docs/source` as the source tree
- **AND** produce build outputs under the documentation build directory

### Requirement: API docs come from docstrings

The system SHALL expose public API documentation through autodoc-compatible docstrings.

#### Scenario: Documenting public APIs

- **WHEN** public classes, functions, or methods are documented
- **THEN** docstrings SHALL remain compatible with the Sphinx/reST style used by the project

#### Scenario: Changing public APIs

- **WHEN** public API behavior or signatures change
- **THEN** related docstrings and generated API documentation SHALL be updated

### Requirement: Examples remain executable

The system SHALL keep documentation examples aligned with executable project workflows.

#### Scenario: Running doctests

- **WHEN** documentation CI runs doctests
- **THEN** source docstrings and selected `.rst` examples SHALL execute successfully with the configured doctest fixtures

#### Scenario: Updating behavior used by examples

- **WHEN** behavior used by a documentation example changes
- **THEN** the corresponding example SHALL be updated or explicitly skipped with a clear reason

### Requirement: Gallery examples are maintained

The system SHALL maintain runnable gallery examples under the documentation examples tree.

#### Scenario: Building gallery documentation

- **WHEN** HTML documentation is built with gallery generation enabled
- **THEN** gallery examples SHALL be processed from `docs/source/examples`

### Requirement: Documentation deployment is controlled

The system SHALL publish documentation artifacts only through the configured CI deployment conditions.

#### Scenario: Building pull request documentation

- **WHEN** documentation CI runs for a pull request
- **THEN** it SHALL build documentation artifacts
- **AND** publish preview artifacts only when required secrets are available

#### Scenario: Building release documentation

- **WHEN** documentation CI runs for a release
- **THEN** it SHALL build release documentation artifacts
- **AND** publish release documentation only when required secrets are available

