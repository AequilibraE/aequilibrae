## ADDED Requirements

### Requirement: Transit capability boundaries are explicit

The transit GTFS capability SHALL cover the current public transport database, GTFS import, route-system persistence, transit graph creation, transit preload, and transit assignment contracts until a separate transit assignment capability is introduced.

#### Scenario: Planning a transit change

- **WHEN** a change affects GTFS import, transit graph persistence, preload, or transit assignment before a separate transit assignment spec exists
- **THEN** the change SHALL update the transit GTFS capability requirements

### Requirement: Transit assignment may be split later

The transit assignment behavior SHALL be eligible for a separate capability when a future change materially changes assignment APIs, algorithms, result contracts, or persistence.

#### Scenario: Introducing a transit assignment change

- **WHEN** a future proposal changes transit assignment behavior independently of GTFS import and transit graph construction
- **THEN** the proposal SHALL consider creating or modifying a dedicated transit assignment capability

