## ADDED Requirements

### Requirement: Route choice sets can be generated

The system SHALL generate route choice sets for compatible graph and demand inputs.

#### Scenario: Running route choice

- **WHEN** route choice is executed with a prepared graph and demand
- **THEN** the system SHALL generate candidate paths according to the selected route choice procedure
- **AND** store route set outputs in the route choice result structures

### Requirement: Path files can be written

The system SHALL support writing generated route paths to supported path-file outputs.

#### Scenario: Saving path files

- **WHEN** path-file saving is requested after route choice execution
- **THEN** the system SHALL write route path records to the requested output location using the supported file format

### Requirement: Link loading can be computed from route choice

The system SHALL compute link loading from route choice demand and generated routes.

#### Scenario: Computing route-choice link loading

- **WHEN** link loading is requested for route choice results
- **THEN** the system SHALL return loading values by link and demand core according to the route choice probabilities

### Requirement: Select-link sets are supported

The system SHALL support select-link definitions as named sets of link conditions for route choice loading outputs.

#### Scenario: Computing select-link outputs

- **WHEN** route choice is executed with named select-link definitions
- **THEN** the system SHALL compute select-link loading and origin-destination outputs for each named definition

### Requirement: Select-link assignment outputs can be persisted

The system SHALL allow supported select-link assignment outputs to be saved through project matrix and result registries.

#### Scenario: Saving select-link results

- **WHEN** select-link assignment results are saved for an open project
- **THEN** the system SHALL register the matrix and result outputs in the project registries

