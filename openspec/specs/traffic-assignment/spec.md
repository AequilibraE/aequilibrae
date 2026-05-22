# Traffic Assignment Specification

## Purpose

This specification captures the current behavioral contract for traffic assignment and related result generation.

## Requirements

### Requirement: Assignments use transport classes

The system SHALL model traffic assignment demand through unique transport classes.

#### Scenario: Adding assignment classes

- **WHEN** transport classes are assigned to a traffic assignment
- **THEN** each class SHALL be unique within that assignment
- **AND** each class SHALL provide a graph and demand matrix suitable for assignment

#### Scenario: Rejecting duplicate classes

- **WHEN** the same transport class is added more than once
- **THEN** the system SHALL reject the assignment setup

### Requirement: Assignment fields are validated

The system SHALL validate graph fields used for travel time, capacity, costs, skims, and volume delay functions.

#### Scenario: Setting a time field

- **WHEN** a time field is set for an assignment
- **THEN** the field SHALL exist for every class graph
- **AND** the field SHALL not contain invalid null, zero, or negative values unless the operation explicitly allows them

### Requirement: Volume delay functions are configurable

The system SHALL support configuring assignment volume delay functions and their parameters.

#### Scenario: Configuring a VDF

- **WHEN** a supported VDF is selected for traffic assignment
- **THEN** the system SHALL use that function to update congested travel times during assignment

#### Scenario: Configuring VDF parameters

- **WHEN** VDF parameters are provided
- **THEN** the system SHALL map those parameters to graph fields or values required by the selected VDF

### Requirement: Assignment algorithms are selectable

The system SHALL expose supported equilibrium assignment algorithms and execute the selected algorithm.

#### Scenario: Selecting an algorithm

- **WHEN** a supported assignment algorithm is selected
- **THEN** the system SHALL prepare the corresponding assignment procedure

#### Scenario: Executing assignment

- **WHEN** a configured assignment is executed
- **THEN** the system SHALL load demand, update link costs, and record convergence information according to the selected algorithm

### Requirement: Assignment results are available

The system SHALL expose assignment outputs as tabular results and support saving them to the project results registry.

#### Scenario: Reading results

- **WHEN** assignment execution completes
- **THEN** the system SHALL make link-level assignment results available as a DataFrame

#### Scenario: Saving results

- **WHEN** assignment results are saved with a table name
- **THEN** the system SHALL write result data to the results database
- **AND** register metadata in the project results table

### Requirement: Skims and reports are available

The system SHALL expose convergence reports and skim outputs generated during assignment.

#### Scenario: Reading convergence report

- **WHEN** a completed assignment is queried for its report
- **THEN** the system SHALL return convergence data as tabular information

#### Scenario: Reading class skims

- **WHEN** assignment classes were configured to skim fields
- **THEN** the system SHALL expose skim results by class identifier

