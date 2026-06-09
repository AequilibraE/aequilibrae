## ADDED Requirements

### Requirement: Assignment warns about unassignable demand

The system SHALL detect and warn when positive traffic-assignment demand cannot be assigned on the selected class graph
because required centroids are absent from the graph or because no path exists for positive OD pairs.

#### Scenario: Positive demand references centroids absent from the graph

- **WHEN** a traffic class has positive productions or attractions for centroids that are not present in its selected graph
- **THEN** the assignment validation SHALL report those centroids as missing from the graph
- **AND** the report SHALL include the positive demand attached to those centroids
- **AND** assignment execution SHALL warn by default without failing

#### Scenario: Positive OD demand has no path

- **WHEN** a traffic class has positive OD demand between centroids that are present in the graph but not mutually reachable
- **THEN** the assignment validation SHALL report the unassignable OD demand
- **AND** the report SHALL include total unassignable demand, percentage of class demand, affected OD-pair count, and bounded OD samples
- **AND** assignment execution SHALL warn by default without failing

#### Scenario: Missing graph centroids have zero demand

- **WHEN** a graph contains configured centroids that are absent from the selected mode graph
- **AND** the traffic class has zero production and attraction for those centroids
- **THEN** the assignment validation SHALL NOT report demand loss for those centroids

#### Scenario: Multiple traffic classes are validated independently

- **WHEN** a traffic assignment contains multiple traffic classes
- **THEN** demand-connectivity validation SHALL produce class-specific warnings and summaries
- **AND** one class with unassignable demand SHALL NOT mask another class with fully assignable demand
