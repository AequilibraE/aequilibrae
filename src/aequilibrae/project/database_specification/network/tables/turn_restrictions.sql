--@ The *turn_restrictions* table holds information about turn restrictions and penalties
--@ for the network.
--@
--@ The **restriction_id** field is the unique identifier for the turn restriction
--@
--@ The **from_node** field is the node at the start of the incoming movement leg.
--@
--@ The **via_node** field is the node where the turn occurs.
--@
--@ The **to_node** field is the node at the end of the outgoing movement leg.
--@
--@ The **geometry** field is a LINESTRING built from three points representing
--@ the turn movement sequence (incoming leg, via node, outgoing leg).
--@
--@ The **penalty** field is the turn penalty in the same time unit as the graph cost field.
--@ NULL indicates a prohibited turn. Positive infinity is also accepted at the API
--@ layer and is normalised to NULL before storage. Negative values are not allowed.
--@
--@ The **modes** field is a concatenation of mode IDs for which this restriction applies.


CREATE TABLE IF NOT EXISTS turn_restrictions (
    restriction_id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
    from_node      INTEGER NOT NULL,
    via_node       INTEGER NOT NULL,
    to_node        INTEGER NOT NULL,
    penalty        REAL,
    modes          TEXT    NOT NULL,
    FOREIGN KEY(from_node) REFERENCES nodes(node_id) ON UPDATE CASCADE,
    FOREIGN KEY(via_node) REFERENCES nodes(node_id) ON UPDATE CASCADE,
    FOREIGN KEY(to_node) REFERENCES nodes(node_id) ON UPDATE CASCADE,
    CHECK(from_node != via_node),
    CHECK(via_node != to_node),
    CHECK(penalty IS NULL OR penalty >= 0),
    CHECK(LENGTH(modes) > 0),
    UNIQUE(from_node, via_node, to_node, modes)
);

--#
select AddGeometryColumn( 'turn_restrictions', 'geometry', 4326, 'LINESTRING', 'XY', 0);

--#
SELECT CreateSpatialIndex( 'turn_restrictions' , 'geometry' );

--#
CREATE INDEX IF NOT EXISTS idx_turn_restrictions_from_node ON turn_restrictions(from_node);

--#
CREATE INDEX IF NOT EXISTS idx_turn_restrictions_to_node ON turn_restrictions(to_node);

--#
CREATE INDEX IF NOT EXISTS idx_turn_restrictions_modes ON turn_restrictions(modes);

--#
CREATE INDEX IF NOT EXISTS idx_turn_restrictions_via_node ON turn_restrictions(via_node);
