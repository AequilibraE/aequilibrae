-- Ensures turn movement is node-consistent as a three-node sequence
CREATE TRIGGER aequilibrae_turn_restrictions_node_consistency_insert BEFORE INSERT ON turn_restrictions
WHEN new.from_node = new.via_node OR new.via_node = new.to_node
BEGIN
    SELECT RAISE(ABORT, 'Turn restriction requires distinct from_node, via_node and to_node legs');
END;

--#
-- Prevents a mode record from being changed when referenced by turn restrictions
CREATE TRIGGER aequilibrae_mode_keep_if_in_use_on_turn_restrictions_updating BEFORE UPDATE OF mode_id ON "modes"
WHEN (Select count(*) from turn_restrictions where instr(modes, old.mode_id) > 0)>0
BEGIN
    SELECT RAISE(ABORT, 'Mode in use on your network. Cannot change it');
END;

--#
-- Prevents a mode record from being removed when referenced by turn restrictions
CREATE TRIGGER aequilibrae_mode_keep_if_in_use_on_turn_restrictions_deleting BEFORE DELETE ON "modes"
WHEN (Select count(*) from turn_restrictions where instr(modes, old.mode_id) > 0)>0
BEGIN
    SELECT RAISE(ABORT, 'Mode in use on your network. Cannot change it');
END;

--#
CREATE TRIGGER aequilibrae_turn_restrictions_node_consistency_update BEFORE UPDATE OF from_node, via_node, to_node ON turn_restrictions
WHEN new.from_node = new.via_node OR new.via_node = new.to_node
BEGIN
    SELECT RAISE(ABORT, 'Turn restriction requires distinct from_node, via_node and to_node legs');
END;

--#
CREATE TRIGGER aequilibrae_turn_restrictions_set_geometry_insert AFTER INSERT ON turn_restrictions
BEGIN
    UPDATE turn_restrictions
    SET geometry = (
        SELECT AddPoint(
            MakeLine(
                CASE
                    WHEN d_in > 10.0 AND d_in > 0.0 THEN Line_Interpolate_Point(vf_line, 10.0 / d_in)
                    ELSE n_from_geom
                END,
                CASE
                    WHEN same_loop = 1 THEN ShiftCoords(n_via_geom, 0.00005, 0.00005)
                    ELSE n_via_geom
                END
            ),
            CASE
                WHEN d_out > 10.0 AND d_out > 0.0 THEN Line_Interpolate_Point(vt_line, 10.0 / d_out)
                ELSE n_to_geom
            END
        )
        FROM (
            SELECT
                nf.geometry AS n_from_geom,
                nv.geometry AS n_via_geom,
                nt.geometry AS n_to_geom,
                MakeLine(nv.geometry, nf.geometry) AS vf_line,
                MakeLine(nv.geometry, nt.geometry) AS vt_line,
                GeodesicLength(MakeLine(nv.geometry, nf.geometry)) AS d_in,
                GeodesicLength(MakeLine(nv.geometry, nt.geometry)) AS d_out,
                CASE WHEN new.from_node = new.to_node THEN 1 ELSE 0 END AS same_loop
            FROM nodes nf
            JOIN nodes nv ON nv.node_id = new.via_node
            JOIN nodes nt ON nt.node_id = new.to_node
            WHERE nf.node_id = new.from_node
        )
    )
    WHERE restriction_id = new.restriction_id;
END;

--#
CREATE TRIGGER aequilibrae_turn_restrictions_set_geometry_update AFTER UPDATE OF from_node, via_node, to_node ON turn_restrictions
BEGIN
    UPDATE turn_restrictions
    SET geometry = (
        SELECT AddPoint(
            MakeLine(
                CASE
                    WHEN d_in > 10.0 AND d_in > 0.0 THEN Line_Interpolate_Point(vf_line, 10.0 / d_in)
                    ELSE n_from_geom
                END,
                CASE
                    WHEN same_loop = 1 THEN ShiftCoords(n_via_geom, 0.00005, 0.00005)
                    ELSE n_via_geom
                END
            ),
            CASE
                WHEN d_out > 10.0 AND d_out > 0.0 THEN Line_Interpolate_Point(vt_line, 10.0 / d_out)
                ELSE n_to_geom
            END
        )
        FROM (
            SELECT
                nf.geometry AS n_from_geom,
                nv.geometry AS n_via_geom,
                nt.geometry AS n_to_geom,
                MakeLine(nv.geometry, nf.geometry) AS vf_line,
                MakeLine(nv.geometry, nt.geometry) AS vt_line,
                GeodesicLength(MakeLine(nv.geometry, nf.geometry)) AS d_in,
                GeodesicLength(MakeLine(nv.geometry, nt.geometry)) AS d_out,
                CASE WHEN new.from_node = new.to_node THEN 1 ELSE 0 END AS same_loop
            FROM nodes nf
            JOIN nodes nv ON nv.node_id = new.via_node
            JOIN nodes nt ON nt.node_id = new.to_node
            WHERE nf.node_id = new.from_node
        )
    )
    WHERE restriction_id = new.restriction_id;
END;

--#
CREATE TRIGGER aequilibrae_turn_restrictions_update_geometry_on_node_move AFTER UPDATE OF geometry ON nodes
BEGIN
    UPDATE turn_restrictions
    SET from_node = from_node
    WHERE from_node = new.node_id
       OR via_node = new.node_id
       OR to_node = new.node_id;
END;

--#
CREATE TRIGGER aequilibrae_turn_restrictions_modes_insert BEFORE INSERT ON turn_restrictions
WHEN
    new.modes IS NULL
    OR LENGTH(new.modes) = 0
    OR (SELECT COUNT(*) FROM modes WHERE INSTR(new.modes, mode_id) > 0) < LENGTH(new.modes)
BEGIN
    SELECT RAISE(ABORT, 'Turn restriction mode codes need to exist in the modes table');
END;

--#
CREATE TRIGGER aequilibrae_turn_restrictions_modes_update BEFORE UPDATE OF modes ON turn_restrictions
WHEN
    new.modes IS NULL
    OR LENGTH(new.modes) = 0
    OR (SELECT COUNT(*) FROM modes WHERE INSTR(new.modes, mode_id) > 0) < LENGTH(new.modes)
BEGIN
    SELECT RAISE(ABORT, 'Turn restriction mode codes need to exist in the modes table');
END;

--#
-- Prevents duplicate mode coverage for the same turn,
-- even when modes are provided in different configurations/order
CREATE TRIGGER aequilibrae_turn_restrictions_no_overlap_insert BEFORE INSERT ON turn_restrictions
WHEN EXISTS (
    SELECT 1
    FROM turn_restrictions tr
    JOIN modes m
      ON INSTR(new.modes, m.mode_id) > 0
     AND INSTR(tr.modes, m.mode_id) > 0
    WHERE tr.from_node = new.from_node
      AND tr.via_node = new.via_node
      AND tr.to_node = new.to_node
)
BEGIN
    SELECT RAISE(ABORT, 'Duplicate turn restriction exists for one or more modes');
END;

--#
-- Prevents updates that create duplicate mode coverage for the same turn,
-- even when modes are provided in different configurations/order
CREATE TRIGGER aequilibrae_turn_restrictions_no_overlap_update BEFORE UPDATE OF from_node, via_node, to_node, modes ON turn_restrictions
WHEN EXISTS (
    SELECT 1
    FROM turn_restrictions tr
    JOIN modes m
      ON INSTR(new.modes, m.mode_id) > 0
     AND INSTR(tr.modes, m.mode_id) > 0
    WHERE tr.restriction_id != old.restriction_id
      AND tr.from_node = new.from_node
      AND tr.via_node = new.via_node
      AND tr.to_node = new.to_node
)
BEGIN
    SELECT RAISE(ABORT, 'Duplicate turn restriction exists for one or more modes');
END;
