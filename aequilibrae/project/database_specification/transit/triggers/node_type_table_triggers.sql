-- Guarantees that the node_type records have a single letter for node_type_id

CREATE TRIGGER node_type_single_letter_update BEFORE UPDATE OF node_type_id ON "node_types"
WHEN
    length(new.node_type_id)!= 1
BEGIN
    SELECT RAISE(ABORT, 'Node_type_id need to be a single letter');
END;

--#
-- Guarantees that the node_type_id field is exactly 1 character long

CREATE TRIGGER node_type_single_letter_insert BEFORE INSERT ON "node_types"
WHEN
    length(new.node_type_id)!= 1
BEGIN
    SELECT RAISE(ABORT, 'Node_type_id need to be a single letter');
END;

--#

-- Prevents a node_type record to be changed when it is in use for any node

CREATE TRIGGER node_type_keep_if_in_use_updating BEFORE UPDATE OF node_type ON "node_types"
WHEN
    (SELECT count(*) FROM nodes WHERE old.node_type = node_type)>0
BEGIN
    SELECT RAISE(ABORT, 'Node_type is in use on your network. Cannot change it');
END;

--#
-- Prevents a node_type record to be removed when it is in use for any node
CREATE TRIGGER node_type_keep_if_in_use_deleting BEFORE DELETE ON "node_types"
WHEN
    (SELECT count(*) FROM nodes WHERE old.node_type = node_type)>0
BEGIN
    SELECT RAISE(ABORT, 'Node_type is in use on your network. Cannot change it');
END;

--#
-- Ensures an ALTERED node does not reference a non existing node_type
CREATE TRIGGER node_type_on_nodes_update BEFORE UPDATE OF 'node_type' ON nodes
WHEN
    (SELECT count(*) FROM node_types WHERE new.node_type = node_type)<1
BEGIN
    SELECT RAISE(ABORT, 'Node_type need to exist in the node_types table in order to be used');
END;

--#
-- Ensures an added node does not reference a non existing mode
CREATE TRIGGER node_type_on_nodes_insert BEFORE INSERT ON nodes
WHEN
    (SELECT count(*) FROM node_types WHERE new.node_type = node_type)<1
BEGIN
    SELECT RAISE(ABORT, 'Node_type need to exist in the node_types table in order to be used');
END;

--#
-- Ensures that we do not delete a protected node type
CREATE TRIGGER node_type_on_nodes_delete_protected_node_type BEFORE DELETE ON node_types
WHEN
    old.node_type = "default" OR old.node_type = "centroid_connector"
BEGIN
    SELECT RAISE(ABORT, 'We cannot delete this node type');
END;

--#
-- Ensures that we do not alter a protected node type
CREATE TRIGGER node_type_keep_if_protected_type BEFORE UPDATE OF node_type ON "node_types"
WHEN
    old.node_type = "default" OR old.node_type = "centroid_connector"
BEGIN
    SELECT RAISE(ABORT, 'We cannot delete this node type');
END;

--#
-- Keeps the two protected items unchanged in the database
CREATE TRIGGER node_type_id_keep_if_protected_type BEFORE UPDATE OF node_type_id ON "node_types"
WHEN
    old.node_type = "default" OR old.node_type = "centroid_connector"
BEGIN
    SELECT RAISE(ABORT, 'We cannot alter this node type');
END;


-- %%%%%%%%%%%% Modified from source `network/triggers/link_type_table_triggers
--#
-- Keeps the list of node_types at a node up-to-date when we try to manually change it in the modes table
CREATE TRIGGER node_type_on_nodes_table_update_nodes_node_type AFTER UPDATE of node_types ON nodes
BEGIN
    UPDATE nodes
          SET node_types = (SELECT GROUP_CONCAT(node_type_id, '')
          FROM node_types WHERE instr((SELECT GROUP_CONCAT(node_types.node_type_id, '')
                                       FROM nodes
                                       INNER JOIN node_types ON nodes.node_type=node_types.node_type
                                       ), node_type_id) > 0)
          WHERE nodes.node_id=new.node_id;
END;
-- %%%%%%%%%%%%

--#
-- Keeps the list of node_types at a node up-to-date when we change the a_node for a node
CREATE TRIGGER node_type_on_nodes_table_update_nodes_a_node AFTER UPDATE of a_node ON nodes
BEGIN
    UPDATE nodes
        SET node_types = (SELECT GROUP_CONCAT(node_type_id, '')
                          FROM node_types
                          WHERE instr((SELECT GROUP_CONCAT(node_types.node_type_id, '')
                                       FROM nodes
                                       INNER JOIN node_types ON nodes.node_type=node_types.node_type
                                       WHERE (nodes.a_node = new.a_node) OR (nodes.b_node = new.a_node)
                                       ), node_type_id) > 0)
        WHERE nodes.node_id=new.a_node;
    
    UPDATE nodes
        SET node_types = (SELECT GROUP_CONCAT(node_type_id, '')
                          FROM node_types
                          WHERE instr((SELECT GROUP_CONCAT(node_types.node_type_id, '')
                                       FROM nodes
                                       INNER JOIN node_types ON nodes.node_type=node_types.node_type
                                       WHERE (nodes.a_node = old.a_node) OR (nodes.b_node = old.a_node)
                                       ), node_type_id) > 0)
        WHERE nodes.node_id=old.a_node;
END;

--#
-- Keeps the list of node_types at a node up-to-date when we change the b_node for a node
CREATE TRIGGER node_type_on_nodes_table_update_nodes_b_node AFTER UPDATE of b_node ON nodes
BEGIN
UPDATE nodes
    SET node_types = (SELECT GROUP_CONCAT(node_type_id, '')
                      FROM node_types
                      WHERE instr((SELECT GROUP_CONCAT(node_types.node_type_id, '')
                                   FROM nodes
                                   INNER JOIN node_types ON nodes.node_type=node_types.node_type
                                   WHERE (nodes.a_node = new.b_node) OR (nodes.b_node = new.b_node)
                                   ), node_type_id) > 0)
    WHERE nodes.node_id=new.b_node;

UPDATE nodes
    SET node_types = (SELECT GROUP_CONCAT(node_type_id, '')
                      FROM node_types WHERE instr((SELECT GROUP_CONCAT(node_types.node_type_id, '')
                                                   FROM nodes
                                                   INNER JOIN node_types ON nodes.node_type=node_types.node_type
                                                   WHERE (nodes.a_node = old.b_node) OR (nodes.b_node = old.b_node)
                                                   ), node_type_id) > 0)
    WHERE nodes.node_id=old.b_node;
END;
