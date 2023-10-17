-- Prevents a link_type record to be changed when it is in use for any link

CREATE TRIGGER link_type_keep_if_in_use_updating BEFORE UPDATE OF link_type ON "link_types"
WHEN
    (SELECT count(*) FROM links WHERE old.link_type = link_type)>0
BEGIN
    SELECT RAISE(ABORT, 'Link_type is in use on your network. Cannot change it');
END;

--#
-- Prevents a link_type record to be removed when it is in use for any link
CREATE TRIGGER link_type_keep_if_in_use_deleting BEFORE DELETE ON "link_types"
WHEN
    (SELECT count(*) FROM links WHERE old.link_type = link_type)>0
BEGIN
    SELECT RAISE(ABORT, 'Link_type is in use on your network. Cannot change it');
END;

--#
-- Ensures an ALTERED link does not reference a non existing link_type
CREATE TRIGGER link_type_on_links_update BEFORE UPDATE OF 'link_type' ON links
WHEN
    (SELECT count(*) FROM link_types WHERE new.link_type = link_type)<1
BEGIN
    SELECT RAISE(ABORT, 'Link_type need to exist in the link_types table in order to be used');
END;

--#
-- Ensures an added link does not reference a non existing mode
CREATE TRIGGER link_type_on_links_insert BEFORE INSERT ON links
WHEN
    (SELECT count(*) FROM link_types WHERE new.link_type = link_type)<1
BEGIN
    SELECT RAISE(ABORT, 'Link_type need to exist in the link_types table in order to be used');
END;

--#
-- Ensures that we do not delete a protected link type
CREATE TRIGGER link_type_on_links_delete_protected_link_type BEFORE DELETE ON link_types
WHEN
    old.link_type = "default" OR old.link_type = "centroid_connector"
BEGIN
    SELECT RAISE(ABORT, 'We cannot delete this link type');
END;

--#
-- Ensures that we do not alter a protected link type
CREATE TRIGGER link_type_keep_if_protected_type BEFORE UPDATE OF link_type ON "link_types"
WHEN
    old.link_type = "default" OR old.link_type = "centroid_connector"
BEGIN
    SELECT RAISE(ABORT, 'We cannot delete this link type');
END;

--#
-- Keeps the two protected items unchanged in the database
CREATE TRIGGER link_type_id_keep_if_protected_type BEFORE UPDATE OF link_type_id ON "link_types"
WHEN
    old.link_type = "default" OR old.link_type = "centroid_connector"
BEGIN
    SELECT RAISE(ABORT, 'We cannot alter this link type');
END;

--#
-- Keeps the list of link_types at a node up-to-date when we try to manually change it in the modes table
CREATE TRIGGER link_type_on_nodes_table_update_nodes_link_type AFTER UPDATE of link_types ON nodes
BEGIN
    UPDATE nodes
          SET link_types = (SELECT GROUP_CONCAT(link_type_id, '') 
          FROM link_types WHERE instr((SELECT GROUP_CONCAT(link_types.link_type_id, '') 
                                       FROM links 
                                       INNER JOIN link_types ON links.link_type=link_types.link_type
                                       WHERE (links.a_node = new.node_id) OR (links.b_node = new.node_id)
                                       ), link_type_id) > 0)
          WHERE nodes.node_id=new.node_id;
END;

--#
-- Keeps the list of link_types at a node up-to-date when we change link type for a link
CREATE TRIGGER link_type_on_nodes_table_update_links_link_type AFTER UPDATE of link_type ON links
BEGIN
    UPDATE nodes
        SET link_types = (SELECT GROUP_CONCAT(link_type_id, '') 
                          FROM link_types 
                          WHERE instr((SELECT GROUP_CONCAT(link_types.link_type_id, '') 
                                       FROM links 
                                       INNER JOIN link_types ON links.link_type=link_types.link_type
                                       WHERE (links.a_node = new.a_node) OR (links.b_node = new.a_node)
                                       ), link_type_id) > 0)
        WHERE nodes.node_id=new.a_node;

    UPDATE nodes
        SET link_types = (SELECT GROUP_CONCAT(link_type_id, '') 
                          FROM link_types 
                          WHERE instr((SELECT GROUP_CONCAT(link_types.link_type_id, '') 
                                       FROM links 
                                       INNER JOIN link_types ON links.link_type=link_types.link_type
                                       WHERE (links.a_node = new.b_node) OR (links.b_node = new.b_node)
                                       ), link_type_id) > 0)
        WHERE nodes.node_id=new.b_node;
END;


--#
-- Keeps the list of link_types at a node up-to-date when we change the a_node for a link
CREATE TRIGGER link_type_on_nodes_table_update_links_a_node AFTER UPDATE of a_node ON links
BEGIN
    UPDATE nodes
        SET link_types = (SELECT GROUP_CONCAT(link_type_id, '') 
                          FROM link_types 
                          WHERE instr((SELECT GROUP_CONCAT(link_types.link_type_id, '') 
                                       FROM links 
                                       INNER JOIN link_types ON links.link_type=link_types.link_type
                                       WHERE (links.a_node = new.a_node) OR (links.b_node = new.a_node)
                                       ), link_type_id) > 0)
        WHERE nodes.node_id=new.a_node;
    
    UPDATE nodes
        SET link_types = (SELECT GROUP_CONCAT(link_type_id, '') 
                          FROM link_types 
                          WHERE instr((SELECT GROUP_CONCAT(link_types.link_type_id, '') 
                                       FROM links
                                       INNER JOIN link_types ON links.link_type=link_types.link_type
                                       WHERE (links.a_node = old.a_node) OR (links.b_node = old.a_node)
                                       ), link_type_id) > 0)
        WHERE nodes.node_id=old.a_node;
END;

--#
-- Keeps the list of link_types at a node up-to-date when we change the b_node for a link
CREATE TRIGGER link_type_on_nodes_table_update_links_b_node AFTER UPDATE of b_node ON links
BEGIN
UPDATE nodes
    SET link_types = (SELECT GROUP_CONCAT(link_type_id, '')
                      FROM link_types
                      WHERE instr((SELECT GROUP_CONCAT(link_types.link_type_id, '')
                                   FROM links
                                   INNER JOIN link_types ON links.link_type=link_types.link_type
                                   WHERE (links.a_node = new.b_node) OR (links.b_node = new.b_node)
                                   ), link_type_id) > 0)
    WHERE nodes.node_id=new.b_node;

UPDATE nodes
    SET link_types = (SELECT GROUP_CONCAT(link_type_id, '')
                      FROM link_types WHERE instr((SELECT GROUP_CONCAT(link_types.link_type_id, '')
                                                   FROM links
                                                   INNER JOIN link_types ON links.link_type=link_types.link_type
                                                   WHERE (links.a_node = old.b_node) OR (links.b_node = old.b_node)
                                                   ), link_type_id) > 0)
    WHERE nodes.node_id=old.b_node;
END;
