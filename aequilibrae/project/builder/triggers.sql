CREATE TRIGGER new_link_a_node BEFORE INSERT ON links
  WHEN
    (SELECT count(*)
    FROM nodes
    WHERE nodes.geometry = StartPoint(new.geometry) AND
    (nodes.ROWID IN (
        SELECT ROWID FROM SpatialIndex WHERE f_table_name = 'nodes' AND
        search_frame = StartPoint(new.geometry)) OR
      nodes.node_id = new.a_node)) = 0
  BEGIN
    INSERT INTO nodes (node_id, geometry)
    VALUES ((SELECT coalesce(max(node_id) + 1,1) from nodes),
            StartPoint(new.geometry));
  END;
#
CREATE TRIGGER new_link_b_node BEFORE INSERT ON links
  WHEN
    (SELECT count(*)
    FROM nodes
    WHERE nodes.geometry = EndPoint(new.geometry) AND
    (nodes.ROWID IN (
        SELECT ROWID FROM SpatialIndex WHERE f_table_name = 'nodes' AND
        search_frame = EndPoint(new.geometry)) OR
      nodes.node_id = new.b_node)) = 0
  BEGIN
    INSERT INTO nodes (node_id, geometry)
    VALUES ((SELECT coalesce(max(node_id) + 1,1) from nodes),
            EndPoint(new.geometry));
  END;
#
-- we use a before ordering here, as it is the only way to guarantee this will run before the nodeid update trigger.
-- when inserting a link endpoint to empty space, create a new node
CREATE TRIGGER update_link_a_node BEFORE UPDATE OF geometry ON links
  WHEN
    (SELECT count(*)
    FROM nodes
    WHERE nodes.geometry = StartPoint(new.geometry) AND
    (nodes.ROWID IN (
        SELECT ROWID FROM SpatialIndex WHERE f_table_name = 'nodes' AND
        search_frame = StartPoint(new.geometry)) OR
      nodes.node_id = new.a_node)) = 0
  BEGIN
    INSERT INTO nodes (node_id, geometry)
    VALUES ((SELECT coalesce(max(node_id) + 1,1) from nodes),
            StartPoint(new.geometry));
  END;
#
CREATE TRIGGER update_link_b_node BEFORE UPDATE OF geometry ON links
  WHEN
    (SELECT count(*)
    FROM nodes
    WHERE nodes.geometry = EndPoint(new.geometry) AND
    (nodes.ROWID IN (
        SELECT ROWID FROM SpatialIndex WHERE f_table_name = 'nodes' AND
        search_frame = EndPoint(new.geometry)) OR
      nodes.node_id = new.b_node)) = 0
  BEGIN
    INSERT INTO nodes (node_id, geometry)
    VALUES ((SELECT coalesce(max(node_id) + 1,1) from nodes),
            EndPoint(new.geometry));
  END;
#
  
CREATE TRIGGER new_link AFTER INSERT ON links
  BEGIN
    -- Update a/b_node AFTER creating a link.
    UPDATE links
    SET a_node = (
      SELECT node_id
      FROM nodes
      WHERE nodes.geometry = StartPoint(new.geometry) AND
      (nodes.ROWID IN (
          SELECT ROWID FROM SpatialIndex WHERE f_table_name = 'nodes' AND
          search_frame = StartPoint(new.geometry)) OR
        nodes.node_id = new.a_node))
    WHERE links.ROWID = new.ROWID;
    UPDATE links
    SET b_node = (
      SELECT node_id
      FROM nodes
      WHERE nodes.geometry = EndPoint(links.geometry) AND
      (nodes.ROWID IN (
          SELECT ROWID FROM SpatialIndex WHERE f_table_name = 'nodes' AND
          search_frame = EndPoint(links.geometry)) OR
        nodes.node_id = new.b_node))
    WHERE links.ROWID = new.ROWID;
  END;
#
CREATE TRIGGER updated_link_geometry AFTER UPDATE OF geometry ON links
  BEGIN
  -- Update a/b_node AFTER moving a link.
  -- Note that if this TRIGGER is triggered by a node move, then the SpatialIndex may be out of date.
  -- This is why we also allow current a_node to persist.
    UPDATE links
    SET a_node = (
      SELECT node_id
      FROM nodes
      WHERE nodes.geometry = StartPoint(new.geometry) AND
      (nodes.ROWID IN (
          SELECT ROWID FROM SpatialIndex WHERE f_table_name = 'nodes' AND
          search_frame = StartPoint(new.geometry)) OR
        nodes.node_id = new.a_node))
    WHERE links.ROWID = new.ROWID;
    UPDATE links
    SET b_node = (
      SELECT node_id
      FROM nodes
      WHERE nodes.geometry = EndPoint(links.geometry) AND
      (nodes.ROWID IN (
          SELECT ROWID FROM SpatialIndex WHERE f_table_name = 'nodes' AND
          search_frame = EndPoint(links.geometry)) OR
        nodes.node_id = new.b_node))
    WHERE links.ROWID = new.ROWID;
    
    -- now delete nodes which no-longer have attached links
    -- limit search to nodes which were attached to this link.
    DELETE FROM nodes
    WHERE (node_id = old.a_node OR node_id = old.b_node)
    --AND NOT (geometry = EndPoint(new.geometry) OR
    --         geometry = StartPoint(new.geometry))
    AND node_id not in (    
      SELECT a_node
      FROM links
      WHERE a_node is NOT NULL
      union all
      SELECT b_node
      FROM links
      WHERE b_node is NOT NULL);
  END;
#

-- delete lonely node AFTER link deleted
CREATE TRIGGER deleted_link AFTER delete ON links
  BEGIN
    DELETE FROM nodes
    WHERE node_id NOT IN (
      SELECT a_node
      FROM links
      union all
      SELECT b_node
      FROM links);
    END;
#
-- when moving OR creating a link, don't allow it to duplicate an existing link.
-- TODO

-- Triggered by change of nodes
--

-- when you move a node, move attached links
CREATE TRIGGER update_node_geometry AFTER UPDATE OF geometry ON nodes
  BEGIN
    UPDATE links
    SET geometry = SetStartPoint(geometry,new.geometry)
    WHERE a_node = new.node_id
    AND StartPoint(geometry) != new.geometry;
    UPDATE links
    SET geometry = SetEndPoint(geometry,new.geometry)
    WHERE b_node = new.node_id
    AND EndPoint(geometry) != new.geometry;
  END;
#
-- when you move a node on top of another node, steal all links FROM that node, AND delete it.
-- be careful of merging the a_nodes of attached links to the new node
-- this may be better as a TRIGGER on links?
CREATE TRIGGER cannibalise_node BEFORE UPDATE OF geometry ON nodes
  WHEN
    -- detect another node in the new location
    (SELECT count(*)
    FROM nodes
    WHERE node_id != new.node_id
    AND geometry = new.geometry AND
    ROWID IN (
      SELECT ROWID FROM SpatialIndex WHERE f_table_name = 'nodes' AND
      search_frame = new.geometry)) > 0
  BEGIN
    -- grab a_nodes belonging to node in same location
    UPDATE links
    SET a_node = new.node_id
    WHERE a_node = (SELECT node_id
                    FROM nodes
                    WHERE node_id != new.node_id
                    AND geometry = new.geometry AND
                    ROWID IN (
                      SELECT ROWID FROM SpatialIndex WHERE f_table_name = 'nodes' AND
                      search_frame = new.geometry));
    -- grab b_nodes belonging to node in same location
    UPDATE links
    SET b_node = new.node_id
    WHERE b_node = (SELECT node_id
                    FROM nodes
                    WHERE node_id != new.node_id
                    AND geometry = new.geometry AND
                    ROWID IN (
                      SELECT ROWID FROM SpatialIndex WHERE f_table_name = 'nodes' AND
                      search_frame = new.geometry));
    -- delete nodes in same location
    DELETE FROM nodes
    WHERE node_id != new.node_id
    AND geometry = new.geometry AND
    ROWID IN (
      SELECT ROWID FROM SpatialIndex WHERE f_table_name = 'nodes' AND
      search_frame = new.geometry);
  END;
#
-- you may NOT CREATE a node on top of another node.
CREATE TRIGGER no_duplicate_node BEFORE INSERT ON nodes
  WHEN
    (SELECT count(*)
    FROM nodes
    WHERE nodes.node_id != new.node_id
    AND nodes.geometry = new.geometry AND
    nodes.ROWID IN (
      SELECT ROWID FROM SpatialIndex WHERE f_table_name = 'nodes' AND
      search_frame = new.geometry)) > 0
  BEGIN
    -- todo: change this to perform a cannibalisation instead.
    SELECT raise(ABORT, 'Cannot create on-top of other node');
  END;
#
-- TODO: cannot CREATE node NOT attached.

-- don't delete a node, unless no attached links
CREATE TRIGGER dont_delete_node BEFORE DELETE ON nodes
  WHEN (SELECT count(*) FROM links WHERE a_node = old.node_id OR b_node = old.node_id) > 0
  BEGIN
    SELECT raise(ABORT, 'Node cannot be deleted, it still has attached links.');
  END;
#
-- don't CREATE a node, unless on a link endpoint
-- TODO
-- CREATE BEFORE WHERE spatial index AND PointN()

-- when editing node_id, UPDATE connected links
CREATE TRIGGER updated_node_id AFTER UPDATE OF node_id ON nodes
  BEGIN
    UPDATE links SET a_node = new.node_id
    WHERE links.a_node = old.node_id;
    UPDATE links SET b_node = new.node_id
    WHERE links.b_node = old.node_id;
  END;
#