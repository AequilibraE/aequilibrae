-- TODO: allow arbitrary CRS
-- TODO: allow arbitrary column AND table names

-- note that sqlite only recognises 5 basic column affinities (TEXT, NUMERIC, INTEGER, REAL, BLOB); more specific declarations are ignored
-- the 'INTEGER PRIMARY KEY' column is always 64-bit signed integer, AND an alias for 'ROWID'.

-- Note that manually editing the ogc_fid will corrupt the spatial index. Therefore, we leave the
-- ogc_fid alone, and have a separate link_id and node_id, for network editors who have specific
-- requirements.

-- it is recommended to use the listed edit widgets in QGIS;

--
-- Triggers are grouped by the table which triggers their execution
-- 

-- Triggered by changes to links.
--

-- we use a before ordering here, as it is the only way to guarantee this will run before the nodeid update trigger.
-- when inserting a link endpoint to empty space, create a new node
--#
create INDEX IF NOT EXISTS links_a_node_idx ON links (a_node);

--#
create INDEX IF NOT EXISTS links_b_node_idx ON links (b_node);

--#
create INDEX IF NOT EXISTS links_link_type ON links (link_type);

--#
create INDEX IF NOT EXISTS nodes_node_id ON nodes (node_id);

--#
create trigger new_link_a_node before insert on links
  when
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
--#
create trigger new_link_b_node before insert on links
  when
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
--#
-- we use a before ordering here, as it is the only way to guarantee this will run before the nodeid update trigger.
-- when inserting a link endpoint to empty space, create a new node
create trigger update_link_a_node before update of geometry on links
  when
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
--#
create trigger update_link_b_node before update of geometry on links
  when
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
--#
  
create trigger new_link after insert on links
  begin
    -- Update a/b_node AFTER creating a link.
    update links
    set a_node = (
      select node_id
      from nodes
      where nodes.geometry = StartPoint(new.geometry) and
      (nodes.rowid in (
          select rowid from SpatialIndex where f_table_name = 'nodes' and
          search_frame = StartPoint(new.geometry)) or
        nodes.node_id = new.a_node))
    where links.rowid = new.rowid;
    update links
    set b_node = (
      select node_id
      from nodes
      where nodes.geometry = EndPoint(links.geometry) and
      (nodes.rowid in (
          select rowid from SpatialIndex where f_table_name = 'nodes' and
          search_frame = EndPoint(links.geometry)) or
        nodes.node_id = new.b_node))
    where links.rowid = new.rowid;
    update links
    set distance = GeodesicLength(new.geometry)
    where links.rowid = new.rowid;

    update links set
        link_id=(select max(link_id)+1 from links)
    where rowid=NEW.rowid and new.link_id is null;

    -- We update the modes for the node ID that just received a new link starting in it
    update nodes
    set modes = (select GROUP_CONCAT(mode_id, '') from modes where instr((
    select GROUP_CONCAT(modes, '') from links where (links.a_node = new.a_node) or (links.b_node = new.a_node))
    , mode_id) > 0)
    where nodes.node_id=new.a_node;

    -- We update the modes for the node ID that just received a new link ending in it
    update nodes
    set modes = (select GROUP_CONCAT(mode_id, '') from modes where instr((
    select GROUP_CONCAT(modes, '') from links where (links.a_node = new.b_node) or (links.b_node = new.b_node))
    , mode_id) > 0)
    where nodes.node_id=new.b_node;
  end;
--#
create trigger updated_link_geometry after update of geometry on links
  begin
  -- Update a/b_node AFTER moving a link.
  -- Note that if this TRIGGER is triggered by a node move, then the SpatialIndex may be out of date.
  -- This is why we also allow current a_node to persist.
    update links
    set a_node = (
      select node_id
      from nodes
      where nodes.geometry = StartPoint(new.geometry) and
      (nodes.rowid in (
          select rowid from SpatialIndex where f_table_name = 'nodes' and
          search_frame = StartPoint(new.geometry)) or
        nodes.node_id = new.a_node))
    where links.rowid = new.rowid;
    update links
    set b_node = (
      select node_id
      from nodes
      where nodes.geometry = EndPoint(links.geometry) and
      (nodes.rowid in (
          select rowid from SpatialIndex where f_table_name = 'nodes' and
          search_frame = EndPoint(links.geometry)) or
        nodes.node_id = new.b_node))
    where links.rowid = new.rowid;
    update links
    set distance = GeodesicLength(new.geometry)
    where links.rowid = new.rowid;

    -- now delete nodes which no-longer have attached links
    -- limit search to nodes which were attached to this link.
    delete from nodes
    where (node_id = old.a_node or node_id = old.b_node)
    --AND NOT (geometry = EndPoint(new.geometry) OR
    --         geometry = StartPoint(new.geometry))
    and node_id not in (
      select a_node
      from links
      where a_node is not null
      union all
      select b_node
      from links
      where b_node is not null);
  end;
--#

create trigger deleted_link after delete on links
  begin
-- delete lonely node AFTER link deleted
	Delete from Nodes
    where node_id = old.a_node and
           is_centroid != 1 and
           (select count(*) from Links where a_node = old.a_node or b_node = old.a_node) < 1;

	Delete from Nodes
    where node_id = old.b_node and
           is_centroid != 1 and
           (select count(*) from Links where a_node = old.b_node or b_node = old.b_node) < 1;

     -- We update the modes for the node ID that just lost a link starting in it
    update nodes
    set modes = (select GROUP_CONCAT(mode_id, '')
                 from modes
                 where instr((select GROUP_CONCAT(modes, '')
                              from links
                              where (links.a_node = old.a_node) or (links.b_node = old.a_node))
                             , mode_id) > 0)
    where nodes.node_id=old.a_node;

    -- We update the modes for the node ID that just lost a link ending in it
    update nodes
    set modes = (select GROUP_CONCAT(mode_id, '')
                 from modes
                 where instr((select GROUP_CONCAT(modes, '')
                              from links
                              where (links.a_node = old.b_node) or (links.b_node = old.b_node))
                             , mode_id) > 0)

    where nodes.node_id=old.b_node;
    end;
--#
-- when you move a node, move attached links
create trigger update_node_geometry after update of geometry on nodes
  begin
    update links
    set geometry = SetStartPoint(geometry,new.geometry)
    where a_node = new.node_id
    and StartPoint(geometry) != new.geometry;
    update links
    set geometry = SetEndPoint(geometry,new.geometry)
    where b_node = new.node_id
    and EndPoint(geometry) != new.geometry;
  end;
--#
-- when you move a node on top of another node, steal all links FROM that node, AND delete it.
-- be careful of merging the a_nodes of attached links to the new node
-- this may be better as a TRIGGER on links?
create trigger cannibalize_node_abort_when_centroid before update of geometry on nodes
  when
    -- detect another node in the new location
    (SELECT count(*)
    FROM nodes
    WHERE node_id != new.node_id
    AND geometry = new.geometry AND
    (is_centroid=1 OR new.is_centroid=1) AND
    ROWID IN (
      SELECT ROWID FROM SpatialIndex WHERE f_table_name = 'nodes' AND
      search_frame = new.geometry)) > 0
  BEGIN
       SELECT RAISE(ABORT,'Cannot cannibalize centroids');
  END;

--#
create trigger cannibalize_node before update of geometry on nodes
  when
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
--#
-- you may NOT CREATE a node on top of another node.
create trigger no_duplicate_node before insert on nodes
  when
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
--#
-- don't delete a node, unless no attached links
create trigger dont_delete_node before delete on nodes
  when (SELECT count(*) FROM links WHERE a_node = old.node_id OR b_node = old.node_id) > 0
  BEGIN
    SELECT raise(ABORT, 'Node cannot be deleted, it still has attached links.');
  END;
--#
-- when editing node_id, UPDATE connected links
create trigger updated_node_id after update of node_id on nodes
  begin
    update links set a_node = new.node_id
    where links.a_node = old.node_id;
    update links set b_node = new.node_id
    where links.b_node = old.node_id;
  end;
--#

-- Guarantees that link direction is one of the required values
create trigger links_direction_update before update of direction on links
when new.direction != -1 AND new.direction != 0 AND new.direction != 1
begin
  select RAISE(ABORT,'Link direction needs to be -1, 0 or 1');
end;

--#
create trigger links_direction_insert before insert on links
when new.direction != -1 AND new.direction != 0 AND new.direction != 1
begin
  select RAISE(ABORT,'Link direction needs to be -1, 0 or 1');
end;

--#
create trigger enforces_link_length_update after update of distance on links
begin
  update links set distance = GeodesicLength(new.geometry)
  where links.rowid = new.rowid;end;

--#
-- Guarantees that link direction is one of the required values
create trigger nodes_iscentroid_update before update of is_centroid on nodes
when new.is_centroid != 0 AND new.is_centroid != 1
begin
  select RAISE(ABORT,'is_centroid flag needs to be 0 or 1');
end;

--#
-- Deletes an empty node when marked no longer as a centroid
create trigger nodes_iscentroid_change_update after update of is_centroid on nodes
when new.is_centroid = 0 AND (SELECT count(*) FROM links WHERE a_node = new.node_id OR b_node = new.node_id) = 0
begin
  delete from nodes where node_id=new.node_id;
end;

--#
create trigger nodes_iscentroid_insert before insert on nodes
when new.is_centroid != 0 AND new.is_centroid != 1
begin
  select RAISE(ABORT,'is_centroid flag needs to be 0 or 1');
end;
