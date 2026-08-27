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
create INDEX IF NOT EXISTS aequilibrae_links_a_node_idx ON links (a_node);

--#
create INDEX IF NOT EXISTS aequilibrae_links_b_node_idx ON links (b_node);

--#
create INDEX IF NOT EXISTS aequilibrae_links_link_type ON links (link_type);

--#
create INDEX IF NOT EXISTS aequilibrae_nodes_node_id ON nodes (node_id);

--#
-- a_node and b_node are derived from link geometry. Reject changes which do not
-- preserve that relationship, while allowing AequilibraE's maintenance triggers
-- to update the fields after a geometry or node-id change.
create trigger aequilibrae_links_a_node_update before update of a_node on links
  when new.a_node is not old.a_node
  and not exists (
    select 1
    from nodes
    where nodes.node_id = new.a_node
    and nodes.geometry = StartPoint(new.geometry))
  begin
    select raise(ABORT, 'a_node does not match the start point of link geometry');
  end;

--#
create trigger aequilibrae_links_b_node_update before update of b_node on links
  when new.b_node is not old.b_node
  and not exists (
    select 1
    from nodes
    where nodes.node_id = new.b_node
    and nodes.geometry = EndPoint(new.geometry))
  begin
    select raise(ABORT, 'b_node does not match the end point of link geometry');
  end;

--#
create trigger aequilibrae_new_link_a_node before insert on links
  when
    (SELECT count(*)
    FROM nodes
    WHERE nodes.geometry = StartPoint(new.geometry) AND
    (nodes.ROWID IN (
        SELECT pkid FROM "idx_nodes_geometry" WHERE xmin <= MbrMaxX(StartPoint(new.geometry)) AND xmax >= MbrMinX(StartPoint(new.geometry)) AND ymin <= MbrMaxY(StartPoint(new.geometry)) AND ymax >= MbrMinY(StartPoint(new.geometry))) OR
      nodes.node_id = new.a_node)) = 0
  BEGIN
    INSERT INTO nodes (node_id, geometry)
    VALUES ((SELECT coalesce(max(node_id) + 1,1) from nodes),
            StartPoint(new.geometry));
  END;
--#
create trigger aequilibrae_new_link_b_node before insert on links
  when
    (SELECT count(*)
    FROM nodes
    WHERE nodes.geometry = EndPoint(new.geometry) AND
    (nodes.ROWID IN (
        SELECT pkid FROM "idx_nodes_geometry" WHERE xmin <= MbrMaxX(EndPoint(new.geometry)) AND xmax >= MbrMinX(EndPoint(new.geometry)) AND ymin <= MbrMaxY(EndPoint(new.geometry)) AND ymax >= MbrMinY(EndPoint(new.geometry))) OR
      nodes.node_id = new.b_node)) = 0
  BEGIN
    INSERT INTO nodes (node_id, geometry)
    VALUES ((SELECT coalesce(max(node_id) + 1,1) from nodes),
            EndPoint(new.geometry));
  END;
--#
-- we use a before ordering here, as it is the only way to guarantee this will run before the nodeid update trigger.
-- when inserting a link endpoint to empty space, create a new node
create trigger aequilibrae_update_link_a_node before update of geometry on links
  when
    (SELECT count(*)
    FROM nodes
    WHERE nodes.geometry = StartPoint(new.geometry) AND
    (nodes.ROWID IN (
        SELECT pkid FROM "idx_nodes_geometry" WHERE xmin <= MbrMaxX(StartPoint(new.geometry)) AND xmax >= MbrMinX(StartPoint(new.geometry)) AND ymin <= MbrMaxY(StartPoint(new.geometry)) AND ymax >= MbrMinY(StartPoint(new.geometry))) OR
      nodes.node_id = new.a_node)) = 0
  BEGIN
    INSERT INTO nodes (node_id, geometry)
    VALUES ((SELECT coalesce(max(node_id) + 1,1) from nodes),
            StartPoint(new.geometry));
  END;
--#
create trigger aequilibrae_update_link_b_node before update of geometry on links
  when
    (SELECT count(*)
    FROM nodes
    WHERE nodes.geometry = EndPoint(new.geometry) AND
    (nodes.ROWID IN (
        SELECT pkid FROM "idx_nodes_geometry" WHERE xmin <= MbrMaxX(EndPoint(new.geometry)) AND xmax >= MbrMinX(EndPoint(new.geometry)) AND ymin <= MbrMaxY(EndPoint(new.geometry)) AND ymax >= MbrMinY(EndPoint(new.geometry))) OR
      nodes.node_id = new.b_node)) = 0
  BEGIN
    INSERT INTO nodes (node_id, geometry)
    VALUES ((SELECT coalesce(max(node_id) + 1,1) from nodes),
            EndPoint(new.geometry));
  END;
--#
  
create trigger aequilibrae_new_link after insert on links
  begin
    -- Update a_node AFTER creating a link.
    update links
    set a_node = (
      select node_id
      from nodes
      where nodes.geometry = StartPoint(new.geometry) and
      (nodes.rowid in (
          SELECT pkid FROM "idx_nodes_geometry" WHERE xmin <= MbrMaxX(StartPoint(new.geometry)) AND xmax >= MbrMinX(StartPoint(new.geometry)) AND ymin <= MbrMaxY(StartPoint(new.geometry)) AND ymax >= MbrMinY(StartPoint(new.geometry))) or
        nodes.node_id = new.a_node))
    where links.rowid = new.rowid;
    update links
    set b_node = (
      select node_id
      from nodes
      where nodes.geometry = EndPoint(new.geometry) and
      (nodes.rowid in (
          SELECT pkid FROM "idx_nodes_geometry" WHERE xmin <= MbrMaxX(EndPoint(new.geometry)) AND xmax >= MbrMinX(EndPoint(new.geometry)) AND ymin <= MbrMaxY(EndPoint(new.geometry)) AND ymax >= MbrMinY(EndPoint(new.geometry))) or
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
create trigger aequilibrae_updated_link_geometry after update of geometry on links
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
          SELECT pkid FROM "idx_nodes_geometry" WHERE xmin <= MbrMaxX(StartPoint(new.geometry)) AND xmax >= MbrMinX(StartPoint(new.geometry)) AND ymin <= MbrMaxY(StartPoint(new.geometry)) AND ymax >= MbrMinY(StartPoint(new.geometry))) or
        nodes.node_id = new.a_node))
    where links.rowid = new.rowid;
    update links
    set b_node = (
      select node_id
      from nodes
      where nodes.geometry = EndPoint(new.geometry) and
      (nodes.rowid in (
          SELECT pkid FROM "idx_nodes_geometry" WHERE xmin <= MbrMaxX(EndPoint(new.geometry)) AND xmax >= MbrMinX(EndPoint(new.geometry)) AND ymin <= MbrMaxY(EndPoint(new.geometry)) AND ymax >= MbrMinY(EndPoint(new.geometry))) or
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

create trigger aequilibrae_deleted_link after delete on links
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
-- Node identity has its own maintenance workflow. Changing both identity and
-- geometry at once would make the sibling AFTER triggers order-dependent.
create trigger aequilibrae_node_id_geometry_update_guard before update of node_id, geometry on nodes
  when new.node_id is not old.node_id
  and new.geometry is not old.geometry
  begin
    select raise(ABORT, 'node_id must be updated separately from node geometry');
  end;

--#
-- Demoting an empty centroid deletes it. If geometry is named in the same UPDATE,
-- SpatiaLite may maintain the spatial index after that deletion and leave a ghost
-- entry. Move the centroid first, then demote it; linked centroids can be moved and
-- demoted atomically because the cleanup trigger will not delete them.
create trigger aequilibrae_empty_centroid_geometry_update before update of geometry on nodes
  when old.is_centroid = 1
  and new.is_centroid = 0
  and not exists (
    select 1
    from links
    where a_node = old.node_id or b_node = old.node_id)
  begin
    select raise(ABORT, 'an empty centroid must be demoted separately from moving it');
  end;

--#
-- Reject merging a centroid with any other node. Keeping this validation BEFORE
-- the row update guarantees that it runs before the mutating AFTER trigger below.
create trigger aequilibrae_cannibalize_node_abort_when_centroid before update of geometry on nodes
  when exists (
    select 1
    from nodes as collision
    where collision.ROWID != old.ROWID
    and collision.geometry = new.geometry
    and (collision.is_centroid = 1 or old.is_centroid = 1 or new.is_centroid = 1)
    and collision.ROWID in (
      SELECT pkid FROM "idx_nodes_geometry" WHERE xmin <= MbrMaxX(new.geometry) AND xmax >= MbrMinX(new.geometry) AND ymin <= MbrMaxY(new.geometry) AND ymax >= MbrMinY(new.geometry)))
  BEGIN
       SELECT RAISE(ABORT,'Cannot cannibalize centroids');
  END;

--#
-- Move a node and, if another node occupies the destination, merge it into the
-- moving node. These statements must remain in one trigger and in this order:
-- transfer links, delete the replaced node, then move attached link geometries.
create trigger aequilibrae_update_node_geometry after update of geometry on nodes
  when new.geometry is not old.geometry
  begin
    UPDATE links
    SET a_node = new.node_id
    WHERE a_node IN (SELECT collision.node_id
                     FROM nodes as collision
                     WHERE collision.ROWID != new.ROWID
                     AND collision.geometry = new.geometry
                     AND collision.ROWID IN (
                       SELECT pkid FROM "idx_nodes_geometry" WHERE xmin <= MbrMaxX(new.geometry) AND xmax >= MbrMinX(new.geometry) AND ymin <= MbrMaxY(new.geometry) AND ymax >= MbrMinY(new.geometry)));

    UPDATE links
    SET b_node = new.node_id
    WHERE b_node IN (SELECT collision.node_id
                     FROM nodes as collision
                     WHERE collision.ROWID != new.ROWID
                     AND collision.geometry = new.geometry
                     AND collision.ROWID IN (
                       SELECT pkid FROM "idx_nodes_geometry" WHERE xmin <= MbrMaxX(new.geometry) AND xmax >= MbrMinX(new.geometry) AND ymin <= MbrMaxY(new.geometry) AND ymax >= MbrMinY(new.geometry)));

    DELETE FROM nodes
    WHERE ROWID != new.ROWID
    AND geometry = new.geometry AND
    ROWID IN (
      SELECT pkid FROM "idx_nodes_geometry" WHERE xmin <= MbrMaxX(new.geometry) AND xmax >= MbrMinX(new.geometry) AND ymin <= MbrMaxY(new.geometry) AND ymax >= MbrMinY(new.geometry));

    UPDATE links
    SET geometry = SetStartPoint(geometry,new.geometry)
    WHERE a_node = new.node_id
    AND StartPoint(geometry) != new.geometry;

    UPDATE links
    SET geometry = SetEndPoint(geometry,new.geometry)
    WHERE b_node = new.node_id
    AND EndPoint(geometry) != new.geometry;
  end;
--#
-- you may NOT CREATE a node on top of another node.
create trigger aequilibrae_no_duplicate_node before insert on nodes
  when
    (SELECT count(*)
    FROM nodes
    WHERE nodes.node_id != new.node_id
    AND nodes.geometry = new.geometry AND
    nodes.ROWID IN (
      SELECT pkid FROM "idx_nodes_geometry" WHERE xmin <= MbrMaxX(new.geometry) AND xmax >= MbrMinX(new.geometry) AND ymin <= MbrMaxY(new.geometry) AND ymax >= MbrMinY(new.geometry))) > 0
  BEGIN
    -- todo: change this to perform a cannibalisation instead.
    SELECT raise(ABORT, 'Cannot create on-top of other node');
  END;
--#
-- don't delete a node, unless no attached links
create trigger aequilibrae_dont_delete_node before delete on nodes
  when (SELECT count(*) FROM links WHERE a_node = old.node_id OR b_node = old.node_id) > 0
  BEGIN
    SELECT raise(ABORT, 'Node cannot be deleted, it still has attached links.');
  END;
--#
-- when editing node_id, UPDATE connected links
create trigger aequilibrae_updated_node_id after update of node_id on nodes
  begin
    update links set a_node = new.node_id
    where links.a_node = old.node_id;
    update links set b_node = new.node_id
    where links.b_node = old.node_id;
  end;
--#

-- Guarantees that link direction is one of the required values
create trigger aequilibrae_links_direction_update before update of direction on links
when new.direction != -1 AND new.direction != 0 AND new.direction != 1
begin
  select RAISE(ABORT,'Link direction needs to be -1, 0 or 1');
end;

--#
create trigger aequilibrae_links_direction_insert before insert on links
when new.direction != -1 AND new.direction != 0 AND new.direction != 1
begin
  select RAISE(ABORT,'Link direction needs to be -1, 0 or 1');
end;

--#
create trigger aequilibrae_enforces_link_length_update after update of distance on links
begin
  update links set distance = GeodesicLength(new.geometry)
  where links.rowid = new.rowid;end;

--#
-- Guarantees that link direction is one of the required values
create trigger aequilibrae_nodes_iscentroid_update before update of is_centroid on nodes
when new.is_centroid != 0 AND new.is_centroid != 1
begin
  select RAISE(ABORT,'is_centroid flag needs to be 0 or 1');
end;

--#
-- Deletes an empty node when marked no longer as a centroid
create trigger aequilibrae_nodes_iscentroid_change_update after update of is_centroid on nodes
when old.is_centroid = 1
AND new.is_centroid = 0
AND not exists (
  SELECT 1
  FROM links
  WHERE a_node IN (old.node_id, new.node_id)
     OR b_node IN (old.node_id, new.node_id))
begin
  delete from nodes where ROWID = new.ROWID;
end;

--#
create trigger aequilibrae_nodes_iscentroid_insert before insert on nodes
when new.is_centroid != 0 AND new.is_centroid != 1
begin
  select RAISE(ABORT,'is_centroid flag needs to be 0 or 1');
end;
