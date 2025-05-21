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
