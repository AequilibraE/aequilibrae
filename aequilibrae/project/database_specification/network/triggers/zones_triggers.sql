-- In AequilibraE a zone's centroid is the node whose node_id matches the zone_id, so any node that
-- shares its ID with an existing zone MUST be flagged as a centroid.
--
-- Nodes that land on an existing zone ID (either because they were created or renumbered) are tagged
-- automatically, and so are the nodes a new zone is created on top of. Removing the centroid flag from
-- a node while its zone exists is blocked, as that would leave the zone without a centroid (and would
-- delete the node altogether when it has no links attached to it).

--#
-- Tags a node created on top of an existing zone ID as a centroid
create trigger aequilibrae_new_node_zone_centroid after insert on nodes
when new.is_centroid != 1 and exists (select 1 from zones where zones.zone_id = new.node_id)
begin
  update nodes set is_centroid = 1 where nodes.rowid = new.rowid;
end;

--#
-- Tags a node renumbered onto an existing zone ID as a centroid
create trigger aequilibrae_updated_node_id_zone_centroid after update of node_id on nodes
when new.is_centroid != 1 and exists (select 1 from zones where zones.zone_id = new.node_id)
begin
  update nodes set is_centroid = 1 where nodes.rowid = new.rowid;
end;

--#
-- Guarantees that a node cannot stop being a centroid while its zone exists
create trigger aequilibrae_nodes_iscentroid_zone_update before update of is_centroid on nodes
when new.is_centroid != 1 and exists (select 1 from zones where zones.zone_id = new.node_id)
begin
  select RAISE(ABORT,'Nodes that share their ID with a zone must be centroids. Delete the zone first');
end;

--#
-- Tags the node a new zone was created on top of as a centroid
create trigger aequilibrae_new_zone_centroid after insert on zones
begin
  update nodes set is_centroid = 1 where nodes.node_id = new.zone_id and nodes.is_centroid != 1;
end;

--#
-- Tags the node a zone was renumbered on top of as a centroid
create trigger aequilibrae_updated_zone_id_centroid after update of zone_id on zones
begin
  update nodes set is_centroid = 1 where nodes.node_id = new.zone_id and nodes.is_centroid != 1;
end;
