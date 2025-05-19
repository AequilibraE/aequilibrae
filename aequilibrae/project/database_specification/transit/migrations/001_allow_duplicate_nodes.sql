-- The transit graph can create a lot of nodes on top of each other, these shouldn't be cannibalised, they also
-- shouldn't have to be scatter to fit in the database. So we remove the related triggers on the transit database. This
-- will make it more cumbersome to edit, however, we doubt anyone would want to edit this network as it is all
-- auto-generated anyway.

DROP TRIGGER IF EXISTS no_duplicate_node;
DROP TRIGGER IF EXISTS cannibalize_node_abort_when_centroid;
DROP TRIGGER IF EXISTS cannibalize_node;
