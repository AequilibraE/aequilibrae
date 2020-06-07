-- Guarantees that the link_type records have a single letter for link_type_id

CREATE TRIGGER link_type_single_letter_update BEFORE UPDATE OF link_type_id ON "link_types"
WHEN
length(new.link_type_id)!= 1
BEGIN
    SELECT RAISE(ABORT, 'Link_type_id need to be a single letter');
END;

#
-- Guarantees that the link_type_id field is exactly 1 character long

CREATE TRIGGER link_type_single_letter_insert BEFORE INSERT ON "link_types"
WHEN
length(new.link_type_id)!= 1
BEGIN
    SELECT RAISE(ABORT, 'Link_type_id need to be a single letter');
END;

#

-- Prevents a link_type record to be changed when it is in use for any link

CREATE TRIGGER link_type_keep_if_in_use_updating BEFORE UPDATE OF link_type ON "link_types"
WHEN
(Select count(*) from links where old.link_type = link_type)>0

BEGIN
    SELECT RAISE(ABORT, 'Link_type is in use on your network. Cannot change it');
END;

#
-- Prevents a link_type record to be removed when it is in use for any link
CREATE TRIGGER link_type_keep_if_in_use_deleting BEFORE DELETE ON "link_types"
WHEN
(Select count(*) from links where old.link_type = link_type)>0

BEGIN
    SELECT RAISE(ABORT, 'Link_type is in use on your network. Cannot change it');
END;

#
-- Ensures an ALTERED link does not reference a non existing link_type
CREATE TRIGGER link_type_on_links_update BEFORE UPDATE OF 'link_type' ON links
WHEN
(select count(*) from link_types where new.link_type = link_type)<1
BEGIN
    SELECT RAISE(ABORT, 'Link_type need to exist in the link_types table in order to be used');
END;

#
-- Ensures an added link does not reference a non existing mode
CREATE TRIGGER link_type_on_links_insert BEFORE INSERT ON links
WHEN
(select count(*) from link_types where new.link_type = link_type)<1
BEGIN
    SELECT RAISE(ABORT, 'Link_type need to exist in the link_types table in order to be used');
END;

#
-- Ensures that we do not delete a protected link type
CREATE TRIGGER link_type_on_links_delete_protected_link_type BEFORE DELETE ON link_types
WHEN
old.link_type = "default" OR old.link_type = "centroid_connector"
BEGIN
    SELECT RAISE(ABORT, 'We cannot delete this link type');
END;

#
-- Ensures that we do not alter a protected link type
CREATE TRIGGER link_type_keep_if_protected_type BEFORE UPDATE OF link_type ON "link_types"
WHEN
old.link_type = "default" OR old.link_type = "centroid_connector"
BEGIN
    SELECT RAISE(ABORT, 'We cannot delete this link type');
END;

#
-- Keeps the two protected items unchanged in the database
CREATE TRIGGER link_type_id_keep_if_protected_type BEFORE UPDATE OF link_type_id ON "link_types"
WHEN
old.link_type = "default" OR old.link_type = "centroid_connector"
BEGIN

    SELECT RAISE(ABORT, 'We cannot alter this link type');
END;

#
-- Keeps the list of link_types at a node up-to-date when we try to manually change it in the modes table
create trigger link_type_on_nodes_table_update_nodes_link_type after update of link_types on nodes
begin

update nodes
set link_types = (select GROUP_CONCAT(link_type_id, '') from link_types where instr((
select GROUP_CONCAT(link_types.link_type_id, '') from links
INNER JOIN link_types ON links.link_type=link_types.link_type
where (links.a_node = new.node_id) or (links.b_node = new.node_id)), link_type_id) > 0)
where nodes.node_id=new.node_id;
end;


#
-- Keeps the list of link_types at a node up-to-date when we change link type for a link
create trigger link_type_on_nodes_table_update_links_link_type after update of link_type on links
begin

update nodes
set link_types = (select GROUP_CONCAT(link_type_id, '') from link_types where instr((
select GROUP_CONCAT(link_types.link_type_id, '') from links
INNER JOIN link_types ON links.link_type=link_types.link_type
where (links.a_node = new.a_node) or (links.b_node = new.a_node)), link_type_id) > 0)
where nodes.node_id=new.a_node;

update nodes
set link_types = (select GROUP_CONCAT(link_type_id, '') from link_types where instr((
select GROUP_CONCAT(link_types.link_type_id, '') from links
INNER JOIN link_types ON links.link_type=link_types.link_type
where (links.a_node = new.b_node) or (links.b_node = new.b_node)), link_type_id) > 0)
where nodes.node_id=new.b_node;

end;


#
-- Keeps the list of link_types at a node up-to-date when we change the a_node for a link
create trigger link_type_on_nodes_table_update_links_a_node after update of a_node on links
begin

update nodes
set link_types = (select GROUP_CONCAT(link_type_id, '') from link_types where instr((
select GROUP_CONCAT(link_types.link_type_id, '') from links
INNER JOIN link_types ON links.link_type=link_types.link_type
where (links.a_node = new.a_node) or (links.b_node = new.a_node)), link_type_id) > 0)
where nodes.node_id=new.a_node;

update nodes
set link_types = (select GROUP_CONCAT(link_type_id, '') from link_types where instr((
select GROUP_CONCAT(link_types.link_type_id, '') from links
INNER JOIN link_types ON links.link_type=link_types.link_type
where (links.a_node = old.a_node) or (links.b_node = old.a_node)), link_type_id) > 0)
where nodes.node_id=old.a_node;

end;

#
-- Keeps the list of link_types at a node up-to-date when we change the b_node for a link
create trigger link_type_on_nodes_table_update_links_b_node after update of b_node on links
begin

update nodes
set link_types = (select GROUP_CONCAT(link_type_id, '') from link_types where instr((
select GROUP_CONCAT(link_types.link_type_id, '') from links
INNER JOIN link_types ON links.link_type=link_types.link_type
where (links.a_node = new.b_node) or (links.b_node = new.b_node)), link_type_id) > 0)
where nodes.node_id=new.b_node;

update nodes
set link_types = (select GROUP_CONCAT(link_type_id, '') from link_types where instr((
select GROUP_CONCAT(link_types.link_type_id, '') from links
INNER JOIN link_types ON links.link_type=link_types.link_type
where (links.a_node = old.b_node) or (links.b_node = old.b_node)), link_type_id) > 0)
where nodes.node_id=old.b_node;

end;
