-- Keeps link_attributes in sync with links: exactly one attribute record per link, matched on link_id.
--
-- The editable links_full view and its INSTEAD OF triggers are not defined here. They depend on the
-- user-defined columns of link_attributes, so they are generated in Python (links_view.py) and rebuilt
-- whenever a link field is added.

--#
create trigger aequilibrae_new_link_attributes after insert on links
  begin
    insert or ignore into link_attributes (link_id) values (new.link_id);
  end;

--#
create trigger aequilibrae_deleted_link_attributes after delete on links
  begin
    delete from link_attributes where link_id = old.link_id;
  end;

--#
create trigger aequilibrae_renumbered_link_attributes after update of link_id on links
  when new.link_id != old.link_id
  begin
    update link_attributes set link_id = new.link_id where link_id = old.link_id;
  end;

--#
-- Attribute records are removed with their links, never on their own
create trigger aequilibrae_link_attributes_delete_guard before delete on link_attributes
  when exists (select 1 from links where link_id = old.link_id)
  begin
    select raise(abort, 'link_attributes records are deleted with their links. Delete the link instead');
  end;

--#
create trigger aequilibrae_link_attributes_insert_guard before insert on link_attributes
  when not exists (select 1 from links where link_id = new.link_id)
  begin
    select raise(abort, 'link_attributes records require an existing link_id in the links table');
  end;

--#
create trigger aequilibrae_link_attributes_link_id_guard before update of link_id on link_attributes
  when not exists (select 1 from links where link_id = new.link_id)
  begin
    select raise(abort, 'link_attributes.link_id follows links.link_id. Change the link_id on links instead');
  end;
