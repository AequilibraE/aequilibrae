create trigger enforces_route_link_length_update after update of distance on route_links
begin
  update route_links set distance = GeodesicLength(new.geometry)
  where route_links.rowid = new.rowid;
end;

--#
create trigger enforces_route_link_geo_update after update of geometry on route_links
begin
  update route_links set distance = GeodesicLength(new.geometry)
  where route_links.rowid = new.rowid;
end;

--#
create trigger new_route_link after insert on route_links
begin
  update route_links set distance = GeodesicLength(new.geometry)
  where route_links.rowid = new.rowid;
end;