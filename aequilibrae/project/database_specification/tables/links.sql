CREATE TABLE 'links' (ogc_fid      INTEGER PRIMARY KEY,
                      link_id      INTEGER NOT NULL UNIQUE,
                      a_node       INTEGER NOT NULL ,
                      b_node       INTEGER NOT NULL ,
                      direction    INTEGER NOT NULL DEFAULT 0,
                      distance     NUMERIC NOT NULL ,
                      modes        TEXT    NOT NULL,
                      link_type    TEXT    REFERENCES link_types(link_type) ON update RESTRICT ON delete RESTRICT,
                      'name'       NUMERIC,
                      speed_ab     NUMERIC,
                      speed_ba     NUMERIC,
                      capacity_ab  NUMERIC,
                      capacity_ba  NUMERIC
                     );

#
select AddGeometryColumn( 'links', 'geometry', 4326, 'LINESTRING', 'XY', 1);
#
INSERT INTO 'attributes_documentation' (name_table, attribute, description) VALUES('links','link_id', 'Unique link ID');
#
INSERT INTO 'attributes_documentation' (name_table, attribute, description) VALUES('links','a_node', 'origin node for the link');
#
INSERT INTO 'attributes_documentation' (name_table, attribute, description) VALUES('links','b_node', 'destination node for the link');
#
INSERT INTO 'attributes_documentation' (name_table, attribute, description) VALUES('links','direction', 'Flow direction allowed on the link');
#
INSERT INTO 'attributes_documentation' (name_table, attribute, description) VALUES('links','distance', 'length of the link');
#
INSERT INTO 'attributes_documentation' (name_table, attribute, description) VALUES('links','modes', 'modes allowed on the link');
#
INSERT INTO 'attributes_documentation' (name_table, attribute, description) VALUES('links','link_type', 'Link type');
#
INSERT INTO 'attributes_documentation' (name_table, attribute, description) VALUES('links','name', 'Name of the street/link');
#
INSERT INTO 'attributes_documentation' (name_table, attribute, description) VALUES('links','speed_ab', 'AB directional speed (if allowed)');
#
INSERT INTO 'attributes_documentation' (name_table, attribute, description) VALUES('links','speed_ba', 'BA directional speed (if allowed)');
#
INSERT INTO 'attributes_documentation' (name_table, attribute, description) VALUES('links','capacity_ab', 'AB directional link capacity (if allowed)');
#
INSERT INTO 'attributes_documentation' (name_table, attribute, description) VALUES('links','capacity_ba', 'BA directional link capacity (if allowed)');

