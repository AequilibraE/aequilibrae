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