CREATE TABLE 'nodes' (ogc_fid     INTEGER PRIMARY KEY,
                      node_id     INTEGER UNIQUE NOT NULL,
                      is_centroid INTEGER        NOT NULL DEFAULT 0,
                      modes       TEXT,
                      link_types  TEXT);

#
SELECT AddGeometryColumn( 'nodes', 'geometry', 4326, 'POINT', 'XY', 1);