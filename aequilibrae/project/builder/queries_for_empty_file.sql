-- TODO: allow arbitrary CRS

-- basic network setup
-- alternatively use ogr2ogr

-- note that sqlite only recognises 5 basic column affinities (TEXT, NUMERIC, INTEGER, REAL, BLOB); more specific declarations are ignored
-- the 'INTEGER PRIMARY KEY' column is always 64-bit signed integer, AND an alias for 'ROWID'.

-- Note that manually editing the ogc_fid will corrupt the spatial index. Therefore, we leave the
-- ogc_fid alone, and have a separate link_id and node_id, for network editors who have specific
-- requirements.

-- Queries create structure for the supply model (network) and for the demand side. It is all under development
--
--
#
-- Creates the metadata table for the AequilibraE project
CREATE TABLE 'model_metadata' (item VARCHAR PRIMARY KEY, --
                      text VARCHAR);
#
-- Creates a log for model changes
CREATE TABLE 'model_log' (ogc_fid INTEGER PRIMARY KEY,
                          log_type VARCHAR, -- Automatic Vs. Manual
                          change_date DATETIME,
                          description VARCHAR -- Long description for change
                          );

#
-- Creates a log for model changes
CREATE TABLE 'model_run_log' (ogc_fid INTEGER PRIMARY KEY,
                              log_type VARCHAR, -- Automatic Vs. Manual
                              run_description VARCHAR, -- Long description for mode
                              run_date DATETIME
                              );


#
CREATE TABLE 'modes' (mode_id VARCHAR UNIQUE PRIMARY KEY NOT NULL, -- 1-character mode identifier
                     description VARCHAR, -- Long description for mode
                     pce REAL NOT NULL,    -- Passenger Car Equivalent
                     CHECK(typeof("mode_id") = "text" AND
                            length("mode_id") = 1 AND
                            pce > 0
                          )
                    );

#
-- it is recommended to use the listed edit widgets in QGIS;
CREATE TABLE 'links' (
  ogc_fid INTEGER PRIMARY KEY, -- Hidden widget
  link_id INTEGER UNIQUE NOT NULL, -- Text edit widget with 'Not null' constraint
  modes VARCHAR, -- Modes allowed on this link
  a_node INTEGER, -- Text edit widget, with 'editable' unchecked
  b_node INTEGER, -- Text edit widget, with 'editable' unchecked
  direction INTEGER, -- Range widget, 'Editable', min=0, max=2, step=1, default=0
  capacity_ab REAL,
  capacity_ba REAL,
  speed_ab REAL,
  speed_ba REAL,
  'length' REAL
);
#
SELECT AddGeometryColumn( 'links', 'geometry', DEFAULT_CRS, 'LINESTRING', 'XY' );
#
SELECT CreateSpatialIndex( 'links' , 'geometry' );
#
CREATE INDEX links_a_node_idx ON links (a_node);
#
CREATE INDEX links_b_node_idx ON links (b_node);
#
-- it is recommended to use the listed edit widgets in QGIS
CREATE TABLE 'nodes' (
  ogc_fid INTEGER PRIMARY KEY, -- Hidden widget
  node_id INTEGER UNIQUE NOT NULL, -- Text edit widget with 'Not null' constraint
  centroid INTEGER UNIQUE NOT NULL default 0 -- Node can be a centroid (centroid=1), or not (centroid=0)
  CHECK(centroid >= 0 AND
        centroid <= 1
        )
);
#
SELECT AddGeometryColumn( 'nodes', 'geometry', DEFAULT_CRS, 'POINT', 'XY' );
#
SELECT CreateSpatialIndex( 'nodes' , 'geometry' );
#
-- Creates the view for the list of centroids
CREATE VIEW centroids AS 
    SELECT *
    FROM nodes
    WHERE nodes.centroid = 1
        AND A.bool=1
        AND B.bool=0;
        
        