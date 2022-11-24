--@ This table is the junction of two different types of links:
--@    * Network links (from the links table) that can be traversed by walk
--@    * Access links connecting stops to the network
--@
--@ This network is static and should be re-created any time there are changes
--@ in the transit data (e.g. import of a new GTFS feed), or the network changes.
--@
--@ It is important to understand that the number of links in this network is
--@ substantially larger than in the **links** table because it has links
--@ connecting each stop to the physical network, as well as links
--@ connecting transit stops stops directly to each other whenever
--@ they are very close.
--@
--@ Physical links are also broken to allow stops to be connected by walk in the
--@ middle of links, where stops are actually located. We attempt to reduce the
--@ number of link breaks by combining access links into the same point whenever
--@ possible and without penalizing agents with substantially longer walk
--@ distances.


CREATE TABLE IF NOT EXISTS walk_network (
	walk_link	INTEGER  NOT NULL PRIMARY KEY AUTOINCREMENT,
	from_node	INTEGER  NOT NULL,
	to_node	    INTEGER  NOT NULL,
	distance    REAL     NOT NULL,
	ref_link	INTEGER  NOT NULL default 0
);

SELECT AddGeometryColumn('transit_walk', 'geometry', SRID_PARAMETER, 'LINESTRING', 'XY', 1);

SELECT CreateSpatialIndex('transit_walk' , 'geometry');

CREATE INDEX IF NOT EXISTS transit_walk_from_node ON transit_walk (from_node);

CREATE INDEX IF NOT EXISTS transit_walk_to_node ON transit_walk (to_node);
