--@ The links table holds the geometry and the mandatory fields of all links available in the
--@ aequilibrae network model, regardless of the modes allowed on them. All other link attributes,
--@ including every user-defined field, live in the **link_attributes** table.
--@
--@ The two tables are presented together as the **links_full** view, which is the layer to read
--@ and edit links with (e.g. in QGIS). The view is editable: inserts, updates and deletes made on
--@ it are routed to the right table by triggers. It is rebuilt whenever a link field is added.
--@
--@ All information on the fields a_node and b_node correspond to a entries in
--@ the node_id field in the nodes table. They are automatically managed with
--@ triggers as the user edits the network. Updates that would make either field
--@ inconsistent with the corresponding geometry endpoint are rejected.
--@
--@ The **modes** field is a concatenation of all the ids (mode_id) of the models allowed
--@ on each link, and map directly to the mode_id field in the **Modes** table. A mode
--@ can only be added to a link if it exists in the **Modes** table.
--@
--@ The **link_type** corresponds to the *link_type* field from the *link_types* table.
--@ As it is the case for modes, a link_type can only be assigned to a link if it exists
--@ in the **link_types** table.
--@
--@ The fields **length**, **node_a** and **node_b** are automatically
--@ updated by triggers based in the links' geometries and node positions. Link length
--@ is always measured in **meters**.
--@
--@ The table is indexed on **link_id** (its primary key), **node_a** and **node_b**.


CREATE TABLE  if not exists links (ogc_fid           INTEGER PRIMARY KEY,
                                   link_id           INTEGER NOT NULL UNIQUE,
                                   a_node            INTEGER,
                                   b_node            INTEGER,
                                   direction         INTEGER NOT NULL DEFAULT 0,
                                   distance          NUMERIC,
                                   modes             TEXT    NOT NULL,
                                   link_type         TEXT    REFERENCES link_types(link_type) ON update RESTRICT ON delete RESTRICT,
                                   CHECK(TYPEOF(link_id) == 'integer'),
                                   CHECK(link_id > 0),
                                   CHECK(TYPEOF(a_node) IN ('integer', 'null')),
                                   CHECK(TYPEOF(b_node) IN ('integer', 'null')),
                                   CHECK(TYPEOF(direction) == 'integer'),
                                   CHECK(LENGTH(modes)>0),
                                   CHECK(direction IN (-1, 0, 1)));

--#
select AddGeometryColumn( 'links', 'geometry', 4326, 'LINESTRING', 'XY', 1);

--#
CREATE UNIQUE INDEX idx_link ON links (link_id);

--#
SELECT CreateSpatialIndex( 'links' , 'geometry' );

--#
CREATE INDEX idx_link_anode ON links (a_node);

--#
CREATE INDEX idx_link_bnode ON links (b_node);

--#
CREATE INDEX idx_link_modes ON links (modes);

--#
CREATE INDEX idx_link_link_type ON links (link_type);

--#
CREATE INDEX idx_links_a_node_b_node ON links (a_node, b_node);

--#
INSERT INTO 'attributes_documentation' (name_table, attribute, description) VALUES('links','link_id', 'Unique link ID');
--#
INSERT INTO 'attributes_documentation' (name_table, attribute, description) VALUES('links','a_node', 'origin node for the link');
--#
INSERT INTO 'attributes_documentation' (name_table, attribute, description) VALUES('links','b_node', 'destination node for the link');
--#
INSERT INTO 'attributes_documentation' (name_table, attribute, description) VALUES('links','direction', 'Flow direction allowed on the link');
--#
INSERT INTO 'attributes_documentation' (name_table, attribute, description) VALUES('links','distance', 'length of the link');
--#
INSERT INTO 'attributes_documentation' (name_table, attribute, description) VALUES('links','modes', 'modes allowed on the link');
--#
INSERT INTO 'attributes_documentation' (name_table, attribute, description) VALUES('links','link_type', 'Link type');
