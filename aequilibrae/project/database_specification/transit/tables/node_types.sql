--@ The *node_types* table holds information about the available
--@ node types in the network.
--@
--@ The **node_type** field corresponds to the node type, and it is the
--@ table's primary key
--@
--@ The **node_type_id** field presents the identification of the node type
--@
--@ The **description** field holds the description of the node type
--@


CREATE TABLE  if not exists node_types (node_type     VARCHAR UNIQUE NOT NULL PRIMARY KEY,
                                        node_type_id  VARCHAR UNIQUE NOT NULL,
                                        description   VARCHAR);

--#
INSERT INTO 'node_types' (node_type, node_type_id, description) VALUES('default', 'y', 'Default general node type');
--#
INSERT INTO 'node_types' (node_type, node_type_id, description) VALUES('od', 'n', 'Origin/Desination node type');
--#
INSERT INTO 'node_types' (node_type, node_type_id, description) VALUES('origin', 'o', 'Origin node type');
--#
INSERT INTO 'node_types' (node_type, node_type_id, description) VALUES('destination', 'd', 'Desination node type');
--#
INSERT INTO 'node_types' (node_type, node_type_id, description) VALUES('stop', 's', 'Stop node type');
--#
INSERT INTO 'node_types' (node_type, node_type_id, description) VALUES('alighting', 'a', 'Alighting node type');
--#
INSERT INTO 'node_types' (node_type, node_type_id, description) VALUES('boarding', 'b', 'Boarding node type');
       --#
INSERT INTO 'node_types' (node_type, node_type_id, description) VALUES('walking', 'w', 'Walking node type');

--@ Attributes follow
--#
INSERT INTO 'attributes_documentation' (name_table, attribute, description) VALUES('node_types','node_type', 'Node type name. E.g stop or boarding');
--#
INSERT INTO 'attributes_documentation' (name_table, attribute, description) VALUES('node_types','node_type_id', 'Single letter identifying the mode. E.g. a, for alighting');
--#
INSERT INTO 'attributes_documentation' (name_table, attribute, description) VALUES('node_types','description', 'Description of the same. E.g. Stop nodes connect ODs and walking nodes to boarding and alighting nodes via boarding and alighting links.');
