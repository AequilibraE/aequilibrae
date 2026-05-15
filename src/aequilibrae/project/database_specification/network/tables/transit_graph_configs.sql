--@ The *transit_graph_configs* table holds configuration parameters for a TransitGraph of a particular `period_id`

CREATE TABLE if not exists transit_graph_configs (period_id INTEGER UNIQUE NOT NULL PRIMARY KEY REFERENCES periods(period_id),
                                                  config    TEXT);

--@ Attributes follow
--#
INSERT INTO 'attributes_documentation' (name_table, attribute, description) VALUES('transit_graph_configs','period_id', 'The period this config is associated with.');
--#
INSERT INTO 'attributes_documentation' (name_table, attribute, description) VALUES('transit_graph_configs','mode_id', 'JSON string containing the configuration parameters.');
