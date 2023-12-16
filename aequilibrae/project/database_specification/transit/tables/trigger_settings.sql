--@ This table intends to allow the enabled and disabling of certain triggers
--@


CREATE TABLE if not exists trigger_settings (name TEXT PRIMARY KEY, enabled INTEGER NOT NULL DEFAULT TRUE);
--#
INSERT INTO trigger_settings (name, enabled) VALUES('new_link_a_or_b_node', TRUE);
--#
INSERT INTO 'attributes_documentation' (name_table, attribute, description) VALUES('trigger_settings', 'name', 'name for trigger to query against');
--#
INSERT INTO 'attributes_documentation' (name_table, attribute, description) VALUES('trigger_settings', 'enabled', 'boolean value');
