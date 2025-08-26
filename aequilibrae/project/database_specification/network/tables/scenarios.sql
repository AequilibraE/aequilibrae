CREATE TABLE IF NOT EXISTS scenarios (scenario_name TEXT UNIQUE NOT NULL, description TEXT);

--#
INSERT INTO 'scenarios' (scenario_name, description) VALUES('root', 'The default, and root, scenario for an AequilbraE project. The name "root" is treated as a special case.');

--@ Attributes follow
--#
INSERT INTO 'attributes_documentation' (name_table, attribute, description) VALUES('scenarios', 'scenario_name', 'The scenario folder name.');
--#
INSERT INTO 'attributes_documentation' (name_table, attribute, description) VALUES('scenarios', 'description', 'Description of the scenario');
