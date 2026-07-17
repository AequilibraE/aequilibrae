ALTER TABLE links ADD COLUMN fixed_cost_ab NUMERIC;
ALTER TABLE links ADD COLUMN fixed_cost_ba NUMERIC;
ALTER TABLE links ADD COLUMN other_attributes TEXT;
ALTER TABLE nodes ADD COLUMN other_attributes TEXT;

INSERT INTO attributes_documentation (name_table, attribute, description) VALUES('links','fixed_cost_*', 'Directional fixed costs (if any). Tolls, for example');
INSERT INTO attributes_documentation (name_table, attribute, description) VALUES('links','other_attributes', 'Other attributes of the link. Preferably in json format');
INSERT INTO attributes_documentation (name_table, attribute, description) VALUES('nodes','other_attributes', 'Other attributes of the node. Preferably in json format');
