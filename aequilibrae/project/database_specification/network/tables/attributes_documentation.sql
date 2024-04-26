--@ The *attributes_documentation* table holds information about attributes
--@ in the tables links, link_types, modes, nodes, and zones.
--@
--@ By default, these attributes are all documented, but further
--@ attribues can be added into the table.
--@
--@ The **name_table** field holds the name of the table that has the attribute
--@
--@ The **attribute** field holds the name of the attribute
--@
--@ The **description** field holds the description of the attribute
--@
--@ It is possible to have one attribute with the same name in two
--@ different tables. However, one cannot have two attibutes with the
--@ same name within the same table.


CREATE TABLE  if not exists attributes_documentation (name_table  TEXT NOT NULL,
                                                      attribute   TEXT NOT NULL,
                                                      description TEXT,
                                                      UNIQUE (name_table, attribute)
                                                      );

--#
CREATE INDEX idx_attributes ON attributes_documentation (name_table, attribute);
