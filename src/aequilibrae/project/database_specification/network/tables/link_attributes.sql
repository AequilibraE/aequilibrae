--@ The link_attributes table holds every link attribute that is not mandatory for the
--@ network topology: names, speeds, capacities, costs and all user-defined fields.
--@ It has exactly one record for each record in the **links** table, matched on **link_id**.
--@
--@ Triggers keep the two tables in sync: a record is created when a link is added,
--@ deleted when its link is deleted and renumbered when its link_id changes. Records
--@ cannot be deleted, or created for a link that does not exist, directly.
--@
--@ Fields added through the API (``project.tables.links.fields``) are created in this
--@ table, and the **links_full** view is rebuilt to show them. Their descriptions are
--@ stored in *attributes_documentation* under the **links** table name, alongside
--@ the descriptions of the mandatory link fields.

CREATE TABLE if not exists link_attributes (link_id           INTEGER PRIMARY KEY,
                                            'name'            TEXT,
                                            lanes_ab          INTEGER,
                                            lanes_ba          INTEGER,
                                            speed_ab          INTEGER,
                                            speed_ba          INTEGER,
                                            fixed_cost_ab     NUMERIC,
                                            fixed_cost_ba     NUMERIC,
                                            travel_time_ab    NUMERIC,
                                            travel_time_ba    NUMERIC,
                                            capacity_ab       NUMERIC,
                                            capacity_ba       NUMERIC);

--#
INSERT INTO 'attributes_documentation' (name_table, attribute, description) VALUES('links','name', 'Name of the street/link');
--#
INSERT INTO 'attributes_documentation' (name_table, attribute, description) VALUES('links','fixed_cost_*', 'Directional fixed costs (if any). Tolls, for example');
--#
INSERT INTO 'attributes_documentation' (name_table, attribute, description) VALUES('links','speed_*', 'Directional speeds (if allowed)');
--#
INSERT INTO 'attributes_documentation' (name_table, attribute, description) VALUES('links','capacity_*', 'Directional link capacities (if allowed)');
--#
INSERT INTO 'attributes_documentation' (name_table, attribute, description) VALUES('links','travel_time_*', 'Directional free-flow travel time (if allowed)');