.. _tables_about:

===========
About table
===========

The **about** table is the simplest of all tables in the AequilibraE project,
but it is the one table that contains the documentation about the project, and 
it is therefore crucial for project management and quality assurance during
modelling projects.

New AequilibraE projects will be created with 12 default metadata fields, listed
in the table below (with examples).

+--------------------------+-----------------------------------------------------------------------+
|  **Metadata**            |                         **Description**                               |
+==========================+=======================================================================+
| **model_name**           | Name of the model. (e.g. Alice Spring Freight Forecasting model)      |
+--------------------------+-----------------------------------------------------------------------+
| **region**               | Alice Springs, Northern Territory - Australia                         |
+--------------------------+-----------------------------------------------------------------------+
| **description**          | Freight model for the 200km circle centered in Alice Springs          |
+--------------------------+-----------------------------------------------------------------------+
| **author**               | John Doe                                                              |
+--------------------------+-----------------------------------------------------------------------+
| **license**              | GPL                                                                   |
+--------------------------+-----------------------------------------------------------------------+
| **scenario_name**        | Optimistic scenario                                                   |
+--------------------------+-----------------------------------------------------------------------+
| **scenario_description** | Contains hypothesis that we will find infinite petroleum here         |
+--------------------------+-----------------------------------------------------------------------+
| **year**                 | 2085                                                                  |
+--------------------------+-----------------------------------------------------------------------+
| **model_version**        | 2025.34.75                                                            |
+--------------------------+-----------------------------------------------------------------------+
| **project_id**           | f3a43347aa0d4755b2a7c4e06ae1dfca                                      |
+--------------------------+-----------------------------------------------------------------------+
| **aequilibrae_version**  | 0.7.4                                                                 |
+--------------------------+-----------------------------------------------------------------------+
| **projection**           | 4326                                                                  |
+--------------------------+-----------------------------------------------------------------------+

However, it is possible to create new information fields programatically. Once
the new field is added, the underlying database is altered and the field will
be present when the project is open during future use.

::

    p = Project()
    p.open('my/project/folder')
    p.about.add_info_field('my_super_relevant_field')
    p.about.my_super_relevant_field = 'super relevant information'
    p.about.write_back()

Changing existing fields can also be done programmatically

::

    p = Project()
    p.open('my/project/folder')
    p.about.scenario_name = 'Just a better scenario name'
    p.about.write_back()

In both cases, the new values for the information fields are only saved to disk
if *write_back()* is invoked.

We strongly recommend not to edit the information on **projection** and
**aequilibrae_version**, as these are fields that might or might not be used by
the software to produce valuable information to the user with regards to
opportunities for version upgrades.