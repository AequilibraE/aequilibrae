.. _tables_about:

===========
About table
===========

The **about** table is the simplest of all tables in the AequilibraE project,
but it is the one table that contains the documentation about the project, and 
it is therefore crucial for project management and quality assurance during
modelling projects.

It is possible to create new information fields programatically. Once
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

.. note::

We strongly recommend not to edit the information on **projection** and
**aequilibrae_version**, as these are fields that might or might not be used by
the software to produce valuable information to the user with regards to
opportunities for version upgrades.