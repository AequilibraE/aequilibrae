.. _tables_zones:

Zones table
===========

The **zones** table exists only for the user's convenience, as it is likely to
be required in a full-blown model. As it is not required to exist, the table
created with each new model has a very limited number of fields, as follows:

* zone_id
* area (in m\ :sup:`2`)
* name
* population
* employment
* geometry

As it is to be expected, zone_id must be unique, but the remaining fields are
not restricted in any form.

The API for manipulation of the zones table and each one of its records is
consistent with what exists to manipulate the other fields in the database.

As it happens with links and nodes, zones also have geometries associated with
them, and in this case they are of the type **MultiPolygon**.

An example of manipulating the zones table follows:

::

    p = Project()
    p.open('my/project/folder')
    zones = p.zones

    # We edit the fields for a particular zone
    zone_downtown = zones.get(1)
    zone_downtown.population = 637
    zone_downtown.employment = 10039
    zone_downtown.save()

    fields = zones.fields

    # We can also add one more field to the table
    fields.add('parking_spots', 'Total licensed parking spots', 'INTEGER')
