Components
===========

An AequilibraE project helds geometric information that can be accessed by the user in 
three different classes: ``Links``, ``Nodes``, and ``Zones``. We'll first cover these classes, and
then we'll go over the project components without geo-spatial information.

``project.network.links``
-------------------------

This method allows you to access the API resources to manipulate the 'links' table.
Each item in the 'links' table is a ``Link`` object.

.. code-block:: python

    >>> from shapely.geometry import LineString

    >>> project = create_example(project_path, "coquimbo")

    >>> project_links = project.network.links

    # Let's add a new field to our 'links' table
    >>> project_links.fields.add("my_field", "This is an example", "TEXT")
    
    # Let's add a new link to our project 
    >>> new_link = project_links.insert(modes="bctw", geometry=LineString([(-71.304754, -29.955233), (-71.304863, -29.954049)]))
    
    # The 'links' table has three fields which cannot be empty (i.e. with `NULL` values):
    # `link_id`, `direction`, and `modes`. When we create a node, `new` automatically
    # creates a `link_id`, and sets the default value (0) for direction. Thus, the modes
    # information should be added, otherwise, it will raise an error.

    # To delete one link from the project, you can use the following
    >>> project_links.delete(21332)

    # The `copy` function creates a copy of a specified link
    # It is very helpful case you want to split a link. 
    # You can check out in one of the usage examples.
    >>> link_copy = project_links.copy(10972)

.. admonition:: References

    * :ref:`modifications_on_links_layer`

.. seealso::
    
    * :func:`aequilibrae.project.network.links.Links`
        Class documentation 
    * :ref:`project_from_link_layer`
        Usage example
    * :ref:`editing_network_splitting_link`
        Usage example

``project.network.nodes``
-------------------------

This method allows you to access the API resources to manipulate the 'nodes' table.
Each item in the 'nodes' table is a ``Node`` object.

.. code-block:: python

    >>> from shapely.geometry import Point

    >>> project_nodes = project.network.nodes

    # To get one 'Node' object
    >>> node = project_nodes.get(10070)

    # We can check the existing fields for each node in the 'nodes' table
    >>> project_nodes.columns
    ('node_id', 'is_centroid', 'modes', 'link_types', 'geometry', 'osm_id')

    # Let's renumber this node
    >>> project_nodes.renumber(node_id=10070, new_id=1000)

    # A node can also be used to add a special generator
    # `new_centroid` returns a `Node` object.
    # Don't forget to add a geometry to your centroid if it's a new node
    # This centroid corresponds to the Port of Coquimbo!
    >>> centroid_id = project_nodes.new_centroid(2000, geometry=Point(-71.32, -29.94))
    >>> centroid = project_nodes.get(centroid_id)

    # As this centroid is not associated with a zone, we must tell AequilibraE the initial area around
    # the centroid to look for candidate nodes to which the centroid can connect.
    >>> project_nodes.connect_mode(node_id=centroid_id, area=centroid.geometry.buffer(0.01), mode_id="c")

    # Last but not less important, you can check your project nodes
    # `project_nodes.data` returns a geopandas GeoDataFrame.
    >>> nodes_data = project_nodes.data

    >>> # or if you want to check the coordinate of each node in the shape of
    >>> # a Pandas DataFrame
    >>> coords = project_nodes.lonlat
    >>> coords.head(3) # doctest: +NORMALIZE_WHITESPACE
      node_id        lon        lat
    0   10037 -71.315117 -29.996804
    1   10064 -71.336604 -29.949050
    2   10065 -71.336517 -29.949062

.. admonition:: References

    * :ref:`modifications_on_nodes_layer`

.. seealso::

    * :func:`aequilibrae.project.network.nodes.Nodes`
        Class documentation
    * :ref:`editing_network_nodes`
        Usage example


.. _project_zones:

``project.network.zones``
-------------------------

This method allows you to access the API resources to manipulate the 'zones' table.
Each item in the 'zones' table is a ``Zone`` object.

.. code-block:: python

    >>> from shapely.geometry import Polygon

    >>> project_zones = project.network.zones

    # Let's start this example by adding a new field to the 'zones' table
    >>> project_zones.fields.add("parking_spots", "Number of public parking spots", "INTEGER")

    # We can check if the new field was indeed created
    >>> project_zones.fields.all_fields() # doctest: +ELLIPSIS
    ['area', 'employment', 'geometry', 'name', 'parking_spots', 'population', 'zone_id']

    # Now let's get a zone and modify it
    >>> zone = project_zones.get(40)
    
    # By disconnecting the transit mode
    >>> project_zones.disconnect_mode("t")
    
    # Connecting the bicycle mode
    >>> project_zones.connect_mode("b")
    
    # And adding the number of public parking spots in the field we just created
    >>> project_zones.update(zone.zone_id, parking_spots=30)
    
    # The changes connecting / disconnecting modes reflect in the zone centroids
    # and can be seen in the 'nodes' table.

    # To return a dictionary with all 'Zone' objects in the model
    >>> {zone.zone_id: zone for zone in project_zones} # doctest: +ELLIPSIS
    {1: ..., ..., 133: ...}

    # If you want to delete a zone
    >>> project_zones.delete(38)

    # Or to add a new one
    >>> zone_extent = Polygon([(-71.3325, -29.9473), (-71.3283, -29.9473), (-71.3283, -29.9539), (-71.3325, -29.9539)])

    >>> new_zone_id = project_zones.insert(zone_id=38, geometry=zone_extent)

    # We can add a centroid to the zone we just created by specifying its location or
    # pass `None` to use the geometric center of the zone 
    >>> project_zones.add_centroid(new_zone_id, Point(-71.33, -29.95))

    # Finally, to return a geopandas GeoDataFrame with the project zones
    >>> zones = project_zones.data

    # To get a Shapely Polygon or Multipolygon with the entire zoning coverage
    >>> boundaries = project_zones.coverage()

    # And to get the nearest zone to a given geometry
    >>> project_zones.get_closest_zone(Point(-71.3336, -29.9490))
    57

    >>> project.close()

.. seealso::

    * :func:`aequilibrae.project.network.zones.Zones`
        Class documentation
    * :ref:`create_zones`
        Usage example

``project.about``
-----------------

This table object provides access to project metadata. Use the standard table API to add metadata
records or update existing values.

.. doctest::

    >>> project = create_example(my_folder_path / "about")

    >>> project.about.insert(infoname="my_new_field")
    'my_new_field'
    >>> project.about.update("my_new_field", infovalue="add some useful information about the field")

    # We can update data in an existing field
    >>> project.about.update("author", infovalue="Your Name")

    # Metadata is available as table data
    >>> "my_new_field" in project.about.data.infoname.values
    True

    >>> project.close()

.. seealso::

    * :func:`aequilibrae.project.about.About`
        Class documentation
    * :ref:`tables_about`
        Table documentation

``project.FieldEditor``
-----------------------

The ``FieldEditor`` allows the user to edit the project data tables, and it has two different purposes:

* Managing data tables, through the addition/deletion of fields
* Editing the tables' metadata (aka the description of each field)

This class is directly accessed from within the corresponding module one wants to edit.

.. doctest::

    >>> project = create_example(my_folder_path / "field_editor", "nauru")

    # We'll edit the fields in the 'nodes' table
    >>> node_fields = project.network.nodes.fields

    # To add a new field to the 'nodes' table
    >>> node_fields.add("my_new_field", "this is an example of AequilibraE's functionalities", "TEXT")

    # Don't forget to save these modifications
    >>> node_fields.save()

    # To edit the description of a field
    >>> node_fields.osm_id = "number of the osm node_id"

    # Or just to access the description of a field
    >>> node_fields.modes
    'Modes connected to the node'

    # One can also check all the fields in the 'nodes' table.
    >>> node_fields.all_fields() # doctest: +ELLIPSIS
    ['is_centroid', ..., 'my_new_field']

    >>> project.close()

All field descriptions are kept in the table 'attributes_documentation'.

.. seealso::

    *  :func:`aequilibrae.project.field_editor.FieldEditor`
        Class documentation

``project.log``
---------------

Every AequilibraE project contains a log file that holds information on all the project procedures.
It is possible to access the log file contents, as presented in the next code block.

.. doctest::

    >>> project = create_example(my_folder_path / "log", "nauru")

    >>> project_log = project.log()

    # Returns a list with all entries in the log file.
    >>> print(project_log.contents()) # doctest: +ELLIPSIS
    ['2021-01-01 15:52:03,945;aequilibrae;INFO ; Created project on D:/release/Sample models/nauru', ...]

    # If your project's log is getting cluttered, it is possible to clear it. 
    # Use this option wisely once the deletion of data in the log file can't be undone.
    >>> project_log.clear()

    >>> project.close()

.. seealso::
    
    * :func:`aequilibrae.log.Log`
        Class documentation
    * :ref:`useful-log-tips`
        Usage example
    
``project.matrices``
--------------------

This table provides access to all the matrices available in the model, which allows us to update the
records in the 'matrices' table. Each item in the 'matrices' table  is a ``MatrixRecord`` object.

.. doctest::

    >>> project = create_example(my_folder_path / "matrices")

    >>> matrices = project.matrices

    # One can also check all the project matrices as a Pandas' DataFrame
    >>> matrices.list() # doctest: +SKIP

    # We can add a new matrix
    >>> matrices.create(...) # doctest: +SKIP
    
    # If you're unsure if you have a matrix in your project, you can check if it exists.
    # This function will return `True` or `False`.
    >>> matrices.file_exists("demand_mc")
    True

    # To delete a matrix from the 'matrices' table, we can delete the record directly
    >>> matrices.delete_matrix("demand_mc")

    # Or by selecting the matrix and deleting it
    >>> mat_record = matrices.get("demand_mc") # doctest: +SKIP
    >>> matrices.delete_matrix(mat_record.name) # doctest: +SKIP

    # If a matrix was added or deleted by an external process, synchronise the
    # 'matrices' table to keep it organised. This will removed matrix records for files
    # that are not present.
    >>> matrices.sync()

    >>> project.close()

.. seealso::

    * :func:`aequilibrae.project.data.matrices.Matrices`
        Class documentation
    * :ref:`matrix_table`
        Table documentation

``project.network.link_types``
------------------------------

This method allows you to access the API resources to manipulate the 'link_types' table.
Each item in the 'link_types' table is a ``LinkType`` object.

.. doctest::

    >>> project = create_example(my_folder_path / "link_types", "coquimbo")

    >>> link_types = project.network.link_types

    # Create a new LinkType with ID 'A'    
    >>> link_types.insert(link_type_id="A", link_type="Arterial")  
    'A'

    # We can update information for the LinkType we just created
    >>> link_types.update("A", description="This is a description", speed=35)

    # To create a new field in the 'link_types' table, you can call the function `fields`
    # to return a FieldEditor instance, which can be edited
    >>> link_types.fields.add("my_new_field", "this is an example of AequilibraE's functionalities", "TEXT")

    # You can also remove a LinkType from a project using its `link_type_id`
    >>> link_types.delete("A")

    # To check all `LinkTypes` in the project as a dictionary whose keys are the `link_type_id`'s
    >>> link_types.all_types() # doctest: +SKIP
    {'z': <aequilibrae.project.network.link_type.LinkType object at 0x...>} 

    # There are two ways to get a LinkType from the 'link_types' table
    # using the `link_type_id`
    >>> get_link = link_types.get("p")

    # or using the `link_type`
    >>> get_link = link_types.get_by_name("primary")

    >>> project.close()

.. seealso::

    * :func:`aequilibrae.project.network.link_types.LinkTypes`
        Class documentation
    * :ref:`tables_link_types`
        Table documentation

``project.network.modes``
-------------------------

This method allows you to access the API resources to manipulate the 'modes' table.
Each item in 'modes' table is a ``Mode`` object.

.. doctest::

    >>> project = create_example(my_folder_path / "modes", "coquimbo")

    >>> modes = project.network.modes

    # We create a new mode
    >>> modes.insert(mode_id="k", mode_name="flying_car")
    'k'

    # We can continue editing the mode after adding it
    >>> modes.update("k", description="Like the one in the cartoons")

    # You can also remove a Mode from a project using its ``mode_id``
    >>> modes.delete("k")

    # To check all `Modes` in the project as a dictionary whose keys are the `mode_id`'s
    >>> modes.all_modes() # doctest: +SKIP
    {'b': <aequilibrae.project.network.mode.Mode object at 0x...>}

    # There are two ways to get a Mode from the 'modes' table
    # using the ``mode_id``
    >>> get_mode = modes.get("c")
    
    # or using the ``mode_name``
    >>> get_mode = modes.get_by_name("car")

    >>> project.close()

.. seealso::

    * :func:`aequilibrae.project.network.modes.Modes`
        Class documentation
    * :ref:`tables_modes`
        Table documentation

``project.network.periods``
---------------------------

This method allows you to access the API resources to manipulate the 'periods' table.
Each item in the 'periods' table is a ``Period`` object.

.. doctest::

    >>> project = create_example(my_folder_path / "periods", "coquimbo")

    >>> periods = project.network.periods

    # Let's add a new field to our 'periods' table
    >>> periods.fields.add("my_field", "This is field description", "TEXT")

    # To see all periods data as a Pandas' DataFrame
    >>> all_periods = periods.data

    # To add a new period
    >>> new_period_id = periods.new_period(2, 21600, 43200, "6AM to noon")

    # We can update the new period with a value for the new field
    >>> periods.update(new_period_id, my_field="hello world")

    # It is also possible to renumber a period
    >>> periods.renumber(new_period_id, 9)

    # And check the existing data fields for each period
    >>> periods.columns
    ('period_id', 'period_start', 'period_end', 'period_description', 'my_field')

    >>> project.close()

.. seealso::

    * :func:`aequilibrae.project.network.periods.Periods`
        Class documentation
    * :ref:`tables_period`
        Table documentation
