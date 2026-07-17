Importing and exporting the network
===================================

AequilibraE can populate a project network from OpenStreetMap, Overture Maps,
GMNS files, and link layers. It can also export an existing project network to
GMNS. This page documents the public import APIs and the behavior of the new
OSM/Overture import pipeline.

.. _network_importer_optional_dependencies:

Optional dependencies
---------------------

The OSM and Overture importers use optional packages. Install them with the
``create`` extra before using these importers::

    pip install aequilibrae[create]

The relevant optional packages are:

* ``osmnx`` for OSM Overpass downloads and OSMnx simplification;
* ``pyrosm`` for local ``.osm.pbf`` imports;
* ``overturemaps`` for Overture Maps cloud downloads;
* ``neatnet`` for neatnet simplification.

.. _importing_from_osm:

Importing from OpenStreetMap
----------------------------

Use :func:`aequilibrae.project.network.network.Network.import_from_osm` to create
an AequilibraE network from OpenStreetMap. Exactly one source selector must be
provided:

* ``place_name`` for an OSM place lookup through Overpass;
* ``model_area`` for an EPSG:4326 polygon downloaded through Overpass;
* ``pbf_path`` for a local ``.osm.pbf`` extract read with ``pyrosm``.

Examples::

    project.network.import_from_osm(place_name="Nauru")

    from shapely.geometry import box

    project.network.import_from_osm(
        model_area=box(-112.185, 36.59, -112.179, 36.60),
        modes=("car", "walk"),
        simplify=False,
    )

    project.network.import_from_osm(
        pbf_path="/path/to/extract.osm.pbf",
        modes=("car", "transit", "bicycle", "walk"),
        simplify="osmnx",
    )

``.osm``, ``.osm.bz2`` and XML inputs are not supported directly. Convert them
to ``.osm.pbf`` first, for example with ``osmium``::

    osmium cat input.osm -o output.osm.pbf

The importer preserves the OSM ``highway`` value as ``link_type``. It does not
apply a link-type allow-list; filtering is controlled by the ``modes`` argument.
OSM tags that do not map to real project table columns are stored as JSON in
``other_attributes``.

Overpass imports write the downloaded payload and a manifest under
``<project>/downloaded data/osm-overpass/``. Local PBF imports do not create a
download cache because the source file is already local.

.. _importing_from_overture:

Importing from Overture Maps
----------------------------

Use :func:`aequilibrae.project.network.network.Network.import_from_overture` to
import the Overture Maps transportation network for an EPSG:4326 polygon::

    from shapely.geometry import box

    project.network.import_from_overture(
        model_area=box(-112.185, 36.59, -112.179, 36.60),
        modes=("car", "walk"),
        simplify="osmnx",
    )

The importer always uses the latest Overture Maps release advertised by the
Overture STAC catalog. The release used for the import is written to the project
``about`` table and to the download-cache manifest.

The Overture importer reads the cloud ``segment`` and ``connector`` themes,
splits segments at intermediate connectors, and derives AequilibraE link
attributes from the Overture schema:

* ``class`` becomes ``link_type``;
* connector order defines ``a_node`` and ``b_node``;
* access restrictions determine ``direction`` where possible;
* global speed-limit rules populate ``speed_ab`` and ``speed_ba``;
* additional Overture properties and rule arrays are stored in
  ``other_attributes``.

The raw Overture tables and a manifest are written under
``<project>/downloaded data/overture-cloud/``.

.. _network_import_modes:

Mode filtering
--------------

All network import methods accept ``modes`` as a sequence of AequilibraE mode
names. Supported names are:

* ``"car"``;
* ``"transit"``;
* ``"bicycle"``;
* ``"walk"``.

Only links with at least one requested mode are kept. The stored ``modes`` field
contains AequilibraE mode codes, not the full mode names.

.. _network_import_simplification:

Network simplification
----------------------

OSM and Overture imports can simplify the staged network before it is written to
the project database. The ``simplify`` argument accepts:

* ``"osmnx"``: simplify with OSMnx. This is the default;
* ``"neatnet"``: simplify with neatnet;
* ``False``: skip simplification.

``consolidate_tolerance`` controls intersection/node consolidation in metres
(after automatic projection to a local UTM CRS) for both simplifiers. Set it to
``None`` to run OSMnx topological simplification without intersection
consolidation; for neatnet, where node consolidation is integral to the
algorithm, ``None`` falls back to the default tolerance of 10 metres::

    project.network.import_from_osm(
        pbf_path="/path/to/extract.osm.pbf",
        simplify="osmnx",
        consolidate_tolerance=None,
    )

    project.network.import_from_overture(
        model_area=model_area,
        simplify=False,
    )

Simplified links retain source provenance in ``other_attributes``. For OSMnx,
merged-link provenance is stored under ``source_ids``.

.. _network_importer_public_api:

Public API summary
------------------

The main entry points are:

* :func:`aequilibrae.project.network.network.Network.import_from_osm`;
* :func:`aequilibrae.project.network.network.Network.import_from_overture`;
* :func:`aequilibrae.project.network.network.Network.import_network` for explicit
  source names: ``"osm-overpass"``, ``"osm-pbf"`` and ``"overture-cloud"``.

``Network.create_from_osm`` was removed. Use ``Network.import_from_osm`` instead.

.. note::

   The OSM/Overture importer writes source-specific attributes to the existing
   ``other_attributes`` column on ``links`` and ``nodes``. New projects contain
   these columns. Existing projects created with older schemas must be upgraded
   or recreated before using the importer.

.. seealso::

    * :ref:`plot_from_overture`
        Network import example (Overture Maps, with OSM as an alternative)
    * :ref:`parameters_file`
        Project parameter file

Importing from link layer
-------------------------

It is possible to create an AequilibraE project from a link layer, such as a \*.csv file that
contains geometry in WKT, for instance. You can check an example with all functions used in
:ref:`the following example <project_from_link_layer>`.

.. _importing_from_gmns_file:

Importing from files in GMNS format
-----------------------------------

Before importing a network from a source in GMNS format, it is imperative to know 
in which spatial reference its geometries (links and nodes) were created. If the SRID
is different than 4326, it must be passed as an input using the argument ``srid``.

.. image:: ../_images/plot_import_from_gmns.png
    :align: center
    :alt: example
    :target: ../_auto_examples/network_manipulation/plot_create_from_gmns.html

It is possible to import the following files from a GMNS source:

* link table;
* node table;
* use_group table;
* geometry table.

You can find the specification for all these tables in the GMNS documentation,
`here <https://github.com/zephyr-data-specs/GMNS/tree/develop/docs/spec>`_.

By default, the method ``create_from_gmns()`` read all required and optional fields
specified in the GMNS link and node tables specification. If you need it to read 
any additional fields as well, you have to modify the AequilibraE parameters as
shown in the :ref:`example <import_from_gmns>`.

When adding a new field to be read in the parameters.yml file, it is important to 
keep the "required" key set to False, since you will always be adding a non-required 
field. Required fields for a specific table are only those defined in the GMNS
specification.

.. note::

    In the AequilibraE nodes table, if a node is to be identified as a centroid, its
    'is_centroid' field has to be set to 1. However, this is not part of the GMNS
    specification. Thus, if you want a node to be identified as a centroid during the
    import process, in the GMNS node table you have to set the field 'node_type' equals
    to 'centroid'.

.. seealso::

    * :func:`aequilibrae.project.network.network.Network.create_from_gmns`
        Function documentation
    * :ref:`import_from_gmns`
        Usage example

.. _aequilibrae_to_gmns:

Exporting AequilibraE model to GMNS format
------------------------------------------

After loading an existing AequilibraE project, you can export it to GMNS format. 

.. image:: ../_images/plot_export_to_gmns.png
    :align: center
    :alt: example
    :target: ../_auto_examples/network_manipulation/plot_export_to_gmns.html

It is possible to export an AequilibraE network to the following tables in GMNS format:

* link table
* node table
* use_definition table

This list does not include the optional 'use_group' table, which is an optional argument
of the GMNS function, because mode groups are not used in the AequilibraE modes table.

In addition to all GMNS required fields for each of the three exported tables, some
other fields are also added as reminder of where the features came from when looking 
back at the AequilibraE project.

.. note::

    When a node is identified as a centroid in the AequilibraE nodes table, this
    information is transmitted to the GMNS node table by means of the field
    'node_type', which is set to 'centroid' in this case. The 'node_type' field
    is an optional field listed in the GMNS node table specification.

You can find the GMNS specification
`here <https://github.com/zephyr-data-specs/GMNS/tree/develop/docs/spec>`_.

.. seealso::

    * :func:`aequilibrae.project.network.network.Network.export_to_gmns`
        Function documentation
    * :ref:`export_to_gmns`
        Usage example
