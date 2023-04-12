===========
API Reference
===========

.. automodule:: aequilibrae

Project
-------
.. current_module:: aequilibrae
.. autosummary::
   :nosignatures:
   :toctree: generated/

   Project
   Project.load
   Project.new
   Project.open

Project Components
~~~~~~~~~~~~~~~~~~
.. currentmodule:: aequilibrae.project
.. autosummary::
   :nosignatures:
   :toctree: generated/

    About
    FieldEditor
    Log
    Matrices
    Network
    Zoning

Project Objects
~~~~~~~~~~~~~~~
.. currentmodule:: aequilibrae.project
.. autosummary::
   :nosignatures:
   :toctree: generated/

    Zone

Network Data
------------
.. currentmodule:: aequilibrae.project.network
.. autosummary::
   :nosignatures:
   :toctree: generated/

    Modes
    LinkTypes
    Links
    Nodes

Network Items
-------------
.. currentmodule:: aequilibrae.project.network

.. autosummary::
   :nosignatures:
   :toctree: generated/

    Mode
    LinkType
    Link
    Node

Parameters
----------
.. currentmodule:: aequilibrae

.. autosummary::
   :nosignatures:
   :toctree: generated/

    Parameters
    Parameters.restore_default
    Parameters.write_back

Distribution
------------

.. currentmodule:: aequilibrae

.. autosummary::
   :nosignatures:
   :toctree: generated/

    Ipf
    Ipf.fit
    Ipf.save_to_project
    GravityApplication
    GravityApplication.apply
    GravityApplication.save_to_project
    GravityCalibration
    GravityCalibration.calibrate
    SyntheticGravityModel
    SyntheticGravityModel.load
    SyntheticGravityModel.save

Matrix
------
.. currentmodule:: aequilibrae.matrix

.. autosummary::
   :nosignatures:
   :toctree: generated/

   AequilibraeData
   AequilibraeData.create_empty
   AequilibraeData.empty
   AequilibraeData.export
   AequilibraeData.load
   AequilibraeData.random_name
   AequilibraeMatrix
   AequilibraeMatrix.close
   AequilibraeMatrix.columns
   AequilibraeMatrix.computational_view
   AequilibraeMatrix.copy
   AequilibraeMatrix.create_empty
   AequilibraeMatrix.create_from_omx
   AequilibraeMatrix.create_from_trip_list
   AequilibraeMatrix.export
   AequilibraeMatrix.get_matrix
   AequilibraeMatrix.is_omx
   AequilibraeMatrix.load
   AequilibraeMatrix.nan_to_num
   AequilibraeMatrix.random_name
   AequilibraeMatrix.rows
   AequilibraeMatrix.save
   AequilibraeMatrix.setDescription
   AequilibraeMatrix.setName
   AequilibraeMatrix.set_index

Paths
-----

.. currentmodule:: aequilibrae.paths

.. autosummary::
   :nosignatures:
   :toctree: generated/

    Graph
    AssignmentResults
    SkimResults
    SkimResults.prepare
    SkimResults.set_cores
    PathResults
    VDF
    TrafficClass
    TrafficAssignment

Transit
-------

.. currentmodule:: aequilibrae.transit

.. autosummary::
   :nosignatures:
   :toctree: generated/

    GTFS
    create_gtfsdb
