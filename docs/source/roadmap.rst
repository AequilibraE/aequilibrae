Roadmap
=======

As AequilibraE is a project with an incredibly small team and no source of
funding, it is not feasible to determine a precise schedule for the development
of new features or even a proper roadmap of specific developments.

However, there are a number of enhancements to the software that we have already
identified and that we intend to dedicate some time to in the future.

* Network data model
    * Introduce centroid connector data type to replace inference that all links
      connected to centroids are connectors

* Traffic assignment

    * Re-development of the path-finding algorithm to allow for turn
      penalties/bans
    * Implementation of network simplification to improve performance of
      path-finding

* Public Transport

    * Import of GTFS map-matching it into a project network
    * Re-development of Public Transport data model for GTFS/AequilibraE
    * Export of GTFS (enables editing of GTFS in QGIS
    * Transit path computation (Likely to be the
      `CSA <https://link.springer.com/chapter/10.1007/978-3-642-38527-8_6>`_ or
      similar)

* Project

    * Inclusion of new table for scalar values
    * Inclusion of new table for vectors based on centroid IDs (plus metadata
      table)
    * Inclusion of new table for vectors based on node IDs (plus metadata table)
    * Inclusion of new table for vectors based on link IDs (plus metadata table)
    * New matrix registry and matrix folder alongside project

* QGIS

    * Inclusion of TSP and more general vehicle routing problems (resource
      constraints, pick-up and delivery, etc.)

If there is any other feature you would like to suggest, please record a new
issue on `GitHub <https://github.com/AequilibraE/aequilibrae/issues>`_, or drop
us a line.

If your organization is making use of AequilibraE, please consider funding some
of the new developments or maintenance of the project.