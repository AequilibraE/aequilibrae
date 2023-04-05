Use examples
============
This page is still under development, so if you have developed interesting use
cases, please consider contributing them.

.. note::
   The examples provided here are not meant as a through description of
   AequilibraE's capabilities. For that, please look into the API documentation
   or email aequilibrae@googlegroups.com

Sample Data
-----------

We have compiled two very distinct example datasets imported from the
`TNTP instances <https://github.com/bstabler/TransportationNetworks/>`_.

* `Sioux Falls <http://www.aequilibrae.com/data/SiouxFalls.7z>`_
* `Chicago Regional <http://www.aequilibrae.com/data/Chicago.7z>`_
* `Birmingham <http://www.aequilibrae.com/data/Birmingham.7z>`_

While the Sioux Falls network is probably the most traditional example network
available for evaluating network algorithms, the Chicago Regional model is a
good example of a real-world sized model, with roughly 1,800 zones.

Each instance contains the following folder structure and contents:

0_tntp_data:

* Data imported from `TNTP instances <https://github.com/bstabler/TransportationNetworks/>`_.
* matrices in openmatrix and AequilibraE formats
* vectors computed from the matrix in question and in AequilibraE format
* No alterations made to the data

1_project

* AequilibraE project result of the import of the links and nodes layers

2_skim_results:

* Skim results for distance and free_flow_travel_time computed by minimizing
  free_flow_travel_time
* Result matrices in openmatrix and AequilibraE formats

3_desire_lines

* Layers for desire lines and delaunay lines,  each one in a separate
  geopackage file
* Desire lines flow map
* Delaunay Lines flow map

4_assignment_results

* Outputs from traffic assignment to a relative gap of 1e-5 and with skimming
  enabled
* Link flows in csv and AequilibraE formats
* Skim matrices in openmatrix and AequilibraE formats
* Assignment flow map in png format

5_distribution_results

* Models calibrated for inverse power and negative exponential deterrence
  functions
* Convergence logs for the calibration of each model
* Trip length frequency distribution chart for original matrix
* Trip length frequency distribution chart for model with negative exponential
  deterrence function
* Trip length frequency distribution chart for model with inverse power
  deterrence function
* Inputs are the original demand matrix and the skim for TIME (final iteration)
  from the ASSIGNMENT

6_forecast

* Synthetic future vectors generated with a random growth from 0 to 10% in each
  cell on top of the original matrix vectors
* Application of both gravity models calibrated plus IPF to the synthetic
  future vectors

7_future_year_assignment

* Traffic assignment

    - Outputs from traffic assignment to a relative gap of 1e-5 and with
      skimming enabled
    - Link flows in csv and AequilibraE formats
    - Skim matrices in openmatrix and AequilibraE formats
* Scenario comparison flow map of absolute differences
* Composite scenario comparison flow map (gray is flow maintained in both
  scenarios, red is flow growth and green is flow decline)

