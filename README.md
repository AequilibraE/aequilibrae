# AequilibraE

[![Downloads](https://img.shields.io/pypi/dm/aequilibrae.svg?maxAge=2592000)](https://pypi.python.org/pypi/aequilibrae)
[![Documentation](https://github.com/AequilibraE/aequilibrae/actions/workflows/documentation.yml/badge.svg)](https://github.com/AequilibraE/aequilibrae/actions/workflows/documentation.yml)
[![unit tests](https://github.com/AequilibraE/aequilibrae/actions/workflows/unit_tests.yml/badge.svg)](https://github.com/AequilibraE/aequilibrae/actions/workflows/unit_tests.yml)
[![Code coverage](https://github.com/AequilibraE/aequilibrae/actions/workflows/test_linux_with_coverage.yml/badge.svg)](https://github.com/AequilibraE/aequilibrae/actions/workflows/test_linux_with_coverage.yml)
[![Packaging](https://github.com/AequilibraE/aequilibrae/actions/workflows/build_wheels.yml/badge.svg)](https://github.com/AequilibraE/aequilibrae/actions/workflows/build_wheels.yml)

AequilibraE is a fully-featured Open-Source transportation modeling package and
the first comprehensive package of its kind for the Python ecosystem, and is 
released under an extremely permissive and business-friendly license.

It is developed as general-purpose modeling software and imposes very little 
underlying structure on models built upon it. This flexibility also extends to
the ability of using all its core algorithms without an actual AequilibraE 
model by simply building very simple memory objects from Pandas DataFrames, and
NumPY arrays, making it the perfect candidate for use-cases where transport is 
one component of a bigger and more general planning or otherwise analytical 
modeling pipeline.

Different than in traditional packages, AequilibraE's network is stored in 
SQLite/Spatialite, a widely supported open format, and its editing capabilities
are built into its data layer through a series of spatial database triggers, 
which allows network editing to be done on Any GIS package supporting SpatiaLite, 
through a dedicated Python API or directly from an SQL console while maintaining
full geographical consistency between links and nodes, as well as data integrity
and consistency with other model tables.

AequilibraE provides full support for OMX matrices, which can be used as input
for any AequilibraE procedure, and makes its outputs, particularly skim matrices 
readily available to other modeling activities.

AequilibraE includes multi-class user-equilibrium assignment with full support
for class-specific networks, value-of-time and generalized cost functions, and 
includes a range of equilibration algorithms, including MSA, the traditional 
Frank-Wolfe as well as the state-of-the-art Bi-conjugate Frank-Wolfe.

AequilibraE's support for public transport includes a GTFS importer that can 
map-match routes into the model network and an optimized version of the
traditional "Optimal-Strategies" transit assignment, and full support in the data 
model for other schedule-based assignments to be implemented in the future.

State-of-the-art computational performance and full multi-threading can be 
expected from all key algorithms in AequilibraE, from cache-optimized IPF, 
to path-computation based on sophisticated data structures and cascading network
loading, which all ensure that AequilibraE performs at par with the best
commercial packages current available on the market.

AequilibraE has also a Graphical Interface for the popular GIS package QGIS, 
which gives access to most AequilibraE procedures and includes a wide range of
visualization tools, such as flow maps, desire and delaunay lines, scenario 
comparison, matrix visualization, etc. This GUI, called QAequilibraE, is 
currently available in English, French and Portuguese and more languages are
continuously being added, which is another substantial point of difference from 
commercial packages.

Finally, AequilibraE is developed 100% in the open and incorporates software-development 
best practices for testing and documentation. AequilibraE's testing includes all 
major operating systems (Windows, Linux and MacOS) and all currently supported versions
of Python. AequilibraE is also supported on ARM-based cloud computation nodes, making 
cloud deployments substantially less expensive.

## Comprehensive documentation

[AequilibraE documentation built with Sphinx ](http://www.aequilibrae.com)


### What is available only in QGIS

Some common resources for transportation modeling are inherently visual, and therefore they make more sense if
available within a GIS platform. For that reason, many resources are available only from AequilibraE's 
[QGIS plugin](http://plugins.qgis.org/plugins/qaequilibrae/),
which uses AequilibraE as its computational workhorse and also provides GUIs for most of AequilibraE's tools. Said tool
is developed independently and a little delayed with relationship to the Python package, and more details can be found in its 
[GitHub repository](https://github.com/AequilibraE/qaequilibrae).
