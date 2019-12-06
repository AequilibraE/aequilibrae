An overview of AequilibraE
==========================

.. toctree::
   :maxdepth: 4

Sub-modules
~~~~~~~~~~~

AequilibraE is organized in submodules that are often derived from the traditional 4-step model. However, other modules
have already been added or are expected to be added in the future. The current modules are:

- :ref:`overview_distribution`
- :ref:`overview_paths`
- :ref:`overview_transit`
- :ref:`overview_matrix`
- :ref:`overview_parameters`

Contributions can be made to the existing modules or in the form of new modules.

.. _overview_distribution:

Trip distribution
~~~~~~~~~~~~~~~~~
Synthetic gravity calib and appl
IPF

.. _overview_paths:

Path computation
~~~~~~~~~~~~~~~~
Traffic assignment
Path computation
Skimming

.. _overview_transit:

Transit
~~~~~~~
GTFS import

.. _overview_matrix:

Matrix
~~~~~~

The matrix submodule has two main components. Datasets and Matrices

Their existence is required for performance purposes and to support consistency across other modules. It also make it
a lot faster to develop new features. Compatibility with de-facto open standards is also pursued as a major requirement.

They are both memory mapped structures, which allows for some nice features, but they still consume all memory
necessary to handle them in full. In the future we will look into dropping that requirement, but the software work is
substantial.

AequilibraE Matrix
------------------

If one looks into how all commercial software handle matrices, it would be clear that there are definitely two different
schools of thought. The first one, is where matrices are those where matrices are simple binary blobs that require the
existence of a model to give it context.

The second one is a more comprehensive take, where a matrix file not only contains its indices and metadata, but one
that can also store multiple matrices. This capability, available in the openmatrix format, is one that was reproduced
in the AequilibraE matrix and API.

Because the `Open Matrix Format <https://github.com/osPlanning/omx>`_ has established itself as the de-factor standard
for matrix exchange in the industry, AequilibraE aims to allow users to never touch the AEM data format if they so
decide. However, all underlying computation will be done using our custom format, so the importing/exporting being done
under the hood will mean additional (although small) overhead.

For programatic applications wher eperformance is critical, we recommend using the AEM format whenever possible.

AequilibraE Data
----------------

AequilibraE datasets are data structures based on NumPy record arrays (arrays with named columns). Its role in the
software is to hold columnar data for all procedures that may use it, such as O/D/P/A vectors used in trip distribution
and link loads from traffic assignment.

AequilibraE data currently supports export to **csv** and **sqlite**. Extending it to other binary files such as HDF5
or `Arrow <https://arrow.apache.org/>`_ are being considered for future development. If you require them, please file
an issue on GitHub.

.. _overview_parameters:

Global parameters
~~~~~~~~~~~~~~~~~
parameters module
