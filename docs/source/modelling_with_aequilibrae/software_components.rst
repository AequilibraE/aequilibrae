

Sub-modules
-----------

AequilibraE is organized in submodules organized around common workflows
used in transport modeling, as well as connected to the maintenance and
operation of models. The current modules are:

- :ref:`overview_project`
- :ref:`overview_parameters`
- :ref:`overview_distribution`
- :ref:`overview_paths`
- :ref:`overview_transit`
- :ref:`overview_matrix`


.. _overview_parameters:

Global parameters
~~~~~~~~~~~~~~~~~
As more features are added to AequilibraE, a large number of parameters start to
be required, so the parameters module has also been growing in importance within
the software.

There are currently 4 main sessions with the parameters file: *Assignment*,
*Distribution*, *Network* and *System*.

The parameters for *assignment* and *distribution* control only convergence
criteria, while the *System* section controls things like the number of CPU
cores used by the software, default directories and Spatialite location for
Windows systems. The *Network* section, however, contains parameters that
control the creation of networks and the import from Open Street Maps.

The parameters file is built using the `YAML <https://yaml.org/>`_ format, and
detailed instructions on how to configure it are provided on
:ref:`parameters_file`.


.. _overview_distribution:

Trip distribution
~~~~~~~~~~~~~~~~~

The trip distribution module is the second oldest piece of code in AequilibraE,
and includes only code for calibration and application of Synthetic gravity
models and Iterative Proportional Fitting. Not much documentation has been
written for this module, but some examples are available on the usage examples
page :ref:`example_usage_distribution`.


.. _overview_paths:

Path computation
~~~~~~~~~~~~~~~~

The path computation module contains some of the oldest code in AequilibraE,
some of which preceed the existance of AequilibraE as a proper Python package.

The package is built around a shortest path algorithm ported from SciPy and
adapted to support proper multi-threading, network loading and multi-field
skimming.

As of now, this module has three main capabilities:

* Traffic assignment
* Regular path computation (one origin to one destination)
* Network skimming

It is worth noting that turn-penalties and turn-prohibitions are currently not
supported in AequilibraE, and that there is no built-in support for multiple
concurrent shortest paths computation, although the path computation path does
release the `GIL <https://wiki.python.org/moin/GlobalInterpreterLock>`_, which
allows the users to get some performance gains using Python's threading module.

A wealth of usage examples are available in the examples page under
:ref:`example_usage_paths`.


.. _overview_transit:

Transit
~~~~~~~

For now the only transit-related capability of AequilibraE is to import GTFS
into SQLite/Spatialite. The results of this import is NOT integrated with the
AequilibraE project.

.. Usage examples can be found on :ref:`example_usage_transit`.


.. _overview_matrix:

Matrix
~~~~~~

The matrix submodule has two main components: *Datasets* and *Matrices*.

Their existence is required for performance purposes and to support consistency
across other modules. It also make it a lot faster to develop new features.
Compatibility with de-facto open standards is also pursued as a major
requirement.

They are both memory mapped structures, which allows for some nice features,
but they still consume all memory necessary to handle them in full. In the
future we will look into dropping that requirement, but the software work is
substantial.

AequilibraE Matrix
------------------

If one looks into how all commercial software handle matrices, it would be
clear that there are definitely two different schools of thought. The first one,
is where matrices are simple binary blobs that require the existence of a model
to give it context.

The second one is a more comprehensive take, where a matrix file not only
contains its indices and metadata, but one that can also store multiple
matrices. This capability, available in the openmatrix format, is one that was
reproduced in the AequilibraE matrix and API.

Because the `Open Matrix Format <https://github.com/osPlanning/omx>`_ has
established itself as the de-facto standard for matrix exchange in the
industry, AequilibraE aims to allow users to never touch the AEM data format if
they so decide. However, all underlying computation will be done using our
custom format, so the importing/exporting being done under the hood will mean
additional (although small) overhead.

For programatic applications where performance is critical, we recommend using
the AEM format whenever possible.

AequilibraE Data
----------------

AequilibraE datasets are data structures based on NumPy record arrays (arrays
with named columns). Its role in the software is to hold columnar data for all
procedures that may use it, such as O/D/P/A vectors used in trip distribution
and link loads from traffic assignment.

AequilibraE data currently supports export to **csv** and **sqlite**. Extending
it to other binary files such as HDF5 or `Arrow <https://arrow.apache.org/>`_
are being considered for future development. If you require them, please file an
issue on GitHub.

