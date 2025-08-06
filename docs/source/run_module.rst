.. _run_module:

Run module
==========

AequilibraE provides a convenient method for defining model entry points and their default
arguments via ``run/__init__.py`` and ``parameters.yml`` respectively. These can be used to
couple model parameters and methods to run models to the model itself.

``run/__init__.py``
-------------------

The run module is a standard Python module that is dynamically imported when the ``project.run``
property is accessed. Objects named within ``parameters.yml`` under the ``run`` heading will have
their arguments partially applied via ``functools.partial`` and return a ``namedtuple``.

Not all objects within the module must be named ``parameters.yml``. If an object is named within
``parameters.yml``, then it must exist within the module otherwise a ``RuntimeError`` will be
raised.

By default an AequilibraE project comes with four example functions: ``matrix_summary``,
``graph_summary``, ``results_summary``, and ``example_function_with_kwargs``. The summary functions
are not named within the default ``parameters.yml`` as they take no arguments.

Functions should use the ``get_active_project()`` function to obtain a reference to the current
project.

State within the module should be avoided as the file may be run multiple times.

.. toctree::
    :maxdepth: 1
    :caption: Run module

    _auto_examples/run_module/index
