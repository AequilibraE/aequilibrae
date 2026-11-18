.. _run_module:

Run module
==========

AequilibraE provides a convenient method for defining model entry points and their default
arguments via ``run/__init__.py`` and ``parameters.yml`` respectively. These can be used to
couple model parameters and methods to run models to the model itself.

``run/__init__.py``
-------------------

AequilibraE imports the run module when you access ``project.run``. The property returns a
dictionary keyed by the function names under ``run`` in ``parameters.yml``. AequilibraE applies
the configured arguments with ``functools.partial``. For example, call
``project.run["results_summary"]()`` to run the configured results summary.

You do not need to list every function in ``parameters.yml``, but each name you list must refer
to a callable in the module.

By default an AequilibraE project comes with four example functions: ``matrix_summary``,
``graph_summary``, ``results_summary``, and ``example_function_with_kwargs``. The default
``parameters.yml`` lists the summary functions without extra arguments.

Functions must accept an explicit ``project`` argument, supplied by ``Project.run``.

State within the module should be avoided as the file may be run multiple times.

Command line interface
----------------------

Functions in the run module can also be invoked from the command line via the ``aeq`` entry
point. ``aeq run --help`` lists the available functions, and each function accepts arguments
derived from its signature, with the defaults from ``parameters.yml`` pre-applied
(``--no-defaults`` disables this).

.. code-block:: bash

    aeq -p /path/to/project run --help
    aeq -p /path/to/project run example_function_with_kwargs --help
    aeq -p /path/to/project run example_function_with_kwargs --arg1 hello iterations=50

Required parameters become positional arguments and parameters with defaults become
``--options``. Parameters whose default value is a boolean become ``--flag``/``--no-flag``
pairs, and functions accepting ``**kwargs`` take trailing ``key=value`` pairs. All values are
parsed as Python literals where possible and kept as strings otherwise, mirroring how
``parameters.yml`` values are typed; quote a value (e.g. ``'"101"'``) to force a string. Type
annotations are shown in the help text but not enforced. If ``-p/--project`` is omitted the
current directory is used.

.. toctree::
    :maxdepth: 1
    :caption: Run module

    _auto_examples/run_module/index
