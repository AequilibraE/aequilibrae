:orphan:

.. _simwrapper:

SimWrapper Extension
=====================

The SimWrapper extension generates a portable dashboard configuration for
visualising an AequilibraE project with the external
`simwrapper.app <https://simwrapper.app/>`_ viewer. SimWrapper itself is a browser-based, client-side tool (originally developed
at TU Berlin) for exploring disaggregate transport simulation outputs; it
accepts simple YAML configuration files describing dashboards of maps, tables
and charts and can run entirely offline. The application is open-source (see
`GitHub <https://github.com/simwrapper/simwrapper>`_) and designed so that even
non-coding stakeholders can open a dashboard in their browser and interact
with results. Users may upload the generated configurations to the public
service or share them via `explore.outerloop.io <https://explore.outerloop.io/>`_.

The extension therefore produces the YAML dashboard file and associated data
files (CSV and Vega-Lite JSON specs) into the chosen output directory so that
SimWrapper can load the project with no further editing.

This document describes how to create the dashboard configuration both
from the command line and from Python. Later sections also remind you where to
upload or share the generated dashboard.
Key points
----------

 - The generator is implemented as :class:`aequilibrae.utils.simwrapper.generate_simwrapper_config.SimwrapperConfigGenerator`.
 - A CLI entry point is installed as the ``aeq-sim`` script (registered in the package entry points).
 - Output: a ``dashboard.yaml`` file and data files (CSV and Vega-Lite JSON) written into the chosen ``<output_dir>``.
   A data subfolder named ``simwrapper_data/`` is created under the output directory
   for CSV and Vega-Lite spec files.

Quickstart — CLI
----------------

Generate a dashboard for the project in the current directory and write
outputs to a `simwrapper` subfolder (or any name you choose for the output
directory):

.. code-block:: console

    $ aeq-sim --project /path/to/project --output-dir simwrapper

If you omit ``--project``, the CLI defaults to the current directory (``.``).
If you omit ``--output-dir``, the CLI will write to a `simwrapper` folder by
default.


CLI options (also available via the Python API constructor)

 - ``--project`` / ``-p``: project root (folder containing ``project_database.sqlite``).
 - ``--output-dir`` / ``-o``: output directory (created inside the project).
 - ``--max-results-tables``: limits the number of results scenarios included. When not
   provided, the generator defaults to three scenarios.
 - ``--results-tables``: explicit list of results table names to include (space-separated, e.g. ``--results-tables table_1 table_2``).
 - ``--centroid-link-types``: explicit link-type names considered centroid connectors (space-separated, e.g. ``--centroid-link-types centroid connector``).
 - ``--quiet`` / ``-q``: suppress informational output.

.. note::
   ``output_dir`` must reside inside the project directory. Absolute paths
   outside the project are rejected; the Python API will raise a ``ValueError``
   and the CLI will report an error and exit with a non-zero return code.


Relative ``output_dir`` values are interpreted as subdirectories under the
project root. Absolute paths are accepted only when they reside within the
project folder.

Example with options

.. code-block:: console

    $ aeq-sim -p /home/user/my_model -o simwrapper --max-results-tables 5 --quiet

Example — CLI selecting specific results and centroid types
---------------------------------------------------------

Pass explicit scenario names and centroid link types to the CLI. Use
space-separated values for multi-valued options:

.. code-block:: console

    $ aeq-sim -p /home/user/my_model -o simwrapper --results-tables res_2023 res_2024 \
       --centroid-link-types centroid connector

Quickstart — Python API
-----------------------

Generate the same outputs programmatically using the
``SimwrapperConfigGenerator`` class. The API is lightweight and suitable for
in-script generation or integration with analyses.

.. code-block:: python

    from aequilibrae.project import Project
    from aequilibrae.utils.simwrapper.generate_simwrapper_config import SimwrapperConfigGenerator

    prj = Project()
    prj.open('/path/to/project')

    gen = SimwrapperConfigGenerator(prj, output_dir='simwrapper', max_results_tables=3)
    gen.write_yamls()

The constructor accepts the same configuration knobs available in the CLI
(``output_dir``, ``max_results_tables``, ``results_tables``,
``centroid_link_types``). ``output_dir`` must be located inside the project
directory; absolute paths outside the project will raise a ``ValueError``.

What the generator writes
-------------------------

All generated files are written into the chosen output directory (``<output_dir>``).

 - ``<output_dir>/dashboard.yaml`` — the SimWrapper dashboard configuration.
 - Data files referenced by the dashboard (CSV and Vega-Lite JSON specs), for example assignment convergence CSVs and ``assignment_convergence.vega.json``.

The generator instance exposes a ``generated_files`` mapping after writing the outputs. This dict-like object maps descriptive keys (for example ``"dashboard"`` or ``"assignment_convergence"``) to ``pathlib.Path`` objects pointing to the written files. It is useful for programmatic inspection or other downstream processing.

Portability
-----------

The YAML dashboard references data files using a relative path (``simwrapper_data/``).
If you move or share the generated output keep the output directory together with
its ``simwrapper_data/`` subfolder so paths in ``dashboard.yaml`` remain valid.

Viewing the dashboard
---------------------

1. Open `simwrapper.app <https://simwrapper.app/>`_ in your browser.
2. Upload the generated ``dashboard.yaml`` or point the app to the generated output folder (keep the output directory together with its ``simwrapper_data/`` subfolder).
3. Interact with the panels: maps, tables and Vega-Lite plots are available according to the panels generated from your project.

Troubleshooting
---------------

 - If the CLI cannot open the project path you supplied, it will print an error and exit with code ``2``. Check the path points to the project folder that contains ``project_database.sqlite``.
 - If the generator encounters an error while writing outputs, it will print an error and exit with code ``1``. Running the same code from Python will raise the underlying exception; you can wrap the call to ``gen.write_yamls()`` in ``try/except`` to diagnose the problem.
 - If you do not want any results scenarios included, pass ``--max-results-tables 0`` to the CLI or ``max_results_tables=0`` to the Python API; the generator will omit results panels from the dashboard in that case.

Exit codes (CLI)
-----------------

 - ``0``: success.
 - ``1``: error generating config (writing outputs).
 - ``2``: error opening project path.

Notes and tips
-------------

- If your project contains many results scenarios, use ``--max-results-tables``
  or ``results_tables`` to control which scenarios are included; the generator
  attempts to pick the most recent scenarios automatically when not specified.
  Selection is based on the ``timestamp`` field when present (most recent first),
  with a stable fallback ordering by ``table_name`` when timestamps are not available.
- When centroid link types are not specified the generator attempts to infer
  them from the project's link types (names containing "centroid" or
  "connector") or from the nodes table.
- The generated dashboard is portable: keep the generated `simwrapper` folder
  with your project to allow others to open the same dashboard locally.

Example — select specific results scenarios (Python)
--------------------------------------------------

Example: include only a subset of available results scenarios in the dashboard.
The snippet below lists available result tables, selects a small subset and
passes their names to the generator.

.. code-block:: python

    from aequilibrae.project import Project
    from aequilibrae.utils.simwrapper.generate_simwrapper_config import SimwrapperConfigGenerator

    prj = Project()
    prj.open('/path/to/project')

    # list available results scenarios
    res_df = prj.results.list()
    available = res_df['table_name'].tolist()

    # pick a subset (example: first two, or filter by name)
    chosen = available[:2]

    gen = SimwrapperConfigGenerator(prj, output_dir='simwrapper', results_tables=chosen)
    gen.write_yamls()

    # inspect generated files
    print(gen.generated_files)

.. note::
   After generation you can open `simwrapper.app` and load the output folder
   (or upload the ``dashboard.yaml``) to view the dashboard interactively.

Programmatic usage
-------------------

Wrap generation in a try/except when calling from scripts so you can handle
errors and inspect outputs programmatically. Example:

.. code-block:: python

    from aequilibrae.project import Project
    from aequilibrae.utils.simwrapper.generate_simwrapper_config import SimwrapperConfigGenerator

    prj = Project()
    prj.open('/path/to/project')

    gen = SimwrapperConfigGenerator(prj, output_dir='simwrapper')
    try:
        gen.write_yamls()
    except Exception as e:
        print('Failed to write simwrapper config:', e)
        raise

    # successful: inspect paths
    for key, path in gen.generated_files.items():
        print(key, path)

Links
-----

- `simwrapper.app <https://simwrapper.app/>`_
- `explore.outerloop.io <https://explore.outerloop.io/>`_

See also
--------

- Code: :mod:`aequilibrae.utils.simwrapper.generate_simwrapper_config`
- Tests: :mod:`tests.aeq.utils.test_simwrapper_config` (unit tests in the
  repository covering behaviour and CLI invocation).
