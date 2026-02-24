
.. _simwrapper:

SimWrapper Extension
=====================

`SimWrapper <https://simwrapper.app/>`_ is a browser-based, client-side tool for exploring transport simulation outputs. It accepts simple YAML configuration files describing dashboards of maps, tables and charts. Nothing is uploaded to the internet, everything is kept local. The application is `open-source <https://github.com/simwrapper/simwrapper>`_ and designed so that non-coding stakeholders can open a dashboard in their browser and interact with results.

The SimWrapper extension generates portable dashboard configuration for visualising an AequilibraE project with the `simwrapper.app <https://simwrapper.app/>`_ or `explore.outerloop.io <https://explore.outerloop.io/>`_. At the time of writing Outer Loop is hosting a preview version of SimWrapper while the required changes are being unstreamed.

You can view an example dashboard at `explore.outerloop.io <https://explore.outerloop.io/>`_ by selecting "Chicago AequilibraE Example" from the Example Dashboards section.

Usage
-----

 - The generator is implemented as :class:`aequilibrae.utils.simwrapper.generate_simwrapper_config.SimwrapperConfigGenerator`.
 - A CLI entry point is installed as the ``aeq-sim`` script (registered in the package entry points).
 - Output: a ``dashboard.yaml`` file and data files (CSV and Vega-Lite JSON) written into the chosen ``<output_dir>``.
   A data subfolder named ``simwrapper_data/`` is created under the output directory
   for CSV and Vega-Lite spec files.

Quickstart — CLI
^^^^^^^^^^^^^^^^

.. code-block:: console

    $ aeq-sim --project /path/to/project --output-dir simwrapper

If you omit ``--project``, the CLI defaults to the current directory (``.``).
If you omit ``--output-dir``, the CLI will write to a `simwrapper` folder by
default.


CLI options

 - ``--project`` / ``-p``: project root (folder containing ``project_database.sqlite``).
 - ``--output-dir`` / ``-o``: output directory (created inside the project).
 - ``--max-results-tables``: limits the number of results scenarios included. When not
   provided, the generator defaults to three results.
 - ``--results-tables``: explicit list of results table names to include (space-separated, e.g. ``--results-tables table_1 table_2``). When not provided, defaults to the first `max-results-tables` tables.
 - ``--centroid-link-types``: explicit link-type names considered centroid connectors (space-separated, e.g. ``--centroid-link-types centroid connector``).
 - ``--quiet`` / ``-q``: suppress informational output.

.. note::
   ``output_dir`` must reside inside the project directory. Absolute paths outside the project are rejected.


Quickstart — Python API
^^^^^^^^^^^^^^^^^^^^^^^

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
^^^^^^^^^^^^^^^^^^^^^^^^^^^

All generated files are written into the chosen output directory (``<output_dir>``).

 - ``<output_dir>/dashboard.yaml`` — the SimWrapper dashboard configuration.
 - Data files referenced by the dashboard (CSV and Vega-Lite JSON specs), for example assignment convergence CSVs and ``assignment_convergence.vega.json``.

The generator instance exposes a ``generated_files`` mapping after writing the outputs. This dict-like object maps descriptive keys (for example ``"dashboard"`` or ``"assignment_convergence"``) to ``pathlib.Path`` objects pointing to the written files. It is useful for programmatic inspection or other downstream processing.



Viewing the dashboard
^^^^^^^^^^^^^^^^^^^^^^

.. warning::
  SimWrapper requires a Chromium-based browser (such as Google Chrome or Microsoft Edge) to access local files.

1. Open `simwrapper.app <https://simwrapper.app/>`_ or `explore.outerloop.io <https://explore.outerloop.io/>`_ in your browser.
2. Select "view local files" then select the project folder (the one containing ``project_database.sqlite``)


Notes
-----


- The YAML dashboard references data files using a relative path, so the dashboard may be shared by sharing the entire project.
- If your project contains many results scenarios, use ``--max-results-tables`` or ``results_tables`` to control which scenarios are included; the generator attempts to pick the most recent scenarios automatically when not specified. Selection is based on the ``timestamp`` field when present (most recent first), with a fallback ordering by ``table_name`` when timestamps are not available. When centroid link types are not specified the generator attempts to infer them from the project's link types (names containing "centroid" or "connector") or from the links table.
- If you do not want any results scenarios included, pass ``--max-results-tables 0`` to the CLI or ``max_results_tables=0`` to the Python API; the generator will omit results panels from the dashboard in that case.


