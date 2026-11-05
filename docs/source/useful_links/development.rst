:orphan:

.. _developing_aequilibrae:

Developing
----------

This section describes how to contribute to AequilibraE's development and what
is our current roadmap.

Contributing to AequilibraE
~~~~~~~~~~~~~~~~~~~~~~~~~~~

This page presents some initial instructions on how to set up your system to start contributing to 
AequilibraE and lists the requirements for all pull requests to be merged into master.

.. note::
   The recommendations on this page are current as of October 2021.

Software Design and requirements
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The most important piece of AequilibraE's backend is, without a doubt, `NumPy <http://numpy.org>`__.

Whenever vectorization is not possible through the use of NumPy functions,
compiled code is developed in order to accelerate computation. We use `Cython
<https://cython.org/>`_ to integrate compiled code written in Cython and
C/C++. You can find documentation of AequilibraE's internal model here
:doc:`Internals <internals>`.

We have not yet found an ideal source of recommendations for developing AequilibraE, but a good initial take can be
found in `this article <https://doi.org/10.1371/journal.pbio.1001745>`_.

Development Install
^^^^^^^^^^^^^^^^^^^

As it goes with most Python packages, we recommend using a dedicated virtual environment to develop AequilibraE.

AequilibraE is currently tested for Python 3.11, 3.12, 3.13 & 3.14, but we recommend using Python 3.12 or 3.13 for
development.

We also assume you are using `PyCharm <https://www.jetbrains.com/pycharm>`_ or 
`VSCode <https://code.visualstudio.com/>`_ which are awesome IDEs for Python.

If you are using a different IDE, we would welcome if you could contribute with instructions to set that up.

Non-Windows
+++++++++++
::

  ./ci.sh setup_dev

Windows
+++++++

Make sure to clone the AequilibraE repository and run the following from within that cloned repo using an elevated command prompt.

Python 3.12 (or whatever version you chose) needs to be installed, and the following instructions assume you are 
using `Chocolatey <https://chocolatey.org/>`_ as a package manager.

::

  cinst python3 --version 3.12
  cinst python

  set PATH=C:\Python312;%PATH%
  python -m venv .venv
  .venv\Scripts\activate
  python -m pip install --upgrade pip
  python -m pip install --group dev -e ".[create]"

Set up PyCharm with the virtual environment you just created.

::

  Settings -> Project -> Project Interpreter -> Gear Icon -> Add -> Existing VEnv

Using Spin
~~~~~~~~~~

AequilibraE uses `Spin <https://scientific-python.org/spin/>`_ to share development commands with GitHub Actions.
Install the build tools and development dependencies in your virtual environment, then create an editable install::

  python -m pip install --upgrade pip
  python -m pip install --group dev -e ".[create]"
  spin install

``spin install`` enables verbose editable rebuilds by default, so importing AequilibraE shows any recompilation.
Keep the build dependencies installed in the same environment, as editable rebuilds need them.

Run the development checks from the repository root::

  spin build
  spin test
  spin lint
  spin doctest

``spin test`` and ``spin doctest`` build the package first. They also work with an editable install created by
``spin install``. Use ``--no-build`` to skip an explicit build when it is not needed.

``spin test`` uses Spin's built-in test command, configured in ``pyproject.toml`` to default to eight workers
(``-j 8``) and pass ``--random-order-bucket=global --random-order --durations=50`` to pytest.
Supplying pytest arguments replaces that entire argument list rather than appending to it. Use ``-j`` to change
the worker count. These defaults do not affect direct pytest invocations. For example::

  spin test tests/aequilibrae/paths
  spin test -j 2 -- -k graph --durations=10

``spin lint`` runs Ruff lint/format checks, cython-lint, clang-tidy and clang-format in check-only mode. Install the LLVM tools
``clang-tidy`` and ``clang-format`` separately using your system package manager. Linting never configures or
builds the package. If there is no compilation database, it warns and skips clang-tidy; the other checks still run.
Use ``spin lint -C BUILD_DIR`` to select an existing build directory.

``spin doctest`` checks the Python docstrings and reStructuredText examples covered by the documentation workflow.
It runs serially by default; pytest arguments can be supplied after ``--``.

The other standard Spin commands are also available:

* ``spin sdist`` and ``spin wheel`` build distribution archives.
* ``spin python``, ``spin ipython`` and ``spin shell`` open an environment using the built package.
* ``spin run`` executes a command using the built package.
* ``spin docs`` builds the Sphinx documentation.
* ``spin gdb`` and ``spin lldb`` launch the corresponding system debugger, which must be installed separately.
* ``spin introspect COMMAND`` shows a command's implementation.

Use ``spin COMMAND --help`` for command-specific options.

Development Guidelines
~~~~~~~~~~~~~~~~~~~~~~

AequilibraE development (tries) to follow a few standards. Since this is largely an after-the-fact concern, several
portions of the code are still not up to such standards.

Style
^^^^^

* Python code should follow (mostly) the `pycodestyle style guide <https://pycodestyle.pycqa.org/en/latest/>`_
* Python docstrings should follow the `reStructuredText Docstring Format <https://www.python.org/latest/peps/pep-0287/>`_
* We are big fans of auto-code formatting. For that, we use `ruff <https://docs.astral.sh/ruff/>`_ and 
  `Black <https://black.readthedocs.io/en/stable/>`_.
* Negating some of what we have said so far, we use maximum line length of 120 characters

Imports
^^^^^^^

* Imports should be one per line.
* Imports should be grouped into standard library, third-party, and intra-library imports 
  (``CTRL + SHIFT + o`` does it automatically on PyCharm).
* Imports of NumPy and Pandas should follow the following convention:

.. code-block:: python

    import numpy as np
    import pandas as pd

Contributing to AequilibraE
^^^^^^^^^^^^^^^^^^^^^^^^^^^

GitHub has a nice visual explanation on how collaboration is done using `GitHub flow
<https://guides.github.com/introduction/flow>`_.

In a nutshell:

1. Fork the repository into your account
2. Write your code
3. Write your tests (and run them locally)
4. Write documentation
5. Issue a Pull request against the ``develop`` branch of AequilibraE

In a more verbose way...

* The ``main`` branch contains the latest release version of AequilibraE.
* The ``develop`` branch contains all new features and bug fixes that will be
  included in the next release. It can be seen as a *release candidate*, so work is not often
  performed on that branch.
* Tests are automatically run on PR's, and thy do not need to be strictly passing for any
  new feature to be merged into the ``develop`` branch, although you shouldn't expect your
  PR to be accepted if it causes too many breaks, lacks tests or documentation.
* The project maintainers have absolute discretion to accept or reject PR's, but reasons
  for refusing contributions will always be made clear on the PR's comments/review.
* Work is done in an issue/feature branch (or a fork) and then pushed to a new branch.
* Automated testing is run using Github Actions. All tests must pass:

  * Unit testing
  * Build/packaging tests
  * Documentation building test

* If the tests pass, then a manual pull request can be approved to merge into develop.
* The ``main`` and ``develop`` branches are protected and therefore can only be written to 
  after the code has been reviewed and approved.
* No individual has the privileges to push to the ``main`` or ``develop`` branches.

Release versions
^^^^^^^^^^^^^^^^

AequilibraE uses the de-facto Python standard for `versioning
<http://the-hitchhikers-guide-to-packaging.readthedocs.io/en/latest/specification.html>`_.

::

  MAJOR.MINOR[.MICRO]

- MAJOR designates a major revision number for the software. Usually, raising a major revision number means that
  you are adding a lot of features, breaking backward-compatibility or drastically changing the API.

- MINOR usually groups moderate changes to the software like bug fixes or minor improvements. Most of the time, end
  users can upgrade with no risks their software to a new minor release. In case an API changes, the end users will be
  notified with deprecation warnings. In other words, API stability is usually a promise between two minor releases.

- Some software use a third level: MICRO. This level is used when the release cycle of minor release is quite long.
  In that case, micro releases are dedicated to bug fixes.

AequilibraE's development is happening mostly within the Minor and Micro levels.

Testing
^^^^^^^

AequilibraE style checking is done with two tools:

* `ruff <https://docs.astral.sh/ruff/>`_, a tool to check Python code style
* `Black <https://black.readthedocs.io/en/stable/>`_, The uncompromising code formatter

And testing is done using `pytest <https://docs.pytest.org/en/stable/>`_.

Testing is done for Windows, MacOs and Ubuntu Linux on all supported Python versions, and we use GitHub Actions
to run these tests. These tests need to pass and additionally somebody has to
manually review the code before merging it into master (or returning for corrections).

In some cases, test targets need to be updated to match the new results produced by the code since these 
are now the correct results.  In order to update the test targets, first determine which tests are 
failing and then review the failing lines in the source files.  These are easy to identify since each 
test ultimately comes down to one of Python's various types of ``assert`` statements.  Once you identify 
which ``assert`` is failing, you can work your way back through the code that creates the test targets in 
order to update it. After updating the test targets, re-run the tests to confirm the new code passes all 
the tests.

Profiling
^^^^^^^^^

As AequilibraE is performance-critical software, changes to its compute-heavy code should be profiled to ensure they
do not degrade performance. Our internal profiling workflows, including how to profile the Cython and C++ extensions
that Python profilers cannot see into, are documented :ref:`in this page <profiling>`.

Documentation
^^^^^^^^^^^^^

All the AequilibraE documentation is (unfortunately) written in
`reStructuredText <http://docutils.sourceforge.net/rst.html>`_  and built with 
`Sphinx <http://www.sphinx-doc.org/en/stable/>`_.
Although reStructuredText is often unnecessarily convoluted to write, Sphinx is capable of converting it to standard-
looking HTML pages, while also bringing the docstring documentation along for the ride.

To build the documentation, first make sure the required packages are installed. If you have set up the development
environment above, then nothing else is needed. In a separate environment, install the development group and create
extra::

    python -m pip install --group dev -e ".[create]"

Next, build the documentation in html format with the following commands run from the ``root`` folder::

    sphinx-apidoc -T -o docs/source/generated aequilibrae
    cd docs
    make html

More information on writing and contributing to AequilibraE's documentation are available
:ref:`in this page <guide_to_docs>`.

Working with progress bars
^^^^^^^^^^^^^^^^^^^^^^^^^^

From version 1.1.0, AequilibraE is capable of displaying progress bars in Jupyter Notebooks using 
`TQDM <https://tqdm.github.io/>`_. For the companion QGIS plugin, `PyQt6 <https://doc.qt.io/qtforpython-6/index.html>`_
is used to emit messages in progress bars.
 
AequilibraE provides a wrapper class `SIGNAL` that will use the appropriate underlying mechanism to display 
the progress bars.

.. code-block:: python

    from aequilibrae.utils.signal import SIGNAL

    class MyClass:
      signal = SIGNAL(object)

      def my_method(self):
        signal.emit(['start', 10, 'running my method']) 
        for i in range(0, 10):
           signal.emit(['update', i, f"Current val: {i}"])
           sleep(0.4)


Calling `MyClass().my_method()` will generate a progress bar in the following form (outside QGIS).

.. code-block:: text

    running my method                                 :  30%|█████▍            | 3/10 [00:01<00:02,  2.50it/s]

The full set of emitted signals which can be used to control progress bars is given in `python_signal.py`

Releases
^^^^^^^^

AequilibraE releases are automatically uploaded to the `Python Package Index
<https://pypi.python.org/pypi/aequilibrae>`_ (PyPi) at each new GitHub release (2 to 6 times per year).

Acknowledgement
^^^^^^^^^^^^^^^

A LOT of the structure around the documentation was borrowed (copied) from the excellent project `ActivitySim
<https://activitysim.github.io/>`_.

Development Roadmap
~~~~~~~~~~~~~~~~~~~

As AequilibraE is a project with an incredibly small team and very little external
funding, it is not feasible to determine a precise schedule for the development
of new features or even a detailed roadmap.

However, there are a number of enhancements to the software that we have already
identified and that we intend to dedicate some time to in the future.

* Network data model

  * Introduce centroid connector data type to replace the inference that all links
    connected to centroids are connectors

* Traffic assignment

  * New origin-based traffic assignment to achieve ultra-converged
    assignment
  * New path-finding algorithm based on contraction-hierarchies

* Public Transport

  * Export of GTFS (enables editing of GTFS in QGIS)

* Project

  * Inclusion of new table for scalar values
  * Inclusion of new table for vectors based on centroid IDs (plus metadata
    table)
  * Inclusion of new table for vectors based on node IDs (plus metadata table)
  * Inclusion of new table for vectors based on link IDs (plus metadata table)

* QGIS

  * Inclusion of TSP and more general vehicle routing problems (resource
    constraints, pick-up, and delivery, etc.)

If there is any other feature you would like to suggest, please record a new
issue on `GitHub <https://github.com/AequilibraE/aequilibrae/issues>`_, or drop
us a line.

If your organization is making use of AequilibraE, please consider funding some
of the new developments or maintenance of the project.
