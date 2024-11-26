Contributing to AequilibraE
===========================

This page presents some initial instructions on how to set up your system to start contributing to 
AequilibraE and lists the requirements for all pull requests to be merged into master.

.. note::
   The recommendations on this page are current as of October 2021.

Software Design and requirements
--------------------------------

The most important piece of AequilibraE's backend is, without a doubt, `NumPy <http://numpy.org>`__.

Whenever vectorization is not possible through the use of NumPy functions, compiled code is developed in order to
accelerate computation. All compiled code is written in `Cython <https://cython.org/>`_.

We have not yet found an ideal source of recommendations for developing AequilibraE, but a good initial take can be
found in `this article <https://doi.org/10.1371/journal.pbio.1001745>`_.

Development Install
-------------------

As it goes with most Python packages, we recommend using a dedicated virtual environment to develop AequilibraE.

AequilibraE is currently tested for Python 3.9, 3.10, 3.11 & 3.12, but we recommend using Python 3.11 or 3.12 for 
development.

We also assume you are using `PyCharm <https://www.jetbrains.com/pycharm>`_ or 
`VSCode <https://code.visualstudio.com/>`_ which are awesome IDEs for Python.

If you are using a different IDE, we would welcome if you could contribute with instructions to set that up.

Non-Windows
~~~~~~~~~~~
::

  ./ci.sh setup_dev

Windows
~~~~~~~

Make sure to clone the AequilibraE repository and run the following from within that cloned repo using an elevated command prompt.

Python 3.12 (or whatever version you chose) needs to be installed, and the following instructions assume you are 
using `Chocolatey <https://chocolatey.org/>`_ as a package manager.

::

  cinst python3 --version 3.12
  cinst python

  set PATH=C:\Python312;%PATH%
  python -m pip install pipenv
  virtualenv .venv #Only if you want to save the virtual environment in the same folder
  python -m pipenv install --dev
  python -m pipenv run pre-commit-install

Setup Pycharm with the virtual environment you just created.

::

  Settings -> Project -> Project Interpreter -> Gear Icon -> Add -> Existing VEnv

Development Guidelines
-----------------------

AequilibraE development (tries) to follow a few standards. Since this is largely an after-the-fact concern, several
portions of the code are still not up to such standards.

Style
~~~~~

* Python code should follow (mostly) the `pycodestyle style guide <https://pycodestyle.pycqa.org/en/latest/>`_
* Python docstrings should follow the `reStructuredText Docstring Format <https://www.python.org/latest/peps/pep-0287/>`_
* We are big fans of auto-code formatting. For that, we use `ruff <https://docs.astral.sh/ruff/>`_ and 
  `Black <https://black.readthedocs.io/en/stable/>`_.
* Negating some of what we have said so far, we use maximum line length of 120 characters

Imports
~~~~~~~

* Imports should be one per line.
* Imports should be grouped into standard library, third-party, and intra-library imports 
  (``CTRL + SHIFT + o`` does it automatically on PyCharm).
* Imports of NumPy and Pandas should follow the following convention:

.. code-block:: python

    import numpy as np
    import pandas as pd

Contributing to AequilibraE
~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
~~~~~~~~~~~~~~~~~

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
~~~~~~~~

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

Documentation
~~~~~~~~~~~~~~

All the AequilibraE documentation is (unfortunately) written in 
`reStructuredText <http://docutils.sourceforge.net/rst.html>`_  and built with 
`Sphinx <http://www.sphinx-doc.org/en/stable/>`_.
Although reStructuredText is often unnecessarily convoluted to write, Sphinx is capable of converting it to standard-
looking HTML pages, while also bringing the docstring documentation along for the ride.

To build the documentation, first make sure the required packages are installed. If you have correctly setup the dev
environment above, then nothing else is needed. However, if you have incorrectly only run::

    python -m pipenv install

Then you will have to run::

    python -m pipenv install --dev

Next, build the documentation in html format with the following commands run from the ``root`` folder::

    sphinx-apidoc -T -o docs/source/generated aequilibrae
    cd docs
    make html

Working with progress bars
~~~~~~~~~~~~~~~~~~~~~~~~~~

From version 1.1.0, AequilibraE is capable of displaying progress bars in Jupyter Notebooks using 
`TQDM <https://tqdm.github.io/>`_. For the companion QGIS plugin, `PyQt5 <https://doc.qt.io/qtforpython-5/>`_
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
~~~~~~~~~

AequilibraE releases are automatically uploaded to the `Python Package Index
<https://pypi.python.org/pypi/aequilibrae>`_ (PyPi) at each new GitHub release (2 to 6 times per year).

Acknowledgement
~~~~~~~~~~~~~~~

A LOT of the structure around the documentation was borrowed (copied) from the excellent project `ActivitySim
<https://activitysim.github.io/>`_.