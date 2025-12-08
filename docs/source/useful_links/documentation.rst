:orphan:

.. _guide_to_docs:

Documentation
=============

AequilibraE's documentation is built with 
`PyData Sphinx Theme <https://pydata-sphinx-theme.readthedocs.io/en/stable/>`_.

In the following sections, we briefly present some useful documentation writing tips.

conf.py file
------------

It stores the configuration of the documentation. For instance what extensions are we using, how to
build the examples gallery, HTML set up, and PDF building, but also where custom theme configs
are stored.

Regarding the extensions, we mostly use Sphinx built-in extensions that should be loaded. However,
to enhance rendering, we use external extensions. When using external extensions, remember to
add/remove them at the TOML.

.. code-block:: python

    extensions = [
        "sphinx.ext.autodoc",
        "sphinx.ext.napoleon",
        "sphinx.ext.mathjax",
        "sphinx.ext.viewcode",
        "sphinx.ext.autosummary",
        "sphinx.ext.doctest",
        "sphinx_gallery.gen_gallery",  # renders the examples gallery
        "sphinx_design",  # adds grids and cards
        "sphinx_copybutton",  # adds the copy button to code blocks
        "sphinx_git",  # adds SHA info to the version history
        "sphinx_tabs.tabs",  # adds tabs 
        "sphinx_subfigure",  # allows placing more than one figure side by side
    ]

LaTeX PDF builder
-----------------

To build the docs PDF file, we use LaTeX engine ``xelatex`` in the conf.py file.

Not everything we have in the docs folder goes to the PDF file. Thus we created an index file
``_latex/index.rst`` containing the sequence and the contents that compose the PDF file. Any
changes in the docs structure need to appear here too.

If you want to build the PDF in your machine, make sure that the following packages are also
installed.

.. code-block::

    sudo apt install -y latexmk texlive-xetex fonts-freefont-otf xindy

API documentation
-----------------

We generate the API documentation automatically using ``_autosummary``. The outputs are placed
into a homonymous folder inside ``docs/source/useful_links``.

The API outputs in autosummary are generated based on the ``aequilibrae`` folder. To use the
API docs within other files, add a reference to the file path based on the same folder structure.
For example, the class documentation for ``Parameters`` is in ``aequilibrae/parameters.py`` so
we refer to it as:

.. code-block:: text

    * :func:`aequilibrae.parameters.Parameters`
        Class documentation

Docstrings
----------

We have some docstrings in the API documentation that are tested at ``documentation.yml`` workflow.
Although the functions and modules the docstrings refer to are properly tested in unit tests, 
testing the docstrings ensure that in-line examples we display to the user are also correct and
up to date.

In AequilibraE, we have docstrings on python and RST files. To write them, you can use the 
`Pandas docstring guide <https://pandas.pydata.org/docs/development/contributing_docstring.html>`_.

To make the code clean when using docstrings, we can use pytest fixtures and
`doctest directives <https://docs.python.org/3/library/doctest.html>`_. For docstrings in RST
files, we have to create the fixtures. In our current workflow, this is done when running the
``docs/create_docs_data.py``. Unlike docstrings in Python files that are independent from each
other, docstrings in the same RST file correspond to one very large code block. So make sure to
open/close projects when needed. 

When writing docstrings, don't forget to add them at the testing workflow in ``documentation.yml``.

Writing examples
----------------

All AequilibraE examples consist in rendered RST from a python file (.py). The extension that
does this job is `Sphinx-Gallery <https://sphinx-gallery.github.io/stable/getting_started.html>`_.

By default only files prefixed with `plot_` will be executed.

When writing the example, we usually add a file reference to the docstring, so the user can
be easily redirected to the example (and you don't relay on relative paths).

Adding a thumbnail is also a plus and we place this information after the libraries imports. 
We use  the option ``# sphinx_gallery_thumbnail_path`` and the path to file. Make sure the path
is correct, otherwise no thumbnail is displayed.

We can add `admonitions and directives <https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html#directives>`_
to the example file.

Check out `Sphinx-Gallery docs <https://sphinx-gallery.github.io/stable/syntax.html>`_ or 
one of the `examples in gallery <https://sphinx-gallery.github.io/stable/auto_examples/index.html>`_
for more information.

Version history
---------------

Version history file is manually updated when we have new releases. Currently we have rows
with three grid-cards each with the version number. 

To follow the current pattern, we can add empty cards (``.. grid-item::``) if the line has less
than 3 cards. Not adding grid-item makes the cards with contents to 'adjust' themselves in the
same column length, leaving the row with more or less than 3 columns.

The code block below presents an example with two rows, represented by ``.. grid::``, where the
latter line has two empty cards.

.. code-block:: text

    .. grid::

        .. grid-item-card:: 1.4.2
            :link:  https://www.aequilibrae.com/docs/python/v1.4.2/
            :link-type: url
            :text-align: center

        .. grid-item-card:: 1.4.3
            :link:  https://www.aequilibrae.com/docs/python/v1.4.3/
            :link-type: url
            :text-align: center

        .. grid-item-card:: 1.5.0
            :link:  https://www.aequilibrae.com/docs/python/v1.5.0/
            :link-type: url
            :text-align: center
 
    .. grid::

        .. grid-item-card:: Upcoming version
            :link:  https://www.aequilibrae.com/develop/python/index.html
            :link-type: url
            :text-align: center

        .. grid-item::

        .. grid-item::

Build documentation
-------------------

In this sections, we go over the statements in the 'build documentation' step of the
``documentation.yml`` file. To check if the documentation is building properly, you can run
these steps in your local machine.

.. code-block:: text

    # We convert the IPF_benchmark in the Jupyter Notebook to a .rst file.
    jupyter nbconvert --to rst docs/source/distribution_procedures/IPF_benchmark.ipynb
    # We build the .tex file without plotting the gallery. Some examples display progress
    # bars, and its characters usually don't render properly. The code blocks are 
    # rendered, but not the outputs.
    sphinx-build -b latex docs/source docs/source/_static/latex -D plot_gallery=False
    # Let's go to the folder where our .tex file is
    cd docs/source/_static/latex
    # And render the PDF.
    LATEXMKOPTS="-xelatex" make all-pdf
    # Go back to docs root
    cd ../../../..
    # We copy the AequilibraE icon to the source/_static folder because it is one of the
    # folders that go to the build
    cp large_icon.png docs/source/_static/large_icon.png
    # We build the documentation at the build folder (this runs all examples)
    sphinx-build -M html docs/source docs/build
    # We zip the entire HTML documentation in a file named 'aequilibrae.zip'
    python -m zipfile -c aequilibrae.zip docs/build/html
    # And copy this file to the build/html folder (because the docs are already rendered
    # and refer to this location).
    cp aequilibrae.zip docs/build/html/aequilibrae.zip

Uploading docs to S3
--------------------

To upload AequilibraE docs to S3, the steps are:

1. Prepare the links
2. Upload the Python documentation
3. Upload the home page

If you look at the ``documentation.yml``, you notice conditional steps for link preparation.
By default, all external links for the AequilibraE webpage refer to LATEST. When uploading
docs to DEV or DEVELOP, these links are changed in one of 'prepare links for DEV' or
'prepare links for DEVELOP' steps.

When preparing links for DEV, each pull request has its own documentation page, so changes
in the documentation from different PR's won't overwrite the one in the branch you're working.

Because we have a home page that does not correspond to ``index.rst`` we have to specifically
declare it when uploading the docs. We use two different `actions <https://github.com/jakejarvis/s3-sync-action>`_
to sync the docs with S3. The code block below is an example. What differentiates the steps
are ``args`` and ``DEST_DIR``. When building the docs with ``make html``, ``home.html`` and
the contents of the so called 'python' folder are at the same level, so we have to upload
them to the right folder.

.. code-block:: text

    # We first upload the contents of the 'python' folder ('DEST_DIR'), but when
    # deleting the files in the bucket that are not present in the latest version of 
    # the repository build, we end up losing our home page. To prevent it, we use the
    # exclude flag to indicate that we don't want to delete the 'home.html' file.
    - name: Upload python to DEV on S3
      if: ${{(github.event_name == 'pull_request') && (env.HAS_SECRETS == 'true')}}
      uses: jakejarvis/s3-sync-action@master
      with:
        args: --acl public-read --follow-symlinks --delete --exclude 'home.html'
      env:
        AWS_S3_BUCKET: ${{ secrets.AWS_S3_BUCKET }}
        AWS_ACCESS_KEY_ID: ${{ secrets.AWS_ACCESS_KEY_ID }}
        AWS_SECRET_ACCESS_KEY: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
        AWS_REGION: 'us-east-1'
        SOURCE_DIR: 'docs/build/html/'
        DEST_DIR: 'dev/${{ github.event.number }}/python/'

    # Then we upload the home page and its related contents. The argument here is different
    # from the one above. Notice that we exclude the existing contents in the 'python' 
    # folder from the command, and we include the 'home.html', its images and indexes
    # (one per flag). All the data is uploaded to the "root" of 'dev/number_of_pull_request'.
    - name: Upload home page to DEV on S3
      if: ${{(github.event_name == 'pull_request') && (env.HAS_SECRETS == 'true')}}
      uses: jakejarvis/s3-sync-action@master
      with:
        args: --acl public-read --follow-symlinks --exclude '*' --include 'home.html' --include '_images/sponsor*' --include '_images/banner*' --include '_static/*' --include 'search*' --include 'genindex.html' --include '_sphinx_design_static/*'
      env:
        AWS_S3_BUCKET: ${{ secrets.AWS_S3_BUCKET }}
        AWS_ACCESS_KEY_ID: ${{ secrets.AWS_ACCESS_KEY_ID }}
        AWS_SECRET_ACCESS_KEY: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
        AWS_REGION: 'us-east-1'
        SOURCE_DIR: 'docs/build/html/'
        DEST_DIR: 'dev/${{ github.event.number }}/'
