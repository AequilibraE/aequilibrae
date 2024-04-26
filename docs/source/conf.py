# -*- coding: utf-8 -*-
#
# Configuration file for the Sphinx documentation builder.
#
# This file does only contain a selection of the most common options. For a
# full list see the documentation:
# http://www.sphinx-doc.org/en/master/config

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

import os
import sys
from datetime import datetime
from pathlib import Path
from sphinx_gallery.sorting import ExplicitOrder

project_dir = Path(__file__).parent.parent.parent
if str(project_dir) not in sys.path:
    sys.path.append(str(project_dir))

# Sometimes this file is exec'd directly from sphinx code...
project_dir = os.path.abspath("../../")
if str(project_dir) not in sys.path:
    sys.path.insert(0, project_dir)

from __version__ import release_version

# -- Project information -----------------------------------------------------

project = "AequilibraE"
copyright = f"{str(datetime.now().date())}, AequilibraE developers"
author = "Pedro Camargo"

# The short X.Y version
version = release_version
if ".dev" in version:
    switcher_version = "dev"
elif "rc" in version:
    switcher_version = version.split("rc")[0] + " (rc)"
else:
    switcher_version = version

# -- General configuration ---------------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
#
# needs_sphinx = '1.0'

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosummary",
    "sphinx_gallery.gen_gallery",
    "sphinx_design",
    "sphinx_copybutton",
    "sphinx_git",
]

# Change plot_gallery to True to start building examples again
sphinx_gallery_conf = {
    "examples_dirs": ["examples"],  # path to your example scripts
    "gallery_dirs": ["_auto_examples"],  # path to where to save gallery generated output
    'capture_repr': ('_repr_html_', '__repr__'),
    'remove_config_comments': True,
    "subsection_order": ExplicitOrder(["examples/creating_models",
                                      "examples/editing_networks",
                                      "examples/trip_distribution",
                                      "examples/visualization",
                                      "examples/aequilibrae_without_a_model",
                                      "examples/full_workflows",
                                      "examples/other_applications"]),
    'plot_gallery': 'False',
}

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# The suffix(es) of source filenames.¶
# You can specify multiple suffix as a list of string:
#
# source_suffix = ['.rst', '.md']
source_suffix = ".rst"

# The master toctree document.
master_doc = "index"

# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
#
# This is also used if you do content translation via gettext catalogs.
# Usually you set "language" from the command line for these cases.
language = "en"

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path .
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "sphinx"
highlight_language = "python3"

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "pydata_sphinx_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

html_theme_options = {
    "show_nav_level": 0,
    "navbar_center": ["navbar-nav"],
    "navbar_end": ["theme-switcher", "navbar-icon-links"],
    "navbar_start": ["navbar-logo", "version-switcher"],
    "navbar_align": "left",
    "switcher": {
        "json_url": "https://www.aequilibrae.com/python/latest/_static/switcher.json",
        "version_match": switcher_version,
    },
    # "check_switcher": False,
    "github_url": "https://github.com/AequilibraE/aequilibrae",
}

# The name for this set of Sphinx documents.  If None, it defaults to
html_title = f"AequilibraE {version}"

# -- Options for HTMLHelp output ---------------------------------------------

# Output file base name for HTML help builder.
htmlhelp_basename = "AequilibraEdoc"

# -- Options for LaTeX output ------------------------------------------------

latex_elements = {}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title,
#  author, documentclass [howto, manual, or own class]).
latex_documents = [(master_doc, "AequilibraE.tex", "AequilibraE Documentation", "Pedro Camargo", "manual")]

latex_appendices = ["getting_started", 
                    "validation_benchmarking/ipf_performance", 
                    "validation_benchmarking/traffic_assignment",
                    "modeling_with_aequilibrae/project_database/importing_from_osm",
                    "modeling_with_aequilibrae/project_database/importing_from_gmns",
                    "modeling_with_aequilibrae/project_database/exporting_to_gmns",
                    ]

# -- Options for manual page output ------------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [(master_doc, "aequilibrae", "AequilibraE Documentation", [author], 1)]

# -- Options for Texinfo output ----------------------------------------------

autodoc_default_options = {
    "members": True, 
    "inherited-members": True,
    "member-order": "bysource",
    "special-members": False,
    "private-members": False,
    "undoc-members": True,
    # "exclude-members": "__weakref__",
}

# autodoc_member_order = "groupwise"

autoclass_content = "class"  # classes should include both the class' and the __init__ method's docstring

autosummary_generate = True

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
    (
        master_doc,
        "AequilibraE",
        "AequilibraE Documentation",
        author,
        "AequilibraE",
        "One line description of project.",
        "Miscellaneous",
    )
]
