"""
.. _avoid_spatialite_download_windows:

Avoiding the automatic download of SpatiaLite binaries on Windows
=================================================================

In this example, we show how to prevent Windows from downloading the SpatiaLite binaries automatically.

This may be relevant to users in corporate environments where the download and use of binaries to
the Windows temporary is restricted.

Spatialite Logo by Massimo Zedda, image from https://www.gaia-gis.it/
"""

# %%

# Imports
import os
from os.path import join
from tempfile import gettempdir
from uuid import uuid4

from aequilibrae.utils.create_example import create_example


# sphinx_gallery_thumbnail_path = '../source/_images/plot_spatialite.png'

# %%
from aequilibrae.utils.spatialite_utils import set_known_spatialite_folder, ensure_spatialite_binaries

# First we prevent Windows from downloading spatialite binaries during this session
# THIS VALUE MUST BE UPPER CASE TO BE EFFECTIVE
os.environ["AEQ_SPATIALITE_DIR"] = r"C:\path\to\existing\download"


# %%
# Now we can go about our business as usual
project = create_example(join(gettempdir(), uuid4().hex))
project.close()
