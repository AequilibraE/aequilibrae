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
os.environ["AEQ_ENSURE_SPATIALITE"] = "NO"


# We can also tell AequilibraE where to find spatialite in your system
# You can permanently set Spatialite in your system by following the instructions in this
# blog post: https://www.xl-optim.com/spatialite-and-python-in-2020/
set_known_spatialite_folder(".")

# Or we can simply tell AequilibraE to download the binaries to a specific folder
# and set it up for use
desired_spatialite_folder = join(gettempdir(), "spatialite")
ensure_spatialite_binaries(desired_spatialite_folder)


# %%
# Now we can go about our business as usual
project = create_example(join(gettempdir(), uuid4().hex))
project.close()
