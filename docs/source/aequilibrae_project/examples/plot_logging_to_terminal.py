"""
.. _logging_to_terminal:

Logging to terminal
===================

In this example, we show how to make all log messages show in the terminal.
"""

# %%

# Imports
from uuid import uuid4
from tempfile import gettempdir
from os.path import join
from aequilibrae.utils.create_example import create_example
import logging
import sys
# sphinx_gallery_thumbnail_path = '../source/images/plot_logging_to_terminal_image.png'

# %%

# We create the example project inside our temp folder
fldr = join(gettempdir(), uuid4().hex)
project = create_example(fldr)
logger = project.logger

# %%
# With the project open, we can tell the logger to direct all messages to the terminal as well
stdout_handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter("%(asctime)s;%(levelname)s ; %(message)s")
stdout_handler.setFormatter(formatter)
logger.addHandler(stdout_handler)

# %%
project.close()
