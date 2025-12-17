"""
.. _logging_to_terminal:

Logging to terminal
===================

This example demonstrates how to configure and use the logging system in AequilibraE.

For details of the logging setup used with AequilibraE see :ref:`logging_internals`.

AequilibraE uses the standard Python `logging` module, but with specific configurations to ensure compatibility with:

1. Progress bars (tqdm), so logs do not break the visual bars in the terminal/notebook.

2. QGIS integration.

3. Multithreaded C++/Cython extensions.
"""

# %%
# Basic Configuration
# -------------------
# By default, AequilibraE does not configure the root logger to avoid interfering with
# other libraries or your own script's settings. To see AequilibraE logs in your
# terminal or notebook, you should use the provided helper.

import logging
from aequilibrae.utils.logging_utils import basic_config

# %%
# This helper installs a custom StreamHandler that writes to ``stdout`` via ``tqdm.write``,
# ensuring that log messages print cleanly above active progress bars.

# Configure logging to INFO level
handler = basic_config(level=logging.INFO)

# Now we can log
logger = logging.getLogger("aequilibrae")
logger.info("This is a standard info message.")

# %%
# How it works
# ------------
# When :py:func:`aequilibrae.utils.logging_utils.basic_config` is called:
#
# 1. It sets the level for the ``"aequilibrae"`` logger.
# 2. It sets ``logger.propagate = False``. This prevents AequilibraE logs from bubbling up
#    to the root logger. This is crucial because standard root handlers might print directly
#    to streams that conflict with the terminal-based progress bars.
# 3. It attaches an ``aequilibrae.utils.logging_utils.AequilibraEStreamHandler`` which preserves the terminal-based
#    progress bars.
#
# You can customise the format and stream:
logger.removeHandler(handler)
basic_config(level=logging.DEBUG, format="%(levelname)s: %(message)s")
logger.debug("This is a debug message with a custom format.")

# %%
# Note that we removed the first handler that ``basic_config`` added otherwise the message will appear twice. If the
# ``"aequilibrae"`` has any loggers then ``basic_config`` will do not do anything.

# %%
# Configuring Specific Modules
# ----------------------------
# All AequilibraE logging happens under the ``"aequilibrae"`` namespace. You can fine-tune
# logging for specific modules if you want to debug only one part of the system.

# Example: Turn on DEBUG logging only for the paths module
paths_logger = logging.getLogger("aequilibrae.paths")
paths_logger.setLevel(logging.DEBUG)

# %%
# The parent ``"aequilibrae"`` logger might be set to WARNING, but this specific module
# will now output DEBUG messages.

# %%
# Scenario-Based Logging
# ----------------------
# AequilibraE Projects handle logging automatically when switching scenarios.
#
# When you open a project athe Project
# attaches a specific `FileHandler` to the logger.
#
# * **One Log per Scenario:** If you run multiple scenarios, AequilibraE automatically
#   switches the log file handler to a new file located in the scenario directory.
# * **Auto-Switching:** You do not need to manually close or remove handlers; the
#   logging system handles the rotation to ensure logs from scenario A don't end up in scenario B's file.
