"""
.. _upgrade_project_db_example:

Upgrade project database
========================

In this example, we show how to upgrade a project database to the latest version.
This is useful when you need to use the latest AequilibraE's database schemas or formats.
"""

# %%
# .. admonition:: References
#
#    * :ref:`database_migration`

# %%
# .. seealso::
#     Functions used in this example:
#
#     * :func:`aequilibrae.project.project.Project.upgrade`

# %%

# Imports
from uuid import uuid4
from tempfile import gettempdir
from os.path import join

from aequilibrae.project.tools import MigrationManager
from aequilibrae.utils.create_example import create_example
from aequilibrae.utils.spatialite_utils import connect_spatialite

# %%

# We create an empty project on an arbitrary folder
fldr = join(gettempdir(), uuid4().hex)

# Let's use Sioux Falls project
project = create_example(fldr)

# %%
# To upgrade all database migrations in a single transaction, we can use:

# project.upgrade()

# %%
# However, it is possible to upgrade only the project database.
project.upgrade(ignore_transit=True, ignore_results=True)

# %%
# Finally, we close the project
project.close()
