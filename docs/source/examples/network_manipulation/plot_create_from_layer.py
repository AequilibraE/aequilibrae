"""
.. _project_from_link_layer:

Create project from a link layer
================================

In this example, we show how to create an empty project and populate it with a
network coming from a link layer we load from a text file. It can easily be
replaced with a different form of loading the data (GeoPandas, for example).

We use Folium to visualize the resulting network.
"""

# %%
# .. admonition:: References
#
#   * :doc:`../../aequilibrae_project/project_components`

# %%
# .. seealso::
#     Several functions, methods, classes and modules are used in this example:
#
#     * :func:`aequilibrae.project.network.links`
#     * :func:`aequilibrae.project.network.nodes`
#     * :func:`aequilibrae.project.network.modes`
#     * :func:`aequilibrae.project.network.link_types`

# %%

# Imports
import urllib.request
from os.path import join
from string import ascii_lowercase
from tempfile import gettempdir
from uuid import uuid4

import geopandas as gpd
import pandas as pd
from shapely.wkt import loads as load_wkt

from aequilibrae import Project

# sphinx_gallery_thumbnail_path = '../source/_images/plot_from_layer.png'

# %%

# We create an empty project on an arbitrary folder
fldr = join(gettempdir(), uuid4().hex)

project = Project()
project.new(fldr)

# %%
# Now we obtain the link data for our example (in this case from a link layer
# we will download from the AequilibraE website).
# With data, we load it on Pandas
dest_path = join(fldr, "queluz.csv")
urllib.request.urlretrieve("https://aequilibrae.com/data/queluz.csv", dest_path)

df = pd.read_csv(dest_path)
df = gpd.GeoDataFrame(df.drop(columns=["fid", "WKT"]), geometry=gpd.GeoSeries.from_wkt(df["WKT"]))

# %%
# Let's see if we have to add new link_types to the model before we add links
# The links we have in the data are:
link_types = df.link_type.unique()

# %%
# And the existing link types are
lt = project.network.link_types
existing_types = [link_type.link_type for link_type in lt]

# %%
# We could also get it directly from the project database

# with project.db_connection as conn:
#     existing_types = [x[0] for x in conn.execute('Select link_type from link_types')]

# %%
# We add the link types that do not exist yet.
# The trickier part is to choose a unique link type ID for each link type.
# You might want to tailor the link type for your use, but here we get letters
# in alphabetical order.

# %%
types_to_add = [ltype for ltype in link_types if ltype not in existing_types]
for i, ltype in enumerate(types_to_add):
    lt.insert(
        link_type_id=ascii_lowercase[i],
        link_type=ltype,
        # description='Your custom description here if you have one',
    )

# %%
# We need to use a similar process for modes
md = project.network.modes
existing_modes = {mode.mode_id: mode.mode_name for mode in md}

# %%
# Now let's see the modes we have in the network that we DON'T have already in
# the model.

# %%
# We get all the unique mode combinations and merge them into a single string
all_variations_string = "".join(df.modes.unique())

# We then get all the unique modes in that string above
all_modes = set(all_variations_string)

# This would all fit nicely in a single line of code, btw. Try it!

# %%
# Now let's add any new mode to the project
modes_to_add = [mode for mode in all_modes if mode not in existing_modes]
for mode_id in modes_to_add:
    # You would need to figure out the right name for each one, but this will do
    md.insert(mode_id=mode_id, mode_name=f"Mode_from_original_data_{mode_id}")
    # description='Your custom description here if you have one'

# %%
# We cannot use the existing link_id, so we create a new field to not loose
# this information
links = project.network.links
link_data = links.fields

# Create the field and add a good description for it
link_data.add("source_id", "link_id from the data source")

# %%
# We can now add all links to the project!
links.insert_from(df.assign(source_id=df["link_id"]))

# %%
# We grab all the links data as a geopandas GeoDataFrame so we can process it easier
links = project.network.links.data

# %%
# Let's plot our network!
links.explore(color="blue", style_kwds={"weight": 2}, tooltip="link_type")

# %%
project.close()
