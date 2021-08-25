"""
Project from a link layer in Geopackage
=======================================

On this example we show how to create an empty project and populate it with a
network coming from a link layer we load with GeoPandas. It can easily be
replaced with a different form of loading the data

We use KeplerGL to visualize the resulting network
"""

# %%
## Imports
from uuid import uuid4
from tempfile import gettempdir
from os.path import join
from aequilibrae import Project
import geopandas as gpd
import folium
import urllib.request
from string import ascii_lowercase

# %%
# We create an empty project on an arbitrary folder
fldr = join(gettempdir(), uuid4().hex)
project = Project()
project.new(fldr)
# %%
# Now we obtain the link data for our example (in this case from a link layer
# we will download from the AequilibraE website)
# With data, we load it on GeoPandas
dest_path = join(fldr, "queluz.gpkg")
urllib.request.urlretrieve('https://aequilibrae.com/data/queluz.gpkg', dest_path)

gdf = gpd.read_file(dest_path)

# %%
# Let's see what we have here

gdf.head()

# %%
# Let's see if we have to add new link_types to the model before we add links
# The links we have in the data are:
link_types = gdf.link_type.unique()

# %%
# And the existing link types are
lt = project.network.link_types
lt_dict = lt.all_types()
existing_types = [ltype.link_type for ltype in lt_dict.values()]

# We could also get it directly from the project database
# existing_types = [x[0] for x in project.conn.execute('Select link_type from link_types')]


# %%
# We add the link types that do not exist yet
# The trickier part is to choose a unique link type ID for each link type
# You might want to tailor the link type for your use, but here we get letters
# in alphabetical order

types_to_add = [ltype for ltype in link_types if ltype not in existing_types]
for i, ltype in enumerate(types_to_add):
    new_type = lt.new(ascii_lowercase[i])
    new_type.link_type = ltype
    # new_type.description = 'Your custom description here if you have one'
    new_type.save()

# %%
# We need to use a similar process for modes

md = project.network.modes
md_dict = md.all_modes()
existing_modes = {k: v.mode_name for k, v in md_dict.items()}

# %%
# Now let's see the modes we have in the network that we DON'T have already in
# the model

# We get all the unique mode combinations and merge into a single string
all_variations_string = ''.join(gdf.modes.unique())

# We then get all the unique modes in that string above
all_modes = set(all_variations_string)

# This would all fit nicely in a single line of code, btw. Try it!

# %%
# Now let's add any new mode to the project

modes_to_add = [mode for mode in all_modes if mode not in existing_modes]
for i, mode_id in enumerate(modes_to_add):
    new_mode = md.new(mode_id)
    # You would need to figure out the right name for each one, but this will do
    new_mode.mode_name = f'Mode_from_original_data_{mode_id}'
    # new_type.description = 'Your custom description here if you have one'

    # It is a little different, because you need to add it to the project
    project.network.modes.add(new_mode)
    new_mode.save()

# %%
## We cannot use the existing link_id, so we create a new field to not loose
# this information
links = project.network.links
link_data = links.fields()
# Create the field and add a good description for it
link_data.add('source_id', 'link_id from the data source')

# We need to refresh the fields so the adding method can see it
links.refresh_fields()


# %%
## We can now add all links to the project!
print('start adding')
for idx, record in gdf.iterrows():
    new_link = links.new()

    # Now let's add all the fields we had
    new_link.source_id = record.link_id
    new_link.direction = record.direction
    new_link.modes = record.modes
    new_link.link_type = record.link_type
    new_link.name = record.name
    new_link.geometry = record.geometry
    new_link.save()

    # We only do this to clear memory
    links.refresh()


# # %%
# # Let's load it on KeplerGL ?
# from keplergl import KeplerGl
#
# # We need a little trick to load the project spatialite in GeoPandas
# sql = "SELECT link_id, name, link_type, modes, Hex(ST_AsBinary(geometry)) geometry FROM links;"
# links_layer = gpd.GeoDataFrame.from_postgis(sql, project.conn, geom_col="geometry")
#
# # Let's add the nodes too for good measure
# sql = "SELECT node_id, Hex(ST_AsBinary(geometry)) geometry FROM nodes;"
# nodes_layer = gpd.GeoDataFrame.from_postgis(sql, project.conn, geom_col="geometry")
#
# # Then we can create the map, add the layer and display it
# map_1 = KeplerGl(height=400)
# map_1.add_data(links_layer, 'links')
# map_1.add_data(nodes_layer, 'nodes')
# map_1
#
#
# # %%
# # We grab all the links data as a Pandas dataframe so we can process it easier
# links = project.network.links.data
#
# # We create a Folium layer
# network_links = folium.FeatureGroup("links")
#
# # We do some Python magic to transform this dataset into the format required by Folium
# # We are only getting link_id and link_type into the map, but we could get other pieces of info as well
# for i, row in links.iterrows():
#     points = row.geometry.to_wkt().replace('LINESTRING ', '').replace('(', '').replace(')', '').split(', ')
#     points = '[[' + '],['.join([p.replace(' ', ', ') for p in points]) + ']]'
#     # we need to take from x/y to lat/long
#     points = [[x[1], x[0]] for x in eval(points)]
#
#     line = folium.vector_layers.PolyLine(points, popup=f'<b>link_id: {row.link_id}</b>', tooltip=f'{row.link_type}',
#                                          color='blue', weight=10).add_to(network_links)
#
# # %%
# # We get the center of the region we are working with some SQL magic
# curr = project.conn.cursor()
# curr.execute('select avg(xmin), avg(ymin) from idx_links_geometry')
# long, lat = curr.fetchone()
#
# # %%
# map_osm = folium.Map(location=[lat, long], zoom_start=14)
# network_links.add_to(map_osm)
# folium.LayerControl().add_to(map_osm)
# map_osm
#
# # %%
# project.close()
#
# # %%
# # **Don't know Queluz? Here is a picture of its most impressive urban structure**
#
# # %%
# from PIL import Image
# import matplotlib.pyplot as plt
#
# pic = 'https://upload.wikimedia.org/wikipedia/commons/2/2c/Ponte_Governador_Mario_Covas_01.jpg'
# pic_local = join(fldr, 'queluz.jpg')
# urllib.request.urlretrieve(pic, pic_local)
#
# img = Image.open(pic_local)
# plt.imshow(img)
#
