# %%
# Imports
from uuid import uuid4
from tempfile import gettempdir
from os.path import join
from aequilibrae import Project
import folium

from aequilibrae.project.network.ovm_downloader import OVMDownloader
# sphinx_gallery_thumbnail_path = 'images/nauru.png'

# %%
# We create an empty project on an arbitrary folder
fldr = join(gettempdir(), uuid4().hex)
project = Project()
project.new(fldr)
# %%
# Now we can download the network from any place in the world (as long as you have memory for all the download
# and data wrangling that will be done)

# We can create from a bounding box or a named place.
# For the sake of this example, we will choose the small nation of Nauru.
project.network.create_from_ovm(west=148.7077, south=-20.2780, east=148.7324, north=-20.2621, data_source=r'C:\Users\penny\git\Aequilibrae\tests\data\overture\theme=transportation', output_dir=r'C:\Users\penny\git\Aequilibrae\tests\data\overture\theme=transportation')
bbox = [148.7077, -20.2780, 148.7324, -20.2621 ]

# project.network.create_from_ovm(west=153.1771, south=-27.6851, east=153.2018, north=-27.6703, data_source=r'C:\Users\penny\git\data\theme=transportation', output_dir=r'C:\Users\penny\git\Aequilibrae\tests\data\overture\theme=transportation')
# brisbane_bbox = [153.1771, -27.6851, 153.2018, -27.6703]
# links = download[0]
# nodes = download[1]
 # %%
links = project.network.links
nodes = project.network.nodes
links

# %%
# We grab all the links data as a Pandas DataFrame so we can process it easier
# links = project.network.links.data

# We create a Folium layer
network_links = folium.FeatureGroup("links")

# We do some Python magic to transform this dataset into the format required by Folium
# We are only getting link_id and link_type into the map, but we could get other pieces of info as well
for i, row in links.iterrows():
    points = row.geometry.wkt.replace("LINESTRING ", "").replace("(", "").replace(")", "").split(", ")
    points = "[[" + "],[".join([p.replace(" ", ", ") for p in points]) + "]]"
    # we need to take from x/y to lat/long
    points = [[x[1], x[0]] for x in eval(points)]

    line = folium.vector_layers.PolyLine(
        points, popup=f"<b>link_id: {row.link_id}</b>", tooltip=f"{row.link_type}", color="blue", weight=10
    ).add_to(network_links)

# %%
# We get the center of the region we are working with some SQL magic
long = (bbox[0]+bbox[2])/2
lat = (bbox[1]+bbox[3])/2
# long = (brisbane_bbox[0]+brisbane_bbox[2])/2
# lat = (brisbane_bbox[1]+brisbane_bbox[3])/2

# %%
map_osm = folium.Map(location=[lat, long], zoom_start=14)
network_links.add_to(map_osm)
folium.LayerControl().add_to(map_osm)
map_osm

# %%
project.close()

# %%
