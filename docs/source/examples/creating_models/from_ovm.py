# %%
# Imports
from pathlib import Path
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
# For the sake of this example, we will choose the small town of Airlie Beach.
dir = str(Path('../../../../').resolve())

# We have stored Airlie Beach's transportation parquet files in the folder with the file path data_source below as using the cloud-native Parquet files takes a much longer time to run
# We recommend downloading these cloud-native Parquet files to drive and replacing the data_source file to match
# The application of Microsoft Azure Storage Explorer was used to download this data. Steps to accieve this:
#  - right click Storage accounts and select 'Connect to Azure Storage', this will cause a window to pop up 
#  - Select 'Blob container or directory'
#  - Select 'Anonymously (my blob container allows public access)'
#  - In the box 'Blob container or directory URL:' paste in the Overture Maps cloud-native Parquet file loction specified on thier github download page with the
#    current url being 'https://github.com/OvertureMaps/data/blob/main/README.md#how-to-access-overture-maps-data'
#    for example the theme transportation the location for Microsoft Azure is 'https://overturemapswestus2.blob.core.windows.net/release/2024-01-17-alpha.0/theme=admins'
#    In the box 'Display name:' this label does not impact the information being downloaded, it's recommend to input the theme of the blob data being imported
#    for this example the display name should be transportation.

data_source = Path(dir) / 'tests' / 'data' / 'overture' / 'theme=transportation'

#The "bbox" parameter specifies the bounding box encompassing the desired geographical location. In the given example, this refers to the bounding box that encompasses Airlie Beach.
bbox = [148.7077, -20.2780, 148.7324, -20.2621 ]
project.network.create_from_ovm(west=bbox[0], south=bbox[1], east=bbox[2], north=bbox[3], data_source=data_source, output_dir=data_source)

# brisbane_bbox = [153.1771, -27.6851, 153.2018, -27.6703]
# project.network.create_from_ovm(west=brisbane_bbox[0], south=brisbane_bbox[1], east=brisbane_bbox[2], north=brisbane_bbox[3], data_source=r'C:\Users\penny\git\data\theme=transportation', output_dir=r'C:\Users\penny\git\Aequilibrae\tests\data\overture\theme=transportation')
# links = download[0]
# nodes = download[1]
 # %%
links = project.network.links.data
nodes = project.network.nodes.data
nodes

# links

# %%
# We grab all the links data as a Pandas DataFrame so we can process it easier

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
