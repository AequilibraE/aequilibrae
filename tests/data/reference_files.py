import tempfile
from os.path import dirname, abspath, join

# For the graph tests
test_network = join(dirname(dirname(abspath(__file__))), "data", "Final_Network.shp")
test_graph = join(dirname(dirname(abspath(__file__))), "data", "test_graph.aeg")
path_test = tempfile.gettempdir()

gtfs_folder = join(dirname(dirname(abspath(__file__))), "data/gtfs")
gtfs_zip = join(dirname(dirname(abspath(__file__))), "data/gtfs.zip")
gtfs_db_output = join(path_test, "test.db")

# For the skimming test


# For the matrix test
omx_example = join(dirname(dirname(abspath(__file__))), "data/test_omx.omx")
no_index_omx = join(dirname(dirname(abspath(__file__))), "data/no_index.omx")

# For project tests
project_file = join(dirname(dirname(abspath(__file__))), "data", "AequilibraE_Project.sqlite")
