import tempfile
from os.path import dirname, abspath, join

data_folder = join(dirname(dirname(abspath(__file__))), "data")

# For the graph tests
test_network = join(dirname(dirname(abspath(__file__))), "data", "Final_Network.shp")
test_graph = join(dirname(dirname(abspath(__file__))), "data", "test_graph.aeg")
path_test = tempfile.gettempdir()
triangle_graph_blocking = join(dirname(dirname(abspath(__file__))), "data", "blocking_triangle_graph_project")

gtfs_folder = join(dirname(dirname(abspath(__file__))), "data/gtfs")
gtfs_zip = join(dirname(dirname(abspath(__file__))), "data/gtfs.zip")
gtfs_db_output = join(path_test, "test.db")

# For the skimming test


# For the matrix test
omx_example = join(dirname(dirname(abspath(__file__))), "data/test_omx.omx")
no_index_omx = join(dirname(dirname(abspath(__file__))), "data/no_index.omx")

# For project tests
project_file = join(dirname(dirname(abspath(__file__))), "data", "AequilibraE_Project.sqlite")

# For Traffic Assignment tests
siouxfalls_project = join(dirname(dirname(abspath(__file__))), "data/SiouxFalls_project")
siouxfalls_demand = join(dirname(dirname(abspath(__file__))), "data/SiouxFalls_project/matrices", "SiouxFalls.omx")
siouxfalls_skims = join(dirname(dirname(abspath(__file__))), "data/SiouxFalls_project/matrices", "sfalls_skims.omx")

#
no_triggers_project = join(dirname(dirname(abspath(__file__))), "data/no_triggers_project")

st_varent_network = join(dirname(abspath(__file__)), "St_Varent_issue307.zip")

# For GMNS tests
gmns_link = join(dirname(dirname(abspath(__file__))), "data/arlington_signals_project/gmns_files", "link.csv")
gmns_node = join(dirname(dirname(abspath(__file__))), "data/arlington_signals_project/gmns_files", "node.csv")
gmns_groups = join(
    dirname(dirname(abspath(__file__))), "data/arlington_signals_project/gmns_files", "use_group_example.csv"
)
