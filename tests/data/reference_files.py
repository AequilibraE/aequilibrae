import tempfile
from pathlib import Path

base_dir = Path(__file__).resolve().parent.parent
data_folder = base_dir / "data"

# For the graph tests
test_network = data_folder / "Final_Network.shp"
path_test = Path(tempfile.gettempdir())
triangle_graph_blocking = data_folder / "blocking_triangle_graph_project"

gtfs_folder = data_folder / "gtfs"
gtfs_zip = data_folder / "gtfs.zip"
gtfs_db_output = path_test / "test.db"

# For the skimming test


# For project tests
project_file = data_folder / "AequilibraE_Project.sqlite"

# For Traffic Assignment tests
siouxfalls_project = data_folder / "SiouxFalls_project"
siouxfalls_demand = siouxfalls_project / "matrices" / "SiouxFalls.omx"
siouxfalls_skims = siouxfalls_project / "matrices" / "sfalls_skims.omx"

#
no_triggers_project = data_folder / "no_triggers_project"
