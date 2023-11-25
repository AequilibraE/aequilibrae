import os
import uuid
from tempfile import gettempdir

from aequilibrae.utils.create_example import create_example


def test_no_centroids():
    proj_path = os.path.join(gettempdir(), uuid.uuid4().hex)
    model = create_example(proj_path, "sioux_falls")

    model.conn.execute("Update Nodes set is_centroid=0")
    model.conn.commit()

    model.network.build_graphs(modes=["c"])
    model.network.build_graphs()
