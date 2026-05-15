import multiprocessing as mp
import random

import numpy as np
from aequilibrae.paths.cython.skimming_core import skimming_single_origin
from aequilibrae.paths.multi_threaded_paths import MultiThreadedPaths
from aequilibrae.paths.results import SkimResults


# Adds the folder with the data to the path and collects the paths to the files
def test_skimming_single_origin(sioux_falls_example):
    sioux_falls_example.network.build_graphs()
    g = sioux_falls_example.network.graphs["c"]
    g.set_blocked_centroid_flows(False)
    g.set_graph(cost_field="distance")
    g.set_skimming("distance")

    orig_idx = random.randint(0, g.centroids.shape[0] - 1)
    origin = g.centroids[orig_idx]

    # skimming results
    res = SkimResults()
    res.prepare(g)
    aux_result = MultiThreadedPaths()
    cores = mp.cpu_count()
    aux_result.prepare_(g, cores, res.nodes)

    a = skimming_single_origin(origin, g, res, aux_result, 0)
    tot = np.sum(res.skims.distance[orig_idx, :])

    assert tot <= 10e10, f"Skimming was not successful. At least one np.inf returned for origin {origin}."
    assert a == origin, f"Skimming returned an error: {a} for origin {origin}"
