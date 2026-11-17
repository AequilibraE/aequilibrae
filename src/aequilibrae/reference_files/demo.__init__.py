"""
AequilibraE run module.

This module is dynamically imported when the `project.run` property is accessed. Objects named
within `parameters.yml` under the `run` heading will have their arguments partially applied via
`functools.partial` and will replace the objects within this module.

Not all objects within this module must be named `parameters.yml`. If an object is named within
`parameters.yml`, then it must exist within this module otherwise a `RuntimeError` will be raised.

Functions must accept an explicit `project` argument, supplied by `Project.run`.

State within this module should be avoided as this file may be run multiple times.
"""

import numpy as np

from aequilibrae.routing.graph import Graph


def matrix_summary(project):
    """
    Compute summary statistics about the matrix files in the supplied project.

    If no matrix files are present an empty dictionary will be returned.
    """
    mats = project.matrices
    mats_to_metadata = mats.get_available_matrices_and_metadata()

    res = {}
    for file_name in mats_to_metadata:
        with mats.get_matrix(file_name) as mat:
            stats = {}
            for nm, array in mat.items():
                stats[nm] = {
                    "total": np.sum(array),
                    "min": np.min(array),
                    "max": np.max(array),
                    "nnz": (array != 0.0).sum(),
                }
        res[file_name] = stats

    return res


def graph_summary(project, graph=None):
    """Return the full and prepared graph counts, or an empty dictionary without a supplied graph."""
    if graph is None:
        return {}
    return {
        graph.mode: {
            "num_links": graph.mapping.network.link_count if isinstance(graph, Graph) else graph.num_links,
            "num_nodes": graph.mapping.network.node_count if isinstance(graph, Graph) else graph.num_nodes,
            "num_zones": graph.num_zones,
            "compact_num_links": graph.num_links if isinstance(graph, Graph) else graph.compact_num_links,
            "compact_num_nodes": graph.num_nodes if isinstance(graph, Graph) else graph.compact_num_nodes,
        }
    }


def results_summary(project):
    """
    Read the metadata table from the results database.
    """

    return project.results.list()


def example_function_with_kwargs(project, arg1: str = None, **kwargs):
    """
    An example function to demonstrate the argument application via parameters.yml.
    """
    if arg1 is None:
        arg1 = "default argument"

    print("arg1:", arg1)
    print("kwargs:", kwargs)
