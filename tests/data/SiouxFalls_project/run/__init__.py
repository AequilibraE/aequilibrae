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


def graph_summary(project):
    """
    Compute summary statistics about the graphs built within the supplied project.

    If no graphs have been built an empty dictionary will be returned.
    """
    graphs = project.tables.graphs

    return {
        k: {
            "num_links": v.num_links,
            "num_nodes": v.num_nodes,
            "num_zones": v.num_zones,
            "compact_num_links": v.compact_num_links,
            "compact_num_nodes": v.compact_num_nodes,
        }
        for k, v in graphs.items()
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
