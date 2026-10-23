"""
AequilibraE run module.

This module is dynamically imported when the `project.run` property is accessed. Objects named
within `parameters.yml` under the `run` heading will have their arguments partially applied via
`functools.partial` and will replace the objects within this module.

Not all objects within this module must be named `parameters.yml`. If an object is named within
`parameters.yml`, then it must exist within this module otherwise a `RuntimeError` will be raised.

Functions should use the `get_active_project()` function to obtain a reference to the current
project.

State within this module should be avoided as this file may be run multiple times.
"""

import numpy as np
import pandas as pd

from aequilibrae.context import get_active_project


def matrix_summary():
    """
    Compute summary statistics about the matrices registered with the active project.

    If no matrices are registered an empty dictionary will be returned.
    """
    project = get_active_project()
    mats = project.matrices
    mats_to_metadata = mats.get_available_matrices_and_metadata()

    res = {}
    for file_name in mats_to_metadata["file_name"]:
        if not file_name.lower().endswith(".omx"):
            continue
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


def graph_summary():
    """
    Compute summary statistics about the graphs built within the active project.

    If no graphs have been built an empty dictionary will be returned.
    """
    project = get_active_project()
    graphs = project.network.graphs

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


def results_summary():
    """
    Read the results table from the project database.
    """
    project = get_active_project()

    sql = """SELECT * from results;"""
    with project.db_connection as conn:
        return pd.read_sql(sql, conn)


def example_function_with_kwargs(arg1: str = None, **kwargs):
    """
    An example function to demonstrate the argument application via parameters.yml.
    """
    if arg1 is None:
        arg1 = "default argument"

    print("arg1:", arg1)
    print("kwargs:", kwargs)
