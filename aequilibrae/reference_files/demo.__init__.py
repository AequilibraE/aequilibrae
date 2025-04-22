import numpy as np
import pandas as pd

from aequilibrae.context import get_active_project


def matrix_summary():
    project = get_active_project()
    mats = project.matrices
    df = mats.list()

    res = {}
    for name in df["name"]:
        mat = mats.get_matrix(name)
        mat.computational_view()

        stats = {}
        for i, nm in enumerate(mat.view_names):
            array = mat.matrix_view[:, :] if len(mat.view_names) == 1 else mat.matrix_view[:, :, i]
            stats[nm] = {
                "total": np.sum(array),
                "min": np.min(array),
                "max": np.max(array),
                "nnz": (array != 0.0).sum(),
            }
        res[name] = stats

    return res


def graph_summary():
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
    project = get_active_project()

    sql = """SELECT * from results;"""
    with project.db_connection as conn:
        return pd.read_sql(sql, conn)


def example_function_with_kwargs(arg1: str = None, **kwargs):
    if arg1 is None:
        arg1 = "default argument"

    print("arg1:", arg1)
    print("kwargs:", kwargs)
