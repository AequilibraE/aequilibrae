import os
from typing import Dict, List
import numpy as np
import pandas as pd
from aequilibrae.context import get_active_project


# TODO: let's make it optional to keep path files in memory, although this can get out of control very quickly it should
# be much quicker when working with a single origin

# TODO: factor out AssignmentResultsTable into different file, also get rid of dirty eval bu changing creation

# FIXME: this is for zone_index and compressed link ids
# for link ids, look up what we are doing in graph - we might want to keep the order of links for a single compressed
# link
# for zones, we need to do the inverse of graph.compact_nodes_to_indices


class TrafficClassIdentifier(object):
    def __init__(self, name: str, id: str):
        self.name = name
        self.id = id


class AssignmentResultsTable(object):
    def __init__(self, table_name: str, project=None) -> None:
        self.project = project or get_active_project()
        self.table_name = table_name
        self.assignment_results = self._read_assignment_results()
        self.table_name = self.assignment_results["table_name"].values[0]
        self.procedure = self.assignment_results["procedure"].values[0]
        self.procedure_id = self.assignment_results["procedure_id"].values[0]
        self.timestamp = self.assignment_results["timestamp"].values[0]
        self.description = self.assignment_results["description"].values[0]
        self.procedure_report = self._parse_procedure_report()

    def _read_assignment_results(self) -> pd.DataFrame:
        conn = self.project.connect()
        results_df = pd.read_sql("SELECT * FROM 'results'", conn)
        conn.close()
        res = results_df.loc[results_df.table_name == self.table_name]
        assert len(res) == 1, f"Found {len(res)} assignment result with this table name, need exactly one"
        return res

    def _parse_procedure_report(self) -> Dict:
        rep_with_replacement = (
            self.assignment_results["procedure_report"].values[0].replace("inf", "np.inf").replace("nan", "np.nan")
        )
        report = eval(rep_with_replacement)
        report["convergence"] = eval(report["convergence"])
        report["setup"] = eval(report["setup"])
        return report

    def get_traffic_class_names_and_id(self) -> List[TrafficClassIdentifier]:
        all_classes = self.procedure_report["setup"]["Classes"]
        return [TrafficClassIdentifier(k, v["network mode"]) for k, v in all_classes.items()]


class AssignmentPaths(object):
    """Class for accessing path files optionally generated during assignment.

    .. code-block:: python

        paths = AssignmentPath(table_name_with_assignment_results)
        paths.get_path_for_destination(origin, destination, iteration, traffic_class_id)
    """

    def __init__(self, table_name: str, project=None) -> None:
        """
        Instantiates the class

        :Arguments:
            **table_name** (str): Name of the traffic assignment result table used to generate the required path files

            **project** (:obj:`Project`, optional): The Project to connect to.
            By default, uses the currently active project

        """
        project = project or get_active_project()
        self.proj_dir = project.project_base_path
        self.table_name = table_name
        self.assignment_results = AssignmentResultsTable(table_name, project)
        self.path_base_dir = os.path.join(self.proj_dir, "path_files", self.assignment_results.procedure_id)
        self.classes = self.assignment_results.get_traffic_class_names_and_id()
        self.compressed_graph_correspondences = self._read_compressed_graph_correspondence()

    def _read_compressed_graph_correspondence(self) -> Dict:
        compressed_graph_correspondences = {}
        for c in self.classes:
            compressed_graph_correspondences[c.id] = pd.read_feather(
                os.path.join(self.path_base_dir, f"correspondence_c{c.id}_{c.name}.feather")
            )
        return compressed_graph_correspondences

    def read_path_file(self, origin: int, iteration: int, traffic_class_id: str) -> (pd.DataFrame, pd.DataFrame):
        possible_traffic_classes = list(filter(lambda x: x.id == traffic_class_id, self.classes))
        assert (
            len(possible_traffic_classes) == 1
        ), f"traffic class id not unique, please choose one of {list(map(lambda x: x.id, self.classes))}"
        traffic_class = possible_traffic_classes[0]
        base_dir = os.path.join(
            self.path_base_dir, f"iter{iteration}", f"path_c{traffic_class.id}_{traffic_class.name}"
        )
        path_o_f = os.path.join(base_dir, f"o{origin}.feather")
        path_o_index_f = os.path.join(base_dir, f"o{origin}_indexdata.feather")
        path_o = pd.read_feather(path_o_f)
        path_o_index = pd.read_feather(path_o_index_f)
        return path_o, path_o_index

    def get_path_for_destination(self, origin: int, destination: int, iteration: int, traffic_class_id: str):
        """Return all link ids, i.e. the full path, for a given destination"""
        path_o, path_o_index = self.read_path_file(origin, iteration, traffic_class_id)
        return self.get_path_for_destination_from_files(path_o, path_o_index, destination)

    @staticmethod
    def get_path_for_destination_from_files(path_o: pd.DataFrame, path_o_index: pd.DataFrame, destination: int):
        """for a given path file and path index file, and a given destination, return the path links in o-d order"""
        if destination == 0:
            lower_incl = 0
        else:
            lower_incl = path_o_index.loc[path_o_index.index == destination - 1].values[0][0]
        upper_non_incl = path_o_index.loc[path_o_index.index == destination].values[0][0]
        links_on_path = path_o.loc[(path_o.index >= lower_incl) & (path_o.index < upper_non_incl)].values.flatten()
        links_on_path = np.flip(links_on_path)
        return links_on_path
