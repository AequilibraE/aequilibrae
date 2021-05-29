import os
from typing import Dict, List
import numpy as np
import pandas as pd
from aequilibrae import logger
from aequilibrae.project.database_connection import ENVIRON_VAR
from aequilibrae.project.database_connection import database_connection


# TODO: let's make it optional to keep path files in memory, although this can get out of control very quickly it should
# be much quicker when working with a single origin

# TODO: factor out AssignmentResultsTable into different file, also get rid of dirty eval bu changing creation


class TrafficClassIdentifier(object):
    def __init__(self, name: str, id: str):
        self.name = name
        self.id = id


class AssignmentResultsTable(object):
    def __init__(self, table_name: str) -> None:
        self.proj_dir = os.environ.get(ENVIRON_VAR)
        self.table_name = table_name
        self.assignment_results = self._read_assignment_results()
        self.table_name = self.assignment_results["table_name"].values[0]
        self.procedure = self.assignment_results["procedure"].values[0]
        self.procedure_id = self.assignment_results["procedure_id"].values[0]
        self.timestamp = self.assignment_results["timestamp"].values[0]
        self.description = self.assignment_results["description"].values[0]
        self.procedure_report = self._parse_procedure_report()

    def _read_assignment_results(self) -> pd.DataFrame:
        conn = database_connection()
        results_df = pd.read_sql("SELECT * FROM 'results'", conn)
        conn.close()
        res = results_df.loc[results_df.table_name == self.table_name]
        assert len(res) == 1, f"Found {len(res)} assignment result with this table name, need exactly one"
        return res

    def _parse_procedure_report(self) -> Dict:
        rep_with_replacement = (
            self.assignment_results["table_name"].values[0].replace("inf", "np.inf").replace("nan", "np.nan")
        )
        report = eval(rep_with_replacement)
        report["convergence"] = eval(report["convergence"])
        report["setup"] = eval(report["setup"])
        return report

    def get_traffic_class_names_and_id(self) -> List[TrafficClassIdentifier]:
        all_classes = self.procedure_report["setup"]["Classes"]
        return [TrafficClassIdentifier(k, v["network mode"]) for k, v in all_classes.items()]


class AssignmentPaths(object):
    """ Class for accessing path files optionally generated during assignment.
    ::
        paths = AssignmentPath(table_name_with_assignment_results)
        paths.get_path_for_destination(origin, destination, traffic_class_name, iteration)
    """

    def __init__(self, table_name: str) -> None:
        """
        Instantiates the class
         Args:
            table_name (str): Name of the traffic assignment result table used to generate the required path files
        """
        self.proj_dir = os.environ.get(ENVIRON_VAR)
        self.table_name = table_name
        self.assignment_results = AssignmentResultsTable(table_name)
        self.classes = self.assignment_results.get_traffic_class_names_and_id()
        self.compressed_graph_correspondences = self._read_compressed_graph_correspondence()
        self.path_base_dir = os.path.join(self.proj_dir, "path_files", self.assignment_results.procedure_id)

    def _read_compressed_graph_correspondence(self) -> Dict:
        for c in self.classes:
            self.compressed_graph_correspondences[c.id] = pd.read_feather(
                os.path.join(self.path_base_dir, f"correspondence_c{c.mode}_{c.id}.feather")
            )

    def _read_path_file(self, iteration: int, traffic_class_id: str, origin: int) -> (pd.DataFrame, pd.DataFrame):
        possible_traffic_classes = list(filter(lambda x: x.id == traffic_class_id, self.classes))
        assert (
            len(possible_traffic_classes) == 1
        ), f"traffic class id not unique, please choose one of {list(map(lambda x: x.id, self.classes))}"
        traffic_class = possible_traffic_classes[0]
        base_dir = os.path.join(
            self.path_base_dir, f"iter{iteration}", f"path_c{traffic_class.mode}_{traffic_class.id}"
        )
        path_o_f = os.path.join(base_dir, f"o{origin}.feather")
        path_o_index_f = os.path.join(base_dir, f"o{origin}_indexdata.feather")
        path_o = pd.read_feather(path_o_f)
        path_o_index = pd.read_feather(path_o_index_f)
        return path_o, path_o_index

    # make this o, d, iter, class
    def get_path_for_destination(self, path_o, path_o_index, destination):
        """ Return all link ids, i.e. the full path, for a given destination """
        if destination == 0:
            lower_incl = 0
        else:
            lower_incl = path_o_index.loc[path_o_index.index == destination - 1].values[0][0]

        upper_non_incl = path_o_index.loc[path_o_index.index == destination].values[0][0]
        links_on_path = path_o.loc[(path_o.index >= lower_incl) & (path_o.index < upper_non_incl)].values.flatten()

        # TODO: do we want d->o, like here, or do we turn it around so links are o to d?

        return links_on_path