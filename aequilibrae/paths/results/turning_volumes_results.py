import re
from pathlib import Path
from typing import Literal
from typing import Optional

import numpy as np
import pandas as pd

from aequilibrae import AequilibraeMatrix
from aequilibrae import TrafficClass

TURNING_VOLUME_GROUPING_COLUMNS = ["network mode", "class_name", "iteration", "a", "b", "c"]
TURNING_VOLUME_COLUMNS = TURNING_VOLUME_GROUPING_COLUMNS + ['demand']
TURNING_VOLUME_OD_COLUMNS = ["network mode", "class_name", "iteration", "a", "b", "c", "id", "id_next", "link_id",
                             "direction", "link_id_next", "direction_next", "origin_idx", "destination_idx",
                             "origin", "destination"]


# TODO: add save turning movements to assignment
class TurningVolumesResults:

    def __init__(
            self,
            class_name: str,
            mode_id: str,
            matrix: AequilibraeMatrix,
            project_dir: Path,
            procedure_id: str,
            iterations: Optional[list[int]] = None,
    ):
        self.class_name = class_name
        self.mode_id = mode_id
        self.matrix = matrix
        self.matrix_mapping = matrix.matrix_hash
        self.project_dir = project_dir
        self.procedure_id = procedure_id
        self.iterations = iterations
        self.procedure_dir = project_dir / "path_files" / procedure_id

    def from_traffic_class(
            self,
            traffic_class: TrafficClass,
            project_dir: Path,
            procedure_id: str,
            iterations: Optional[list[int]] = None,
    ):
        class_name = traffic_class.__id__
        mode_id = traffic_class.mode
        matrix = traffic_class.matrix
        return TurningVolumesResults(class_name, mode_id, matrix, project_dir, procedure_id, iterations)

    def calculate_turning_volumes(self, turns_df: pd.DataFrame, betas: list[float]) -> pd.DataFrame:
        """

        :param turns_df: Dataframe containing turns' abc nodes required fields: [a, b, c]
        :return: dataframe containing the turning volumes
        :param betas: parameters to aggregate volumes by iteration
        """
        node_to_index_df = self.read_path_aux_file("node_to_index")
        correspondence_df = self.read_path_aux_file("correspondence")

        formatted_paths = self.format_paths(self.read_paths_for_iterations())

        formatted_turns = self.format_turns(turns_df, formatted_paths, node_to_index_df, correspondence_df)
        turning_movement_list = []
        for matrix_name in self.matrix.view_names:
            turns_demand = self.get_turns_demand(matrix_name, formatted_turns)
            turn_volumes_by_iteration = self.get_turns_movements(turns_demand)
            turning_movements = self.aggregate_iteration_volumes(turn_volumes_by_iteration, betas)
            turning_movements["matrix_name"] = matrix_name
            turning_movement_list.append(turning_movements)
        return pd.concat(turning_movement_list)

    def read_path_aux_file(self, file_type: Literal["node_to_index", "correspondence"]) -> pd.DataFrame:

        if file_type == "node_to_index":
            return pd.read_feather(self.procedure_dir / f"nodes_to_indices_c{self.mode_id}_{self.class_name}.feather")
        elif file_type == "correspondence":
            return pd.read_feather(self.procedure_dir / f"correspondence_c{self.mode_id}_{self.class_name}.feather")
        else:
            raise ValueError(
                f"Don't know what to do with {file_type}. Expected values are node_to_index or correspondence")

    def read_paths_for_iterations(self) -> pd.DataFrame:
        iter_folder_regex = re.compile("iter([0-9]+)$")
        paths_list = []
        for iter_folder in self.procedure_dir.glob("iter*"):
            match_iter_folder = re.match(iter_folder_regex, str(iter_folder.stem))
            if (not iter_folder.is_dir()) | (match_iter_folder is None):
                continue

            iteration = int(match_iter_folder.groups(0)[0])
            if (self.iterations is not None) & iteration not in self.iterations:
                continue

            paths_folder = self.procedure_dir / f"iter{iteration}" / f"path_c{self.mode_id}_{self.class_name}"
            self.read_paths_from_folder(paths_folder, iteration)
        return pd.concat(paths_list)

    def read_paths_from_folder(self, paths_dir: Path, iteration: int) -> pd.DataFrame:
        path_file_regex = re.compile("^o([0-9]+).feather$")

        files = {int(re.match(path_file_regex, p_file.name).groups()[0]): p_file for p_file in
                 paths_dir.glob("*.feather") if re.match(path_file_regex, p_file.name) is not None}
        path_list = []
        for origin in files.keys():
            o_paths_df = pd.read_feather(paths_dir / f"o{origin}.feather")
            o_idx_paths_df = pd.read_feather(paths_dir / f"o{origin}_indexdata.feather")

            # looks like the indices dataframe is ffilled for missing destinations
            # grab destinations only from first index
            path_starts = o_idx_paths_df.reset_index().groupby("data", as_index=False).first().set_index("index")
            # looks like indices are offset by 1
            path_starts["data"] -= 1
            # only keeping destinations with indices > 0
            path_starts = path_starts[path_starts["data"] > 0].copy()

            # add info to paths df
            o_paths_df["origin_idx"] = origin
            o_paths_df.loc[path_starts["data"], "destination_idx"] = path_starts.index.values
            o_paths_df["destination_idx"] = o_paths_df["destination_idx"].bfill()

            path_list.append(o_paths_df)

        all_paths = pd.concat(path_list)
        all_paths[["network mode", 'class_name', "iteration"]] = [self.mode_id, self.class_name, iteration]
        return all_paths

    def get_turns_demand(self, matrix_values:np.array, turns_df: pd.DataFrame) -> pd.DataFrame:
        turns_df["demand"] = turns_df.apply(self.get_o_d_demand, arguments=matrix_values, axis=1)
        return turns_df

    def format_turns(self, turns_df: pd.DataFrame, formatted_paths: pd.DataFrame, node_to_index_df: pd.DataFrame,
                     correspondence_df: pd.DataFrame) -> pd.DataFrame:
        turns_indices = self.get_turn_indices(turns_df, node_to_index_df)
        turns_w_links = self.get_turn_links(turns_indices, correspondence_df)
        return self.get_turns_ods(turns_w_links, formatted_paths, node_to_index_df)

    def get_o_d_demand(self, row: pd.Series, matrix_values: np.array) -> float:
        return matrix_values[self.matrix_mapping[row["origin"]], self.matrix_mapping[row["destination"]]]

    def format_paths(self, paths: pd.DataFrame) -> pd.DataFrame:
        paths.rename(columns={"data": "id"}, inplace=True)
        paths["id_next"] = paths.groupby(
            ["origin_idx", "destination_idx", "network mode", "class_name", "iteration"]).shift(1)
        return paths.dropna(subset="id_next")

    def get_turn_indices(self, turns_df: pd.DataFrame, node_to_index_df: pd.DataFrame) -> pd.DataFrame:
        # get nodes indices from paths aux file
        for node in ["a", "b", "c"]:
            turns_df[f"{node}_index"] = node_to_index_df.loc[turns_df[node]].values

        return turns_df

    def get_turn_links(self, turns_df: pd.DataFrame, correspondence_df: pd.DataFrame) -> pd.DataFrame:
        # get the first and second links for each turn
        return (
            turns_df.merge(
                correspondence_df[["a_node", "b_node", "link_id", "direction", "id"]],
                left_on=["a_index", "b_index"],
                right_on=["a_node", "b_node"],
            )
            .merge(
                correspondence_df[["a_node", "b_node", "link_id", "direction", "id"]],
                left_on=["b_index", "c_index"],
                right_on=["a_node", "b_node"],
                suffixes=(None, "_next"),
            ).drop(columns=["a_index", "b_index", "c_index", "b_node", "a_node", "a_node_next", "b_node_next"])
        )

    def get_turns_ods(self, turns_w_links: pd.DataFrame, formatted_paths: pd.DataFrame,
                      node_to_index_df) -> pd.DataFrame:
        index_to_node = node_to_index_df.reset_index()
        turns_w_od_idx = formatted_paths.merge(turns_w_links, on=["id", "id_next"])
        turns_w_od = turns_w_od_idx.merge(index_to_node, left_on="origin_idx", right_on="node_index", how="left").merge(
            index_to_node, left_on="destination_idx", right_on="node_index", how="left",
            suffixes=("_origin", "_destination"))
        turns_w_od.rename(columns={"index_origin": "origin", "index_destination": "destination"}, inplace=True)
        return turns_w_od[TURNING_VOLUME_OD_COLUMNS]

    def get_turns_movements(self, turns_demand: pd.DataFrame) -> pd.DataFrame:
        return turns_demand[TURNING_VOLUME_COLUMNS].groupby(TURNING_VOLUME_GROUPING_COLUMNS, as_index=False).sum()

    def calculate_volume(self, df: pd.DataFrame, betas: pd.Series) -> pd.Series:
        volume = df['demand']
        iterations = df["iteration"].max()
        for it in range(1, iterations + 1):
            min_idx = max(0, it - betas.size)
            max_idx = min_idx + min(it, betas.size)
            window = range(min_idx, max_idx)
            volume.iloc[it - 1] = (volume.iloc[window] * betas[0:min(it, betas.size)].values).sum()
        return volume

    def aggregate_iteration_volumes(self, turns_volumes: pd.DataFrame, betas: list[float]) -> pd.DataFrame:
        grouping_cols = [col for col in TURNING_VOLUME_GROUPING_COLUMNS if col != 'iteration']
        result = turns_volumes.groupby(grouping_cols).apply(lambda x: self.calculate_volume(x, pd.Series(betas)))
        idx_results = result.reset_index(level=list(range(0, len(grouping_cols))))
        turns_volumes['volume'] = idx_results["demand"]
        return turns_volumes.groupby(grouping_cols).last()