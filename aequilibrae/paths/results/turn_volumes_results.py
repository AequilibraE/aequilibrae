import re
import sqlite3
from os import path
from pathlib import Path
from typing import Literal
from typing import Optional

import numpy as np
import pandas as pd

from aequilibrae.matrix.aequilibrae_matrix import AequilibraeMatrix
from aequilibrae.paths.traffic_class import TrafficClass
from aequilibrae.project.project import Project
from context import get_active_project

TURNING_VOLUME_GROUPING_COLUMNS = ["matrix_name", "network mode", "class_name", "iteration", "a", "b", "c"]
TURNING_VOLUME_COLUMNS = TURNING_VOLUME_GROUPING_COLUMNS + ["demand"]
TURNING_VOLUME_OD_COLUMNS = [
    "network mode",
    "class_name",
    "iteration",
    "a",
    "b",
    "c",
    "id",
    "id_next",
    "link_id",
    "direction",
    "link_id_next",
    "direction_next",
    "origin_idx",
    "destination_idx",
    "origin",
    "destination",
]


class TurnVolumesResults:
    def __init__(
        self,
        class_name: str,
        mode_id: str,
        matrix: AequilibraeMatrix,
        procedure_id: str,
        project: Project = None,
        iteration: Optional[int] = None,
        blend_iterations: bool = True,
    ):
        self.project = project or get_active_project()
        self.class_name = class_name
        self.mode_id = mode_id
        self.matrix = matrix
        self.matrix_mapping = matrix.matrix_hash
        self.project_dir = Path(project.project_base_path)
        self.procedure_id = procedure_id
        self.procedure_dir = self.project_dir / "path_files" / procedure_id
        self.iteration = self.get_iteration(iteration)
        self.blend_iterations = False if self.iteration == 1 else blend_iterations

    @staticmethod
    def from_traffic_class(
        traffic_class: TrafficClass,
        procedure_id: str,
        project: Project = None,
        iteration: Optional[list[int]] = None,
        blend_iterations: bool = True,
    ):
        class_name = traffic_class.__id__
        mode_id = traffic_class.mode
        matrix = traffic_class.matrix
        return TurnVolumesResults(class_name, mode_id, matrix, procedure_id, project, iteration, blend_iterations)

    @staticmethod
    def calculate_from_result_table(
        project: Project,
        turns_df: pd.DataFrame,
        asgn_result_table_name: str,
        class_to_matrix: dict[str, AequilibraeMatrix],
        user_classes: Optional[list[str]] = None,
        blend_iterations: bool = True,
    ):
        conn = sqlite3.connect(path.join(project.project_base_path, "project_database.sqlite"))
        df = pd.read_sql_query(f"select * from results where table_name='{asgn_result_table_name}'", conn)
        conn.close()

        procedure_id = df.at[0, "procedure_id"]

        # Requires multi eval as json.loads fails to read the procedure report.
        # inf is not recognised in eval, replacing with np.inf to allow eval
        procedure_report = eval(df.at[0, "procedure_report"])
        convergence_report = eval(procedure_report["convergence"].replace("inf", "np.inf"))
        setup_report = eval(procedure_report["setup"])

        convergence_report = pd.DataFrame(convergence_report)

        asgn_classes = setup_report["Classes"]

        betas_df = TurnVolumesResults.get_betas_df(convergence_report)

        if user_classes is None:
            user_classes = list(asgn_classes.keys())

        missing_classes = []
        ta_turn_vol_list = []
        for class_name, class_specs in asgn_classes.items():
            if class_name not in user_classes:
                missing_classes.append(class_name)
                continue

            tc_turns = TurnVolumesResults(
                class_name=class_name,
                mode_id=class_specs["network mode"],
                matrix=class_to_matrix[class_name],
                procedure_id=procedure_id,
                iteration=betas_df.index.max(),
                blend_iterations=blend_iterations,
                project=project,
            )
            ta_turn_vol_list.append(tc_turns.calculate_turn_volumes(turns_df, betas_df))

        return pd.concat(ta_turn_vol_list).reset_index(drop=True)

    @staticmethod
    def get_betas_df(convergence_report: pd.DataFrame) -> pd.DataFrame:
        if convergence_report.empty:
            convergence_report = pd.DataFrame({"iteration": [1]})

        if "beta0" not in convergence_report.columns:
            # if betas are not in the report, create dummy betas, where beta0 is 1
            # This allows to calculate volumes for AoN.
            convergence_report["beta0"] = 1
            convergence_report[["beta1", "beta2"]] = 0

        if "alpha" not in convergence_report.columns:
            convergence_report["alpha"] = 1

        betas_df = convergence_report[["iteration", "alpha", "beta0", "beta1", "beta2"]].copy()

        for beta_field in ["beta0", "beta1", "beta2"]:
            betas_df.loc[betas_df[beta_field] == -1, beta_field] = np.nan

        betas_df = betas_df.ffill().set_index("iteration").copy()

        return betas_df

    def calculate_turn_volumes(self, turns_df: pd.DataFrame, betas: pd.DataFrame) -> pd.DataFrame:
        """

        :param turns_df: Dataframe containing turns' abc nodes required fields: [a, b, c]
        :param betas: dataframe with betas to aggregate volumes by iterations. Must be indexed by iteration
        :return: dataframe containing the turning volumes
        """
        node_to_index_df = self.read_path_aux_file("node_to_index")
        correspondence_df = self.read_path_aux_file("correspondence")

        formatted_paths = self.format_paths(self.read_paths_for_iterations())

        formatted_turns = self.format_turns(turns_df, formatted_paths, node_to_index_df, correspondence_df)
        turn_volume_list = []
        for matrix_name in self.matrix.view_names:
            turns_demand = self.get_turns_demand(matrix_name, self.matrix.get_matrix(matrix_name), formatted_turns)
            turn_volumes_by_iteration = self.get_turn_volumes(turns_demand, turns_df)
            turn_volume_list.append(self.aggregate_iteration_volumes(turn_volumes_by_iteration, betas))
        return pd.concat(turn_volume_list)

    def read_path_aux_file(self, file_type: Literal["node_to_index", "correspondence"]) -> pd.DataFrame:
        if file_type == "node_to_index":
            return pd.read_feather(self.procedure_dir / f"nodes_to_indices_c{self.mode_id}_{self.class_name}.feather")
        elif file_type == "correspondence":
            return pd.read_feather(self.procedure_dir / f"correspondence_c{self.mode_id}_{self.class_name}.feather")
        else:
            raise ValueError(
                f"Don't know what to do with {file_type}. Expected values are node_to_index or correspondence"
            )

    def get_iteration(self, max_iter):
        if max_iter is not None:
            return max_iter

        iterations = []
        iter_folder_regex = re.compile("iter([0-9]+)$")
        for iter_folder in self.procedure_dir.glob("iter*"):
            match_iter_folder = re.match(iter_folder_regex, str(iter_folder.stem))

            if (not iter_folder.is_dir()) | (match_iter_folder is None):
                continue

            iterations.append(int(match_iter_folder.groups(0)[0]))
        return max(iterations)

    def read_paths_for_iterations(self) -> pd.DataFrame:
        if not self.blend_iterations:
            paths_folder = self.procedure_dir / f"iter{self.iteration}" / f"path_c{self.mode_id}_{self.class_name}"
            return self.read_paths_from_folder(paths_folder, self.iteration)

        paths_list = []
        for iteration in range(1, int(self.iteration) + 1):
            paths_folder = self.procedure_dir / f"iter{iteration}" / f"path_c{self.mode_id}_{self.class_name}"
            paths_list.append(self.read_paths_from_folder(paths_folder, iteration))
        return pd.concat(paths_list)

    def read_paths_from_folder(self, paths_dir: Path, iteration: int) -> pd.DataFrame:
        path_file_regex = re.compile("^o([0-9]+).feather$")

        files = {
            int(re.match(path_file_regex, p_file.name).groups()[0]): p_file
            for p_file in paths_dir.glob("*.feather")
            if re.match(path_file_regex, p_file.name) is not None
        }
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
        all_paths[["network mode", "class_name", "iteration"]] = [self.mode_id, self.class_name, iteration]
        return all_paths

    def get_turns_demand(self, matrix_name: str, matrix_values: np.array, turns_df: pd.DataFrame) -> pd.DataFrame:
        turns_df["demand"] = turns_df.apply(self.get_o_d_demand, args=(matrix_values,), axis=1)
        turns_df["matrix_name"] = matrix_name
        return turns_df

    def format_turns(
        self,
        turns_df: pd.DataFrame,
        formatted_paths: pd.DataFrame,
        node_to_index_df: pd.DataFrame,
        correspondence_df: pd.DataFrame,
    ) -> pd.DataFrame:
        turns_indices = self.get_turn_indices(turns_df, node_to_index_df)
        turns_w_links = self.get_turn_links(turns_indices, correspondence_df)
        return self.get_turns_ods(turns_w_links, formatted_paths, node_to_index_df)

    def get_o_d_demand(self, row: pd.Series, matrix_values: np.array) -> float:
        return matrix_values[self.matrix_mapping[row["origin"]]][self.matrix_mapping[row["destination"]]]

    def format_paths(self, paths: pd.DataFrame) -> pd.DataFrame:
        paths.rename(columns={"data": "id"}, inplace=True)
        paths["id_next"] = paths.groupby(
            ["origin_idx", "destination_idx", "network mode", "class_name", "iteration"]
        ).shift(1)
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
            )
            .drop(columns=["a_index", "b_index", "c_index", "b_node", "a_node", "a_node_next", "b_node_next"])
        )

    def get_turns_ods(
        self, turns_w_links: pd.DataFrame, formatted_paths: pd.DataFrame, node_to_index_df
    ) -> pd.DataFrame:
        index_to_node = node_to_index_df.reset_index()
        turns_w_od_idx = formatted_paths.merge(turns_w_links, on=["id", "id_next"])
        turns_w_od = turns_w_od_idx.merge(index_to_node, left_on="origin_idx", right_on="node_index", how="left").merge(
            index_to_node,
            left_on="destination_idx",
            right_on="node_index",
            how="left",
            suffixes=("_origin", "_destination"),
        )
        turns_w_od.rename(columns={"index_origin": "origin", "index_destination": "destination"}, inplace=True)
        return turns_w_od[TURNING_VOLUME_OD_COLUMNS]

    def get_turn_volumes(self, turns_demand: pd.DataFrame, turn_df: pd.DataFrame) -> pd.DataFrame:
        agg_turns_demand = (
            turns_demand[TURNING_VOLUME_COLUMNS].groupby(TURNING_VOLUME_GROUPING_COLUMNS, as_index=False).sum()
        )

        full_index = self.get_full_index(agg_turns_demand, turn_df)

        agg_turns_demand.set_index(TURNING_VOLUME_GROUPING_COLUMNS, inplace=True)

        return agg_turns_demand.reindex(full_index, fill_value=0).reset_index()

    def get_full_index(self, agg_turns_demand: pd.DataFrame, turn_df: pd.DataFrame) -> pd.MultiIndex:
        # Create the index to fill in missing iterations
        # the first part of the index comes from the aggregated turning volumes
        col_names = [col for col in TURNING_VOLUME_GROUPING_COLUMNS if col not in ["a", "b", "c", "iteration"]]
        idx_df = agg_turns_demand[col_names].drop_duplicates()
        idx_df["dummy"] = 1

        # the iteration comes from a range if blending, otherwise from a single value
        if self.blend_iterations:
            iteration_idx = pd.DataFrame(pd.Series(range(1, self.iteration + 1)), columns=["iteration"])
        else:
            iteration_idx = pd.DataFrame(data=self.iteration, columns=["iteration"], index=[0])

        iteration_idx["dummy"] = 1

        # abc nodes come from the input turn df
        dummy_turns = turn_df.copy()
        dummy_turns["dummy"] = 1

        return (
            idx_df.merge(dummy_turns, on="dummy", how="outer")
            .merge(iteration_idx, on="dummy", how="outer")
            .set_index(TURNING_VOLUME_GROUPING_COLUMNS)
            .index
        )

    def calculate_volume(self, df: pd.DataFrame, ta_report: pd.DataFrame) -> pd.Series:
        # have to loop through the rows to update volumes for each iteration to calculate the vol fractions.
        # cannot be done with pandas rolling, as it doesn't take in account updated values within the window.
        aon_volume = df.set_index("iteration")["demand"].sort_index()
        iterations = df["iteration"].max()

        # initialise blended volumes with first iteration equal to AoN volume.
        blended_volumes = pd.Series(data=0, index=aon_volume.index)
        blended_volumes.loc[1] = aon_volume.loc[1]

        # calculate the new volumes using betas
        for it in range(2, iterations + 1):
            betas_for_it = pd.Series(ta_report.loc[it, ["beta0", "beta1", "beta2"]]).sort_index(ascending=True)
            alpha_for_it = ta_report.at[it, "alpha"]
            if (betas_for_it != -1).any():
                # only calculate the new volume if betas are all not -1
                min_idx = max(0, it - betas_for_it.size) + 1
                max_idx = min_idx + min(it, betas_for_it.size)
                window = range(min_idx, max_idx)
                it_volume = (aon_volume.loc[window] * betas_for_it[0 : min(it, betas_for_it.size)].values).sum()
            else:
                it_volume = aon_volume.loc[it]

            blended_volumes.loc[it] = (it_volume * alpha_for_it) + (blended_volumes.loc[it - 1] * (1 - alpha_for_it))

        return blended_volumes

    def aggregate_iteration_volumes(self, turns_volumes: pd.DataFrame, ta_report: pd.DataFrame) -> pd.DataFrame:
        if not self.blend_iterations:
            return turns_volumes.rename(columns={"demand": "volume"})

        grouping_cols = [col for col in TURNING_VOLUME_GROUPING_COLUMNS if col != "iteration"]
        result = turns_volumes.groupby(grouping_cols, as_index=False).apply(
            lambda x: self.calculate_volume(x, ta_report)
        )
        return (
            result.melt(id_vars=grouping_cols, value_vars=result.columns, var_name="iteration", value_name="volume")
            .reset_index()
            .groupby(grouping_cols, as_index=False, sort=True)
            .last()
            .drop(columns="index")
        )
