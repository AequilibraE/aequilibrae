import re
from pathlib import Path
from typing import Literal
from typing import Optional
from typing import Union

import numpy as np
import openmatrix as omx
import pandas as pd


# TODO: deal with multiple iterations
# TODO: add save turning movements to assignment
# TODO: deal with multiclass
def calculate_turning_volumes(
        project_dir: Path,
        procedure_id: str,
        turns_df: pd.DataFrame,
        demand_matrix: Union[omx, dict[int, int]],
        mode_id: str,
        class_name: str,
        iterations: Optional[list] = None
) -> pd.DataFrame:
    """

    :param project_dir: project directory
    :param procedure_id: procedure_id this can be retrieved from the assignment info or the results table
    :param turns_df: Dataframe containing turns' abc nodes required fields: [a, b, c]
    :param demand_matrix: {'matrix': matrix, "mapping": matrix_mapping}
    :param mode_id:
    :param class_name:
    :param iterations: Subset of iterations to use. If None uses all available
    :return: dataframe containing the turning volumes
    """
    procedure_dir = project_dir / "path_files" / procedure_id
    node_to_index_df = read_path_aux_file(project_dir, procedure_id, "node_to_index", mode_id, class_name)
    correspondence_df = read_path_aux_file(project_dir, procedure_id, "correspondence", mode_id, class_name)

    # only reads iteration 1
    formatted_paths = format_paths(
        read_paths_for_iterations(procedure_dir, mode_id, class_name, iterations))

    formatted_turns = format_turns(turns_df, formatted_paths, node_to_index_df, correspondence_df)
    turns_demand = get_turns_demand(formatted_turns, demand_matrix)

    return get_turns_movements(turns_demand)


def read_path_aux_file(
        project_dir: Path,
        procedure_id: str,
        file_type: Literal["node_to_index", "correspondence"],
        mode_id: str,
        class_name: str
) -> pd.DataFrame:
    procedure_folder = project_dir / "path_files" / procedure_id
    if file_type == "node_to_index":
        return pd.read_feather(procedure_folder / f"nodes_to_indices_c{mode_id}_{class_name}.feather")
    elif file_type == "correspondence":
        return pd.read_feather(procedure_folder / f"correspondence_c{mode_id}_{class_name}.feather")
    else:
        raise ValueError(f"Don't know what to do with {file_type}. Expected values are node_to_index or correspondence")


def read_paths_for_iterations(procedure_dir: Path, mode_id: str, class_name: str,
                              iterations: Optional[list[int]] = None) -> pd.DataFrame:
    iter_folder_regex = re.compile("iter([0-9]+)$")
    paths_list = []
    for iter_folder in procedure_dir.glob("iter*"):
        match_iter_folder = re.match(iter_folder_regex, str(iter_folder.stem))
        if (not iter_folder.is_dir()) | (match_iter_folder is None):
            continue

        iteration = int(match_iter_folder.groups(0)[0])
        if (iterations is not None) & iteration not in iterations:
            continue

        paths_folder = procedure_dir / f"iter{iteration}" / f"path_c{mode_id}_{class_name}"
        read_paths_from_folder(paths_folder, mode_id, class_name, iteration)
    return pd.concat(paths_list)


def read_paths_from_folder(paths_dir: Path, mode_id: str, class_name: str, iteration: int) -> pd.DataFrame:
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
    all_paths[["network mode", 'class_name', "iteration"]] = [mode_id, class_name, iteration]
    return all_paths


def format_paths(paths: pd.DataFrame) -> pd.DataFrame:
    paths.rename(columns={"data": "id"}, inplace=True)
    paths["id_next"] = paths.groupby(
        ["origin_idx", "destination_idx", "network mode", "class_name", "iteration"]).shift(1)
    return paths.dropna(subset="id_next")


def format_turns(turns_df: pd.DataFrame, formatted_paths: pd.DataFrame, node_to_index_df: pd.DataFrame,
                 correspondence_df: pd.DataFrame) -> pd.DataFrame:
    turns_indices = get_turn_indices(turns_df, node_to_index_df)
    turns_w_links = get_turn_links(turns_indices, correspondence_df)
    return get_turns_ods(turns_w_links, formatted_paths, node_to_index_df)


def get_turn_indices(turns_df: pd.DataFrame, node_to_index_df: pd.DataFrame) -> pd.DataFrame:
    # get nodes indices from paths aux file
    for node in ["a", "b", "c"]:
        turns_df[f"{node}_index"] = node_to_index_df.loc[turns_df[node]].values

    return turns_df


def get_turn_links(turns_df: pd.DataFrame, correspondence_df: pd.DataFrame) -> pd.DataFrame:
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


def get_turns_ods(turns_w_links: pd.DataFrame, formatted_paths: pd.DataFrame, node_to_index) -> pd.DataFrame:
    index_to_node = node_to_index.reset_index()
    turns_w_od_idx = formatted_paths.merge(turns_w_links, on=["id", "id_next"])
    turns_w_od = turns_w_od_idx.merge(index_to_node, left_on="origin_idx", right_on="node_index", how="left").merge(
        index_to_node, left_on="destination_idx", right_on="node_index", how="left",
        suffixes=("_origin", "_destination"))
    turns_w_od.rename(columns={"index_origin": "origin", "index_destination": "destination"}, inplace=True)
    return turns_w_od[
        [
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
    ]


# TODO: change functions to grab the correct matrix from class name and network mode
def get_o_d_demand(row: pd.Series, matrix_dict: dict[str, np.array]) -> float:
    origin_idx = matrix_dict["mapping"][row["origin"]]
    destination_idx = matrix_dict["mapping"][row["destination"]]
    return matrix_dict["matrix"][origin_idx, destination_idx]


def get_turns_demand(turns_df: pd.DataFrame, matrix_dict: dict[str, np.array]) -> pd.DataFrame:
    turns_df["demand"] = turns_df.apply(get_o_d_demand, matrix_dict=matrix_dict, axis=1)
    return turns_df


def get_turns_movements(turns_demand: pd.DataFrame) -> pd.DataFrame:
    grouping_cols = [
        "network mode",
        "class_name",
        "iteration",
        "a",
        "b",
        "c"
    ]
    required_cols = grouping_cols + ["demand"]
    return turns_demand[required_cols].groupby(grouping_cols, as_index=False).sum()
