"""Route tables, path-file round trips and project-backed result persistence."""

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest
from aequilibrae.paths.cython.route_choice_set import RouteChoiceSet
from aequilibrae.paths.cython.route_choice_set_results import RouteChoiceSetResults
from aequilibrae.utils.cython.bridge import Bridge

from aequilibrae.matrix import AequilibraeSparseMatrix, from_dict
from aequilibrae.paths import Network, RouteChoice

from .route_choice_helpers import compressed_graph, configured_choice, diamond, single_od_demand


def test_set_save_routes_requires_existing_directory(route_choice_setup, tmp_path):
    with pytest.raises(ValueError, match="Path does not exist"):
        route_choice_setup["rc"].set_save_routes(tmp_path / "missing")


@pytest.mark.parametrize("recompute_psl", [False, True])
def test_path_file_assignment_round_trip(route_choice_setup, tmp_path, recompute_psl):
    path = tmp_path / "routes"
    path.mkdir()

    choice = route_choice_setup["rc"]
    choice.set_choice_set_generation("link-penalisation", max_routes=5, penalty=1.1)
    choice.set_demand(route_choice_setup["mat"])
    choice.set_select_links({"selected": [(23, 1), (26, 1)]})
    choice.execute()
    choice.save_path_files(path)

    imported = configured_choice(route_choice_setup["graph"])
    imported.set_demand(route_choice_setup["mat"])
    imported.set_select_links({"selected": [(23, 1), (26, 1)]})
    imported.execute_from_path_files(path, recompute_psl=recompute_psl)

    # All generated fields must match after PSL recomputation. Without recomputation,
    # supplied costs, masks and probabilities are kept, as are the assignment results.
    result = choice.get_results()
    pd.testing.assert_frame_equal(result, imported.get_results(), check_like=True)
    pd.testing.assert_frame_equal(choice.get_load_results(), imported.get_load_results(), check_like=True)
    pd.testing.assert_frame_equal(
        choice.get_select_link_loading_results(), imported.get_select_link_loading_results(), check_like=True
    )


def test_path_file_assignment_recosts_routes(route_choice_setup, tmp_path):
    project = route_choice_setup["project"]
    graph = route_choice_setup["graph"]
    path = tmp_path / "routes"
    path.mkdir()

    source = route_choice_setup["rc"]
    source.set_choice_set_generation("link-penalisation", max_routes=5, penalty=1.1)
    source.set_demand(route_choice_setup["mat"])
    source.execute()

    before = source.get_results()
    source.save_path_files(path)

    graph.set_costs(project.network.build(fields=["free_flow_time"]).field_values("free_flow_time"))
    choice = configured_choice(graph)
    choice.set_demand(route_choice_setup["mat"])
    choice.execute_from_path_files(path, recompute_psl=True)
    after = choice.get_results()

    # Recomputing PSL with a changed cost field keeps routes, but recosts them and changes link loads.
    identity = ["origin id", "destination id", "route set", "route directions"]
    pd.testing.assert_frame_equal(before[identity], after[identity], check_like=True)

    costs = project.network.links.data.set_index("link_id")["free_flow_time"]
    expected_costs = [costs.loc[route].sum() for route in after["route set"]]
    np.testing.assert_allclose(after["cost"], expected_costs)
    np.testing.assert_allclose(after.groupby(["origin id", "destination id"])["probability"].sum(), 1.0)

    assert not np.allclose(before["probability"], after["probability"])
    before_loads = source.get_load_results().sort_index(axis=1)
    after_loads = choice.get_load_results().sort_index(axis=1)
    assert not np.allclose(before_loads, after_loads)


def test_save_project_results(route_choice_setup):
    project = route_choice_setup["project"]
    choice = route_choice_setup["rc"]
    choice.set_choice_set_generation("link-penalisation", max_routes=5, penalty=1.1)
    choice.set_select_links({"sl1": [(23, 1), (26, 1)], "sl2": [(11, 1)]})
    choice.set_demand(route_choice_setup["mat"])
    choice.execute()
    loads = choice.get_load_results()
    selected = choice.get_select_link_loading_results()

    choice.save_link_flows("ll")
    choice.save_select_link("sl")
    metadata = project.matrices.get_available_matrices_and_metadata()["sl.omx"]
    assert metadata["procedure"] == "Route choice OD select link"
    assert metadata["procedure_id"] == choice.procedure_id

    with project.results_connection as conn:
        for table, expected in [("ll_uncompressed", loads), ("sl_uncompressed", selected)]:
            stored = pd.read_sql(f"select * from {table}", conn).set_index("link_id")
            retrieved = project.results.get_results(table).set_index("link_id")
            # SQLite reports integer IDs as signed, unlike the shared network identity.
            stored.index = stored.index.astype(np.uint64)
            retrieved.index = retrieved.index.astype(np.uint64)
            pd.testing.assert_frame_equal(stored, expected, check_like=True)
            pd.testing.assert_frame_equal(retrieved, expected, check_like=True)

    matrices = AequilibraeSparseMatrix.from_omx(
        Path(project.project_base_path) / "matrices" / "sl.omx", index_name="route_choice_sl_index"
    )
    for selection, demands in choice.get_select_link_od_matrix_results().items():
        for demand_name, matrix in demands.items():
            np.testing.assert_allclose(matrix.toarray(), matrices[selection + "_" + demand_name].toarray())


def parquet_compressions(where):
    files = list(where.rglob("*.parquet"))
    assert files, f"no parquet files written to `{where}`"
    codecs = set()
    for file in files:
        metadata = pq.ParquetFile(file).metadata
        for group in range(metadata.num_row_groups):
            for column in range(metadata.num_columns):
                codecs.add(metadata.row_group(group).column(column).compression)
    return codecs


@pytest.mark.parametrize("kwargs,codec", [(None, "ZSTD"), ({"compression": "snappy"}, "SNAPPY")])
def test_streamed_routes_match_memory(route_choice_setup, tmp_path, kwargs, codec):
    path = tmp_path / "routes"
    path.mkdir()

    choice = route_choice_setup["rc"]
    choice.set_demand(route_choice_setup["mat"])
    choice.set_choice_set_generation("link-penalisation", max_routes=5, penalty=1.1)
    choice.execute()
    reference = choice.get_results()

    # Without to_parquet_kwargs the default compression applies
    choice.set_save_routes(path, to_parquet_kwargs=kwargs)
    choice.execute()
    assert parquet_compressions(path) == {codec}

    # Non-overridden defaults still apply: the dataset remains hive partitioned by origin.
    assert all(directory.name.startswith("origin id=") for directory in path.iterdir())
    order = ["origin id", "destination id", "cost"]
    actual = choice.get_results()
    pd.testing.assert_frame_equal(
        reference.sort_values(order).reset_index(drop=True),
        actual.sort_values(order).reset_index(drop=True),
        check_like=True,
    )


def test_save_path_files_forwards_parquet_kwargs(route_choice_setup, tmp_path):
    choice = route_choice_setup["rc"]
    choice.set_choice_set_generation("bfsle", max_routes=2, max_depth=5)
    choice.prepare([(20, 4)])
    choice.execute(perform_assignment=False)
    # save_path_files forwards to_parquet_kwargs in the same manner as streamed routes.
    choice.save_path_files(tmp_path, compression="gzip")
    assert parquet_compressions(tmp_path) == {"GZIP"}


def test_path_file_without_psl_rejects_invalid_masked_routes(tmp_path):
    choice = configured_choice(diamond())
    choice.set_demand(single_od_demand(10, 40))
    file = tmp_path / "invalid_route.parquet"
    pd.DataFrame(
        {
            "origin id": pd.Series([10], dtype="uint64"),
            "destination id": pd.Series([40], dtype="uint64"),
            "route set": [[71, 24]],
            "route directions": [[1, 1]],
            "probability": [1.0],
            "mask": [False],
        }
    ).to_parquet(file, index=False)
    with pytest.raises(ValueError, match="disconnected links"):
        choice.execute_from_path_files(file)


@pytest.mark.parametrize(
    "compact,turns,heuristic", [(False, False, None), (True, False, "euclidean"), (True, True, "haversine")]
)
def test_uint64_identity_and_path_files(compact, turns, heuristic, tmp_path):
    origin, middle, destination, link = 2**63 + 7, 2**63 + 8, 2**64 - 1, 2**64 - 3
    links = pd.DataFrame(
        {
            "link_id": np.array([0, link], dtype=np.uint64),
            "a_node": np.array([origin, middle], dtype=np.uint64),
            "b_node": np.array([middle, destination], dtype=np.uint64),
            "direction": [0, 0],
            "time": [1.0, 2.0],
        }
    )
    restrictions = (
        pd.DataFrame({"from_node": [origin], "via_node": [middle], "to_node": [destination], "penalty": [0.5]})
        if turns
        else None
    )
    graph = Network(links, centroids=np.array([destination, origin], dtype=np.uint64)).build_graph(
        "c", "time", compact=compact, turn_restrictions=restrictions
    )
    columns = ("lon", "lat") if heuristic == "haversine" else ("x", "y")
    coordinates = pd.DataFrame({columns[0]: np.arange(graph.num_nodes), columns[1]: 0.0}, index=graph.all_nodes)
    choice = RouteChoice(graph, coordinates=coordinates)
    choice.set_cores(1)
    choice.set_choice_set_generation(
        "bfsle", max_routes=1, a_star=heuristic is not None, heuristic=heuristic or "euclidean", heuristic_scale=0.0
    )

    reverse_ids, reverse_directions = choice.execute_single(destination, origin)[0]
    np.testing.assert_array_equal(reverse_ids, np.array([link, 0], dtype=np.uint64))
    np.testing.assert_array_equal(reverse_directions, [-1, -1])

    ids, directions = choice.execute_single(origin, destination, demand=2.0)[0]
    np.testing.assert_array_equal(ids, np.array([0, link], dtype=np.uint64))
    np.testing.assert_array_equal(directions, [1, 1])
    assert ids.dtype == np.uint64
    assert directions.dtype == np.int8

    demand = from_dict(
        {"demand": np.array([[0.0, 2.0], [0.0, 0.0]])}, index=np.array([origin, destination], dtype=np.uint64)
    )
    choice.set_demand(demand)
    choice.set_select_links({"large": [(link, 1)]})
    choice.execute()

    result = choice.get_results()
    assert result["origin id"].dtype == result["destination id"].dtype == np.uint64
    assert result["cost"].iloc[0] == (3.5 if turns else 3.0)
    assert choice.get_load_results()["demand_tot"].tolist() == [2.0, 2.0]
    assert choice.get_select_link_loading_results()["demand_large_tot"].tolist() == [2.0, 2.0]

    choice.save_path_files(tmp_path)
    imported = configured_choice(graph)
    imported.set_demand(demand)
    imported.execute_from_path_files(tmp_path, recompute_psl=True)

    pd.testing.assert_frame_equal(imported.get_results(), result, check_like=True)
    pd.testing.assert_frame_equal(imported.get_load_results(), choice.get_load_results(), check_like=True)


@pytest.mark.parametrize("path_size_logit", [False, True])
@pytest.mark.parametrize("bfsle,turn", [(False, False), (True, True)])
def test_rebuild_results_table(turn, bfsle, path_size_logit):
    choice = RouteChoiceSet(compressed_graph(turn))
    with Bridge() as bridge:
        choice.batched(
            nodes=[(10, 60), (60, 10)],
            max_routes=2,
            max_depth=6,
            penalty=4.0,
            bfsle=bfsle,
            path_size_logit=path_size_logit,
            cores=1,
            bridge=bridge,
        )

    cached = choice.get_results()
    results = choice.get_results(rebuild=True)
    pd.testing.assert_frame_equal(results, cached, check_like=True)

    assert set(zip(results["origin id"], results["destination id"], strict=True)) == {(10, 60)}
    assert results["origin id"].dtype == np.uint64
    assert results["destination id"].dtype == np.uint64
    assert {tuple(route) for route in results["route set"]} == {(71, 72, 12), (55, 56, 24)}
    assert all(route.dtype == np.uint64 for route in results["route set"])
    assert all(directions.dtype == np.int8 and (directions == 1).all() for directions in results["route directions"])

    if path_size_logit:
        assert results["mask"].dtype == bool
        assert (results[["cost", "path overlap", "probability"]].dtypes == np.float64).all()
    else:
        assert set(results.columns) == {"origin id", "destination id", "route set", "route directions"}


def test_rebuild_requires_results():
    choice = RouteChoiceSet(compressed_graph(False))
    with pytest.raises(RuntimeError, match="results not computed yet"):
        choice.get_results(rebuild=True)


def test_json_routes_are_decoded_as_integer_arrays(monkeypatch):
    table = pd.DataFrame(
        {"origin id": [10], "destination id": [40], "route set": ["[11, 12]"], "route directions": ["[-1, 1]"]}
    )
    monkeypatch.setattr(pd, "read_parquet", lambda *args, **kwargs: table)
    result = RouteChoiceSetResults.read_dataset("unused")
    np.testing.assert_array_equal(result["route set"].iloc[0], [11, 12])
    assert result["route set"].iloc[0].dtype == np.uint64
    np.testing.assert_array_equal(result["route directions"].iloc[0], [-1, 1])
