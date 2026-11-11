import pandas as pd
import pytest

from aequilibrae.matrix import AequilibraeMatrix


@pytest.mark.parametrize(
    "project_fixture", ["empty_project", "sioux_falls_example", "nauru_example", "coquimbo_example"]
)
def test_matrix_summary_uses_filenames_and_matrix_stores(request, project_fixture):
    project = request.getfixturevalue(project_fixture)
    if project_fixture == "empty_project":
        assert project.run["matrix_summary"]() == {}

    matrix = AequilibraeMatrix(index=[10, 20])
    matrix["demand"] = [[0, 3], [2, 0]]
    project.matrices.save_matrix("summary.omx", matrix)

    summary = project.run["matrix_summary"]()
    assert summary["summary.omx"] == {"demand": {"total": 5, "min": 0, "max": 3, "nnz": 2}}
    assert set(summary) == set(project.matrices.get_available_matrices_and_metadata())


@pytest.mark.parametrize(
    "project_fixture", ["empty_project", "sioux_falls_example", "nauru_example", "coquimbo_example"]
)
def test_results_summary_reads_the_results_database(request, project_fixture):
    project = request.getfixturevalue(project_fixture)
    pd.testing.assert_frame_equal(project.run["results_summary"](), project.results.list())

    project.results.create("summary", pd.DataFrame({"value": [1]}), description="Summary test")
    summary = project.run["results_summary"]()

    pd.testing.assert_frame_equal(summary, project.results.list())
    assert "summary" in summary["table_name"].tolist()
