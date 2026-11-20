import json

import pandas as pd
import pytest

from aequilibrae.utils.simwrapper.simwrapper_utils import export_convergence_csv, parse_convergence_json

REPORT = {"convergence": {"iteration": [1, 2, 3], "rgap": [float("inf"), 0.1, 0.01]}, "setup": {"Algorithm": "bfw"}}
SINGLE_ENCODED = json.dumps(REPORT)
DOUBLE_ENCODED = json.dumps(SINGLE_ENCODED)  # As written by versions 1.5 to 1.7 and 2.0.0rc1


@pytest.mark.parametrize("stored", [SINGLE_ENCODED, DOUBLE_ENCODED], ids=["single-encoded", "double-encoded"])
def test_parse_convergence_json_reads_single_and_double_encoded_reports(stored):
    assert parse_convergence_json(stored) == ([1, 2, 3], [float("inf"), 0.1, 0.01])


@pytest.mark.parametrize("stored", [None, "", "null", "not json", json.dumps("plain text"), json.dumps([1, 2])])
def test_parse_convergence_json_ignores_reports_without_convergence(stored):
    assert parse_convergence_json(stored) == ([], [])


def test_export_convergence_csv_reads_old_and_new_rows(tmp_path):
    results = pd.DataFrame(
        {"table_name": ["new", "old", "empty"], "procedure_report": [SINGLE_ENCODED, DOUBLE_ENCODED, "null"]}
    )

    written = pd.read_csv(export_convergence_csv(results, tmp_path))

    assert written["series"].tolist() == ["new"] * 3 + ["old"] * 3
    assert written["iteration"].tolist() == [1, 2, 3] * 2
    assert written["rgap"].tolist() == [float("inf"), 0.1, 0.01] * 2
