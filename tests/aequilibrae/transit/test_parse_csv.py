import io

import pytest

from aequilibrae.transit.parse_csv import parse_csv

COLUMNS = {"stop_id": str, "stop_name": str, "zone_id": str}


@pytest.mark.parametrize(
    "text, rows",
    [
        ("stop_id,stop_name,zone_id\nA,Plaza\nB,Mercado,Z1\n", [("A", "Plaza", ""), ("B", "Mercado", "Z1")]),
        ("stop_id\nA\n", [("A", "", "")]),
    ],
)
def test_rows_and_files_without_optional_fields_are_read(text, rows):
    """Test that rows leaving out trailing fields, and one-row files lacking optional columns, are read with blanks."""
    assert [tuple(row) for row in parse_csv(io.BytesIO(text.encode()), COLUMNS)] == rows
