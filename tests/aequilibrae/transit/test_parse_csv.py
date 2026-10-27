import io

from aequilibrae.transit.column_order import column_order
from aequilibrae.transit.parse_csv import parse_csv


def parse(text: str, table: str):
    return parse_csv(io.BytesIO(text.encode()), column_order[table])


def test_empty_nullable_fields_are_kept_as_none():
    """Test that empty fare_attributes numeric fields are read as None, while given values keep their own type."""
    text = """fare_id,price,currency_type,payment_method,transfers,transfer_duration
free,0,USD,0,0,
unlimited,1.50,USD,1,,
two,2,USD,,2,3600
"""
    fares = parse(text, "fare_attributes.txt")
    rows = {row["fare_id"]: row for row in fares}

    assert rows["free"]["price"] == 0 and rows["free"]["price"] is not None
    assert rows["free"]["transfers"] == 0 and rows["free"]["transfers"] is not None
    assert rows["free"]["transfer_duration"] is None
    assert rows["unlimited"]["price"] == 1.5
    assert rows["unlimited"]["transfers"] is None
    assert rows["two"]["payment_method"] is None
    assert rows["two"]["transfers"] == 2
    assert rows["two"]["transfer_duration"] == 3600
    assert all(type(rows[fare]["transfers"]) is int for fare in ["free", "two"])
    assert type(rows["two"]["transfer_duration"]) is float


def test_missing_nullable_columns_are_read_as_none():
    """Test that an optional nullable column absent from the file is read as None, not 0."""
    fares = parse("fare_id,price,currency_type,payment_method,transfers\nF1,3,USD,0,1\n", "fare_attributes.txt")
    assert fares[0]["transfers"] == 1
    assert fares[0]["transfer_duration"] is None


def test_empty_fields_of_other_numeric_columns_are_still_read_as_zero():
    """Test that empty numeric fields of tables without nullable columns are still read as integer zeros."""
    trips = parse("route_id,service_id,trip_id,direction_id\nR1,all,T1,\nR1,all,T2,1.0\n", "trips.txt")
    assert trips["direction_id"].tolist() == [0, 1]
    assert trips.dtype["direction_id"].kind == "i"


def test_rows_missing_trailing_fields_are_padded():
    """Test that a row omitting its trailing empty fields is read as if they were present and empty."""
    text = "fare_id,price,currency_type,payment_method,transfers,transfer_duration\nF1,3,USD,0\n"
    fares = parse(text, "fare_attributes.txt")
    assert fares[0]["payment_method"] == 0
    assert fares[0]["transfers"] is None
    assert fares[0]["transfer_duration"] is None
