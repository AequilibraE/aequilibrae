import sqlite3
from dataclasses import fields

import pytest


@pytest.fixture(params=["link_types", "links", "modes", "nodes"])
def table_name(request):
    return request.param


@pytest.fixture(scope="function")
def table(empty_project, table_name):
    return getattr(empty_project.tables, table_name).fields


@pytest.fixture(scope="function")
def field_name(table):
    return table.all_fields()[0]


@pytest.fixture(scope="function")
def attribute_count(empty_project, table_name):
    def count():
        sql = "SELECT count(*) FROM attributes_documentation WHERE name_table=?"
        with empty_project.db_connection as conn:
            return conn.execute(sql, (table_name,)).fetchone()[0]

    return count


def test_reading_fields_does_not_change_metadata(empty_project, table_name, attribute_count):
    count_before = attribute_count()
    editor = getattr(empty_project.tables, table_name).fields
    for field_name in editor.all_fields():
        editor.get_description(field_name)
    assert attribute_count() == count_before


def test_error_when_adding_existing_attribute(table, field_name):
    with pytest.raises(ValueError, match="attribute_name already exists"):
        table.add(field_name, "some_value")


@pytest.mark.parametrize(
    "attribute_name, error",
    [
        ("with space", 'attribute_name can only contain letters, numbers and "_"'),
        ("", 'attribute_name can only contain letters, numbers and "_"'),
        ("0starts_with_digit", "attribute_name cannot begin with a digit"),
    ],
)
def test_add_invalid_attribute(attribute_name, error, table):
    with pytest.raises(ValueError, match=error):
        table.add(attribute_name, "some description")


def test_add_valid_attribute(empty_project, table, table_name, attribute_count):
    new_attribute = "new_attribute"
    count_before = attribute_count()
    table.add(new_attribute, "some description")

    project_table = getattr(empty_project.tables, table_name)
    assert attribute_count() == count_before + 1
    assert table.get_description(new_attribute) == "some description"
    assert new_attribute in {field.name for field in fields(project_table.record_type)}
    assert new_attribute in project_table.data.columns


@pytest.mark.parametrize(
    "table_name, attribute, description",
    [
        ("link_types", "lanes", "Default number of lanes in each direction. E.g. 2"),
        ("nodes", "is_centroid", "Flag identifying centroids"),
    ],
)
def test_retrieve_existing_field(table, attribute, description):
    assert attribute in table.all_fields()
    assert table.get_description(attribute) == description


@pytest.mark.parametrize(
    "table_name, attribute",
    [
        ("links", "link_id"),
        ("nodes", "node_id"),
    ],
)
def test_set_description(empty_project, table, table_name, attribute):
    other_editor = getattr(empty_project.tables, table_name).fields
    new_description = "some_value"
    table.set_description(attribute, new_description)
    assert other_editor.get_description(attribute) == new_description


@pytest.mark.parametrize("add_through_editor", [True, False])
def test_editor_sees_new_mixed_case_column(empty_project, add_through_editor):
    zones = empty_project.tables.zones
    editor = zones.fields
    if add_through_editor:
        zones.fields.add("Parking_Spots", "Public parking spots", "INTEGER")
        description = "Public parking spots"
    else:
        with empty_project.db_connection.transaction() as conn:
            conn.execute('ALTER TABLE zones ADD COLUMN "Parking_Spots" INTEGER')
        description = "not provided"

    assert editor.get_description("parking_spots") == description
    assert "Parking_Spots" in zones.columns

    editor.set_description("PARKING_SPOTS", "Total licensed parking spots")
    with empty_project.db_connection as conn:
        rows = conn.execute(
            "SELECT attribute, description FROM attributes_documentation "
            "WHERE name_table=? AND attribute=? COLLATE NOCASE",
            ("zones", "parking_spots"),
        ).fetchall()
    assert rows == [("Parking_Spots", "Total licensed parking spots")]


@pytest.mark.parametrize("add_as_pair", [True, False])
def test_directional_link_descriptions(empty_project, add_as_pair):
    links = empty_project.tables.links
    editor = links.fields
    assert editor.get_description("speed_ab") == "Directional speeds (if allowed)"

    if add_as_pair:
        editor.add("Toll", "Directional toll", directional=True)
    else:
        editor.add("Toll_AB", "Directional toll")
        editor.add("toll_ba", "This does not replace the shared description")

    assert {"toll_ab", "toll_ba"} <= {column.lower() for column in links.columns}
    assert editor.all_fields().count("toll") == 1
    assert editor.get_description("toll") == "Directional toll"
    assert editor.get_description("toll_ab") == editor.get_description("toll_ba") == "Directional toll"

    editor.add("toll", "Standalone toll")
    editor.set_description("toll_ba", "Directional charge")
    assert editor.get_description("toll_*") == editor.get_description("toll_ab") == "Directional charge"
    assert editor.get_description("toll") == "Standalone toll"

    editor.add("toll_ab_extra", "Ordinary field")
    assert "toll_ab_extra" in editor.all_fields()
    assert editor.get_description("toll_ab_extra") == "Ordinary field"


def test_bulk_field_edits(empty_project, monkeypatch):
    links = empty_project.tables.links
    editor = links.fields
    refresh_record_type = links._refresh_record_type
    schema_refreshes = []

    def track_refresh():
        schema_refreshes.append(links.columns)
        refresh_record_type()

    monkeypatch.setattr(links, "_refresh_record_type", track_refresh)
    editor.add_many(
        [
            ("notes", "Link notes", "TEXT"),
            ("toll", "Directional toll", "REAL", True),
            ("lanes_AB", "Directional lanes"),
            ("lanes_ba", "Keep the first description"),
        ]
    )
    assert schema_refreshes == [links.columns]
    assert {"notes", "toll_ab", "toll_ba", "lanes_AB", "lanes_ba"} <= set(links.columns)
    assert editor.get_description("lanes_ba") == "Directional lanes"
    assert editor.get_description("toll_*") == "Directional toll"

    schema_refreshes.clear()
    editor.set_descriptions([("toll_ba", "New toll description"), ("notes", "New notes description")])
    assert schema_refreshes == [links.columns]
    assert editor.get_description("toll_ab") == "New toll description"
    assert editor.get_description("notes") == "New notes description"


def test_bulk_field_edits_roll_back_on_later_error(empty_project):
    links = empty_project.tables.links
    editor = links.fields
    with pytest.raises(ValueError, match="attribute_name already exists"):
        editor.add_many([("toll", "Directional toll", "REAL", True), ("Toll_AB", "Duplicate")])

    assert {"toll_ab", "toll_ba"}.isdisjoint(links.columns)
    with empty_project.db_connection as conn:
        row = conn.execute(
            "SELECT 1 FROM attributes_documentation WHERE name_table=? AND attribute=?", ("links", "toll_*")
        ).fetchone()
    assert row is None

    original_description = editor.get_description("speed_ab")
    with pytest.raises(ValueError, match="links has no field 'missing'"):
        editor.set_descriptions([("speed_ba", "New speed description"), ("missing", "Missing field")])

    assert editor.get_description("speed_ba") == original_description


def test_directional_add_rolls_back_if_metadata_write_fails(empty_project):
    links = empty_project.tables.links
    with empty_project.db_connection as conn:
        conn.execute(
            "CREATE TRIGGER reject_toll BEFORE INSERT ON attributes_documentation "
            "WHEN NEW.attribute = 'toll_*' BEGIN SELECT RAISE(ABORT, 'metadata insert rejected'); END"
        )

    with pytest.raises(sqlite3.IntegrityError, match="metadata insert rejected"):
        links.fields.add("toll", "Directional toll", directional=True)

    assert {"toll_ab", "toll_ba"}.isdisjoint(links.columns)
    assert "toll" not in links.fields.all_fields()


def test_record_type_is_correct_after_rollback_and_schema_version_reuse(empty_project):
    modes = empty_project.tables.modes
    with pytest.raises(RuntimeError, match="roll back the field edit"):
        with empty_project.transaction():
            modes.fields.add("temporary_field", "Rolled back field")
            assert "temporary_field" in modes.data.columns
            raise RuntimeError("roll back the field edit")

    with empty_project.db_connection as conn:
        conn.execute("ALTER TABLE modes ADD COLUMN final_field TEXT")

    assert "temporary_field" not in modes.fields.all_fields()
    assert {field.name for field in fields(modes.record_type)} == set(modes.columns)
    assert tuple(modes.data.columns) == modes.columns
