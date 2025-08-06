import pytest

from aequilibrae.project.field_editor import FieldEditor


@pytest.fixture(params=["link_types", "links", "modes", "nodes"])
def table_name(request):
    return request.param


@pytest.fixture(scope="function")
def table(empty_project, table_name):
    return FieldEditor(empty_project, table_name)


@pytest.fixture(scope="function")
def field_name(table):
    return next(iter(table._original_values.keys()))


@pytest.fixture(scope="function")
def attribute_count(table, table_name):
    qry = f'select count(*) from "attributes_documentation" where name_table="{table_name}"'
    with table.project.db_connection as conn:
        return conn.execute(qry).fetchone()[0]


def test_building(table, attribute_count):
    assert attribute_count == len(table._original_values), "Meta table populated with the wrong number of elements"


def test_error_when_adding_existing_attribute(table, field_name):
    with pytest.raises(ValueError, match="attribute_name already exists"):
        table.add(field_name, "some_value")


@pytest.mark.parametrize(
    "attribute_name, error",
    [
        ("with space", 'attribute_name can only contain letters, numbers and "_"'),
        ("0starts_with_digit", "attribute_name cannot begin with a digit"),
    ],
)
def test_add_invalid_attribute(attribute_name, error, table):
    with pytest.raises(ValueError, match=error):
        table.add(attribute_name, "some description")


def test_add_valid_attribute(empty_project, table, table_name, attribute_count):
    new_attribute = "new_attribute"
    table.add(new_attribute, "some description")

    with empty_project.db_connection as conn:
        sql = f'select count(*) from "attributes_documentation" where name_table="{table_name}"'
        q2 = conn.execute(sql).fetchone()[0]
        assert q2 == attribute_count + 1, "Adding element did not work"

        sql = f'select "{new_attribute}" from "attributes_documentation" where name_table="{table_name}"'
        result = conn.execute(sql).fetchone()[0]
        assert result == new_attribute


@pytest.mark.parametrize(
    "table_name, attribute, description",
    [
        ("link_types", "lanes", "Default number of lanes in each direction. E.g. 2"),
        ("nodes", "is_centroid", "Flag identifying centroids"),
    ],
)
def test_retrieve_existing_field(table, attribute, description):
    assert attribute in table._original_values
    assert getattr(table, attribute) == description


@pytest.mark.parametrize(
    "table_name, attribute",
    [
        ("links", "link_id"),
        ("nodes", "node_id"),
    ],
)
def test_save(empty_project, table_name, attribute):
    table = FieldEditor(empty_project, table_name)
    random_val = "some_value"
    setattr(table, attribute, random_val)
    table.save()
    table2 = FieldEditor(empty_project, table_name)
    assert getattr(table2, attribute) == random_val, "Did not save values properly"
