from pathlib import Path

import pytest
from aequilibrae.project.field_editor import FieldEditor


class TestFieldEditor:
    @pytest.fixture(scope="class")
    def project_session(self, create_empty_project_session):
        return create_empty_project_session()

    @pytest.fixture(scope="class")
    def database_backup(self, project_session):
        return Path(project_session.project_base_path).joinpath("project_database.sqlite").read_bytes()

    @pytest.fixture
    def cleanup_database(self, project_session, database_backup):
        """An optional fixture that can be used to revert the project's database in case the test
        modifies it. This adds some extra overhead to the test (teardown)
        """
        yield
        Path(project_session.project_base_path).joinpath("project_database.sqlite").write_bytes(database_backup)

    @pytest.fixture(params=["link_types", "links", "modes", "nodes"])
    def table_name(self, request):
        return request.param

    @pytest.fixture
    def table(self, project_session, table_name):
        return FieldEditor(project_session, table_name)

    @pytest.fixture
    def field_name(self, table):
        return next(iter(table._original_values.keys()))

    @pytest.fixture
    def attribute_count(self, table, table_name):
        qry = f'select count(*) from "attributes_documentation" where name_table="{table_name}"'
        return table.project.conn.execute(qry).fetchone()[0]

    def test_building(self, table, attribute_count):
        assert attribute_count == len(table._original_values), "Meta table populated with the wrong number of elements"

    def test_error_when_adding_existing_attribute(self, table, field_name):
        with pytest.raises(ValueError, match="attribute_name already exists"):
            table.add(field_name, "some_value")

    @pytest.mark.parametrize(
        "attribute_name, error",
        [
            ("with space", 'attribute_name can only contain letters, numbers and "_"'),
            ("0starts_with_digit", "attribute_name cannot begin with a digit"),
        ],
    )
    def test_add_invalid_attribute(self, attribute_name, error, table):
        with pytest.raises(ValueError, match=error):
            table.add(attribute_name, "some description")

    @pytest.mark.usefixtures("cleanup_database")
    def test_add_valid_attribute(self, table, table_name, attribute_count):
        project = table.project
        new_attribute = "new_attribute"
        table.add(new_attribute, "some description")
        project.conn.commit()
        curr = project.conn.cursor()
        curr.execute(f'select count(*) from "attributes_documentation" where name_table="{table_name}"')
        q2 = curr.fetchone()[0]
        assert q2 == attribute_count + 1, "Adding element did not work"

        result = curr.execute(
            f'select "{new_attribute}" from "attributes_documentation" where name_table="{table_name}"'
        ).fetchone()[0]
        assert result == new_attribute

    # Here we override the `table_name` fixture. This fixture is not directly used in the test but
    # given as input for the `table` fixture, which we do consume here. This way we change the
    # `table_name` parametrization from four cases to just two, but only for this test.
    @pytest.mark.parametrize(
        "table_name, attribute, description",
        [
            ("link_types", "lanes", "Default number of lanes in each direction. E.g. 2"),
            ("nodes", "is_centroid", "Flag identifying centroids"),
        ],
    )
    def test_retrieve_existing_field(self, table, attribute, description):
        assert attribute in table._original_values
        assert getattr(table, attribute) == description

    @pytest.mark.parametrize(
        "table_name, attribute",
        [
            ("links", "link_id"),
            ("nodes", "node_id"),
        ],
    )
    def test_save(self, table_name, attribute, project_session, table):
        random_val = "some_value"
        setattr(table, attribute, random_val)
        table.save()
        table2 = FieldEditor(project_session, table_name)
        assert getattr(table2, attribute) == random_val, "Did not save values properly"
