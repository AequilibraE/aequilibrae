import random
import sqlite3
import string

import pytest
from aequilibrae import Project


class TestAbout:
    @pytest.fixture
    def project(self, create_project):
        return create_project()

    def test_create_and_list(self, project: Project):
        project.about.create()
        list = project.about.list_fields()
        expected = [
            "model_name",
            "region",
            "description",
            "author",
            "license",
            "scenario_name",
            "year",
            "scenario_description",
            "model_version",
            "project_id",
            "aequilibrae_version",
            "projection",
        ]
        assert not set(list) ^ set(expected), "About table does not have all expected fields"

    def test_warning_when_creating_twice(self, project: Project):
        project.about.create()
        project.about.create()
        last_log = project.log().contents()[-1]
        assert "About table already exists" in last_log

    # idea from https://stackoverflow.com/a/2030081/1480643
    def randomword(self, length):
        letters = string.ascii_lowercase + "_"
        return "".join(random.choice(letters) for i in range(length))

    def test_add_info_field(self, project: Project):
        project.about.create()

        all_added = set()
        for t in range(30):
            k = self.randomword(random.randint(1, 15))
            if k not in all_added:
                all_added.add(k)
                project.about.add_info_field(k)

        curr = project.conn.cursor()
        curr.execute("select infoname from 'about'")

        charac = [x[0] for x in curr.fetchall()]
        for k in all_added:
            if k not in charac:
                self.fail(f"Failed to add {k}")

        # Should fail when trying to add a repeated guy
        with pytest.raises(sqlite3.IntegrityError):
            project.about.add_info_field("description")

        # Should fail when trying to add a repeated guy
        with pytest.raises(ValueError):
            project.about.add_info_field("descr1ption")

    def test_write_back(self, project: Project):
        base_path = project.project_base_path
        project.about.create()
        project.about.add_info_field("good_info_field_perhaps")

        val = self.randomword(random.randint(1, 15))
        project.about.good_info_field_perhaps = val

        val2 = self.randomword(random.randint(30, 250))
        project.about.description = val2

        project.about.write_back()

        project.close()
        del project

        project = Project()
        project.open(base_path)
        assert val == project.about.good_info_field_perhaps, "failed to save data to about table"
        assert val2 == project.about.description, "failed to save data to about table"
