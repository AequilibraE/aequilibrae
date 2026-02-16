import random
import sqlite3
import string

import pytest

from aequilibrae.project.project import Project
from aequilibrae.utils.db_utils import read_and_close


def randomword(length):
    letters = string.ascii_lowercase + "_"
    return "".join(random.choice(letters) for _ in range(length))


def test_create_and_list(sioux_falls_example):
    sioux_falls_example.about.create()
    fields = sioux_falls_example.about.list_fields()
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
    assert not set(fields) ^ set(expected), "About table does not have all expected fields"


def test_warning_when_creating_twice(sioux_falls_example: Project):
    sioux_falls_example.about.create()
    sioux_falls_example.about.create()
    last_log = sioux_falls_example.log().contents()[-1]
    assert "About table already exists" in last_log


def test_add_info_field(sioux_falls_example: Project):
    sioux_falls_example.about.create()
    all_added = set()
    for _ in range(30):
        k = randomword(random.randint(1, 15))
        if k not in all_added:
            all_added.add(k)
            sioux_falls_example.about.add_info_field(k)

    with read_and_close(sioux_falls_example.path_to_file) as conn:
        charac = [x[0] for x in conn.execute("select infoname from 'about'").fetchall()]

    for k in all_added:
        assert k in charac, f"Failed to add {k}"

    with pytest.raises(sqlite3.IntegrityError):
        sioux_falls_example.about.add_info_field("description")

    with pytest.raises(ValueError):
        sioux_falls_example.about.add_info_field("descr1ption")


def test_write_back(sioux_falls_example: Project):
    base_path = sioux_falls_example.project_base_path
    sioux_falls_example.about.create()
    sioux_falls_example.about.add_info_field("good_info_field_perhaps")

    val = randomword(random.randint(1, 15))
    sioux_falls_example.about.good_info_field_perhaps = val

    val2 = randomword(random.randint(30, 250))
    sioux_falls_example.about.description = val2

    sioux_falls_example.about.write_back()

    sioux_falls_example.close()

    project = Project()
    project.open(base_path)
    assert val == project.about.good_info_field_perhaps, "failed to save data to about table"
    assert val2 == project.about.description, "failed to save data to about table"
