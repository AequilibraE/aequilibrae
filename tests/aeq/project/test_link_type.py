import random
import string
from sqlite3 import IntegrityError

import pytest

from aequilibrae.utils.db_utils import read_and_close


@pytest.fixture
def random_string():
    letters = [random.choice(string.ascii_letters + "_") for x in range(20)]
    return "".join(letters)


@pytest.fixture
def link_types(empty_no_triggers_project):
    return empty_no_triggers_project.network.link_types


def test_changing_link_type_id(no_triggers_test):
    lt = random.choice(list(no_triggers_test.network.link_types.all_types().values()))

    with pytest.raises(ValueError):
        lt.link_type_id = "test my description"

    with pytest.raises(ValueError):
        lt.link_type_id = "K"


def test_empty(link_types):
    newt = link_types.new("Z")
    with pytest.raises(IntegrityError):
        newt.save()


def test_save(empty_no_triggers_project, link_types, random_string):
    newt = link_types.new("Z")
    newt.link_type = random_string
    newt.description = random_string[::-1]
    newt.save()

    with read_and_close(empty_no_triggers_project.path_to_file) as conn:
        sql = 'select description, link_type from link_types where link_type_id="Z"'
        desc, mname = conn.execute(sql).fetchone()

    assert desc == random_string[::-1], "Didn't save the mode description correctly"
    assert mname == random_string, "Didn't save the mode name correctly"
