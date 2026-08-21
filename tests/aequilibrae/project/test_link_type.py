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
    link_types = no_triggers_test.network.link_types
    lt = random.choice(list(link_types))
    other = next(candidate for candidate in link_types if candidate.link_type_id != lt.link_type_id)

    with pytest.raises(IntegrityError, match="UNIQUE constraint failed: link_types.link_type_id"):
        link_types.update(lt.link_type_id, link_type_id=other.link_type_id)


def test_empty(link_types):
    with pytest.raises(IntegrityError, match="NOT NULL constraint failed: link_types.link_type"):
        link_types.insert(link_type_id="Z")


def test_save(empty_no_triggers_project, link_types, random_string):
    link_types.insert(link_type_id="Z", link_type=random_string, description=random_string[::-1])

    with read_and_close(empty_no_triggers_project.path_to_file) as conn:
        sql = 'select description, link_type from link_types where link_type_id="Z"'
        desc, mname = conn.execute(sql).fetchone()

    assert desc == random_string[::-1], "Didn't save the mode description correctly"
    assert mname == random_string, "Didn't save the mode name correctly"
