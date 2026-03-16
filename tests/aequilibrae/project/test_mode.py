import random
import string
import pytest

from aequilibrae.project.network.mode import Mode


def get_random_string():
    letters = [random.choice(string.ascii_letters + "_") for _ in range(20)]
    return "".join(letters)


def test_build(empty_no_triggers_project):
    for val in ["1", "ab", "", None]:
        with pytest.raises(ValueError):
            Mode(val, empty_no_triggers_project)
    for _ in range(10):
        letter = random.choice(string.ascii_letters)
        m = Mode(letter, empty_no_triggers_project)
        del m


def test_changing_mode_id(empty_no_triggers_project):
    m = Mode("c", empty_no_triggers_project)
    with pytest.raises(ValueError):
        m.mode_id = "test my description"


def test_save(sioux_falls_example):
    random_string = get_random_string()
    with sioux_falls_example.db_connection as conn:
        letter = random.choice([x[0] for x in conn.execute("select mode_id from 'modes'").fetchall()])
        m = Mode(letter, sioux_falls_example)
        m.mode_name = random_string
        m.description = random_string[::-1]
        m.save()
        desc, mname = conn.execute(f'select description, mode_name from modes where mode_id="{letter}"').fetchone()
        assert desc == random_string[::-1], "Didn't save the mode description correctly"
        assert mname == random_string, "Didn't save the mode name correctly"


def test_empty(empty_no_triggers_project):
    a = Mode("k", empty_no_triggers_project)
    a.mode_name = "just a_test"
    with pytest.raises(ValueError):
        a.save()
    a = Mode("l", empty_no_triggers_project)
    a.mode_name = "just_a_test_test_with_l"
    with pytest.raises(ValueError):
        a.save()
