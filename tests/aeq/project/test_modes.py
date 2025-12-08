import random
import string
from sqlite3 import IntegrityError

import pytest

from aequilibrae.project.network.mode import Mode
from aequilibrae.utils.db_utils import read_and_close


def test_add(sioux_falls_example):
    new_mode = Mode("F", sioux_falls_example)
    name = "".join([random.choice(string.ascii_letters + "_") for _ in range(random.randint(1, 20))])
    new_mode.mode_name = name
    sioux_falls_example.network.modes.add(new_mode)

    with read_and_close(sioux_falls_example.path_to_file) as conn:
        mode = conn.execute('select mode_name from modes where mode_id="F"').fetchone()[0]

    assert mode == name, "Could not save the mode properly to the database"


def test_drop(sioux_falls_example):
    sioux_falls_example.network.modes.delete("b")

    with pytest.raises(IntegrityError):
        sioux_falls_example.network.modes.delete("c")


def test_get(sioux_falls_example):
    c = sioux_falls_example.network.modes.get("c")
    assert c.description == "All motorized vehicles"
    del c

    with pytest.raises(ValueError):
        _ = sioux_falls_example.network.modes.get("f")


def test_new(sioux_falls_example):
    modes = sioux_falls_example.network.modes
    assert isinstance(modes.new("h"), Mode), "Returned wrong type"

    m = list(modes.all_modes().keys())[0]
    with pytest.raises(ValueError):
        modes.new(m)


def test_fields(sioux_falls_example):
    fields = sioux_falls_example.network.modes.fields
    fields.all_fields()
    assert fields._table == "modes", "Returned wrong table handler"
