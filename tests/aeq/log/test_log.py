import os
from logging import FileHandler

import pytest
from aequilibrae import Project


def test_contents(sioux_falls_test):
    log = sioux_falls_test.log()
    cont = log.contents()
    assert len(cont) == 4, "Returned the wrong amount of data from the log"


def test_clear(sioux_falls_test):
    log = sioux_falls_test.log()
    log.clear()

    proj_dir = sioux_falls_test.project_base_path
    with open(os.path.join(proj_dir, "aequilibrae.log"), "r") as file:
        q = file.readlines()
    assert len(q) == 0, "Failed to clear the log file"
