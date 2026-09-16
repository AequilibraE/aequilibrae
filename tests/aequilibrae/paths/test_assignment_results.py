from aequilibrae.utils.cython.openmp_helper import omp_get_max_threads

import pytest

from aequilibrae.paths.results import AssignmentResults


def test_set_cores():
    a = AssignmentResults()
    a.set_cores(10)

    with pytest.raises(ValueError):
        a.set_cores(1.3)

    a.set_cores(-2)
    assert a.cores == max(1, omp_get_max_threads() - 2)


def test_set_save_path_file():
    a = AssignmentResults()

    # Never save by default
    assert a.save_path_file is False
