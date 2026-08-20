"""Tests that two simultaneously open Project instances do not affect each other.

Two independent ``Project`` objects must be completely independent: writes to
one must not be visible in the other, and closing one must not disturb the
other.
"""

import pytest

from aequilibrae.project.project import Project


@pytest.fixture
def two_projects(tmp_path):
    p1 = Project.new(tmp_path / "proj1")
    p2 = Project.new(tmp_path / "proj2")
    yield p1, p2
    p1.close()
    p2.close()


def test_two_projects_have_distinct_paths(two_projects):
    p1, p2 = two_projects
    assert p1.project_base_path != p2.project_base_path


def test_write_to_one_does_not_appear_in_other(two_projects):
    """Inserting a mode in p1 must not make it appear in p2."""
    p1, p2 = two_projects

    p1.network.modes.insert(mode_id="x", mode_name="extra_mode")

    modes_p1 = {m.mode_id for m in p1.network.modes}
    modes_p2 = {m.mode_id for m in p2.network.modes}

    assert "x" in modes_p1
    assert "x" not in modes_p2


def test_closing_one_does_not_affect_other(two_projects):
    """Closing p1 must not close or corrupt p2's connections."""
    p1, p2 = two_projects
    p1.close()

    # p2 must still be usable
    with p2.db_connection as conn:
        count = conn.execute("SELECT count(*) FROM links").fetchone()[0]
    assert count == 0


def test_two_projects_context_managers(tmp_path):
    """Both projects can be used as context managers simultaneously."""
    with Project.new(tmp_path / "x") as p1:
        with Project.new(tmp_path / "y") as p2:
            # Both usable at the same time
            with p1.db_connection as c1:
                links1 = c1.execute("SELECT count(*) FROM links").fetchone()[0]
            with p2.db_connection as c2:
                links2 = c2.execute("SELECT count(*) FROM links").fetchone()[0]
    assert links1 == 0
    assert links2 == 0
