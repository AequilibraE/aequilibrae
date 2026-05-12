from copy import copy, deepcopy
from random import randint

import pytest


def test_get(sioux_falls_test):
    links = sioux_falls_test.network.links
    with pytest.raises(ValueError):
        _ = links.get(123456)

    link = links.get(1)
    assert link.capacity_ab == 25900.20064, "Did not populate link correctly"


def test_new(sioux_falls_test):
    links = sioux_falls_test.network.links
    new_link = links.new()

    with sioux_falls_test.db_connection as conn:
        id = conn.execute("Select max(link_id) + 1 from Links").fetchone()[0]
    assert new_link.link_id == id, "Did not populate new link ID properly"
    assert new_link.geometry is None, "Did not populate new geometry properly"


def test_copy_link(sioux_falls_test):
    links = sioux_falls_test.network.links

    with pytest.raises(ValueError):
        _ = links.copy_link(11111)

    new_link = links.copy_link(11)
    old_link = links.get(11)

    assert new_link.geometry == old_link.geometry
    assert new_link.a_node == old_link.a_node
    assert new_link.b_node == old_link.b_node
    assert new_link.direction == old_link.direction
    assert new_link.distance == old_link.distance
    assert new_link.modes == old_link.modes
    assert new_link.link_type == old_link.link_type
    new_link.save()


def test_delete(sioux_falls_test):
    links = sioux_falls_test.network.links

    _ = links.get(10)

    with sioux_falls_test.db_connection as conn:
        tot = conn.execute("Select count(*) from Links").fetchone()[0]
        links.delete(10)
        links.delete(11)
        tot2 = conn.execute("Select count(*) from Links").fetchone()[0]

    assert tot == tot2 + 2, "Did not delete the link properly"

    with pytest.raises(ValueError):
        links.delete(123456)

    with pytest.raises(ValueError):
        _ = links.get(10)


def test_fields(sioux_falls_test):
    links = sioux_falls_test.network.links
    f_editor = links.fields

    fields = sorted(f_editor.all_fields())
    with sioux_falls_test.db_connection as conn:
        dt = conn.execute("pragma table_info(links)").fetchall()

    actual_fields = sorted({x[1].replace("_ab", "").replace("_ba", "") for x in dt if x[1] != "ogc_fid"})
    assert fields == actual_fields, "Table editor is weird for table links"


def test_refresh(sioux_falls_test):
    links = sioux_falls_test.network.links

    link1 = links.get(1)
    val = randint(1, 99999999)
    original_value = link1.capacity_ba

    link1.capacity_ba = val
    link1_again = links.get(1)
    assert link1_again.capacity_ba == val, "Did not preserve correctly"

    links.refresh()
    link1 = links.get(1)
    assert link1.capacity_ba == original_value, "Did not reset correctly"


def test_copy(sioux_falls_test):
    nodes = sioux_falls_test.network.nodes
    with pytest.raises(Exception):
        _ = copy(nodes)
    with pytest.raises(Exception):
        _ = deepcopy(nodes)
