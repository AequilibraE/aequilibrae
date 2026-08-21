from random import randint

import pytest


def test_get(sioux_falls_test):
    links = sioux_falls_test.network.links
    with pytest.raises(ValueError, match="links has no record with link_id=123456"):
        _ = links.get(123456)

    link = links.get(1)
    assert link.capacity_ab == 25900.20064, "Did not populate link correctly"


def test_new(sioux_falls_test):
    links = sioux_falls_test.network.links
    new_id = links.insert(modes="c")

    with sioux_falls_test.db_connection as conn:
        expected_id = conn.execute("Select max(link_id) from Links").fetchone()[0]
    new_link = links.get(new_id)
    assert new_link.link_id == expected_id, "Did not allocate a new link ID properly"
    assert new_link.geometry is None, "Did not populate new geometry properly"


def test_copy_link(sioux_falls_test):
    links = sioux_falls_test.network.links

    with pytest.raises(ValueError, match="links has no record with link_id=11111"):
        _ = links.copy(11111)

    new_id = links.copy(11)
    new_link = links.get(new_id)
    old_link = links.get(11)

    assert new_link.geometry == old_link.geometry
    assert new_link.a_node == old_link.a_node
    assert new_link.b_node == old_link.b_node
    assert new_link.direction == old_link.direction
    assert new_link.distance > 0
    assert new_link.modes == old_link.modes
    assert new_link.link_type == old_link.link_type


def test_delete(sioux_falls_test):
    links = sioux_falls_test.network.links

    _ = links.get(10)

    with sioux_falls_test.db_connection as conn:
        tot = conn.execute("Select count(*) from Links").fetchone()[0]
        links.delete(10)
        links.delete(11)
        tot2 = conn.execute("Select count(*) from Links").fetchone()[0]

    assert tot == tot2 + 2, "Did not delete the link properly"

    with pytest.raises(ValueError, match="links has no record with link_id=123456"):
        links.delete(123456)

    with pytest.raises(ValueError, match="links has no record with link_id=10"):
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

    links.update(1, capacity_ba=val)
    assert links.get(1).capacity_ba == val, "Did not update correctly"

    links.update(1, capacity_ba=original_value)
    assert links.get(1).capacity_ba == original_value, "Did not restore correctly"
