from random import randint, random

import pytest
from shapely.ops import substring

from aequilibrae.utils.db_utils import read_and_close


@pytest.fixture
def links(sioux_falls_test):
    return sioux_falls_test.network.links


@pytest.fixture
def modes(sioux_falls_test):
    return sioux_falls_test.network.modes


@pytest.fixture
def link_id():
    return randint(1, 24)


@pytest.fixture
def link(links, link_id):
    return links.get(link_id)


def check_mode(sioux_falls_test, link_id):
    with read_and_close(sioux_falls_test.path_to_file) as conn:
        return conn.execute(f"Select modes from links where link_id={link_id}").fetchone()[0]


def test_delete(sioux_falls_test, links, link_id, link):
    link.delete()

    with pytest.raises(Exception):
        _ = links.get(link_id)

    with read_and_close(sioux_falls_test.path_to_file) as conn:
        lid = conn.execute(f"Select count(*) from links where link_id={link_id}").fetchone()[0]

    assert lid == 0, f"Failed to delete link {link_id}"


def test_save(sioux_falls_test, links, link_id, link):
    link.save()
    extension = random()
    name = "just a non-important value"

    geo = substring(link.geometry, 0, extension, normalized=True)

    link.name = name
    link.geometry = geo

    link.save()
    links.refresh()
    link2 = links.get(link_id)

    assert link2.name == name, "Failed to save the link name"
    assert link2.geometry.equals_exact(geo, 0.001), "Failed to save the link geometry"

    tot_prev = sioux_falls_test.network.count_links()
    lnk = links.new()
    lnk.geometry = substring(link.geometry, 0, 0.88, normalized=True)
    lnk.modes = "c"
    lnk.save()

    assert sioux_falls_test.network.count_links() == tot_prev + 1, "Failed to save new link"


def test_set_modes(sioux_falls_test, link, link_id):
    link.set_modes("cbt")

    assert link.modes == "cbt", "Did not set modes correctly"
    link.save()

    assert check_mode(sioux_falls_test, link_id) == "cbt"


def test_add_mode(sioux_falls_test, link, link_id, modes):
    for mode in [1, ["cbt"]]:
        with pytest.raises(TypeError):
            link.add_mode(mode)
    with pytest.raises(ValueError):
        link.add_mode("bt")

    link.add_mode("b")
    link.save()
    assert check_mode(sioux_falls_test, link_id) == "cb"

    mode = modes.get("t")
    link.add_mode(mode)
    link.save()
    assert check_mode(sioux_falls_test, link_id) == "cbt"


def test_drop_mode(sioux_falls_test, link, link_id, modes):
    link.set_modes("cbt")
    link.save()
    assert check_mode(sioux_falls_test, link_id) == "cbt"

    link.drop_mode("t")
    link.save()
    assert check_mode(sioux_falls_test, link_id) == "cb"

    mode = modes.get("b")
    link.drop_mode(mode)
    link.save()
    assert check_mode(sioux_falls_test, link_id) == "c"


def test_data_fields(sioux_falls_test, links, link):
    link2 = links.get(randint(1, 24))
    while link2.link_id == link.link_id:
        link2 = links.get(randint(1, 24))

    assert link2.data_fields() == link.data_fields(), "Different links have different data fields"

    fields = sorted(link2.data_fields())

    with read_and_close(sioux_falls_test.path_to_file) as conn:
        dt = conn.execute("pragma table_info(links)").fetchall()

    data_fields = sorted([x[1] for x in dt if x[1] != "ogc_fid"])

    assert sorted(fields) == sorted(data_fields), "Link has unexpected set of fields"
