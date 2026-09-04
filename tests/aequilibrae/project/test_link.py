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
    links.delete(link_id)

    with pytest.raises(ValueError, match=rf"links has no record with link_id={link_id}"):
        _ = links.get(link_id)

    with read_and_close(sioux_falls_test.path_to_file) as conn:
        lid = conn.execute(f"Select count(*) from links where link_id={link_id}").fetchone()[0]

    assert lid == 0, f"Failed to delete link {link_id}"


def test_save(sioux_falls_test, links, link_id, link):
    extension = random()
    name = "just a non-important value"

    geo = substring(link.geometry, 0, extension, normalized=True)

    links.update(link_id, name=name, geometry=geo)

    link2 = links.get(link_id)

    assert link2.name == name, "Failed to save the link name"
    assert link2.geometry.equals_exact(geo, 0.001), "Failed to save the link geometry"

    tot_prev = sioux_falls_test.network.count_links()
    links.insert(geometry=substring(link.geometry, 0, 0.88, normalized=True), modes="c")

    assert sioux_falls_test.network.count_links() == tot_prev + 1, "Failed to save new link"


def test_set_modes(sioux_falls_test, links, link_id):
    links.update(link_id, modes="cbt")

    assert links.get(link_id).modes == "cbt", "Did not set modes correctly"

    assert check_mode(sioux_falls_test, link_id) == "cbt"


def test_add_mode(sioux_falls_test, links, link_id, modes):
    for mode in [1, ["cbt"]]:
        with pytest.raises(TypeError, match="mode_id"):
            links.add_mode(link_id, mode)
    with pytest.raises(ValueError, match="single character"):
        links.add_mode(link_id, "bt")

    links.add_mode(link_id, "b")
    assert check_mode(sioux_falls_test, link_id) == "cb"

    mode = modes.get("t")
    links.add_mode(link_id, mode)
    assert check_mode(sioux_falls_test, link_id) == "cbt"


def test_drop_mode(sioux_falls_test, links, link_id, modes):
    links.update(link_id, modes="cbt")
    assert check_mode(sioux_falls_test, link_id) == "cbt"

    links.drop_mode(link_id, "t")
    assert check_mode(sioux_falls_test, link_id) == "cb"

    mode = modes.get("b")
    links.drop_mode(link_id, mode)
    assert check_mode(sioux_falls_test, link_id) == "c"


def test_data_fields(sioux_falls_test, links, link):
    link2 = links.get(randint(1, 24))
    while link2.link_id == link.link_id:
        link2 = links.get(randint(1, 24))

    assert link2.__dataclass_fields__ == link.__dataclass_fields__, "Different links have different data fields"

    fields = sorted(links.columns)

    with read_and_close(sioux_falls_test.path_to_file) as conn:
        dt = conn.execute("pragma table_info(links)").fetchall()

    data_fields = sorted([x[1] for x in dt if x[1] != "ogc_fid"])

    assert sorted(fields) == sorted(data_fields), "Link has unexpected set of fields"
