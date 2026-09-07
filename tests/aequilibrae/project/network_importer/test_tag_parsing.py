"""Lane-splitting and maxspeed-parsing regression tests for OSM tag parsing."""

import pytest

from aequilibrae.project.network.importer.sources.osm.tags_to_ir import (
    directional_lanes,
    parse_speed,
)


@pytest.mark.parametrize(
    "tags, expected",
    [
        # Bidirectional with only a total lane count must be split, never doubled.
        ({"lanes": "2"}, (1, 1)),
        ({"lanes": "4"}, (2, 2)),
        # Odd totals put the remainder on the AB direction.
        ({"lanes": "3"}, (2, 1)),
        # A single shared lane is reported as 1 in each direction (not 0).
        ({"lanes": "1"}, (1, 1)),
        # Explicit directional tags win.
        ({"lanes:forward": "2", "lanes:backward": "1"}, (2, 1)),
        # One explicit side + total: derive the other from the total.
        ({"lanes": "5", "lanes:forward": "3"}, (3, 2)),
        ({"lanes": "5", "lanes:backward": "2"}, (3, 2)),
        # No data.
        ({}, (None, None)),
    ],
)
def test_bidirectional_lane_splitting(tags, expected):
    assert directional_lanes(tags) == expected


def test_bidirectional_lanes_never_doubled():
    ab, ba = directional_lanes({"lanes": "2"})
    assert (ab, ba) != (2, 2)


def test_oneway_lanes_keep_total_on_single_direction():
    assert directional_lanes({"oneway": "yes", "lanes": "3"}) == (3, None)
    assert directional_lanes({"oneway": "-1", "lanes": "3"}) == (None, 3)


@pytest.mark.parametrize(
    "value, expected",
    [
        ("50", 50.0),
        ("30.5", 30.5),
        ("30 mph", 30 * 1.609344),
    ],
)
def test_parse_speed_valid(value, expected):
    assert parse_speed(value) == pytest.approx(expected)


@pytest.mark.parametrize(
    "value",
    ["50; 40", "50 (variable)", "walk", "none", "signals", "RO:urban", "", None],
)
def test_parse_speed_rejects_messy_values(value):
    assert parse_speed(value) is None
