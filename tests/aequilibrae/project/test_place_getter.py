import os
from random import random

import pytest

from aequilibrae.project.network.osm.place_getter import placegetter


def test_placegetter():
    thresh = 0.05
    if os.environ.get("GITHUB_WORKFLOW", "ERROR") == "Code coverage":
        thresh = 1.01

    if random() < thresh:
        place, report = placegetter("Vatican City")
        if place is None:
            pytest.skip("Skipping... either Vatican City doesn't exist anymore or there was a network failure")
        place = [round(x, 1) for x in place]
        assert place == [12.4, 41.9, 12.5, 41.9], "Returned the wrong boundingbox for Vatican City"

        place, report = placegetter("Just a random place with no bear in reality")
        assert place is None, "Returned a bounding box for a place that does not exist"
    else:
        pytest.skip("Skipped check to not load OSM servers")
