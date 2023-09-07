from unittest import TestCase
from aequilibrae.project.network.osm_utils.place_getter import placegetter
from random import random
import os


class Test(TestCase):
    def test_placegetter(self):
        thresh = 0.05
        if os.environ.get("GITHUB_WORKFLOW", "ERROR") == "Code coverage":
            thresh = 1.01

        if random() < thresh:
            place, report = placegetter("Vatican City")
            if place is None:
                self.skipTest("Skipping... either Vatican City doesn't exist anymore or there was a network failure")
            place = [round(x, 1) for x in place]
            if place != [12.4, 41.9, 12.5, 41.9]:
                self.fail("Returned the wrong boundingbox for Vatican City")

            place, report = placegetter("Just a random place with no bear in reality")
            if place is not None:
                self.fail("Returned a bounding box for a place that does not exist")
        else:
            self.skipTest("Skipped check to not load OSM servers")
