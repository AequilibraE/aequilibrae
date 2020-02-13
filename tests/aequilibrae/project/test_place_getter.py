from unittest import TestCase
from aequilibrae.project.network.osm_utils.place_getter import placegetter
from random import random


class Test(TestCase):
    def test_placegetter(self):

        if random() < 0.05:
            place, report = placegetter("China")
            place = [round(x, 1) for x in place]
            if place != [73.5, 8.8, 134.8, 53.6]:
                self.fail("Returned the wrong boundingbox for china")

            place, report = placegetter("Just a random place with no bear in reality")
            if place is not None:
                self.fail("Returned a bounding box for a place that does not exist")
        else:
            print('Skipped check to not load OSM servers')
