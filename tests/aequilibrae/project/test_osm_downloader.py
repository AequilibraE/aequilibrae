import importlib.util as iutil
from unittest import TestCase
from time import sleep
from aequilibrae.project.network.osm_downloader import OSMDownloader
from random import random

spec = iutil.find_spec("PyQt5")
pyqt = spec is not None


class TestOSMDownloader(TestCase):
    def test_do_work(self):
        if random() < 0.05:
            self.o = OSMDownloader([[0.0, 0.0, 0.1, 0.1]], ["car"])
            self.o.doWork()
            if self.o.json:
                self.fail("It found links in the middle of the ocean")
        else:
            print('Skipped check to not load OSM servers')

    def test_do_work2(self):

        if random() < 0.05:
            # LITTLE PLACE IN THE MIDDLE OF THE Grand Canyon North Rim
            self.o = OSMDownloader([[-112.185, 36.59, -112.179, 36.60]], ["car"])
            self.o.doWork()
            if len(self.o.json[0]["elements"]) > 1000:
                self.fail("It found too many elements in the middle of the Grand Canyon")

            if len(self.o.json[0]["elements"]) < 10:
                self.fail("It found too few elements in the middle of the Grand Canyon")
        else:
            print('Skipped check to not load OSM servers')
