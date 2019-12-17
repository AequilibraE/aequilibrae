import importlib.util as iutil
from unittest import TestCase
from time import sleep
from aequilibrae.project.network.osm_downloader import OSMDownloader

spec = iutil.find_spec("PyQt5")
pyqt = spec is not None


class TestOSMDownloader(TestCase):
    def test_do_work(self):
        self.o = OSMDownloader([[0.0, 0.0, 0.1, 0.1]], ["car"])
        self.o.doWork()
        if self.o.json:
            self.fail("It found links in the middle of the ocean")

    def test_do_work2(self):
        # LITTLE PLACE IN THE MIDDLE OF THE Grand Canyon north rim
        self.o = OSMDownloader([[-112.185, 36.59, -112.179, 36.60]], ["car"])
        self.o.doWork()
        if len(self.o.json[0]["elements"]) > 1000:
            self.fail("It found too many elements in the middle of the Grand Canyon")
