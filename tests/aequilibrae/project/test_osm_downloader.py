from unittest import TestCase
from aequilibrae.project.network.osm_downloader import OSMDownloader


class TestOSMDownloader(TestCase):
    def test_do_work(self):
        o = OSMDownloader([[0.0, 0.0, 0.1, 0.1]], ["car"])
        o.doWork()
        if o.json[0]["elements"]:
            self.fail("It found links in the middle of the ocean")

        # LITTLE PLACE IN THE MIDDLE OF THE Grand Canyon north rim
        o = OSMDownloader([[-112.185, 36.59, -112.179, 36.60]], ["car"])
        o.doWork()

        if len(o.json[0]["elements"]) > 1000:
            self.fail("It found too many elements in the middle of the Grand Canyon")
