import importlib.util as iutil
from tempfile import gettempdir
from unittest import TestCase
import os
from aequilibrae.project.network.osm_downloader import OSMDownloader
from random import random

spec = iutil.find_spec("PyQt5")
pyqt = spec is not None


class TestOSMDownloader(TestCase):
    def setUp(self) -> None:
        os.environ["PATH"] = os.path.join(gettempdir(), "temp_data") + ";" + os.environ["PATH"]

    def test_do_work(self):
        if not self.should_do_work():
            return
        o = OSMDownloader([[0.0, 0.0, 0.1, 0.1]], ["car"])
        o.doWork()
        if o.json:
            self.fail("It found links in the middle of the ocean")

    def test_do_work2(self):
        if not self.should_do_work():
            return

        # LITTLE PLACE IN THE MIDDLE OF THE Grand Canyon North Rim
        o = OSMDownloader([[-112.185, 36.59, -112.179, 36.60]], ["car"])
        o.doWork()

        if "elements" not in o.json[0]:
            return

        if len(o.json[0]["elements"]) > 1000:
            self.fail("It found too many elements in the middle of the Grand Canyon")

        if len(o.json[0]["elements"]) < 10:
            self.fail("It found too few elements in the middle of the Grand Canyon")

    def should_do_work(self):
        thresh = 1.01 if os.environ.get("GITHUB_WORKFLOW", "ERROR") == "Code coverage" else 0.02
        return random() < thresh
