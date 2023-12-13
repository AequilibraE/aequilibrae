import importlib.util as iutil
import tempfile
from pathlib import Path
from tempfile import gettempdir, mkdtemp
from unittest import TestCase
import os
import geopandas as gpd
from aequilibrae.project.network.ovm_downloader import OVMDownloader
from random import random

spec = iutil.find_spec("PyQt5")
pyqt = spec is not None

test_data_dir = Path(__file__)

data_dir = Path(__file__).parent.parent.parent.parent / "data" /"overture"/"theme=transportation"/"type=segment"
print(data_dir)

class TestOVMDownloader(TestCase):
    def setUp(self) -> None:
        os.environ["PATH"] = os.path.join(gettempdir(), "temp_data") + ";" + os.environ["PATH"]
        self.pth = Path(mkdtemp(prefix="aequilibrae"))

    def test_download(self):
        # if not self.should_do_work():
        #     return
       
        o = OVMDownloader(["car"], self.pth)
        with tempfile.TemporaryDirectory() as output_dir:
            output_dir = Path(output_dir)
            o.downloadTransportation([148.72095, -20.26910, 148.72405, -20.26711], data_source=data_dir, output_dir=output_dir)

            expected_file = output_dir / "transportation_data_segment.parquet"
            assert expected_file.exists()

            
            gdf = gpd.read_parquet(expected_file)
            assert gdf.shape[0] > 0

            assert 'road' in gdf.columns
            

        # if o.parquet:
        #     self.fail("It found links in the middle of the ocean")

    def test_do_work2(self):
        # if not self.should_do_work():
        #     return

        # LITTLE PLACE IN THE MIDDLE OF THE Grand Canyon North Rim
        o = OVMDownloader(["car"], self.pth)
        o.load_links()

        if "elements" not in o.json[0]:
            self.fail

        if len(o.json[0]["elements"]) > 1000:
            self.fail("It found too many elements in the middle of the Grand Canyon")

        if len(o.json[0]["elements"]) < 10:
            self.fail("It found too few elements in the middle of the Grand Canyon")

    def should_do_work(self):
        thresh = 1.01 if os.environ.get("GITHUB_WORKFLOW", "ERROR") == "Code coverage" else 0.02
        return random() < thresh
