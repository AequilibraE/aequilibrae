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

data_dir = Path(__file__).parent.parent.parent.parent / "data" /"overture"/"theme=transportation"
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
            list_element = 0
            for t in ['segment', 'connector']:
                output_dir = Path(output_dir) 
                bbo =[148.71641, -20.27082, 148.71861, -20.27001]
                box1 = [148.713909, -20.272261, 148.7206475,-20.2702697]
                woolworths_parkinglot = [148.718, -20.27049, 148.71889, -20.27006]
                list_gdf = o.downloadTransportation(bbox=box1, data_source=data_dir, output_dir=output_dir)
                                        
                expected_file = output_dir / f"type={t}" / f"transportation_data_{t}.parquet"
                assert expected_file.exists()

                
                # gdf = gpd.read_parquet(expected_file)[list_element]
                gdf = list_gdf[list_element]
                assert gdf.shape[0] > 0

                assert 'link_type' in gdf.columns
                
                
                assert ['unknown', 'secondary', 'residential', 'parkingAisle', None] == list(list_gdf[0]['link_type'].unique())

                list_element+=1


        # if o.parquet:
        #     self.fail("It found links in the middle of the ocean")

    def test_format(self):
        # if not self.should_do_work():
        #     return

        # LITTLE PLACE IN THE MIDDLE OF THE Grand Canyon North Rim
        o = OVMDownloader(["car"], self.pth)
        # o.load_links()

    def should_do_work(self):
        thresh = 1.01 if os.environ.get("GITHUB_WORKFLOW", "ERROR") == "Code coverage" else 0.02
        return random() < thresh
