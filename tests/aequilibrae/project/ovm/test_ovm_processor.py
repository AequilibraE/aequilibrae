import copy
from pathlib import Path
from tempfile import gettempdir, mkdtemp
import geopandas as gpd
import shapely
from aequilibrae.project.network.ovm_downloader import OVMDownloader
from unittest import TestCase
import os

class TestOVMProcessor(TestCase):
    def setUp(self) -> None:
        os.environ["PATH"] = os.path.join(gettempdir(), "temp_data") + ";" + os.environ["PATH"]
        self.pth = Path(mkdtemp(prefix="aequilibrae"))

    def test_link_geo_trimmer(self):
        node1 = (148.7165148, -20.273062)
        node2 = (148.7164104, -20.2730078)
        geo = shapely.LineString([(148.7165748, -20.2730668), node1, (148.7164585, -20.2730418), node2])
        link_gdf = gpd.GeoDataFrame([[1, 2, geo]], columns=["a_node", "b_node", "geometry"])
        new_geom = copy.copy(link_gdf)

        node_lu = {1: {'lat': node1[1], 'long': node1[0], 'coord': node1},
                   2: {'lat': node2[1], 'long': node2[0], 'coord': node2}}
    
        o = OVMDownloader(["car"], self.pth)
        
        # Iterate over the correct range
        new_geom['geometry'] = [o.trim_geometry(node_lu, row) for e, row in link_gdf.iterrows()]


        # Assuming you want to assert the length of the new geometry
        assert len(new_geom['geometry'][0].coords) == 3

        # Assuming you want to assert the correctness of the new geometry
        # If you don't need the difference operation, you can skip it
        
        for i in range(0, len(link_gdf)):
            if i > 0:  
                assert new_geom["geometry"][i] == shapely.LineString([node1, (148.7164585, -20.2730418), node2])
