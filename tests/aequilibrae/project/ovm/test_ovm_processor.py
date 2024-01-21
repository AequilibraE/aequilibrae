import copy
import json
import pandas as pd
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
        
    def test_link_lanes(self):
        """
        segment and node infomation is currently [1] element of links when running from_ovm.py
        """
        o = OVMDownloader(["car"], self.pth)

        no_info = None
        simple = [{"direction": "backward"}, 
                  {"direction": "forward"}]
        
        lanes_3 = [{'direction': 'forward', 'restrictions': {'access': [{'allowed': {'when': {'mode': ['hov']}}}], 
                                                            'minOccupancy': {'isAtLeast': 3}}}, 
                  {'direction': 'forward'}, 
                  {'direction': 'forward'}]
        
        def road(lane):
            road_info = str({"class":"secondary",
                "surface":"paved",
                "restrictions":{"speedLimits":{"maxSpeed":[70,"km/h"]}},
                "roadNames":{"common":[{"language":"local","value":"Shute Harbour Road"}]},
                "lanes": lane})
            return road_info
        
        a_node = {'ovm_id': '8f9d0e128cd9709-167FF64A37F1BFFB', 'geometry': shapely.Point(148.72460, -20.27472)}
        b_node = {'ovm_id': '8f9d0e128cd98d6-15FFF68E65613FDF', 'geometry': shapely.Point(148.72471, -20.27492)}
        node_df = gpd.GeoDataFrame(data=[a_node,b_node])

        def segment(direction, road):
            segment = {'ovm_id': '8b9d0e128cd9fff-163FF6797FC40661', 'connectors': ['8f9d0e128cd9709-167FF64A37F1BFFB', '8f9d0e128cd98d6-15FFF68E65613FDF'], 'direction': direction, 
                   'link_type': 'secondary', 'name': 'Shute Harbour Road', 'speed':  '{"maxSpeed":[70,"km/h"]}', 'road': road, 
                   'geometry': shapely.LineString([(148.7245987, -20.2747175), (148.7246504, -20.2747531), (148.724688, -20.274802), (148.7247077, -20.2748593), (148.7247078, -20.2749195)])}
            return segment

        o.create_node_ids(node_df)

        for lane_type in [no_info, simple, lanes_3]:
            assert type(road(lane_type)) == str
            assert type(o.split_connectors(segment(lane_type, road(lane_type)))) == gpd.GeoDataFrame

        gdf_no_info = o.split_connectors(segment(no_info, road(no_info)))

        assert gdf_no_info['direction'][0] == 0
        assert gdf_no_info['lanes_ab'][0] == 1
        assert gdf_no_info['lanes_ba'][0] == 1

        gdf_simple = o.split_connectors(segment(simple, road(simple)))

        assert len(simple) == 2
        assert gdf_simple['direction'][0] == 0
        assert gdf_simple['lanes_ab'][0] == 1
        assert gdf_simple['lanes_ab'][0] == 1

        gdf_lanes_3 = o.split_connectors(segment(lanes_3, road(lanes_3)))
        
        assert len(lanes_3) == 3
        assert gdf_lanes_3['direction'][0] == 1
        assert gdf_lanes_3['lanes_ab'][0] == 3
        assert gdf_lanes_3['lanes_ba'][0] == None

        
