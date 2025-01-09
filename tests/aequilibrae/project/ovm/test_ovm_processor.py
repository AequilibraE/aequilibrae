import copy
import tempfile
from pathlib import Path
import geopandas as gpd
import shapely
from aequilibrae import global_logger

from aequilibrae import Project
from aequilibrae.project.network.ovm_builder import OVMBuilder


def test_link_geo_trimmer():
    node1 = (148.7165148, -20.273062)
    node2 = (148.7164104, -20.2730078)
    geo = shapely.LineString([(148.7165748, -20.2730668), node1, (148.7164585, -20.2730418), node2])
    link_gdf = gpd.GeoDataFrame([[1, 2, geo]], columns=["a_node", "b_node", "geometry"])
    new_geom = copy.copy(link_gdf)

    node_lu = {
        1: {"lat": node1[1], "long": node1[0], "coord": node1},
        2: {"lat": node2[1], "long": node2[0], "coord": node2},
    }

    with tempfile.TemporaryDirectory() as output_dir:
        project_dir = str(Path(output_dir) / "project")
        project = Project()
        project.new(project_dir)
        o = OVMBuilder(link_gdf, gpd.GeoDataFrame(), project_path=project_dir, project=project)

        # Iterate over the correct range
        new_geom["geometry"] = [o.trim_geometry(node_lu, row) for e, row in link_gdf.iterrows()]

        # Assuming you want to assert the length of the new geometry
        assert len(new_geom["geometry"][0].coords) == 3

        # Assuming you want to assert the correctness of the new geometry
        # If you don't need the difference operation, you can skip it

        for i in range(0, len(link_gdf)):
            if i > 0:
                assert new_geom["geometry"][i] == shapely.LineString([node1, (148.7164585, -20.2730418), node2])

        project.close()

def test_link_lanes():
    """
    segment and node infomation is currently [1] element of links when running from_ovm.py
    """

    no_info = None
    simple = [{"direction": "backward"}, {"direction": "forward"}]

    lanes_3 = [
        {
            "direction": "forward",
            "restrictions": {
                "access": [{"allowed": {"when": {"mode": ["hov"]}}}],
                "minOccupancy": {"isAtLeast": 3},
            },
        },
        {"direction": "forward"},
        {"direction": "forward"},
    ]

    highway = [
        {"direction": "backward"},
        {"direction": "backward"},
        {"direction": "backward"},
        {"direction": "backward"},
        {"direction": "forward"},
        {"direction": "forward"},
        {"direction": "forward"},
        {"direction": "forward"},
    ]

    lane_ends = [
        [{
            "at": [0, 0.67],
            "value": [{"direction": "backward"}, {"direction": "forward"}, {"direction": "forward"}],
            }
        ],
        [{"at": [0.67, 1], "value": [{"direction": "backward"}, {"direction": "forward"}]}],
    ]

    lane_begins = [
        [{"at": [0, 0.2], "value": [{"direction": "backward"}, {"direction": "forward"}]}],
        [
            {
                "at": [0.2, 1],
                "value": [{"direction": "backward"}, {"direction": "forward"}, {"direction": "forward"}],
            }
        ],
    ]

    lane_merge_twice = [
        [
            {
                "at": [0, 0.2],
                "value": [
                    {"direction": "backward"},
                    {"direction": "backward"},
                    {"direction": "forward"},
                    {"direction": "forward"},
                ],
            }
        ],
        [
            {
                "at": [0.2, 0.8],
                "value": [{"direction": "backward"}, {"direction": "forward"}, {"direction": "forward"}],
            }
        ],
        [{"at": [0.8, 1], "value": [{"direction": "backward"}, {"direction": "forward"}]}],
    ]

    equal_dis = [
        [
            {
                "at": [0, 0.5],
                "value": [{"direction": "backward"}, {"direction": "forward"}, {"direction": "forward"}],
            }
        ],
        [{"at": [0.5, 1], "value": [{"direction": "backward"}, {"direction": "forward"}]}],
    ]

    # def road(lane):
    #     road_info = str(
    #         {
    #             "class": "secondary",
    #             "surface": "paved",
    #             "restrictions": {"speedLimits": {"maxSpeed": [70, "km/h"]}},
    #             "roadNames": {"common": [{"language": "local", "value": "Shute Harbour Road"}]},
    #             "lanes": lane,
    #         }
    #     )
    #     return road_info

    a_node = {"ovm_id": "8f9d0e128cd9709-167FF64A37F1BFFB", "geometry": shapely.Point(148.72460, -20.27472)}
    b_node = {"ovm_id": "8f9d0e128cd98d6-15FFF68E65613FDF", "geometry": shapely.Point(148.72471, -20.27492)}
    node_df = gpd.GeoDataFrame(data=[a_node, b_node])

    def segment(direction):
        segment = {
            "ovm_id": "8b9d0e128cd9fff-163FF6797FC40661",
            "connectors": [["8f9d0e128cd9709-167FF64A37F1BFFB", "8f9d0e128cd98d6-15FFF68E65613FDF"]],
            "direction": direction,
            "link_type": "secondary",
            "name": '[{"value": "Shute Harbour Road"}]',
            "speed": '{"maxSpeed":[70,"km/h"]}',
            "geometry": shapely.LineString(
                [
                    (148.7245987, -20.2747175),
                    (148.7246504, -20.2747531),
                    (148.724688, -20.274802),
                    (148.7247077, -20.2748593),
                    (148.7247078, -20.2749195),
                ]
            ),
        }
        return segment
    
    # def link_gdf(lane_info):
    #     return gpd.GeoDataFrame(segment(lane_info, road(lane_info)))
    
    def set_up_ovmbuilder(lane_info, output_dir, project):
        print(lane_info)
        print()
        print(segment(lane_info))
        links = gpd.GeoDataFrame(segment(lane_info))
        print(links)
        o = OVMBuilder(links, node_df, project_path=output_dir / "project", project=project)
        o.create_node_ids(node_df)
        o._worksetup()
        link_gdf = o.formatting(links, node_df, output_dir)
        print(link_gdf)
        return link_gdf


    with tempfile.TemporaryDirectory() as output_dir:
        output_dir = Path(output_dir)
        project = Project()
        project.new(output_dir / "project")

        gdf_no_info = set_up_ovmbuilder(no_info, output_dir, project)

        assert gdf_no_info["direction"][0] == 0
        assert gdf_no_info["lanes_ab"][0] == 1
        assert gdf_no_info["lanes_ba"][0] == 1

        gdf_simple = set_up_ovmbuilder(simple, output_dir, project)

        assert len(simple) == 2
        assert gdf_simple["direction"][0] == 0
        assert gdf_simple["lanes_ab"][0] == 1
        assert gdf_simple["lanes_ab"][0] == 1

        gdf_lanes_3 = set_up_ovmbuilder(lanes_3, output_dir, project)

        assert len(lanes_3) == 3
        assert gdf_lanes_3["direction"][0] == 1
        assert gdf_lanes_3["lanes_ab"][0] == 3
        assert gdf_lanes_3["lanes_ba"][0] == None

        gdf_highway = set_up_ovmbuilder(highway, output_dir, project)

        assert len(highway) == 8
        assert gdf_highway["direction"][0] == 0
        assert gdf_highway["lanes_ab"][0] == 4
        assert gdf_highway["lanes_ba"][0] == 4

        gdf_lane_ends = set_up_ovmbuilder(lane_ends, output_dir, project)

        assert len(lane_ends) == 2
        assert len(lane_ends[0][0]["value"]) == 3
        assert len(lane_ends[1][0]["value"]) == 2
        assert gdf_lane_ends["direction"][0] == 0
        assert gdf_lane_ends["lanes_ab"][0] == 2
        assert gdf_lane_ends["lanes_ba"][0] == 1

        gdf_lane_begins = set_up_ovmbuilder(lane_begins, output_dir, project)

        assert len(lane_begins) == 2
        assert len(lane_begins[0][0]["value"]) == 2
        assert len(lane_begins[1][0]["value"]) == 3
        assert gdf_lane_begins["direction"][0] == 0
        assert gdf_lane_begins["lanes_ab"][0] == 2
        assert gdf_lane_begins["lanes_ba"][0] == 1

        gdf_lane_merge_twice = set_up_ovmbuilder(lane_merge_twice, output_dir, project)

        assert len(lane_merge_twice) == 3
        assert len(lane_merge_twice[0][0]["value"]) == 4
        assert len(lane_merge_twice[1][0]["value"]) == 3
        assert len(lane_merge_twice[2][0]["value"]) == 2
        assert gdf_lane_merge_twice["direction"][0] == 0
        assert gdf_lane_merge_twice["lanes_ab"][0] == 2
        assert gdf_lane_merge_twice["lanes_ba"][0] == 1

        gdf_equal_dis = set_up_ovmbuilder(equal_dis, output_dir, project)

        assert len(equal_dis) == 2
        assert len(equal_dis[0][0]["value"]) == 3
        assert len(equal_dis[1][0]["value"]) == 2
        assert gdf_equal_dis["direction"][0] == 0
        assert gdf_equal_dis["lanes_ab"][0] == 2
        assert gdf_equal_dis["lanes_ba"][0] == 1
