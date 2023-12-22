from pathlib import Path
import geopandas as gpd
from shapely.ops import substring
import shapely


data_dir = Path(__file__).parent.parent.parent.parent / "data" / "overture" / "theme=transportation"


def test_link_geo_trimmer():
    node1 = (148.7165148, -20.273062)
    node2 = (148.7164104, -20.2730078)
    geo = shapely.LineString([(148.7165748, -20.2730668), node1, (148.7164585, -20.2730418), node2])
    link_gdf = gpd.GeoDataFrame([[1, 2, geo]], columns=["a_node", "b_node", "geometry"])
    # node_gdf = gpd.GeoDataFrame({"node_id": [1, 2]}, geometry=[shapely.Point(node1), shapely.Point(node2)])

    def trim_geometry(node_lu, row):
        lat_long_a = node_lu[row["a_node"]]
        lat_long_b = node_lu[row["b_node"]]

        for i, coord in enumerate(row.geometry.coords):
            if lat_long_a == coord:
                new_list = row.geometry.coords[i:]
                if lat_long_b == coord:
                    new_list[:i]
        return shapely.LineString(new_list)
      
    node_lu = {1: node1, 2: node2}  # node_gdf[...].set_index(...).to_dict()
    new_geom = trim_geometry(node_lu, link_gdf.iloc[0, :])

    assert len(new_geom.coords) == 3

    assert new_geom == shapely.LineString([node1, (148.7164585, -20.2730418), node2])
