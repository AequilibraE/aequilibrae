from pathlib import Path
import geopandas as gpd
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
        return shapely.LineString([x for x in row.geometry.coords if x == lat_long_a or x == lat_long_b])

    node_lu = {1: node1, 2: node2}  # node_gdf[...].set_index(...).to_dict()
    new_geom = trim_geometry(node_lu, link_gdf.iloc[0, :])
    assert len(new_geom.coords) == 3
