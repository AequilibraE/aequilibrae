import geopandas as gpd
import pytest
from geopandas.testing import assert_geodataframe_equal
from shapely.geometry import Point

from aequilibrae.project.project_table import SpatialProjectTable
from aequilibrae.utils.db_utils import ConnectionClosure


class Places(SpatialProjectTable):
    name = "places"
    key = "place_id"
    record_name = "PlaceRecord"
    has_numeric_key = True


@pytest.fixture
def places():
    closure = ConnectionClosure(":memory:")
    connection = closure.db_connection._connection
    connection.execute("SELECT InitSpatialMetadata(1)")
    connection.execute("CREATE TABLE places (place_id INTEGER PRIMARY KEY, name TEXT)")
    connection.execute("SELECT AddGeometryColumn('places', 'geometry', 4326, 'POINT', 'XY')")
    connection.commit()

    yield Places(closure.db_connection)
    closure.close()


def test_data_returns_geodataframe_with_active_geometry(places):
    places.insert(place_id=1, name="first", geometry=Point(1, 2))

    frame = places.data

    assert isinstance(frame, gpd.GeoDataFrame)
    assert frame.geometry.name == "geometry"
    assert frame.crs.to_epsg() == places.srid
    assert frame.geometry.to_list() == [Point(1, 2)]


def test_bulk_insert_accepts_geodataframe_and_preserves_geometry(places):
    additions = gpd.GeoDataFrame(
        {"name": ["first", "second"]},
        geometry=[Point(1, 2), Point(3, 4)],
        crs=places.srid,
    )
    original = additions.copy()

    assert places.insert_from(additions) == [1, 2]

    assert_geodataframe_equal(additions, original)
    result = places.data.sort_values("place_id").reset_index(drop=True)
    assert isinstance(result, gpd.GeoDataFrame)
    expected = additions.assign(place_id=[1, 2])[["place_id", "name", "geometry"]]
    assert_geodataframe_equal(result, expected)


def test_bulk_update_accepts_geodataframe_and_preserves_geometry(places):
    places.insert(place_id=1, name="first", geometry=Point(1, 2))
    places.insert(place_id=2, name="second", geometry=Point(3, 4))
    updates = gpd.GeoDataFrame(
        {"place_id": [1, 2]},
        geometry=[Point(5, 6), Point(7, 8)],
        crs=places.srid,
    )
    original = updates.copy()

    assert places.update_from(updates) == 2

    assert_geodataframe_equal(updates, original)
    result = places.data.sort_values("place_id").reset_index(drop=True)
    assert isinstance(result, gpd.GeoDataFrame)
    assert result.geometry.name == "geometry"
    assert result.crs.to_epsg() == places.srid
    assert result.geometry.to_list() == updates.geometry.to_list()
