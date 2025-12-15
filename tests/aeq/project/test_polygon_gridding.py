from pathlib import Path

import geopandas as gpd
import pytest

from aequilibrae.project.network.osm.model_area_gridding import geometry_grid


def test_geometry_grid():
    pth = Path(__file__).parent / "data"

    # Small simple polygon
    polygon = gpd.read_parquet(pth / "wynnum.parquet").geometry[0]
    grid = geometry_grid(polygon, 4326)
    assert grid.geometry.area.sum() == pytest.approx(polygon.area, 0.000000001)
    assert grid.shape[0] == 1

    # Bigger polygon
    polygon = gpd.read_parquet(pth / "porto_rico.parquet").geometry[0]
    grid = geometry_grid(polygon, 4326)
    assert grid.geometry.area.sum() == pytest.approx(polygon.area, 0.000000001)
    assert grid.shape[0] == 16
