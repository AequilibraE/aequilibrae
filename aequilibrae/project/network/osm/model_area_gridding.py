# Inspired by https://www.matecdev.com/posts/shapely-polygon-gridding.html
from math import ceil

import geopandas as gpd
import numpy as np
from shapely.geometry import Polygon


def geometry_grid(model_area, srid) -> gpd.GeoDataFrame:
    minx, miny, maxx, maxy = model_area.bounds
    # Some rough heuristic to get the number of points per sub-polygon in the 2 digits range
    subd = ceil((len(model_area.exterior.coords) / 32) ** 0.5)
    dx = (maxx - minx) / subd
    dy = (maxy - miny) / subd
    elements = []
    x1 = minx
    for i in range(subd):
        j1 = miny
        for j in range(subd):
            elements.append(Polygon([[x1, j1], [x1, j1 + dy], [x1 + dx, j1 + dy], [x1 + dx, j1]]))
            j1 += dy
        x1 += dx

    gdf = gpd.GeoDataFrame({"id": np.arange(len(elements))}, geometry=elements, crs=srid)

    return gdf.clip(model_area)
