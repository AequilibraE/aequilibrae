# Inspired by https://www.matecdev.com/posts/shapely-polygon-gridding.html
import numpy as np
from shapely.geometry import Polygon
import geopandas as gpd


def geometry_grid(model_area, srid) -> gpd.GeoDataFrame:
    minx, miny, maxx, maxy = model_area.bounds
    subd = min(0.01, abs(maxy - miny) / 3, abs(maxx - minx) / 3)
    space_x = int((maxx - minx) / subd)
    space_y = int((maxy - miny) / subd)
    combx, comby = np.linspace(minx, maxx, space_x), np.linspace(miny, maxy, space_y)
    elements = []
    for i in range(len(combx) - 1):
        for j in range(len(comby) - 1):
            elements.append(
                Polygon(
                    [
                        [combx[i], comby[j]],
                        [combx[i], comby[j + 1]],
                        [combx[i + 1], comby[j + 1]],
                        [combx[i + 1], comby[j]],
                    ]
                )
            )

    gdf = gpd.GeoDataFrame({"id": np.arange(len(elements))}, geometry=elements, crs=srid)

    return gdf.clip(model_area)
