"""Convenience functions for working with geospatial data."""

import numpy as np
from pyproj import CRS


def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    """
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2

    c = 2.0 * np.arcsin(np.sqrt(a))
    distance_m = 6367000.0 * c
    return distance_m


def metre_crs_for_gdf(gdf):
    length_unit = gdf.crs.axis_info[0].unit_name
    if length_unit and length_unit.lower() == "metre":
        return gdf.crs

    xmin, ymin, xmax, ymax = gdf.total_bounds
    lon = (xmin + xmax) / 2
    lat = (ymin + ymax) / 2

    # Gets the UTM zone from longitude and latitude
    zone = int((lon + 180) / 6) + 1
    south = lat < 0

    return CRS.from_dict({"proj": "utm", "zone": zone, "south": south, "datum": "WGS84", "units": "m"})
