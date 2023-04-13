import numpy as np


def compute_line_bearing(point_a: tuple, point_b: tuple) -> float:
    """
    Computes line bearing for projected (cartesian) coordinates.
    For non-projected coordinates, see: https://gist.github.com/jeromer/2005586

    :Arguments:
        **point_a** (:obj:`tuple`): first point coordinates (lat, lon)
        **point_b** (:obj:`tuple`): second point coordinates (lat, lon)
    """

    delta_lat = abs(point_a[1] - point_b[1])
    delta_long = abs(point_a[0] - point_b[0])
    return np.arctan2(delta_lat, delta_long) * 180 / np.pi
