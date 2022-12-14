import numpy as np


def compute_line_bearing(point_a: tuple, point_b: tuple) -> float:
    # THIS FUNCTIONS APPROPRIATELY FOR PROJECTED COORDINATES ONLY
    # FOR NON-PROJECTED COORDINATE SYSTEMS, SEE https://gist.github.com/jeromer/2005586

    delta_lat = abs(point_a[1] - point_b[1])
    delta_long = abs(point_a[0] - point_b[0])
    return np.arctan2(delta_lat, delta_long) * 180 / np.pi
