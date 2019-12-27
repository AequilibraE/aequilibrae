import math
from warnings import warn
from aequilibrae.project.network import OSMDownloader
from aequilibrae.project.network.osm_builder import OSMBuilder
from aequilibrae.project.network.osm_utils.place_getter import placegetter
from aequilibrae.project.network.osm_utils.osm_params import max_query_area_size
from aequilibrae.project.network.haversine import haversine
from ...utils import WorkerThread


class Network(WorkerThread):
    def __init__(self, project):
        self.conn = project.conn

    def create_from_osm(
        self,
        west: float = None,
        south: float = None,
        east: float = None,
        north: float = None,
        place_name: str = None,
        modes=("car, transit, bike, walk"),
    ):

        if isinstance(modes, (tuple, list)):
            modes = list(modes)
        elif isinstance(modes, str):
            modes = [modes]
        else:
            raise ValueError("'modes' needs to be string or list/tuple of string")

        if place_name is None:
            if min(east, west) < -180 or max(east, west) > 180 or min(north, south) < -90 or max(north, south) > 90:
                raise ValueError("Coordinates out of bounds")
            bbox = [west, south, east, north]
        else:
            bbox, report = placegetter(place_name)
            west, south, east, north = bbox
            if bbox is None:
                warn('We could not find a reference for place name "{}"'.format(place_name))
                return
            for i in report:
                if "PLACE FOUND" in i:
                    print(i)

        # Need to compute the size of the bounding box to not exceed it too much
        height = haversine((east + west) / 2, south, (east + west) / 2, north)
        width = haversine(east, (north + south) / 2, west, (north + south) / 2)
        area = height * width

        if area < max_query_area_size:
            polygons = [bbox]
        else:
            polygons = []
            parts = math.ceil(area / max_query_area_size)
            horizontal = math.ceil(math.sqrt(parts))
            vertical = math.ceil(parts / horizontal)
            dx = east - west
            dy = north - south
            for i in range(horizontal):
                xmin = west + i * dx
                xmax = west + (i + 1) * dx
                for j in range(vertical):
                    ymin = south + j * dy
                    ymax = south + (j + 1) * dy
                    box = [xmin, ymin, xmax, ymax]
                    polygons.append(box)

        self.downloader = OSMDownloader(polygons, modes)
        self.downloader.doWork()

        self.builder = OSMBuilder(self.downloader.json, self.conn)
        self.builder.doWork()

    def count_links(self):
        print("quick query on number of links")

    def count_nodes(self):
        print("quick query on number of nodes")

    def many_more_queries_like_that(self):
        print("With all the work that goes with it")
