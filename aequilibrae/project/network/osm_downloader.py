""""
Large portions of this code were adopted from OSMNx, by Geoff Boeing.

Although attempts to use OSMNx were made (including refactoring its
entire code base as a contribution to that package), it became clear
that its integration with libraries not available with QGIS' Python
distribution was too tight, and was therefore not practical to
detach them in order to use OSMNx as a dependency or submodule

For the original work, please see https://github.com/gboeing/osmnx
"""
import time
import math
import re
from typing import List
import requests
from .osm_utils.osm_params import overpass_endpoint, timeout, http_headers
from ...utils import WorkerThread


class OSMDownloader(WorkerThread):
    def __init__(self, poly, modes: List[str]):
        """
        Download OSM ways and nodes within some bounding box from the Overpass API.

        Parameters
        ----------
        north : float
            northern latitude of bounding box
        south : float
            southern latitude of bounding box
        east : float
            eastern longitude of bounding box
        west : float
            western longitude of bounding box
        network_type : string
            {'walk', 'bike', 'drive', 'drive_service', 'all', 'all_private'} what
            type of street network to get
        timeout : int
            the timeout interval for requests and to pass to API
        memory : int
            server memory allocation size for the query, in bytes. If none, server
            will use its default allocation size
        max_query_area_size : float
            max area for any part of the geometry, in the units the geometry is in:
            any polygon bigger will get divided up for multiple queries to API
            (default is 50,000 * 50,000 units [ie, 50km x 50km in area, if units are
            meters])
        infrastructure : string
            download infrastructure of given type. default is streets, ie,
            'way["highway"]') but other infrastructures may be selected like power
            grids, ie, 'way["power"~"line"]'
        custom_filter : string
            a custom network filter to be used instead of the network_type presets

        Returns
        -------
        response_jsons : list
        """
        self.poly = poly
        self.filter = self.get_osm_filter(modes)
        self.report = []
        self.json = {}

    def doWork(self):

        infrastructure = 'way["highway"]'

        # define the query to send the API
        # specifying way["highway"] means that all ways returned must have a highway
        # key. the {filters} then remove ways by key/value. the '>' makes it recurse
        # so we get ways and way nodes. maxsize is in bytes.
        # turn bbox into a polygon and project to local UTM

        # loop through each polygon rectangle in the geometry (there will only
        # be one if original bbox didn't exceed max area size)

        # represent bbox as south,west,north,east and round lat-longs to 6
        # decimal places (ie, ~100 mm) so URL strings aren't different
        # due to float rounding issues (for consistent caching)
        west = self.poly.xMinimum()
        east = self.poly.xMaximum()
        south = self.poly.yMinimum()
        north = self.poly.yMaximum()
        query_template = "[out:json][timeout:{timeout}];({infrastructure}{filters}({south:.6f},{west:.6f},{north:.6f},{east:.6f});>;);out;"
        query_str = query_template.format(
            north=north,
            south=south,
            east=east,
            west=west,
            infrastructure=infrastructure,
            filters=self.filter,
            timeout=timeout,
        )
        self.json = self.overpass_request(data={"data": query_str}, timeout=timeout)

    def overpass_request(self, data, pause_duration=None, timeout=180, error_pause_duration=None):
        """
        Send a request to the Overpass API via HTTP POST and return the JSON
        response.

        Parameters
        ----------
        data : dict or OrderedDict
            key-value pairs of parameters to post to the API
        pause_duration : int
            how long to pause in seconds before requests, if None, will query API
            status endpoint to find when next slot is available
        timeout : int
            the timeout interval for the requests library
        error_pause_duration : int
            how long to pause in seconds before re-trying requests if error

        Returns
        -------
        dict
        """

        # define the Overpass API URL, then construct a GET-style URL as a string to
        url = overpass_endpoint.rstrip("/") + "/interpreter"
        if pause_duration is None:
            time.sleep(10)
        start_time = time.time()
        self.report.append('Posting to {} with timeout={}, "{}"'.format(url, timeout, data))
        response = requests.post(url, data=data, timeout=timeout, headers=http_headers)

        # get the response size and the domain, log result
        size_kb = len(response.content) / 1000.0
        domain = re.findall(r"(?s)//(.*?)/", url)[0]
        self.report.append(
            "Downloaded {:,.1f}KB from {} in {:,.2f} seconds".format(size_kb, domain, time.time() - start_time)
        )

        try:
            response_json = response.json()
            if "remark" in response_json:
                self.report.append('Server remark: "{}"'.format(response_json["remark"]))
        except Exception:
            # 429 is 'too many requests' and 504 is 'gateway timeout' from server
            # overload - handle these errors by recursively calling
            # overpass_request until we get a valid response
            if response.status_code in [429, 504]:
                # pause for error_pause_duration seconds before re-trying request
                if error_pause_duration is None:
                    error_pause_duration = 10
                self.report.append(
                    "Server at {} returned status code {} and no JSON data. Re-trying request in {:.2f} seconds.".format(
                        domain, response.status_code, error_pause_duration
                    )
                )
                time.sleep(error_pause_duration)
                response_json = self.overpass_request(data=data, pause_duration=pause_duration, timeout=timeout)

            # else, this was an unhandled status_code, throw an exception
            else:
                self.report.append(
                    "Server at {} returned status code {} and no JSON data".format(domain, response.status_code)
                )
                raise Exception(
                    "Server returned no JSON data.\n{} {}\n{}".format(response, response.reason, response.text)
                )

        return response_json

    def get_osm_filter(self, modes: list) -> str:
        """
        loosely adapted from http://www.github.com/gboeing/osmnx
        """

        car_only = [
            "motor",
            "motorway",
            "trunk",
            "primary",
            "secondary",
            "tertiary",
            "unclassified",
            "residential",
            "motorway_link",
            "trunk_link",
            "primary_link",
            "secondary_link",
            "tertiary_link",
            "living_street",
            "service",
            "pedestrian",
            "track",
            "bus_guideway",
            "escape",
            "road",
        ]

        transit = car_only + ["bus_guideway"]

        walk = [
            "cycleway",
            "footway",
            "steps",
            "corridor",
            "pedestrian",
            "elevator",
            "escalator",
            "path",
            "track",
            "trail",
            "bridleway",
        ]

        bike = ["cycleway", "corridor", "pedestrian", "path", "track", "trail"]

        all_tags = [
            "secondary_link",
            "escalator",
            "trail",
            "cycleway",
            "path",
            "trunk_link",
            "secondary",
            "escape",
            "track",
            "road",
            "motorway_link",
            "primary",
            "corridor",
            "residential",
            "footway",
            "motorway",
            "primary_link",
            "unclassified",
            "bus_guideway",
            "tertiary_link",
            "living_street",
            "pedestrian",
            "bridleway",
            "elevator",
            "motor",
            "trunk",
            "tertiary",
            "service",
            "steps",
            "proposed",
            "raceway",
            "construction",
            "abandoned",
            "platform",
        ]

        # Default to remove
        service = '["service"!~"parking|parking_aisle|driveway|private|emergency_access"]'

        access = '["access"!~"private"]'

        tags_to_keep = []
        if "car" in modes:
            tags_to_keep += car_only
        if "transit" in modes:
            tags_to_keep += transit
        if "bike" in modes:
            tags_to_keep += bike
        if "walk" in modes:
            tags_to_keep += walk

        tags_to_keep = list(set(tags_to_keep))
        filtered = [x for x in all_tags if x not in tags_to_keep]

        filtered = "|".join(filtered)

        filter = ('["area"!~"yes"]["highway"!~"{}"]{}{}').format(filtered, service, access)

        return filter
