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
import re
import requests
from PyQt5.QtCore import pyqtSignal, QObject
from .osm_utils.osm_params import overpass_endpoint, timeout, http_headers
from aequilibrae.parameters import Parameters


class OSMDownloader(QObject):
    downloading = pyqtSignal(object)

    def __init__(self, polygons, modes):
        QObject.__init__(self, None)
        self.polygons = polygons
        self.filter = self.get_osm_filter(modes)
        self.report = []
        self.json = []

    def doWork(self):
        infrastructure = 'way["highway"]'
        query_template = (
            "[out:json][timeout:{timeout}];({infrastructure}{filters}({south:.6f},{west:.6f},"
            "{north:.6f},{east:.6f});>;);out;"
        )
        self.downloading.emit(["text", "Downloading polygon {} of {}".format(1, len(self.polygons))])
        self.downloading.emit(["maxValue", len(self.polygons)])
        self.downloading.emit(["Value", 0])

        for counter, poly in enumerate(self.polygons):
            self.downloading.emit(["Value", counter])
            west, south, east, north = poly
            query_str = query_template.format(
                north=north,
                south=south,
                east=east,
                west=west,
                infrastructure=infrastructure,
                filters=self.filter,
                timeout=timeout,
            )
            json = self.overpass_request(data={"data": query_str}, timeout=timeout)
            if json["elements"]:
                self.json.append(json)
        self.downloading.emit(["Value", len(self.polygons)])

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
                    "Server at {} returned status code {} and no JSON data. Re-trying request in "
                    "{:.2f} seconds.".format(domain, response.status_code, error_pause_duration)
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

        p = Parameters().parameters["network"]["osm"]
        all_tags = p["all_link_types"]

        p = p["modes"]
        all_modes = list(p.keys())

        tags_to_keep = []
        for m in modes:
            if m not in all_modes:
                raise ValueError("Mode {} not listed in the parameters file".format(m))
            tags_to_keep += p[m]["link_types"]
        tags_to_keep = list(set(tags_to_keep))

        # Default to remove
        service = '["service"!~"parking|parking_aisle|driveway|private|emergency_access"]'
        access = '["access"!~"private"]'

        filtered = [x for x in all_tags if x not in tags_to_keep]
        filtered = "|".join(filtered)

        filter = '["area"!~"yes"]["highway"!~"{}"]{}{}'.format(filtered, service, access)

        return filter
