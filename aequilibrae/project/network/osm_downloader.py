""""
Large portions of this code were adopted from OSMNx, by Geoff Boeing.

Although attempts to use OSMNx were made (including refactoring its
entire code base as a contribution to that package), it became clear
that its integration with libraries not available with QGIS' Python
distribution was too tight, and was therefore not practical to
detach them in order to use OSMNx as a dependency or submodule

For the original work, please see https://github.com/gboeing/osmnx
"""
import logging
import time
import re
import requests
from .osm_utils.osm_params import http_headers, memory
from aequilibrae.parameters import Parameters
from aequilibrae.context import get_logger
import gc
import importlib.util as iutil
from ...utils import WorkerThread

spec = iutil.find_spec("PyQt5")
pyqt = spec is not None
if pyqt:
    from PyQt5.QtCore import pyqtSignal


class OSMDownloader(WorkerThread):
    if pyqt:
        downloading = pyqtSignal(object)

    def __emit_all(self, *args):
        if pyqt:
            self.downloading.emit(*args)

    def __init__(self, polygons, modes, logger: logging.Logger = None):
        WorkerThread.__init__(self, None)
        self.logger = logger or get_logger()
        self.polygons = polygons
        self.filter = self.get_osm_filter(modes)
        self.report = []
        self.json = []
        par = Parameters().parameters["osm"]
        self.overpass_endpoint = par["overpass_endpoint"]
        self.timeout = par["timeout"]
        self.sleeptime = par["sleeptime"]

    def doWork(self):
        infrastructure = 'way["highway"]'
        query_template = (
            "{memory}[out:json][timeout:{timeout}];({infrastructure}{filters}({south:.6f},{west:.6f},"
            "{north:.6f},{east:.6f});>;);out;"
        )
        self.__emit_all(["maxValue", len(self.polygons)])
        self.__emit_all(["Value", 0])
        m = ""
        if memory > 0:
            m = f"[maxsize: {memory}]"
        for counter, poly in enumerate(self.polygons):
            msg = f"Downloading polygon {counter + 1} of {len(self.polygons)}"
            self.logger.debug(msg)
            self.__emit_all(["Value", counter])
            self.__emit_all(["text", msg])
            west, south, east, north = poly
            query_str = query_template.format(
                north=north,
                south=south,
                east=east,
                west=west,
                infrastructure=infrastructure,
                filters=self.filter,
                timeout=self.timeout,
                memory=m,
            )
            json = self.overpass_request(data={"data": query_str}, timeout=self.timeout)
            if json["elements"]:
                self.json.extend(json["elements"])
            del json
            gc.collect()
        self.__emit_all(["Value", len(self.polygons)])
        self.__emit_all(["FinishedDownloading", 0])

    def overpass_request(self, data, pause_duration=None, timeout=180, error_pause_duration=None):
        """Send a request to the Overpass API via HTTP POST and return the JSON response.

        :Arguments:
            **data**(:obj:`dict` or `OrderedDict`): key-value pairs of parameters to post to the API
            **pause_duration** (:obj:`int`): how long to pause in seconds before requests, if None, will query API
            status endpoint to find when next slot is available
            **timeout** (:obj:`int`): the timeout interval for the requests library
            **error_pause_duration**(:obj:`int`): how long to pause in seconds before re-trying requests if error

        :Returns:
            :obj:`dict`
        """

        # define the Overpass API URL, then construct a GET-style URL as a string to
        url = self.overpass_endpoint.rstrip("/") + "/interpreter"
        if pause_duration is None:
            time.sleep(self.sleeptime)
        start_time = time.time()
        self.report.append(f'Posting to {url} with timeout={timeout}, "{data}"')
        self.logger.debug(f'Posting to {url} with timeout={timeout}, "{data}"')
        response = requests.post(url, data=data, timeout=timeout, headers=http_headers)

        # get the response size and the domain, log result
        size_kb = len(response.content) / 1000.0
        domain = re.findall(r"(?s)//(.*?)/", url)[0]
        msg = "Downloaded {:,.1f}KB from {} in {:,.2f} seconds".format(size_kb, domain, time.time() - start_time)
        self.report.append(msg)
        self.logger.info(msg)

        try:
            response_json = response.json()
            if "remark" in response_json:
                msg = f'Server remark: "{response_json["remark"]}"'
                self.report.append(msg)
                self.logger.info(msg)
        except Exception:
            # 429 is 'too many requests' and 504 is 'gateway timeout' from server
            # overload - handle these errors by recursively calling
            # overpass_request until we get a valid response
            if response.status_code in [429, 504]:
                # pause for error_pause_duration seconds before re-trying request
                if error_pause_duration is None:
                    error_pause_duration = self.sleeptime + 1
                msg = "Server at {} returned status code {} and no JSON data. Re-trying request in {:.2f} seconds.".format(
                    domain, response.status_code, error_pause_duration
                )
                self.report.append(msg)
                self.logger.info(msg)
                time.sleep(error_pause_duration)
                response_json = self.overpass_request(data=data, pause_duration=pause_duration, timeout=timeout)

            # else, this was an unhandled status_code, throw an exception
            else:
                self.report.append(f"Server at {domain} returned status code {response.status_code} and no JSON data")
                raise Exception(f"Server returned no JSON data.\n{response} {response.reason}\n{response.text}")

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
                raise ValueError(f"Mode {m} not listed in the parameters file")
            tags_to_keep += p[m]["link_types"]
        tags_to_keep = list(set(tags_to_keep))

        # Default to remove
        service = '["service"!~"parking|parking_aisle|driveway|private|emergency_access"]'
        access = '["access"!~"private"]'

        filtered = [x for x in all_tags if x not in tags_to_keep]
        filtered = "|".join(filtered)

        filter = f'["area"!~"yes"]["highway"!~"{filtered}"]{service}{access}'

        return filter
