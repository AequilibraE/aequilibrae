import time
import re
from typing import List, Union, Tuple
import requests
from aequilibrae.parameters import Parameters
from .osm_params import http_headers


def placegetter(place: str) -> Tuple[Union[None, List[float]], list]:
    """
    Send a request to the Nominatim API via HTTP GET and return a geometry polygon
    for the region we are querying

    Parameters
    ----------
    place : str
        Name of the place we want to download a network for

    Adapted from http://www.github.com/gboeing/osmnx
    """

    par = Parameters().parameters["osm"]
    nominatim_endpoint = par["nominatim_endpoint"]
    max_attempts = par["max_attempts"]

    params = {"q": place, "format": "json"}

    report = []
    pause_duration = 1
    timeout = 30
    error_pause_duration = 180

    # prepare the Nominatim API URL
    url = nominatim_endpoint.rstrip("/") + "/search"
    prepared_url = requests.Request("GET", url, params=params).prepare().url
    # Pause, then request it
    report.append("Pausing {:,.2f} seconds before making API GET request".format(pause_duration))
    time.sleep(pause_duration)
    start_time = time.time()
    report.append(f"Requesting {prepared_url} with timeout={timeout}")
    response = requests.get(url, params=params, timeout=timeout, headers=http_headers)

    # get the response size and the domain, log result
    size_kb = len(response.content) / 1000.0
    domain = re.findall(r"(?s)//(.*?)/", url)[0]
    report.append("Downloaded {:,.1f}KB from {} in {:,.2f} seconds".format(size_kb, domain, time.time() - start_time))

    bbox = None
    for attempts in range(max_attempts):
        report.append(f"Attempt: {attempts}")
        if response.status_code != 200:
            report.append(
                "Server at {} returned status code {} and no JSON data. Re-trying request in {:.2f} seconds.".format(
                    domain, response.status_code, error_pause_duration
                )
            )

        if response.status_code in [429, 504]:
            # SEND MESSAGE
            time.sleep(error_pause_duration)
            continue
        elif response.status_code == 200:
            response_json = response.json()
            report.append("COMPLETE QUERY RESPONSE FOR PLACE:")
            report.append(str(response_json))
            if len(response_json):
                bbox = [float(x) for x in response_json[0]["boundingbox"]]
                bbox = [bbox[2], bbox[0], bbox[3], bbox[1]]
                report.append(f"PLACE FOUND:{response_json[0]['display_name']}")
            return (bbox, report)
        else:
            bbox = None

        if attempts == max_attempts - 1 and bbox is None:
            report.append("Reached maximum download attempts. Please wait a few minutes and try again")
        else:
            report.append("We got an error for place query.")

        return (bbox, report)
