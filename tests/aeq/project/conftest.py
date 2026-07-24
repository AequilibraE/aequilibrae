from unittest.mock import patch

import pytest

# Canned Overpass API responses used to avoid making real HTTP requests during
# tests. The Overpass/osm3s JSON schema is versioned ("version": 0.6) and has
# been stable for years, so it's pinned here rather than left to whatever a
# live server happens to return.
#
# Two ways sharing a node (a T-intersection) so the network builder actually
# splits "Main St" into two link segments at the shared node - this is what
# gives a distinct-osm_id count lower than the link count, and a node count
# that isn't exceeded by the link count, matching what test_create_from_osm
# asserts about a real network.
_MAIN_ST_NODES = [
    {"type": "node", "id": 1, "lat": 36.591, "lon": -112.184, "tags": {}},
    {"type": "node", "id": 2, "lat": 36.592, "lon": -112.184, "tags": {}},
    {"type": "node", "id": 3, "lat": 36.593, "lon": -112.184, "tags": {}},  # intersection with side_st
    {"type": "node", "id": 4, "lat": 36.594, "lon": -112.184, "tags": {}},
    {"type": "node", "id": 5, "lat": 36.595, "lon": -112.184, "tags": {}},
]
_SIDE_ST_NODES = [
    {"type": "node", "id": 6, "lat": 36.593, "lon": -112.183, "tags": {}},
    {"type": "node", "id": 7, "lat": 36.593, "lon": -112.182, "tags": {}},
    {"type": "node", "id": 8, "lat": 36.593, "lon": -112.181, "tags": {}},
]
_WAYS = [
    {
        "type": "way",
        "id": 100,
        "nodes": [1, 2, 3, 4, 5],
        "tags": {"highway": "residential", "name": "Main St"},
    },
    {
        "type": "way",
        "id": 101,
        "nodes": [3, 6, 7, 8],
        "tags": {"highway": "residential", "name": "Side St"},
    },
]
_GRID_RESPONSE = {
    "version": 0.6,
    "generator": "Overpass API (mock)",
    "elements": _MAIN_ST_NODES + _SIDE_ST_NODES + _WAYS,
}

# No roads in the middle of the ocean.
_EMPTY_RESPONSE = {"version": 0.6, "generator": "Overpass API (mock)", "elements": []}


class _MockOverpassResponse:
    status_code = 200
    reason = "OK"
    text = ""
    content = b""

    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return self._payload


def _patch_overpass(payload):
    def _mock_post(url, data=None, timeout=None, headers=None):
        return _MockOverpassResponse(payload)

    return patch("aequilibrae.project.network.osm.osm_downloader.requests.post", side_effect=_mock_post)


@pytest.fixture(scope="function")
def mock_overpass_empty():
    """Replaces real HTTP calls to the Overpass API with an empty result set."""
    with _patch_overpass(_EMPTY_RESPONSE):
        yield


@pytest.fixture(scope="function")
def mock_overpass_grid():
    """Replaces real HTTP calls to the Overpass API with a small street grid.

    OSMDownloader.overpass_request calls requests.post directly, so patching it
    there covers both direct OSMDownloader usage and Network.create_from_osm,
    which goes through the same downloader.
    """
    with _patch_overpass(_GRID_RESPONSE):
        yield


# Canned Nominatim response for "Vatican City", using its real, public
# bounding box (south, north, west, east) - Nominatim's own JSON schema.
_VATICAN_CITY_RESPONSE = [
    {
        "display_name": "Vatican City",
        "boundingbox": ["41.9002044", "41.9073912", "12.4457442", "12.4583658"],
    }
]

# Nominatim returns an empty list when nothing matches the query.
_NO_MATCH_RESPONSE = []


class _MockNominatimResponse:
    status_code = 200
    reason = "OK"
    text = ""
    content = b""

    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return self._payload


def _mock_get(url, params=None, timeout=None, headers=None):
    query = (params or {}).get("q", "")
    payload = _VATICAN_CITY_RESPONSE if query == "Vatican City" else _NO_MATCH_RESPONSE
    return _MockNominatimResponse(payload)


@pytest.fixture(scope="function")
def mock_nominatim_api():
    """Replaces real HTTP calls to the Nominatim API with canned responses.

    placegetter calls requests.get directly, keyed here off the "q" query
    param so different place names can resolve to different canned results.
    """
    with patch("aequilibrae.project.network.osm.place_getter.requests.get", side_effect=_mock_get):
        yield
