from unittest.mock import patch

import pytest

# Canned Overpass API response used to avoid making real HTTP requests during
# tests. The Overpass/osm3s JSON schema is versioned ("version": 0.6) and has
# been stable for years, so it's pinned here rather than left to whatever a
# live server happens to return.
#
# Two ways sharing a node (a T-intersection) so the network builder actually
# splits "main_st" into two link segments at the shared node - this is what
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
        "nodes": [3, 6, 7],
        "tags": {"highway": "residential", "name": "Side St"},
    },
]
_MOCK_RESPONSE = {
    "version": 0.6,
    "generator": "Overpass API (mock)",
    "elements": _MAIN_ST_NODES + _SIDE_ST_NODES + _WAYS,
}


class _MockOverpassResponse:
    status_code = 200
    reason = "OK"
    text = ""
    content = b""

    def json(self):
        return _MOCK_RESPONSE


def _mock_post(url, data=None, timeout=None, headers=None):
    return _MockOverpassResponse()


@pytest.fixture(scope="function")
def mock_overpass_api():
    """Replaces real HTTP calls to the Overpass API with a canned response.

    OSMDownloader.overpass_request calls requests.post directly, so patching it
    there covers both direct OSMDownloader usage and Network.create_from_osm,
    which goes through the same downloader.
    """
    with patch("aequilibrae.project.network.osm.osm_downloader.requests.post", side_effect=_mock_post):
        yield
