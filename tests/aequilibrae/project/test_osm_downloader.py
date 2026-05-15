import os
import pytest
from random import random

from shapely.geometry import box

from aequilibrae.project.network.osm.osm_downloader import OSMDownloader


@pytest.fixture(scope="function")
def should_do_work():
    thresh = 1.01 if os.environ.get("GITHUB_WORKFLOW", "ERROR") == "Code coverage" else 0.02
    return random() < thresh


def test_do_work(should_do_work):
    if not should_do_work:
        pytest.skip("Skipping test based on random chance")

    o = OSMDownloader([box(0.0, 0.0, 0.1, 0.1)], ["car"])
    o.doWork()
    assert not o.json, "It found links in the middle of the ocean"


def test_do_work2(should_do_work):
    if not should_do_work:
        pytest.skip("Skipping test based on random chance")

    # LITTLE PLACE IN THE MIDDLE OF THE Grand Canyon North Rim
    o = OSMDownloader([box(-112.185, 36.59, -112.179, 36.60)], ["car"])
    o.doWork()

    if len(o.json) == 0 or "elements" not in o.json[0]:
        pytest.skip("No elements found in response")

    assert len(o.json[0]["elements"]) <= 1000, "It found too many elements in the middle of the Grand Canyon"
    assert len(o.json[0]["elements"]) >= 10, "It found too few elements in the middle of the Grand Canyon"
