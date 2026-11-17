import os
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Polygon

from aequilibrae import Project
from aequilibrae.distribution import GravityApplication, GravityCalibration, Ipf, SyntheticGravityModel
from aequilibrae.examples import create_example
from aequilibrae.matrix import AequilibraeMatrix
from aequilibrae.transit.transit import Transit


@pytest.fixture(autouse=True)
def doctest_fixtures(doctest_namespace, tmp_path_factory, tmp_path):
    doctest_namespace["project_path"] = tmp_path / "p"
    doctest_namespace["my_folder_path"] = tmp_path_factory.mktemp(uuid.uuid4().hex)
    doctest_namespace["create_example"] = create_example
    doctest_namespace["Project"] = Project
    doctest_namespace["Transit"] = Transit
    doctest_namespace["AequilibraeMatrix"] = AequilibraeMatrix
    doctest_namespace["GravityApplication"] = GravityApplication
    doctest_namespace["GravityCalibration"] = GravityCalibration
    doctest_namespace["Ipf"] = Ipf
    doctest_namespace["SyntheticGravityModel"] = SyntheticGravityModel

    doctest_namespace["os"] = os
    doctest_namespace["pd"] = pd
    doctest_namespace["np"] = np
    doctest_namespace["Path"] = Path
    doctest_namespace["Polygon"] = Polygon


@pytest.fixture(scope="session", autouse=True)
def set_env():
    os.environ["AEQ_SHOW_PROGRESS"] = "FALSE"
