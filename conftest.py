import os
import uuid
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc

# Register Arrow's at-fork handler before the test suite loads other native libraries.
pc.filter(pa.array([True]), pa.array([True]))

import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Polygon

from aequilibrae.matrix import AequilibraeMatrix
from aequilibrae.project.project import Project
from aequilibrae.transit import Transit
from aequilibrae.utils.create_example import create_example


@pytest.fixture(autouse=True)
def doctest_fixtures(doctest_namespace, tmp_path_factory, tmp_path):
    doctest_namespace["project_path"] = tmp_path / "p"
    doctest_namespace["my_folder_path"] = tmp_path_factory.mktemp(uuid.uuid4().hex)
    doctest_namespace["create_example"] = create_example
    doctest_namespace["Project"] = Project
    doctest_namespace["Transit"] = Transit
    doctest_namespace["AequilibraeMatrix"] = AequilibraeMatrix

    doctest_namespace["os"] = os
    doctest_namespace["pd"] = pd
    doctest_namespace["np"] = np
    doctest_namespace["Path"] = Path
    doctest_namespace["Polygon"] = Polygon


@pytest.fixture(scope="session", autouse=True)
def set_env():
    os.environ["AEQ_SHOW_PROGRESS"] = "FALSE"
