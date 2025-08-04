import os
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Polygon

from aequilibrae import Project
from aequilibrae.matrix import AequilibraeMatrix
from aequilibrae.transit import Transit
from aequilibrae.utils.create_example import create_example


@pytest.fixture(autouse=True)
def doctest_fixtures(doctest_namespace, tmp_path_factory, tmp_path):
    doctest_namespace["project_path"] = str(tmp_path / "p")
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
