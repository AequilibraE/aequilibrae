import logging
import shutil
import tempfile
import uuid
import zipfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from aequilibrae import Project
from aequilibrae.utils.create_example import create_example


@pytest.fixture(scope="session")
def centroids():
    return np.arange(27) + 1


@pytest.fixture(scope="session")
def cache_path(test_base):
    return test_base / "cache"


@pytest.fixture(scope="session")
def test_base():
    return Path(tempfile.gettempdir()) / "aequilibrae_testing"


@pytest.fixture(scope="function")
def test_folder(test_base):
    right_now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    dir = test_base / f"{right_now}--{uuid.uuid4().hex[:4]}"
    while dir.exists():
        dir = test_base / f"{right_now}--{uuid.uuid4().hex[:4]}"
    dir.mkdir(parents=True, exist_ok=True)
    return dir


@pytest.fixture(scope="session")
def test_data_path():
    return Path(__file__).parent / "data"


@pytest.fixture(scope="function")
def omx_example(test_data_path, test_folder):
    test_folder.mkdir(parents=True, exist_ok=True)
    shutil.copy(test_data_path / "test_omx.omx", test_folder / "test_omx.omx")
    return test_folder / "test_omx.omx"


@pytest.fixture(scope="function")
def no_index_omx(test_data_path, test_folder):
    test_folder.mkdir(parents=True, exist_ok=True)
    shutil.copy(test_data_path / "no_index.omx", test_folder / "no_index.omx")
    return test_folder / "test_omx.omx"


@pytest.fixture(scope="function")
def sioux_falls_example(cache_path, test_folder):
    source = cache_path / "sioux_falls"
    shutil.copytree(source, test_folder, dirs_exist_ok=True)
    project = Project.from_path(test_folder)
    yield project
    project.close()


@pytest.fixture(scope="function")
def sioux_falls_test(test_data_path, test_folder) -> Project:
    source = test_data_path / "SiouxFalls_project"
    shutil.copytree(source, test_folder, dirs_exist_ok=True)
    project = Project.from_path(test_folder)
    yield project
    project.close()


@pytest.fixture(scope="function")
def sioux_falls_single_class(cache_path, test_folder) -> Project:
    source = cache_path / "sioux_falls_single_class"
    shutil.copytree(source, test_folder, dirs_exist_ok=True)
    project = Project.from_path(test_folder)
    yield project
    project.close()


@pytest.fixture(scope="function")
def triangle_graph_blocking(test_data_path, test_folder) -> Project:
    source = test_data_path / "blocking_triangle_graph_project"
    shutil.copytree(source, test_folder, dirs_exist_ok=True)
    project = Project.from_path(test_folder)
    yield project
    project.close()


@pytest.fixture(scope="function")
def coquimbo_example(cache_path, test_folder):
    source = cache_path / "coquimbo"
    shutil.copytree(source, test_folder, dirs_exist_ok=True)
    project = Project.from_path(test_folder)
    yield project
    project.close()


def pytest_sessionstart(session):
    test_base = Path(tempfile.gettempdir()) / "aequilibrae_testing"
    tgt = test_base / "cache" / "sioux_falls"
    if not tgt.exists():
        create_example(tgt, "sioux_falls").close()

    tgt = test_base / "cache" / "coquimbo"
    if not tgt.exists():
        create_example(tgt, "coquimbo").close()

    tgt = test_base / "cache" / "sioux_falls_single_class"
    if not tgt.exists():
        zipfile.ZipFile(Path(__file__).parent / "data" / "sioux_falls_single_class.zip").extractall(tgt)

    right_now = datetime.now().strftime("%Y-%m-%d_%H")
    for item in test_base.glob("*"):
        if item.is_dir():
            try:
                if right_now not in item.name and "cache" not in item.name:
                    shutil.rmtree(item)
            except Exception as e:
                # Skip folders with non-matching name pattern
                logging.error(f"Couldn't delete dir {item}, reason: {e}")
