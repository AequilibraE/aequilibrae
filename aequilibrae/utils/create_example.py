import zipfile
import os
from os.path import dirname, join
from aequilibrae import Project


def create_example(path: str) -> Project:
    """Copies the Sioux Falls model as a new example project and returns the project handle"""
    if os.path.isdir(path):
        raise FileExistsError('Cannot overwrite an existing directory')

    os.mkdir(path)
    zipfile.ZipFile(join(dirname(__file__), '../reference_files/sioux_falls.zip')).extractall(path)
    proj = Project()
    proj.open(path)
    return proj
