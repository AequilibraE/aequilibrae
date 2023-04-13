import zipfile
import os
from os.path import dirname, join
from aequilibrae.project import Project


def create_example(path: str, from_model="sioux_falls") -> Project:
    """Copies an example model to a new project project and returns the project handle

    :Arguments:
        **path** (:obj:`str`): Path where to create a new model. must be a non-existing folder/directory.
        **from_model** (:obj:`str`, `Optional`): Example to create from *sioux_falls*, *nauru* or *coquimbo*.
        Defaults to *sioux_falls*
    :Returns:
        **project** (:obj:`Project`): Aequilibrae Project handle (open)

    """
    if os.path.isdir(path):
        raise FileExistsError("Cannot overwrite an existing directory")

    if not os.path.isfile(join(dirname(__file__), f"../reference_files/{from_model}.zip")):
        raise FileExistsError("Example not found")

    os.makedirs(path, exist_ok=True)
    zipfile.ZipFile(join(dirname(__file__), f"../reference_files/{from_model}.zip")).extractall(path)
    proj = Project()
    proj.open(path)
    return proj
