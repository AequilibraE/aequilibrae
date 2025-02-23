import zipfile
import os
from os.path import dirname, join
from pathlib import Path
from typing import List

from aequilibrae.project import Project


def create_example(path: os.PathLike, from_model="sioux_falls") -> Project:
    """Copies an example model to a new project project and returns the project handle

    :Arguments:
        **path** (:obj:`str`): Path where to create a new model. must be a non-existing folder/directory.

        **from_model** (:obj:`str`, *Optional*): Example to create from *sioux_falls*, *nauru* or *coquimbo*.
        Defaults to *sioux_falls*

    :Returns:
        **project** (:obj:`Project`): Aequilibrae Project handle (open)

    """
    pth = Path(path)
    if pth.is_dir() and pth.exists():
        raise FileExistsError("Cannot overwrite an existing directory")

    source = Path(__file__).parent.parent / "reference_files" / f"{from_model}.zip"
    if not source.exists():
        raise FileExistsError("Example not found")

    pth.mkdir(parents=True, exist_ok=True)
    zipfile.ZipFile(source).extractall(pth)
    proj = Project()
    proj.open(str(pth))
    return proj


def list_examples() -> List[str]:
    pth = Path(__file__).parent.parent / "reference_files"
    return [str(x.stem) for x in pth.glob("*.zip")]
