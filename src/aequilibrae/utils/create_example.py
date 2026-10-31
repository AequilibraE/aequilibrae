import os
import zipfile
from pathlib import Path
from typing import List

from aequilibrae.project.project import Project


def create_example(path: os.PathLike | str, from_model="sioux_falls") -> Project:
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
        raise FileExistsError(f"Example not found '{path}'")

    pth.mkdir(parents=True, exist_ok=True)
    zipfile.ZipFile(source).extractall(pth)
    project = Project.from_path(str(pth))
    project.upgrade()

    # This refreshes the metadata of the project, so that the project is ready to be used.
    # It is a workaround for a bug in sqlite3 that does not refresh the metadata after an upgrade.
    with project.db_connection as conn:
        _ = conn.execute("SELECT count(*) FROM sqlite_master").fetchone()
    return project


def list_examples() -> List[str]:
    pth = Path(__file__).parent.parent / "reference_files"
    return [str(x.stem) for x in pth.glob("*.zip")]
