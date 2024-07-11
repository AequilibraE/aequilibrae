from uuid import uuid4
from os import remove
from os.path import join, exists
from tempfile import gettempdir
from zipfile import ZIP_DEFLATED, ZipFile
from pathlib import Path

from aequilibrae import __file__ as aeq_init
from aequilibrae.transit import Transit
from aequilibrae.utils.create_example import create_example


def rebuid_coquimbo_example(dest_folder):
    """
    dest_folder: where the zip file with the complete model is stored
    gtfs_path: path to gtfs_file
    """

    fldr = join(gettempdir(), uuid4().hex)
    project = create_example(fldr, "coquimbo")
    gtfs_path = join(fldr, "gtfs_coquimbo.zip")

    remove(join(fldr, "public_transport.sqlite"))

    data = Transit(project)

    transit = data.new_gtfs_builder(agency="Lisanco", file_path=gtfs_path)

    transit.load_date("2016-04-13")

    transit.set_allow_map_match(True)
    transit.map_match()

    transit.save_to_disk()

    cursor = Transit(project).pt_con.cursor()
    cursor.execute("VACUUM;")

    with ZipFile(f"{dest_folder}/coquimbo.zip", "w", compression=ZIP_DEFLATED, compresslevel=9) as zip_object:
        zip_object.write(f"{fldr}/project_database.sqlite", "project_database.sqlite")
        zip_object.write(f"{fldr}/public_transport.sqlite", "public_transport.sqlite")
        zip_object.write(f"{fldr}/gtfs_coquimbo.zip", "gtfs_coquimbo.zip")
        zip_object.write(f"{fldr}/parameters.yml", "parameters.yml")

    if exists(f"{dest_folder}/coquimbo.zip"):
        print("ZIP file created!")


if __name__ == "__main__":
    aeq_dir = Path(aeq_init).parent
    rebuid_coquimbo_example(aeq_dir / "reference_files")
