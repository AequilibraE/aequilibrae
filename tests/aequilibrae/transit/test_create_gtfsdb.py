from unittest import TestCase

from aequilibrae.transit.gtfs import create_gtfsdb
from ...data import gtfs_folder
from ...data import gtfs_zip
from ...data import gtfs_db_output


class TestCreate_gtfsdb(TestCase):
    def test_create_database(self):
        # TODO: Fix Travis-ci configuration in order to properly run the test with spatialite enabled
        self.gtfs = create_gtfsdb(gtfs_zip, save_db=gtfs_db_output, overwrite=True, spatialite_enabled=False)
        self.gtfs.create_database()
        self.gtfs.conn.close()

    def test_import_gtfs(self):
        # self.gtfs = create_gtfsdb(gtfs_folder, save_db=gtfs_db_output, overwrite=True, spatialite_enabled=True)
        self.gtfs = create_gtfsdb(gtfs_folder, save_db=gtfs_db_output, overwrite=True, spatialite_enabled=False)
        self.gtfs.import_gtfs()

        self.gtfs = create_gtfsdb(gtfs_zip, save_db=gtfs_db_output, overwrite=True, spatialite_enabled=False)
        self.gtfs.import_gtfs()
