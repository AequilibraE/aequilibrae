from unittest import TestCase
from aequilibrae.transit.gtfs import create_gtfsdb
import os, sys

lib_path = os.path.abspath(os.path.join('..', '..'))
sys.path.append(lib_path)
from data import gtfs_folder, gtfs_zip, gtfs_db_output


class TestCreate_gtfsdb(TestCase):
    def test_create_database(self):
        self.gtfs = create_gtfsdb(gtfs_zip, save_db=gtfs_db_output, overwrite=True, spatialite_enabled=True)
        self.gtfs.create_database()
        self.gtfs.conn.close()

    def test_import_gtfs(self):
        # self.gtfs = create_gtfsdb(gtfs_folder, save_db=gtfs_db_output, overwrite=True, spatialite_enabled=True)
        self.gtfs = create_gtfsdb(gtfs_folder, save_db=gtfs_db_output, overwrite=True, spatialite_enabled=True)
        self.gtfs.import_gtfs()

        self.gtfs = create_gtfsdb(gtfs_zip, save_db=gtfs_db_output, overwrite=True, spatialite_enabled=True)
        self.gtfs.import_gtfs()
