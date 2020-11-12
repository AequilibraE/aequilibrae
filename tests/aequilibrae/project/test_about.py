from unittest import TestCase
import sqlite3
import tempfile
import random
import os
import uuid
from shutil import copytree, rmtree
from aequilibrae import Project
import string
from ...data import siouxfalls_project


class TestAbout(TestCase):
    def setUp(self) -> None:
        os.environ['PATH'] = os.path.join(tempfile.gettempdir(), 'temp_data') + ';' + os.environ['PATH']
        self.temp_proj_folder = os.path.join(tempfile.gettempdir(), uuid.uuid4().hex)
        copytree(siouxfalls_project, self.temp_proj_folder)
        self.proj = Project()
        self.proj.open(self.temp_proj_folder)
        self.curr = self.proj.conn.cursor()

    def tearDown(self) -> None:
        self.proj.close()

    def test_create_and_list(self):
        self.proj.about.create()

        with self.assertWarns(expected_warning=Warning):
            self.proj.about.create()

        list = self.proj.about.list_fields()
        expected = ['model_name', 'region', 'description', 'author', 'license', 'scenario_name', 'year',
                    'scenario_description', 'model_version', 'project_id', 'aequilibrae_version', 'projection']

        failed = set(list).symmetric_difference(set(expected))
        if failed:
            self.fail('About table does not have all expected fields')

    # idea from https://stackoverflow.com/a/2030081/1480643
    def randomword(self, length):
        letters = string.ascii_lowercase + '_'
        return ''.join(random.choice(letters) for i in range(length))

    def test_add_info_field(self):
        self.proj.about.create()

        all_added = set()
        for t in range(30):
            k = self.randomword(random.randint(1, 15))
            if k not in all_added:
                all_added.add(k)
                self.proj.about.add_info_field(k)

        curr = self.proj.conn.cursor()
        curr.execute("select infoname from 'about'")

        charac = [x[0] for x in curr.fetchall()]
        for k in all_added:
            if k not in charac:
                self.fail(f'Failed to add {k}')

        # Should fail when trying to add a repeated guy
        with self.assertRaises(sqlite3.IntegrityError):
            self.proj.about.add_info_field('description')

        # Should fail when trying to add a repeated guy
        with self.assertRaises(ValueError):
            self.proj.about.add_info_field('descr1ption')

    def test_write_back(self):
        self.proj.about.create()
        self.proj.about.add_info_field('good_info_field_perhaps')

        val = self.randomword(random.randint(1, 15))
        self.proj.about.good_info_field_perhaps = val

        val2 = self.randomword(random.randint(30, 250))
        self.proj.about.description = val2

        self.proj.about.write_back()

        self.proj.close()
        del self.proj

        self.proj = Project()
        self.proj.open(self.temp_proj_folder)
        self.assertEqual(val, self.proj.about.good_info_field_perhaps, 'failed to save data to about table')
        self.assertEqual(val2, self.proj.about.description, 'failed to save data to about table')
