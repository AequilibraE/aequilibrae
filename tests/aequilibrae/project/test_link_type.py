from unittest import TestCase
import string
import random
import os
from shutil import copytree, rmtree
import tempfile
import uuid
from aequilibrae.project.network import LinkType
from aequilibrae.project import Project
from ...data import no_triggers_project


class TestLinkType(TestCase):
    def setUp(self) -> None:
        self.temp_proj_folder = os.path.join(tempfile.gettempdir(), uuid.uuid4().hex)
        copytree(no_triggers_project, self.temp_proj_folder)
        self.proj = Project()
        self.proj.open(self.temp_proj_folder)
        self.curr = self.proj.conn.cursor()

        letters = [random.choice(string.ascii_letters + '_') for x in range(20)]
        self.random_string = ''.join(letters)

    def tearDown(self) -> None:
        self.proj.close()
        print(self.temp_proj_folder)
        rmtree(self.temp_proj_folder)

    def test_build(self):
        for val in ['1', 'ab', '', None]:
            with self.assertRaises(ValueError):
                m = LinkType(val)

        for letter in range(10):
            letter = random.choice(string.ascii_letters)
            m = LinkType(letter)
            del m

    def test_changing_link_type_id(self):
        lt = LinkType('X')
        with self.assertRaises(ValueError):
            lt.link_type_id = 'test my description'

    def test_empty(self):
        a = LinkType('k')
        a.link_type = 'just a_test'
        with self.assertRaises(ValueError):
            a.save()

        a = LinkType('l')
        a.link_type = 'just_a_test_test_with_l'
        with self.assertRaises(ValueError):
            a.save()

    def test_save(self):
        self.curr.execute("select link_type_id from 'link_types'")

        letter = random.choice([x[0] for x in self.curr.fetchall()])
        m = LinkType(letter)
        m.link_type = self.random_string
        m.description = self.random_string[::-1]
        m.save()

        self.curr.execute(f'select description, link_type from link_types where link_type_id="{letter}"')

        desc, mname = self.curr.fetchone()
        self.assertEqual(desc, self.random_string[::-1], "Didn't save the mode description correctly")
        self.assertEqual(mname, self.random_string, "Didn't save the mode name correctly")
