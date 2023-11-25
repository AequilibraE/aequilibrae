import random
import string
from sqlite3 import IntegrityError
from unittest import TestCase

from tests.models_for_test import ModelsTest


class TestLinkType(TestCase):
    def setUp(self) -> None:
        tm = ModelsTest()
        self.proj = tm.no_triggers()
        self.curr = self.proj.conn.cursor()

        letters = [random.choice(string.ascii_letters + "_") for x in range(20)]
        self.random_string = "".join(letters)

    def tearDown(self) -> None:
        self.proj.close()

    def test_changing_link_type_id(self):
        ltypes = self.proj.network.link_types

        lt = random.choice([x for x in ltypes.all_types().values()])

        with self.assertRaises(ValueError):
            lt.link_type_id = "test my description"

        with self.assertRaises(ValueError):
            lt.link_type_id = "K"

    def test_empty(self):
        ltypes = self.proj.network.link_types

        newt = ltypes.new("Z")
        # a.link_type = 'just a_test'
        with self.assertRaises(IntegrityError):
            newt.save()

    def test_save(self):
        ltypes = self.proj.network.link_types

        newt = ltypes.new("Z")
        newt.link_type = self.random_string
        newt.description = self.random_string[::-1]
        newt.save()

        self.curr.execute('select description, link_type from link_types where link_type_id="Z"')

        desc, mname = self.curr.fetchone()
        self.assertEqual(desc, self.random_string[::-1], "Didn't save the mode description correctly")
        self.assertEqual(mname, self.random_string, "Didn't save the mode name correctly")
