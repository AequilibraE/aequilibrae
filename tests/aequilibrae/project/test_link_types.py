import random
import string
from unittest import TestCase

from tests.models_for_test import ModelsTest


class TestLinkTypes(TestCase):
    def setUp(self) -> None:
        tm = ModelsTest()
        self.proj = tm.no_triggers()
        self.curr = self.proj.conn.cursor()

        letters = [random.choice(string.ascii_letters + "_") for x in range(20)]
        self.random_string = "".join(letters)

    def tearDown(self) -> None:
        self.proj.close()

    def test_add(self):
        lt = self.proj.network.link_types
        existing = list(lt.all_types().keys())

        newlt = lt.new("G")
        newlt.link_type = "unique_link_type"
        newlt.save()

        nowexisting = list(lt.all_types().keys())

        n = [x for x in nowexisting if x not in existing][0]
        self.assertEqual("G", n, "Failed to add link type")

    def test_delete(self):
        lt = self.proj.network.link_types
        existing = list(lt.all_types().keys())
        deleted = random.choice(existing)
        lt.delete(deleted)
        remaining = list(lt.all_types().keys())

        difference = [x for x in existing if x not in remaining]

        self.assertEqual(deleted, difference[0], "Failed to delete link type")

    def test_get_and_get_by_name(self):
        lt = self.proj.network.link_types
        ltget = lt.get("y")
        ltgetbn = lt.get_by_name("default")

        self.assertEqual(ltget.link_type_id, ltgetbn.link_type_id, "Get methods returned different things")
        self.assertEqual(ltget.link_type, ltgetbn.link_type, "Get methods returned different things")
        self.assertEqual(ltget.description, ltgetbn.description, "Get methods returned different things")
        self.assertEqual(ltget.lanes, ltgetbn.lanes, "Get methods returned different things")
        self.assertEqual(ltget.lane_capacity, ltgetbn.lane_capacity, "Get methods returned different things")
        self.assertEqual(ltget.link_type, "default", "Get methods returned different things")

    def test_all_types(self):
        lt = self.proj.network.link_types
        all_lts = set([x for x in lt.all_types().keys()])

        c = self.proj.conn.cursor()
        c.execute("select link_type_id from link_types")
        reallts = set([x[0] for x in c.fetchall()])

        diff = all_lts.symmetric_difference(reallts)
        self.assertEqual(diff, set(), "Getting all link_types failed")
