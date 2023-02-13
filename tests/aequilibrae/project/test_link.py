from unittest import TestCase
from shapely.ops import substring
from copy import copy, deepcopy
import os
from shutil import copytree, rmtree
import uuid
from random import randint, random
from tempfile import gettempdir
from aequilibrae.project import Project

from ...data import siouxfalls_project


class TestLink(TestCase):
    def setUp(self) -> None:
        os.environ["PATH"] = os.path.join(gettempdir(), "../../temp_data") + ";" + os.environ["PATH"]

        self.proj_dir = os.path.join(gettempdir(), uuid.uuid4().hex)
        copytree(siouxfalls_project, self.proj_dir)

        self.project = Project()
        self.project.open(self.proj_dir)
        self.network = self.project.network
        self.curr = self.project.conn.cursor()

        self.links = self.network.links
        self.modes = self.network.modes
        self.lid = randint(1, 24)
        self.link = self.links.get(self.lid)

    def tearDown(self) -> None:
        self.project.close()
        del self.curr
        try:
            rmtree(self.proj_dir)
        except Exception as e:
            print(f"Failed to remove at {e.args}")

    def test_delete(self):
        self.link.delete()

        with self.assertRaises(Exception):
            _ = self.links.get(self.lid)

        self.curr.execute("Select count(*) from links where link_id=?", [self.lid])

        self.assertEqual(0, self.curr.fetchone()[0], f"Failed to delete link {self.lid}")

    def test_save(self):
        self.link.save()
        extension = random()
        name = "just a non-important value"

        geo = substring(self.link.geometry, 0, extension, normalized=True)

        self.link.name = name
        self.link.geometry = geo

        self.link.save()
        self.links.refresh()
        link2 = self.links.get(self.lid)

        self.assertEqual(link2.name, name, "Failed to save the link name")
        self.assertAlmostEqual(link2.geometry, geo, 3, "Failed to save the link geometry")

        tot_prev = self.network.count_links()
        lnk = self.links.new()
        lnk.geometry = substring(self.link.geometry, 0, 0.88, normalized=True)
        lnk.modes = "c"
        lnk.save()

        self.assertEqual(tot_prev + 1, self.network.count_links(), "Failed to save new link")

    def test_set_modes(self):
        self.link.set_modes("cbt")

        self.assertEqual(self.link.modes, "cbt", "Did not set modes correctly")
        self.link.save()

        self.assertEqual(self.__check_mode(), "cbt")

    def test_add_mode(self):
        for mode in [1, ["cbt"]]:
            with self.assertRaises(TypeError):
                self.link.add_mode(mode)
        with self.assertRaises(ValueError):
            self.link.add_mode("bt")

        self.link.add_mode("b")
        self.link.save()
        self.assertEqual(self.__check_mode(), "cb")

        mode = self.modes.get("t")
        self.link.add_mode(mode)
        self.link.save()
        self.assertEqual(self.__check_mode(), "cbt")

    def test_drop_mode(self):
        self.link.set_modes("cbt")
        self.link.save()
        self.assertEqual(self.__check_mode(), "cbt")

        self.link.drop_mode("t")
        self.link.save()
        self.assertEqual(self.__check_mode(), "cb")

        mode = self.modes.get("b")
        self.link.drop_mode(mode)
        self.link.save()
        self.assertEqual(self.__check_mode(), "c")

    def test_data_fields(self):
        link2 = self.links.get(randint(1, 24))
        while link2.link_id == self.link.link_id:
            link2 = self.links.get(randint(1, 24))

        self.assertEqual(link2.data_fields(), self.link.data_fields(), "Different links have different data fields")

        fields = sorted(link2.data_fields())
        self.curr.execute("pragma table_info(links)")
        dt = self.curr.fetchall()

        data_fields = sorted([x[1] for x in dt if x[1] != "ogc_fid"])

        self.assertEqual(sorted(fields), sorted(data_fields), "Link has unexpected set of fields")

    def __check_mode(self):
        sql = "Select modes from links where link_id=?"
        self.curr.execute(sql, [self.lid])
        return self.curr.fetchone()[0]
