import random
import string
from unittest import TestCase

from aequilibrae.project.network.mode import Mode
from tests.models_for_test import ModelsTest


class TestMode(TestCase):
    def setUp(self) -> None:
        tm = ModelsTest()
        self.proj = tm.no_triggers()
        self.curr = self.proj.conn.cursor()

        letters = [random.choice(string.ascii_letters + "_") for x in range(20)]
        self.random_string = "".join(letters)

    def tearDown(self) -> None:
        self.proj.close()

    def test_build(self):
        for val in ["1", "ab", "", None]:
            with self.assertRaises(ValueError):
                m = Mode(val, self.proj)

        for letter in range(10):
            letter = random.choice(string.ascii_letters)
            m = Mode(letter, self.proj)
            del m

    def test_changing_mode_id(self):
        m = Mode("c", self.proj)
        with self.assertRaises(ValueError):
            m.mode_id = "test my description"

    def test_save(self):
        self.curr.execute("select mode_id from 'modes'")

        letter = random.choice([x[0] for x in self.curr.fetchall()])
        m = Mode(letter, self.proj)
        m.mode_name = self.random_string
        m.description = self.random_string[::-1]
        m.save()

        self.curr.execute(f'select description, mode_name from modes where mode_id="{letter}"')

        desc, mname = self.curr.fetchone()
        self.assertEqual(desc, self.random_string[::-1], "Didn't save the mode description correctly")
        self.assertEqual(mname, self.random_string, "Didn't save the mode name correctly")

    def test_empty(self):
        a = Mode("k", self.proj)
        a.mode_name = "just a_test"
        with self.assertRaises(ValueError):
            a.save()

        a = Mode("l", self.proj)
        a.mode_name = "just_a_test_test_with_l"
        with self.assertRaises(ValueError):
            a.save()
