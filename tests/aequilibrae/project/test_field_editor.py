import os
from uuid import uuid4
from random import choice
from shutil import copytree
from os.path import join
from tempfile import gettempdir
from unittest import TestCase
from aequilibrae.project.field_editor import FieldEditor, ALLOWED_CHARACTERS
from aequilibrae import Project
from ...data import siouxfalls_project


class TestFieldEditor(TestCase):
    my_tables = ["link_types", "links", "modes", "nodes"]

    def setUp(self) -> None:
        os.environ["PATH"] = os.path.join(gettempdir(), "temp_data") + ";" + os.environ["PATH"]
        self.temp_proj_folder = join(gettempdir(), uuid4().hex)
        copytree(siouxfalls_project, self.temp_proj_folder)
        self.proj = Project()
        self.proj.open(self.temp_proj_folder)

    def tearDown(self) -> None:
        self.proj.close()

    def randomword(self, length):
        val = "".join(choice(ALLOWED_CHARACTERS) for i in range(length))
        if val[0] == "_" or val[-1] == "_":
            return self.randomword(length)
        return val

    def test_building(self):
        for tab in ["modes", "links", "nodes", "link_types"]:
            table = FieldEditor(self.proj, tab)
            qry = f'select count(*) from "attributes_documentation" where name_table="{tab}"'
            q = self.proj.conn.execute(qry).fetchone()[0]
            self.assertEqual(q, len(table._original_values), "Meta table populated with the wrong number of elements")

    def test_add(self):
        for tab in self.my_tables:
            table = FieldEditor(self.proj, tab)
            qry = f'select count(*) from "attributes_documentation" where name_table="{tab}"'
            q = self.proj.conn.execute(qry).fetchone()[0]
            one = choice(list(table._original_values.keys()))
            self.proj.conn.commit()
            with self.assertRaises(ValueError) as em:
                table.add(one, self.randomword(30))
            self.assertEqual("attribute_name already exists", str(em.exception), "failed in the wrong place")

            with self.assertRaises(ValueError) as em:
                table.add(f"{self.randomword(5)} {5}", self.randomword(30))
            self.assertEqual(
                'attribute_name can only contain letters, numbers and "_"',
                str(em.exception),
                "failed in the wrong place",
            )

            with self.assertRaises(ValueError):
                table.add(choice("0123456789") + self.randomword(20), self.randomword(30))
            new_one = self.randomword(20)
            while new_one[0] in "0123456789":
                new_one = self.randomword(20)
            table.add(new_one, self.randomword(30))
            self.proj.conn.commit()
            curr = self.proj.conn.cursor()
            curr.execute(f'select count(*) from "attributes_documentation" where name_table="{tab}"')
            q2 = curr.fetchone()[0]
            self.assertEqual(q + 1, q2, "Adding element did not work")

            # If query fails, we failed to add new field to the database
            curr.execute(f'select "{new_one}" from "attributes_documentation" where name_table="{tab}"')

            if "alpha" in table._original_values.keys():
                self.assertEqual(table.alpha, "Available for user convenience", "not being able to retrieve values")

            self.proj.conn.commit()
            del curr

    def test_check_completeness(self):
        for table in self.my_tables:
            curr = self.proj.conn.cursor()

            # We add a bogus record to the attribute list
            val = self.randomword(30).lower()
            qry = 'INSERT INTO "attributes_documentation" VALUES (?,?," ");'
            curr.execute(qry, (table, val))
            self.proj.conn.commit()

            curr.execute(f'Select name_table from "attributes_documentation" where attribute="{val}"')
            self.assertEqual(curr.fetchone()[0], table, "Failed to insert bogus value")

            # Then we add a new field to the table
            val2 = self.randomword(10)
            curr.execute(f'Alter table "{table}" add column "{val2}" NUMERIC;')
            curr.execute(f"pragma table_info({table})")
            fields = [x[1] for x in curr.fetchall() if x[1] == val2]
            self.assertEqual([val2], fields, "failed to add a new field")

            table = FieldEditor(self.proj, table)
            self.proj.conn.commit()
            curr = self.proj.conn.cursor()
            curr.execute(f'Select count(*) from "attributes_documentation" where attribute="{val}"')
            self.assertEqual(curr.fetchone()[0], 0, f"clean the table on loading failed {val}")

            curr.execute(f'Select count(*) from "attributes_documentation" where attribute="{val2}"')
            self.assertEqual(curr.fetchone()[0], 1, "clean the table on loading failed")
            print(table._original_values[val2])

            self.proj.conn.commit()
            del curr

    def test_save(self):
        for tab in ["modes", "links", "nodes", "link_types"]:
            table = FieldEditor(self.proj, tab)
            random_val = self.randomword(30)
            if "alpha" in table._original_values.keys():
                table.alpha = random_val
                table.save()
                table2 = FieldEditor(self.proj, tab)

                self.assertEqual(table2.alpha, random_val, "Did not save values properly")

            if "link_id" in table._original_values.keys():
                table.link_id = random_val
                table.save()
                table2 = FieldEditor(self.proj, tab)

                self.assertEqual(table2.link_id, random_val, "Did not save values properly")

            if "node_id" in table._original_values.keys():
                table.node_id = random_val
                table.save()
                table2 = FieldEditor(self.proj, tab)

                self.assertEqual(table2.node_id, random_val, "Did not save values properly")

            self.proj.conn.commit()
