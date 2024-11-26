import shutil
import sys
from os.path import join, realpath
from pathlib import Path
from tempfile import gettempdir
from typing import List
from uuid import uuid4

project_dir = Path(__file__).parent.parent
if str(project_dir) not in sys.path:
    sys.path.append(str(project_dir))


class CreateTablesSRC:
    def __init__(self, component: str, tgt_fldr: str):
        from aequilibrae.project import Project
        from aequilibrae.project.database_connection import database_connection
        from aequilibrae.transit import Transit

        # Create a new project
        self.proj_path = join(gettempdir(), f"aequilibrae_{uuid4().hex[:6]}")
        self.proj = Project()
        self.proj.new(self.proj_path)
        Transit(self.proj)

        self.__folder = "network" if component == "project_database" else "transit"
        self.stub = "data_model"
        # Get the appropriate data for the database we are documenting
        self.conn = database_connection(db_type=self.__folder, project_path=self.proj_path)
        self.path = join(
            *Path(realpath(__file__)).parts[:-1],
            f"../aequilibrae/project/database_specification/{self.__folder}/tables",
        )
        self.doc_path = str(
            Path(realpath(__file__)).parent / "source" / "modeling_with_aequilibrae" / tgt_fldr
        )

        Path(join(self.doc_path, self.stub)).mkdir(exist_ok=True, parents=True)

    def create(self):
        datamodel_rst = join(self.doc_path, self.stub, "datamodel.rst")
        shutil.copyfile(join(self.doc_path, "datamodel.rst.template"), datamodel_rst)
        placeholder = "LIST_OF_TABLES"
        tables_txt = "table_list.txt"
        all_tables = [e for e in self.readlines(join(self.path, tables_txt)) if e != "migrations"]

        for table_name in all_tables:
            descr = self.conn.execute(f"pragma table_info({table_name})").fetchall()

            # Data model reference
            reference = f".. _{table_name}_{self.__folder}_data_model:\n"

            # Title of the page
            title = f'**{table_name.replace("_", " ")}** table structure'
            txt = [reference, title, "=" * len(title), ""]

            docstrings = self.__get_docstrings(table_name)
            sql_code = self.__get_sql_code(table_name)

            txt.extend(docstrings)

            txt.append("")
            txt.append(".. csv-table:: ")
            txt.append('   :header: "Field", "Type", "NULL allowed", "Default Value"')
            txt.append("   :widths:    30,     20,         20,          20")
            txt.append("")

            for dt in descr:
                data = list(dt[1:-1])
                if dt[-1] == 1:
                    data[0] += "*"

                if data[-1] is None:
                    data[-1] = ""

                if data[2] == 1:
                    data[2] = "NO"
                else:
                    data[2] = "YES"

                txt.append("   " + ",".join([str(x) for x in data]))
            txt.append("\n\n(* - Primary key)")

            txt.extend(sql_code)

            output = join(self.doc_path, self.stub, f"{table_name}.rst")
            with open(output, "w") as f:
                for line in txt:
                    f.write(line + "\n")

            all_tables = [f"   {x.rstrip()}.rst\n" for x in sorted(all_tables, reverse=True)]

            with open(datamodel_rst, "r") as lst:
                datamodel = lst.readlines()

            for i, line in enumerate(datamodel):
                if placeholder in line:
                    datamodel[i] = ""
                    for tb in all_tables:
                        datamodel.insert(i, tb)
                    break

            with open(datamodel_rst, "w") as lst:
                for line in datamodel:
                    lst.write(line)

    def __get_docstrings(self, table_name: str) -> List[str]:
        with open(join(self.path, table_name + ".sql"), "r") as f:
            lines = f.readlines()

        docstring = []
        for line in lines:
            if "--@" == line[:3]:
                text = line[3:].rstrip()
                docstring.append(text.strip().rstrip().lstrip())
        return docstring

    def __get_sql_code(self, table_name: str) -> List[str]:
        with open(join(self.path, table_name + ".sql"), "r") as f:
            lines = f.readlines()

        sql_code = ["\n\n", "The SQL statement for table and index creation is below.\n\n", "::\n"]
        for line in lines:
            if "--" not in line:
                sql_code.append(f"   {line.rstrip()}")
        return sql_code

    @staticmethod
    def readlines(filename):
        with open(filename, "r") as f:
            return [x.strip() for x in f.readlines()]


tables = [("project_database", "project_database"), ("transit_database", "transit_database")]

for table, pth in tables:
    s = CreateTablesSRC(table, pth)
    s.create()
