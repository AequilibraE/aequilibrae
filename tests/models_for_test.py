import uuid
from datetime import datetime
from os.path import join
from pathlib import Path
from shutil import copytree
from tempfile import gettempdir

import pandas as pd

from aequilibrae.project import Project
from aequilibrae.project.project_creation import remove_triggers
from aequilibrae.utils.db_utils import read_and_close
from .data import no_triggers_project


class ModelsTest:
    today = datetime.today().strftime("%Y-%m-%d")
    path_no_trigger = Path(gettempdir()) / f"aeq_test_no_trigger_base_{today}"

    def __init__(self):
        pass

    def __create_no_triggers(self):
        if not self.path_no_trigger.exists():
            proj = Project()
            proj.new(str(self.path_no_trigger))
            remove_triggers(proj.conn, proj.logger, db_type="network")
            tables = ["link_types", "nodes", "links"]
            with read_and_close(join(no_triggers_project, "project_database.sqlite")) as conn:
                for tbl in tables:
                    df = pd.read_sql(f"Select * from {tbl}", conn)
                    cols = pd.read_sql(f"Select * from {tbl}", proj.conn).columns
                    columns = [col for col in df.columns if col in cols]
                    df[columns].to_sql(tbl, proj.conn, if_exists="append", index=False)

    def no_triggers(self) -> Project:
        self.__create_no_triggers()
        temp_proj_folder = join(gettempdir(), f"aeq_test_{uuid.uuid4().hex}")
        copytree(self.path_no_trigger, temp_proj_folder)
        return Project.from_path(temp_proj_folder)
