import string
import uuid
from os.path import join, dirname, realpath

from aequilibrae.project.project_creation import run_queries_from_sql_file
from aequilibrae.utils.db_utils import commit_and_close


class About:
    """Provides an interface for querying and editing the **about** table of an AequilibraE project

    .. code-block:: python

        >>> project = create_example(project_path)

        # Adding a new field and saving it
        >>> project.about.add_info_field('my_super_relevant_field')
        >>> project.about.my_super_relevant_field = 'super relevant information'
        >>> project.about.write_back()

        # changing the value for an existing value/field
        >>> project.about.scenario_name = 'Just a better scenario name'
        >>> project.about.write_back()

    """

    def __init__(self, project):
        self.__characteristics = []
        self.__original = {}
        self.__path_to_file = project.path_to_file
        self.logger = project.logger

        with project.db_connection as conn:
            if self.__has_about(conn):
                self.__load(conn)

    def create(self):
        """Creates the 'about' table for project files that did not previously contain it"""

        with commit_and_close(self.__path_to_file, spatial=True) as conn:
            if not self.__has_about(conn):
                qry_file = join(dirname(realpath(__file__)), "database_specification", "tables", "about.sql")
                run_queries_from_sql_file(conn, self.logger, qry_file)

            sql = "SELECT count(*) as num_records from about;"
            if conn.execute(sql).fetchone()[0] == 0:
                conn.execute(f"UPDATE 'about' set infovalue='{uuid.uuid4().hex}' where infoname='project_ID'")
                conn.execute("UPDATE 'about' set infovalue='right' where infoname='driving_side'")
                self.__load(conn)
            else:
                self.logger.warning("About table already exists. Nothing was done")

    def list_fields(self) -> list:
        """Returns a list of all characteristics the about table holds"""

        return list(self.__characteristics)

    def add_info_field(self, info_field: str) -> None:
        """Adds new information field to the model

        :Arguments:
            **info_field** (:obj:`str`): Name of the desired information field to be added. Has to be a valid
            Python VARIABLE name (i.e. letter as first character, no spaces and no special characters)

        .. code-block:: python

            >>> project = create_example(project_path)

            >>> project.about.add_info_field('a_cool_field')
            >>> project.about.a_cool_field = 'super relevant information'
            >>> project.about.write_back()
        """
        allowed = string.ascii_lowercase + "_"
        has_forbidden = [x for x in info_field if x not in allowed]

        if has_forbidden:
            raise ValueError(f"{info_field} is not valid as a metadata field. Should be a lower case ascii letter or _")

        with commit_and_close(self.__path_to_file) as conn:
            conn.execute("INSERT INTO 'about' (infoname) VALUES(?)", [info_field])
        self.__characteristics.append(info_field)
        self.__original[info_field] = None

    def write_back(self):
        """Saves the information parameters back to the project database

        .. code-block:: python

            >>> project = create_example(project_path)

            >>> project.about.description = 'This is the example project. Do not use for forecast'
            >>> project.about.write_back()
        """
        with commit_and_close(self.__path_to_file) as conn:
            for k in self.__characteristics:
                v = self.__dict__[k]
                if v != self.__original[k]:
                    conn.execute("UPDATE 'about' set infovalue = ? where infoname=?", [v, k])
                    self.logger.info(f"Updated {k} on About_Table to {v}")

    def __has_about(self, conn):
        sql = "SELECT name FROM sqlite_master WHERE type='table';"
        return any("about" in x[0] for x in conn.execute(sql).fetchall())

    def __load(self, conn):
        self.__characteristics = []
        sql = "select infoname, infovalue from 'about'"
        for x in conn.execute(sql).fetchall():
            self.__characteristics.append(x[0])
            self.__dict__[x[0]] = x[1]
            self.__original[x[0]] = x[1]
