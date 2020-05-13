import sqlite3
from copy import deepcopy
from aequilibrae.project.project_creation import create_about_table


class About:
    """Provides an interface for querying and editing the **about** table of an AequilibraE project"""

    def __init__(self, conn: sqlite3.Connection):
        self.__characteristics = []
        self.__conn = conn
        if self.__has_about():
            self.__load()

    def create(self):
        """Creates the 'about' table for project files that did not previously contain it"""
        if not self.__has_about():
            create_about_table(self.__conn)
            self.__load()

    def add_info_field(self, info_field: str) -> None:
        """Adds new information field to the model

            Args:
                *info_field* (:obj:`str`): Name of the desired information field to be added.  Has to be a valid
                Python VARIABLE name (i.e. letter as first character, no spaces and no special characters)

            ::

                p = Project()
                p.open('my/project/folder')
                p.about.add_info_field('my_super_relevant_field')
                p.about.my_super_relevant_field = 'super relevant information'
                p.about.write_back()
        """

        if info_field[0].isalpha() and ' ' not in info_field:
            sql = "INSERT INTO 'about' (infoname) VALUES(?)"
            curr = self.__conn.cursor()
            curr.execute(sql, info_field)
            self.__conn.commit()
        else:
            raise ValueError(f'{info_field} is not valid as a metadata field.')

    def write_back(self):
        """Saves the information parameters back to the project database

            ::

                p = Project()
                p.open('my/project/folder')
                p.about.description = 'This is the example project. Do not use for forecast'
                p.about.write_back()
        """
        curr = self.__conn.cursor()
        for k in self.__characteristics:
            v = self.__dict__[k]
            curr.execute(f"UPDATE 'about' set infovalue = '{v}' where infoname='{k}'")
        self.__conn.commit()

    def __has_about(self):
        curr = self.__conn.cursor()
        curr.execute("SELECT name FROM sqlite_master WHERE type='table';")
        if 'about' in [x[0] for x in curr.fetchall()]:
            return True
        return False

    def __load(self):
        self.__characteristics = []
        curr = self.__conn.cursor()
        curr.execute("select infoname, infovalue from 'about'")

        for x in curr.fetchall():
            self.__characteristics.append(x[0])
            self.__dict__[x[0]] = x[1]
