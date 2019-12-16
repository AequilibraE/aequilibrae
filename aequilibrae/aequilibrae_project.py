"""
AequilibraE Project
"""
import importlib.util as iutil
import sqlite3
import os
import numpy as np
import logging
from tempfile import gettempdir
from reference_files import spatialite_database
from utils import WorkerThread

# from ..utils import WorkerThread
from parameters import Parameters

spec = iutil.find_spec("PyQt5")
pyqt = spec is not None
if pyqt:
    from PyQt5.QtCore import pyqtSignal as SIGNAL


class AequilibraEProject(WorkerThread):
    """
    The AequilibraE Project is a wrapper around a the SQLite file that holds all the project information
    and Python
    """

    if pyqt:
        assignment = SIGNAL(object)

    def __init__(self):
        self.model_database = None

        # class objects that the user should not be access to
        self.__model_conn = None

    def load_model(self, model_file: str):
        self.model_database = None
        if not os.path.isfile(model_file):
            return "File does not exist"

        self.model_database = model_file
        self.__model_conn = sqlite3.connect(self.model_database)
        self.__cursor = self.__model_conn.cursor()

    def metadata(self, item: str) -> str:
        self.__run_query__("select * from model_metadata")
        data1 = self.__cursor.fetchall()
        for db_item, db_text in data1:
            if db_item.upper() == item.upper():
                return db_text
        raise ValueError("No metadata item corresponds to {}".format(item))

    def write_metadata(self, item: str, text: str):
        self.__cursor.execute("DELETE from model_metadata WHERE item=?", (item,))
        self.__cursor.execute("insert into model_metadata values (?,?)", (item, text))

    def __run_query__(self, qry):
        self.__cursor.execute(qry)
