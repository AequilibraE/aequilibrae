import os
import sqlite3
import tempfile
import uuid
import warnings

import numpy as np
import pandas as pd
from numpy.lib.format import open_memmap

MEMORY = 1
DISK = 0
data_export_types = ["aed", "csv", "sqlite"]


class AequilibraeData(object):
    """AequilibraE dataset. This is deprecated in favor of Pandas DataFrames

    .. code-block:: python

        >>> from aequilibrae.matrix.aequilibrae_data import AequilibraeData

        >>> args = {"file_path": os.path.join(my_folder_path, "vectors.aed"),
        ...         "entries": 3,
        ...         "field_names": ["production"],
        ...         "data_types": [np.float64]}

        >>> vectors = AequilibraeData()
        >>> vectors.create_empty(**args)

        >>> vectors.production[:] = [1, 4, 7]
        >>> vectors.index[:] = [1, 2, 3]

        >>> vectors.export(os.path.join(my_folder_path, "vectors.csv"))

        >>> reload_dataset = AequilibraeData()
        >>> reload_dataset.load(os.path.join(my_folder_path, "vectors.aed"))
        >>> reload_dataset.production
        memmap([1., 4., 7.])
    """

    def __init__(self):
        self.data = None
        self.file_path = None
        self.entries = None
        self.fields = None
        self.num_fields = None
        self.data_types = None
        self.aeq_index_type = None
        self.memory_mode = None

    @classmethod
    def empty(cls, *args, **kwargs):
        instance = cls()
        instance.create_empty(*args, **kwargs)
        return instance

    def create_empty(
        self, file_path=None, entries=1, field_names=None, data_types=None, memory_mode=False, fill=None, index=None
    ):
        """
        Creates a new empty dataset

        :Arguments:
            **file_path** (:obj:`str`, *Optional*): Full path for the output data file. If 'memory_mode' is ``False``
            and path is missing, then the file is created in the temp folder

            **entries** (:obj:`int`, *Optional*): Number of records in the dataset. Default is 1

            **field_names** (:obj:`list`, *Optional*): List of field names for this dataset. If no list is provided, the
            field 'data' will be created

            **data_types** (:obj:`np.dtype`, *Optional*): List of data types for the dataset.
            Types need to be NumPy data types (e.g. ``np.int16``, ``np.float64``).
            If no list of types are provided, type will be ``np.float64``

            **memory_mode** (:obj:`bool`, *Optional*): If ``True``, dataset will be kept in memory.
            If ``False``, the dataset will be a memory-mapped numpy array
        """

        if file_path is not None or memory_mode:
            if field_names is None:
                field_names = ["data"]

            if data_types is None:
                data_types = [np.float64] * len(field_names)

            self.file_path = file_path
            self.entries = entries
            self.fields = field_names
            self.data_types = data_types
            self.aeq_index_type = np.uint64

            if memory_mode:
                self.memory_mode = MEMORY
            else:
                self.memory_mode = DISK
                if self.file_path is None:
                    self.file_path = self.random_name()

            # Consistency checks
            if not isinstance(self.fields, list):
                raise ValueError('Titles for fields, "field_names", needs to be a list')

            if not isinstance(self.data_types, list):
                raise ValueError('Data types, "data_types", needs to be a list')

            for field in self.fields:
                if not isinstance(field, str):
                    raise TypeError(f"{field} is not a string. You cannot use it as a field name")
                if not field.isidentifier():
                    raise Exception(field + " is a not a valid identifier name. You cannot use it as a field name")
                if field in object.__dict__:
                    raise Exception(field + " is a reserved name. You cannot use it as a field name")

            self.num_fields = len(self.fields)

            dtype = [("index", self.aeq_index_type)] + list(zip(self.fields, self.data_types))

            # the file
            if self.memory_mode:
                self.data = np.recarray((self.entries,), dtype=dtype)
            else:
                self.data = open_memmap(self.file_path, mode="w+", dtype=dtype, shape=(self.entries,))

            if fill is not None:
                [self.data[f].fill(fill) for f in self.fields]
            if index is not None:
                self.index[:] = index[:]

    def __getattr__(self, field_name):
        if field_name in object.__dict__:
            return self.__dict__[field_name]

        if self.data is not None:
            if field_name in self.fields:
                return self.data[field_name]

            if field_name == "index":
                return self.data["index"]

            raise AttributeError("No such method or data field! --> " + str(field_name))
        else:
            raise AttributeError("Data container is empty")

    def load(self, file_path):
        """
        Loads dataset from file

        :Arguments:
            **file_path** (:obj:`str`): Full file path to the AequilibraeData to be loaded
        """
        f = open(file_path)
        self.file_path = os.path.realpath(f.name)
        f.close()

        # Map in memory and load data names plus dimensions
        self.data = open_memmap(self.file_path, mode="r+")

        self.entries = self.data.shape[0]
        self.fields = [x for x in self.data.dtype.fields if x != "index"]
        self.num_fields = len(self.fields)
        self.data_types = [self.data[x].dtype.type for x in self.fields]

    def to_pandas(self) -> pd.DataFrame:
        """
        Converts the dataset to a Pandas DataFrame

        :Returns:
            **data** (:obj:`pd.DataFrame`): Pandas DataFrame with the dataset
        """
        return pd.DataFrame(self.data, index=self.index)

    def export(self, file_name, table_name="aequilibrae_table"):
        """
        Exports the dataset to another format. Supports CSV and SQLite

        :Arguments:
            **file_name** (:obj:`str`): File name with PATH and extension (csv, or sqlite3, sqlite or db)

            **table_name** (:obj:`str`): It only applies if you are saving to an SQLite table. Otherwise ignored
        """
        DeprecationWarning("This method is deprecated and will be removed in the future. Use Pandas DataFrames instead")
        file_type = os.path.splitext(file_name)[1]
        headers = ["index"]
        headers.extend(self.fields)

        if file_type.lower() == ".aed":
            dtype = [("index", self.aeq_index_type)]
            dtype.extend([(self.fields[i], self.data_types[i]) for i in range(self.num_fields)])
            data = open_memmap(file_name, mode="w+", dtype=dtype, shape=(self.entries,))
            for field in data.dtype.names:
                data[field] = self.data[field]
            data.flush()
            del data

        elif file_type.lower() == ".csv":
            fmt = "%d"
            for dt in self.data_types:
                if np.issubdtype(dt, np.floating):
                    fmt += ",%f"
                elif np.issubdtype(dt, np.integer):
                    fmt += ",%d"
            data = np.array(self.data, copy=True)
            for nm in self.data.dtype.names:
                np.nan_to_num(data[nm], copy=False)

            np.savetxt(file_name, data[np.newaxis, :][0], delimiter=",", fmt=fmt, header=",".join(headers), comments="")

        elif file_type.lower() in [".sqlite", ".sqlite3", ".db"]:
            # Connecting to the database file
            conn = sqlite3.connect(file_name)
            c = conn.cursor()
            # Creating the table, but before deletes if the table exists
            c.execute("""DROP TABLE IF EXISTS """ + table_name)
            fi = ""
            qm = "?"
            for f in headers[1:]:
                fi += ", " + f + " REAL"
                qm += ", ?"

            c.execute("""CREATE TABLE """ + table_name + """ (link_id INTEGER PRIMARY KEY""" + fi + ")" "")
            c.execute("BEGIN TRANSACTION")
            c.executemany("INSERT INTO " + table_name + " VALUES (" + qm + ")", self.data)
            c.execute("END TRANSACTION")
            conn.commit()
            conn.close()

    @staticmethod
    def random_name():
        """
        Returns a random name for a dataset with root in the temp directory of the user

        .. code-block:: python

            >>> from aequilibrae.matrix.aequilibrae_data import AequilibraeData

            >>> dataset = AequilibraeData()
            >>> dataset.random_name() # doctest: +ELLIPSIS
            '/tmp/Aequilibrae_data_...'
        """
        return os.path.join(tempfile.gettempdir(), f"Aequilibrae_data_{uuid.uuid4()}.aed")
