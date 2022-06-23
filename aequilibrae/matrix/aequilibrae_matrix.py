from functools import reduce
import importlib.util as iutil
import os
import tempfile
import uuid
from shutil import copyfile
from typing import List
import functools
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix

# Checks if we can display OMX
spec = iutil.find_spec("openmatrix")
has_omx = spec is not None
if has_omx:
    import openmatrix as omx

# CONSTANTS
VERSION = 1  # VERSION OF THE MATRIX FORMAT
INT = 0
FLOAT = 1
COMPLEX = 2
CORE_NAME_MAX_LENGTH = 50
INDEX_NAME_MAX_LENGTH = 20
MATRIX_NAME_MAX_LENGTH = 20
MATRIX_DESCRIPTION_MAX_LENGTH = 144  # Why not a tweet?
NOT_COMPRESSED = 0
COMPRESSED = 1

# TODO:  Add an aggregate method
# TODO: Add check that the index 1 (or zero) has only unique values?


# Matrix structure
#
# What:  Version | Compress flag | # cells: compressed matrix | # of zones | # of cores | # of indices (Y) | Data type |
# Size:   uint8  |    uint8      |        uint64              |  uint32    |   uint8    |     uint8        |  uint8    |
# Shape:     1   |       1       |           1                |     1      |            |       1          |     1     |
# Offset:    0   |       1       |           2                |     10     |     14     |      15          |    16     |
#
#
# What:    Data size |  matrix name | matrix description | Core names |   index names    |
# Size:     uint8    |     S20      |          S144      |    S50     |      S20         |
# Shape:      1      |      1       |            1       |  [cores]   |    [indices]     |
# Offset:     17     |     18       |          38        |     182    |  182 + 50*cores  |
#
# What:         indices          |             Matrices                 |
# Size:         uint64           |      f(Data type, Data size)         |
# Shape:     [zones, indices]    |       [zones, zones, cores]          |
# Offset:  18 + 50*cores + Y*20  |   18 + 50*cores + Y*20 + Y*zones*8   |


matrix_export_types = ["Aequilibrae matrix (*.aem)", "Comma-separated file (*.csv)"]
if has_omx:
    matrix_export_types.append("Open matrix (*.omx)")


class AequilibraeMatrix(object):
    """
    Matrix class
    """

    def __init__(self):
        """
        Creates a memory instance for a matrix, that can be used to load an existing matrix or to create an empty one
        """
        self.file_path = None
        self.dtype = None
        self.num_indices = None
        self.index_names = None
        self.compressed = NOT_COMPRESSED
        self.matrix_view = None
        self.view_names = None
        self.matrix_hash = {}
        self.index = None
        self.indices = None
        self.matrix = None
        self.matrices = None
        self.cores = 0
        self.zones = None
        self.dtype = None
        self.names = []  # type: [str]
        self.name = ""
        self.description = ""
        self.current_index = None
        self.__omx = False
        self.omx_file = None  # type: omx.File
        self.__version__ = VERSION  # Writes file version

    def save(self, names=()) -> None:
        """Saves matrix data back to file.

        If working with AEM file, it flushes data to disk.  If working with OMX, requires new names

        Args:
            *names* (:obj:`tuple(str)`, `Optional`): New names for the matrices. Required if working with OMX files"""

        if not self.__omx:
            self.matrices.flush()
            return

        if isinstance(names, str):
            names = [names]

        if len(names) != len(self.view_names):
            raise ValueError("Number of names needs to be equal to computational view")

        exists = [n for n in names if n in self.names]
        if exists:
            raise ValueError(f'Matrix(ces) "{".".join(exists)}" already exist. Choose (a) new name(s)')

        if len(self.view_names) == 1:
            self.omx_file[names[0]] = self.matrix_view[:, :]
        else:
            for i, name in enumerate(names):
                self.omx_file[name] = self.matrix_view[:, :, i]

        self.names = self.omx_file.list_matrices()
        self.computational_view(names)

    def create_empty(
        self,
        file_name: str = None,
        zones: int = None,
        matrix_names: List[str] = None,
        data_type: np.dtype = np.float64,
        index_names: List[str] = None,
        compressed: bool = False,
    ):
        """
        Creates an empty matrix in the AequilibraE format

        Args:
            *file_name* (:obj:`str`): Local path to the matrix file

            *zones* (:obj:`int`): Number of zones in the model (Integer). Maximum number of zones in a matrix is
            4,294,967,296

            *matrix_names* (:obj:`list`): A regular Python list of names of the matrix. Limit is 50 characters each.
            Maximum number of cores per matrix is 256

            *file_name* (:obj:`str`): Local path to the matrix file

            *data_type* (:obj:`np.dtype`, optional): Data type of the matrix as NUMPY data types (NP.int32, np.int64,
            np.float32, np.float64). Defaults to np.float64

            *index_names* (:obj:`list`, optional):  A regular Python list of names for indices. Limit is 20 characters
            each). Maximum number of indices per matrix is 256

            *compressed* (:obj:`bool`, optional): Whether it is a flat matrix or a compressed one (Boolean - Not yet
            implemented)

        ::

            zones_in_the_model = 3317
            names_list = ['Car trips', 'pt trips', 'DRT trips', 'bike trips', 'walk trips']

            mat = AequilibraeMatrix()
            mat.create_empty(file_name='my/path/to/file',
                             zones=zones_in_the_model,
                             matrix_names= names_list)
            mat.num_indices
          1
            mat.zones
          3317
            np.sum(mat[trips])
          0.0
        """

        self.file_path = file_name
        self.zones = zones
        self.index_names = index_names
        self.dtype = data_type

        # Matrix compression still not supported
        if compressed:
            compressed = False
            raise Warning("Matrix compression not yet supported")

        if compressed:
            self.compressed = COMPRESSED
        else:
            self.compressed = NOT_COMPRESSED

        if index_names is None:
            self.index_names = ["main_index"]
        else:
            if not isinstance(index_names, (list, tuple)):
                raise Exception("Index names need to be provided as a list")

            self.index_names = index_names
            for ind_name in index_names:
                if isinstance(ind_name, str):
                    if len(ind_name) > INDEX_NAME_MAX_LENGTH:
                        raise ValueError(
                            "Index names need to be be shorter " "than {}: {}".format(INDEX_NAME_MAX_LENGTH, ind_name)
                        )
                else:
                    raise ValueError("Index names need to be strings: " + str(ind_name))

        self.num_indices = len(self.index_names)

        if matrix_names is None:
            matrix_names = ["mat"]
        else:
            if isinstance(matrix_names, list) or isinstance(matrix_names, tuple):
                for mat_name in matrix_names:
                    if isinstance(mat_name, str):
                        if mat_name in object.__dict__:
                            raise ValueError(mat_name + " is a reserved name")
                        if len(mat_name) > CORE_NAME_MAX_LENGTH:
                            raise ValueError(
                                "Matrix names need to be be shorter "
                                "than {}: {}".format(CORE_NAME_MAX_LENGTH, mat_name)
                            )
                    else:
                        raise ValueError("Matrix core names need to be strings: " + str(mat_name))
            else:
                raise Exception("Matrix names need to be provided as a list")

        self.names = [x for x in matrix_names]
        self.cores = len(self.names)
        if None not in [self.file_path, self.zones]:
            self.__write__()

    def get_matrix(self, core: str, copy=False) -> np.ndarray:
        """
            Returns the data for a matrix core

        Args:
            *core* (:obj:`str`): name of the matrix core to be returned

            *copy* (:obj:`bool`, optional): return a copy of the data. Defaults to False

        :Returns:

            *object* (:obj:`np.ndarray`): NumPy array
        """
        if core not in self.names:
            raise AttributeError("Matrix core does not exist in this matrix")
        if self.__omx:
            return np.array(self.omx_file[core])
        else:
            mat = self.matrix[core]
            if copy:
                mat = np.copy(mat)
            return mat

    def create_from_omx(
        self,
        file_path: str,
        omx_path: str,
        cores: List[str] = None,
        mappings: List[str] = None,
        robust: bool = True,
        compressed: bool = False,
    ) -> None:
        """
        Creates an AequilibraeMatrix from an original OpenMatrix

        Args:
            *file_path* (:obj:`str`): Path for the output AequilibraEMatrix

            *omx_path* (:obj:`str`): Path to the OMX file one wants to import

            *cores* (:obj:`list`): List of matrix cores to be imported

            *mappings* (:obj:`list`): List of the matrix mappings (i.e. indices, centroid numbers) to be imported

            *robust* (:obj:`bool`, optional): Boolean for whether AequilibraE should try to adjust the names for cores
            and indices in case they are too long. Defaults to True

            *compressed* (:obj:`bool`, optional): Boolean for whether we should compress the output matrix.
            Not yet implemented


        """

        def robust_name(input_name: str, max_length: int, forbiden_names: List[str]) -> str:
            return_name = input_name
            if len(input_name) > max_length:
                return_name = input_name[:max_length]

            if return_name not in forbiden_names:
                return return_name

            for i in range(999999):
                x = len(str(i))

                if x + len(return_name) > max_length:
                    trial_name = return_name[: max_length - x] + str(i)
                else:
                    trial_name = return_name + str(i)

                if trial_name not in forbiden_names:
                    return trial_name

        if not has_omx:
            print("Open Matrix is not installed. Cannot continue")
            return

        src = omx.open_file(omx_path, "a")

        avail_cores = src.list_matrices()

        if cores is None:
            do_cores = avail_cores
        else:
            do_cores = [x for x in cores if x in avail_cores]
            if cores != do_cores:
                do_cores = [x for x in cores if x not in avail_cores]
                raise ValueError("Cores listed not available in the OMX file: {}".format(do_cores))

        avail_idx = src.list_mappings()
        if mappings is None:
            do_idx = avail_idx
            if not avail_idx:
                do_idx = ["zero_base_index"]
        else:
            do_idx = [x for x in mappings if x in avail_idx]
            if mappings != do_idx:
                do_idx = [x for x in mappings if x not in avail_idx]
                raise ValueError("Mappings/indices listed not available in the OMX file: {}".format(do_idx))

        shp = src.shape()
        if shp[0] != shp[1]:
            raise ValueError("AequilibraE only supports square matrices")
        zones = shp[0]

        if robust:
            # Use reduce as we have to keep track of which generated names have been used (to avoid collisions)
            core_names = functools.reduce(
                lambda acc, n: acc + [robust_name(n, CORE_NAME_MAX_LENGTH, acc)], do_cores, []
            )
            idx_names = functools.reduce(lambda acc, n: acc + [robust_name(n, INDEX_NAME_MAX_LENGTH, acc)], do_idx, [])
        else:
            core_names = [x for x in do_cores]
            idx_names = [x for x in do_idx]

        self.create_empty(
            file_name=file_path, zones=zones, matrix_names=core_names, index_names=idx_names, compressed=compressed
        )

        # Copy all cores
        for ncore, core in zip(core_names, do_cores):
            self.matrix[ncore][:, :] = np.array(src[core])[:, :]
        self.matrices.flush()

        # copy all indices
        if avail_idx:
            for nidx, idx in enumerate(do_idx):
                ix = np.array(list(src.mapping(idx).keys()))
                self.indices[:, nidx] = ix[:]
        else:
            self.index[:, 0] = np.arange(zones)

        self.indices.flush()

    def create_from_trip_list(self, path_to_file: str, from_column: str, to_column: str, list_cores: List[str]) -> str:
        """
        Creates an AequilibraeMatrix from a trip list csv file
        The output is saved in the same folder as the trip list file

        Args:
            *path_to_file* (:obj:`str`): Path for the trip list csv file

            *from_column* (:obj:`str`): trip list file column containing the origin zones numbers

            *from_column* (:obj:`str`): trip list file column containing the destination zones numbers

            *list_cores* (:obj:`list`): list of core columns in the trip list file

        """

        # Loading file
        trip_df = pd.read_csv(path_to_file)

        # Creating zone indices
        zones_list = sorted(list(set(list(trip_df[from_column].unique()) + list(trip_df[to_column].unique()))))
        zones_df = pd.DataFrame({"zone": zones_list, "idx": list(np.arange(len(zones_list)))})

        trip_df = trip_df.merge(
            zones_df.rename(columns={"zone": from_column, "idx": from_column + "_idx"}), on=from_column, how="left"
        ).merge(zones_df.rename(columns={"zone": to_column, "idx": to_column + "_idx"}), on=to_column, how="left")

        new_mat = AequilibraeMatrix()
        nb_of_zones = len(zones_list)
        new_mat.create_empty(file_name=path_to_file[:-4] + ".aem", zones=nb_of_zones, matrix_names=list_cores)

        for idx, core in enumerate(list_cores):
            m = (
                coo_matrix(
                    (trip_df[core], (trip_df[from_column + "_idx"], trip_df[to_column + "_idx"])),
                    shape=(nb_of_zones, nb_of_zones),
                )
                .toarray()
                .astype(np.float64)
            )
            new_mat.matrix[new_mat.names[idx]][:, :] = m[:, :]

        new_mat.save()

        print(f"AequilibraE matrix saved at {path_to_file[:-4]}.aem")
        return

    def __load_aem__(self):
        # GET File version
        self.__version__ = np.memmap(self.file_path, dtype="uint8", offset=0, mode="r+", shape=1)[0]

        if self.__version__ != VERSION:
            raise ValueError("Matrix formats do not match")

        # If matrix is compressed or not
        self.compressed = np.memmap(self.file_path, dtype="uint8", offset=1, mode="r+", shape=1)[0]

        # number matrix cells if compressed
        matrix_cells = np.memmap(self.file_path, dtype="uint64", offset=2, mode="r+", shape=1)[0]

        # Zones
        self.zones = np.memmap(self.file_path, dtype="uint32", offset=10, mode="r+", shape=1)[0]

        # Matrix cores
        self.cores = np.memmap(self.file_path, dtype="uint8", offset=14, mode="r+", shape=1)[0]

        # Matrix indices
        self.num_indices = np.memmap(self.file_path, dtype="uint8", offset=15, mode="r+", shape=1)[0]

        # Data type
        data_class = np.memmap(self.file_path, dtype="uint8", offset=16, mode="r+", shape=1)[0]

        # Data size
        data_size = np.memmap(self.file_path, dtype="uint8", offset=17, mode="r+", shape=1)[0]

        if data_class == INT:
            if data_size == 1:
                self.dtype = np.int8
            elif data_size == 2:
                self.dtype = np.int16
            elif data_size == 4:
                self.dtype = np.int32
            elif data_size == 8:
                self.dtype = np.int64
            elif data_size == 16:
                self.dtype = np.int128

        if data_class == FLOAT:
            if data_size == 2:
                self.dtype = np.float16
            elif data_size == 4:
                self.dtype = np.float32
            elif data_size == 8:
                self.dtype = np.float64
            elif data_size == 16:
                self.dtype = np.float128

        # matrix name
        self.name = np.memmap(self.file_path, dtype="S" + str(MATRIX_NAME_MAX_LENGTH), offset=18, mode="r+", shape=1)[0]

        # matrix description
        offset = 18 + MATRIX_NAME_MAX_LENGTH
        self.description = np.memmap(
            self.file_path, dtype="S" + str(MATRIX_DESCRIPTION_MAX_LENGTH), offset=offset, mode="r+", shape=1
        )[0]

        # core names
        offset += MATRIX_DESCRIPTION_MAX_LENGTH
        self.names = list(
            np.memmap(self.file_path, dtype="S" + str(CORE_NAME_MAX_LENGTH), offset=offset, mode="r+", shape=self.cores)
        )

        self.names = [x.decode("utf-8") for x in self.names]

        # Index names
        offset += CORE_NAME_MAX_LENGTH * self.cores
        self.index_names = list(
            np.memmap(
                self.file_path, dtype="S" + str(INDEX_NAME_MAX_LENGTH), offset=offset, mode="r+", shape=self.num_indices
            )
        )
        self.index_names = [x.decode("utf-8") for x in self.index_names]

        # Index
        offset += self.num_indices * INDEX_NAME_MAX_LENGTH
        self.indices = np.memmap(
            self.file_path, dtype="uint64", offset=offset, mode="r+", shape=(self.zones, self.num_indices)
        )
        self.set_index(self.index_names[0])

        # DATA
        offset += self.zones * 8 * self.num_indices
        if self.compressed:
            self.matrices = np.memmap(
                self.file_path, dtype=self.dtype, offset=offset, mode="r+", shape=(matrix_cells, self.cores + 2)
            )
        else:
            self.matrices = np.memmap(
                self.file_path, dtype=self.dtype, offset=offset, mode="r+", shape=(self.zones, self.zones, self.cores)
            )

        self.matrix = {}
        for i, v in enumerate(self.names):
            self.matrix[v] = self.matrices[:, :, i]
        self.matrix_hash = self.__builds_hash__()

    def __load_omx__(self):
        # GET File version
        self.__version__ = VERSION

        self.name = self.file_path
        self.description = "OMX MATRIX"
        self.names = self.omx_file.list_matrices()

        if len(self.omx_file) == 0:
            raise LookupError("Matrix file has no cores")

        self.index_names = self.omx_file.list_mappings()

        if len(self.index_names) == 0:
            raise LookupError("Matrix file has no indices (mappings). If you can't index it, you can't use it")

        self.num_indices = len(self.index_names)
        self.zones = len(list(self.omx_file.mapping(self.index_names[0]).keys()))

        self.cores = len(self.names)
        self.set_index(self.index_names[0])

        self.matrices = np.zeros(1)
        self.matrix = {}
        self.matrix_hash = self.__builds_hash__()

    def __write__(self):
        np.memmap(self.file_path, dtype="uint8", offset=0, mode="w+", shape=1)[0] = self.__version__

        # If matrix is compressed or not
        np.memmap(self.file_path, dtype="uint8", offset=1, mode="r+", shape=1)[0] = self.compressed

        # number matrix cells if compressed
        matrix_cells = self.zones * self.zones
        np.memmap(self.file_path, dtype="uint64", offset=2, mode="r+", shape=1)[0] = matrix_cells

        # Zones
        np.memmap(self.file_path, dtype="uint32", offset=10, mode="r+", shape=1)[0] = self.zones

        # Matrix cores
        np.memmap(self.file_path, dtype="uint8", offset=14, mode="r+", shape=1)[0] = self.cores

        # Matrix indices
        np.memmap(self.file_path, dtype="uint8", offset=15, mode="r+", shape=1)[0] = self.num_indices

        # Data type
        data_class = self.__define_data_class()
        np.memmap(self.file_path, dtype="uint8", offset=16, mode="r+", shape=1)[0] = data_class

        # Data size
        data_size = np.dtype(self.dtype).itemsize
        np.memmap(self.file_path, dtype="uint8", offset=17, mode="r+", shape=1)[0] = data_size

        # matrix name
        np.memmap(self.file_path, dtype="S" + str(MATRIX_NAME_MAX_LENGTH), offset=18, mode="r+", shape=1)[0] = self.name

        # matrix description
        offset = 18 + MATRIX_NAME_MAX_LENGTH
        np.memmap(self.file_path, dtype="S" + str(MATRIX_DESCRIPTION_MAX_LENGTH), offset=offset, mode="r+", shape=1)[
            0
        ] = self.description

        # core names
        offset += MATRIX_DESCRIPTION_MAX_LENGTH
        fp = np.memmap(
            self.file_path, dtype="S" + str(CORE_NAME_MAX_LENGTH), offset=offset, mode="r+", shape=self.cores
        )

        for i, v in enumerate(self.names):
            fp[i] = v
        fp.flush()
        del fp

        # Index names
        offset += CORE_NAME_MAX_LENGTH * self.cores
        fp = np.memmap(
            self.file_path, dtype="S" + str(INDEX_NAME_MAX_LENGTH), offset=offset, mode="r+", shape=self.num_indices
        )

        for i, v in enumerate(self.index_names):
            fp[i] = v
        fp.flush()
        del fp

        # Index
        offset += self.num_indices * INDEX_NAME_MAX_LENGTH
        self.indices = np.memmap(
            self.file_path, dtype="uint64", offset=offset, mode="r+", shape=(self.zones, self.num_indices)
        )
        self.indices.fill(0)
        self.indices.flush()
        self.set_index(self.index_names[0])

        offset += self.zones * 8 * self.num_indices
        if self.compressed:
            self.matrices = np.memmap(
                self.file_path, dtype=self.dtype, offset=offset, mode="r+", shape=(matrix_cells, self.cores + 2)
            )
        else:
            self.matrices = np.memmap(
                self.file_path, dtype=self.dtype, offset=offset, mode="r+", shape=(self.zones, self.zones, self.cores)
            )

        if np.issubdtype(self.dtype, np.integer):
            self.matrices.fill(np.iinfo(self.dtype).min)
        else:
            self.matrices.fill(np.nan)
        self.matrices.flush()

        self.matrix = {}
        for i, v in enumerate(self.names):
            self.matrix[v] = self.matrices[:, :, i]

    def set_index(self, index_to_set: str) -> None:
        """
        Sets the standard index to be the one the user wants to have be the one being used in all operations
        during run time. The first index is ALWAYS the default one every time the matrix is instantiated

        Args:
            index_to_set (:obj:`str`): Name of the index to be used. The default index name is 'main_index'

        ::

            zones_in_the_model = 3317
            names_list = ['Car trips', 'pt trips', 'DRT trips', 'bike trips', 'walk trips']
            index_list = ['tazs',  'census']

            mat = AequilibraeMatrix()
            mat.create_empty(file_name='my/path/to/file',
                             zones=zones_in_the_model,
                             matrix_names=names_list,
                             index_names =index_list )
            mat.num_indices
          2
            mat.current_index
          'tazs'
            mat.set_index('census')
            mat.current_index
          'census'
        """
        if self.__omx:
            self.index = np.array(list(self.omx_file.mapping(index_to_set).keys()))
            self.current_index = index_to_set
        else:
            if index_to_set in self.index_names:
                ind_index = self.index_names.index(index_to_set)
                self.index = self.indices[:, ind_index]
                self.current_index = index_to_set
            else:
                raise ValueError("Index {} needs to be a string or its integer index.".format(str(index_to_set)))

    def __getattr__(self, mat_name: str):
        if mat_name in object.__dict__:
            return self.__dict__[mat_name]

        if mat_name in self.names:
            return self.get_matrix(mat_name)

        raise AttributeError("No such method or matrix core! --> " + str(mat_name))

    # Transforms matrix from dense to CSR
    # def compress(self):
    #     print(self.file_path + '. Method not implemented yet. All matrices are dense')

    # Transforms matrix from CSR to dense
    # def decompress(self):
    #     print(self.file_path + '. Method not implemented yet. All matrices are dense')

    # Adds index to matrix
    # def add_index(self):
    #     print(self.file_path + '. Method not implemented yet. All indices need to exist during the matrix creation')

    # Adds index to matrix
    # def remove_index(self, index_number):
    #     print(self.file_path + '. Method not implemented. Indices are fixed on matrix creation: ' + str(index_number))

    def close(self):
        """
        Removes matrix from memory and flushes all data to disk, or closes the OMX file if that is the case
        """

        if self.__omx:
            self.omx_file.close()
        else:
            self.matrices.flush()
            self.index.flush()
        del self.matrices
        del self.index

    def export(self, output_name: str, cores: List[str] = None):
        """
        Exports the matrix to other formats. Formats currently supported: CSV, OMX

        When exporting to AEM or OMX, the user can chose to export only a set of cores, but all indices are exported

        When exporting to CSV, the active index will be used, and all cores will be exported as separate columns in
        the output file

        Args:
            *output_name* (:obj:`str`): Path to the output file

            *cores* (:obj:`list`): Names of the cores to be exported.

        ::

            zones_in_the_model = 3317
            names_list = ['Car trips', 'pt trips', 'DRT trips', 'bike trips', 'walk trips']

            mat = AequilibraeMatrix()
            mat.create_empty(file_name='my/path/to/file', zones=zones_in_the_model, matrix_names= names_list)
            mat.cores
          ['Car trips', 'pt trips', 'DRT trips', 'bike trips', 'walk trips']

            mat.export('my_new_path', ['Car trips', 'bike trips'])

            mat2 = AequilibraeMatrix()
            mat2.load('my_new_path')
            mat2.cores
          ['Car trips', 'bike trips']
        """

        if self.__omx:
            raise NotImplementedError("This operation does not make sense for OMX matrices")

        fname, file_extension = os.path.splitext(output_name.upper())

        if file_extension == ".OMX":
            if not has_omx:
                raise ValueError("Open Matrix is not installed. Cannot continue")

        if file_extension not in [".AEM", ".CSV", ".OMX"]:
            raise ValueError("File extension {} not implemented yet".format(file_extension))

        if cores is None:
            cores = self.names

        if file_extension == ".AEM":
            self.copy(output_name=output_name, cores=cores)

        elif file_extension == ".OMX":
            omx_export = omx.open_file(output_name, "w")
            for c in cores:
                omx_export[c] = self.matrix[c]

            for i, idx in enumerate(self.index_names):
                omx_export.create_mapping(idx, self.indices[:, i])
            omx_export.close()

        elif file_extension == ".CSV":

            def f(name):
                coo = coo_matrix(self.matrix[name])
                data = {"row": self.index[coo.row], "column": self.index[coo.col], name: coo.data}
                return pd.DataFrame(data).set_index(["row", "column"])

            dfs = [f(name) for name in self.names]
            df = reduce(lambda a, b: a.join(b, how="outer"), dfs)
            df.to_csv(output_name, index=True)

    def load(self, file_path: str):
        """
        Loads matrix from disk. All cores and indices are load. First index is default

        Args:
            file_path (:obj:`str`): Path to AEM or OMX file on disk

        ::

            zones_in_the_model = 3317
            names_list = ['Car trips', 'pt trips', 'DRT trips', 'bike trips', 'walk trips']

            mat = AequilibraeMatrix()
            mat.create_empty(file_name='my/path/to/file', zones=zones_in_the_model, matrix_names= names_list)
            mat.close()

            mat2 = AequilibraeMatrix()
            mat2.load('my/path/to/file.omx')
            mat2.zones
          3317
        """

        self.file_path = file_path

        if os.path.splitext(file_path)[-1].upper() == ".OMX":
            self.__omx = True
            self.omx_file = omx.open_file(file_path, "a")
            self.__load_omx__()
        else:
            self.__load_aem__()

    def is_omx(self):
        """Returns True if matrix data source is OMX, False otherwise"""
        return self.__omx

    def computational_view(self, core_list: List[str] = None):
        """
        Creates a memory view for a list of matrices that is compatible with Cython memory buffers

        It allows for AequilibraE matrices to be used in all parallelized algorithms within AequilibraE

        In case of OMX matrices, the computational view is held only in memory

        Args:
            *core_list* (:obj:`list`): List with the names of all matrices that need to be in the buffer

        ::

            zones_in_the_model = 3317
            names_list = ['Car trips', 'pt trips', 'DRT trips', 'bike trips', 'walk trips']

            mat = AequilibraeMatrix()
            mat.create_empty(file_name='my/path/to/file', zones=zones_in_the_model, matrix_names= names_list)
            mat.computational_view(['bike trips', 'walk trips'])
            mat.view_names
          ['bike trips', 'walk trips']
        """

        self.matrix_view = None
        self.view_names = None
        if core_list is None:
            core_list = self.names
        else:
            if isinstance(core_list, str):
                core_list = [core_list]
            if isinstance(core_list, list):
                for i in core_list:
                    if i not in self.names:
                        raise ValueError("Matrix core   {}   no available on this matrix".format(i))
            else:
                raise TypeError("Please provide a list of matrices")

        if self.__omx:
            self.view_names = core_list
            if len(core_list) == 1:
                self.matrix_view = np.array(self.omx_file[core_list[0]])
            else:
                self.matrix_view = np.empty((self.zones, self.zones, len(core_list)))
                for i, core in enumerate(core_list):
                    self.matrix_view[:, :, i] = np.array(self.omx_file[core])
        else:
            # Check if matrices are adjacent
            if len(core_list) > 1:
                for i, x in enumerate(core_list[1:]):
                    k = self.names.index(x)  # index of the first element
                    k0 = self.names.index(core_list[i])  # index of the first element
                    if k - k0 != 1:
                        raise ValueError("Matrix cores {} and {} are not adjacent".format(core_list[i - 1], x))

            self.view_names = core_list
            idx1 = self.names.index(core_list[0])
            if len(core_list) == 1:
                self.matrix_view = self.matrices[:, :, idx1]
            elif len(core_list) > 1:
                idx2 = self.names.index(core_list[-1])
                self.matrix_view = self.matrices[:, :, idx1 : idx2 + 1]

    def copy(
        self, output_name: str = None, cores: List[str] = None, names: List[str] = None, compress: bool = None
    ) -> None:
        """
        Copies a list of cores (or all cores) from one matrix file to another one

        Args:
            *output_name* (:obj:`str`): Name of the new matrix file

            *cores* (:obj:`list`):List of the matrix cores to be copied

            *names* (:obj:`list`, optional): List with the new names for the cores. Defaults to current names

            *compress* (:obj:`bool`, optional): Whether you want to compress the matrix or not. Defaults to False
            Not yet implemented

        ::

            zones_in_the_model = 3317
            names_list = ['Car trips', 'pt trips', 'DRT trips', 'bike trips', 'walk trips']

            mat = AequilibraeMatrix()
            mat.create_empty(file_name='my/path/to/file', zones=zones_in_the_model, matrix_names= names_list)
            mat.copy('my/new/path/to/file', cores=['bike trips', 'walk trips'], names=['bicycle', 'walking'])

            mat2 = AequilibraeMatrix()
            mat2.load('my/new/path/to/file')
            mat.cores
          ['bicycle', 'walking']
        """

        if output_name is None:
            output_name = self.random_name()

        if self.__omx:
            mcores = cores
            if mcores is None:
                mcores = self.names
            temp = AequilibraeMatrix()

            fp = self.random_name()
            temp.create_from_omx(file_path=fp, omx_path=self.file_path, cores=mcores, mappings=self.index_names)
            output = temp.copy(output_name, cores, names, compress)
            if names is None:
                names = mcores
            if self.view_names is not None:
                view_names = [names[self.view_names.index(nm)] for nm in self.view_names]
                output.computational_view(view_names)
            temp.close()
            del temp
        else:
            if cores is None:
                copyfile(self.file_path, output_name)
                output = AequilibraeMatrix()
                output.load(output_name)
                if self.view_names is not None:
                    output.computational_view(self.view_names)

                if compress is not None:
                    if compress != self.compressed:
                        if compress:
                            output.compress()
                        else:
                            output.decompress()
            else:
                if compress is None:
                    compress = self.compressed

                if not isinstance(cores, list):
                    raise ValueError("Cores need to be presented as list")

                for i in cores:
                    if i not in self.names:
                        raise ValueError("Matrix core {} not available on this matrix".format(i))

                if names is None:
                    names = cores
                else:
                    if not isinstance(names, list):
                        raise ValueError("Names need to be presented as list")

                    if len(names) != len(cores):
                        raise ValueError("Number of cores to cpy and list of names are not compatible")

                output = AequilibraeMatrix()
                output.create_empty(
                    file_name=output_name,
                    zones=self.zones,
                    matrix_names=names,
                    index_names=self.index_names,
                    data_type=self.dtype,
                    compressed=bool(compress),
                )

                output.indices[:] = self.indices[:]
                for i, c in enumerate(cores):
                    output.matrices[:, :, i] = self.matrices[:, :, self.names.index(c)]
                if self.view_names is not None:
                    view_names = [names[self.view_names.index(nm)] for nm in self.view_names]
                    output.computational_view(view_names)

        output.matrix_hash = output.__builds_hash__()
        output.matrices.flush()
        return output

    def rows(self) -> np.ndarray:
        """
        Returns row vector for the matrix in the computational view

        Computational view needs to be set to a single matrix core

        :Returns:

            *object* (:obj:`np.ndarray`): the row totals for the matrix currently on the computational view

        ::

            mat = AequilibraeMatrix()
            mat.load('my/path/to/file')
            mat.computational_view(mat.cores[0])
            mat.rows()
          array([0.,...,0.])
        """
        return self.__vector(axis=0)

    def columns(self) -> np.ndarray:
        """
        Returns column vector for the matrix in the computational view

        Computational view needs to be set to a single matrix core

        :Returns:

            *object* (:obj:`np.ndarray`): the column totals for the matrix currently on the computational view

        ::

            mat = AequilibraeMatrix()
            mat.load('my/path/to/file')
            mat.computational_view(mat.cores[0])
            mat.columns()
          array([0.34,.0.,...,14.03])
        """
        return self.__vector(axis=1)

    def nan_to_num(self):
        """
        Converts all NaN values in all cores in the computational view to zeros

        ::

            mat = AequilibraeMatrix()
            mat.load('my/path/to/file')
            mat.computational_view(mat.cores[0])
            mat.nan_to_num()
        """

        if self.__omx:
            raise NotImplementedError("This operation does not make sense for OMX matrices")

        if np.issubdtype(self.dtype, np.floating) and self.matrix_view is not None:
            for m in self.view_names:
                self.matrix[m][:, :] = np.nan_to_num(self.matrix[m])[:, :]

    def __vector(self, axis: int):
        if self.view_names is None:
            raise ReferenceError("Matrix is not set for computation")
        if len(self.view_names) > 1:
            raise ValueError("Vector for a multi-core matrix is ambiguous")
        return self.matrix_view.astype(float).sum(axis=axis)[:]

    def __builds_hash__(self):
        return {self.index[i]: i for i in range(self.zones)}

    def __define_data_class(self):
        if self.__omx:
            raise NotImplementedError("This operation does not make sense for OMX matrices")

        if np.issubdtype(self.dtype, np.floating):
            data_class = FLOAT
        elif np.issubdtype(self.dtype, np.integer):
            data_class = INT
        else:
            raise ValueError(
                "Data type not supported. You can choose Integers of 8, 16, 32 and 64 bits, "
                "or floats with 16, 32 or 64 bits"
            )
        return data_class

    def setName(self, matrix_name: str):
        """
        Sets the name for the matrix itself

        Args:
            *matrix_name* (:obj:`str`): matrix name. Maximum length is 50 characters

        ::

            mat = AequilibraeMatrix()
            mat.load('my/path/to/file')
            mat.setName('This is my example')
            mat.name
          'This is my example'
        """
        if self.__omx:
            raise NotImplementedError("This operation does not make sense for OMX matrices")

        if matrix_name is not None:
            if len(str(matrix_name)) > MATRIX_NAME_MAX_LENGTH:
                matrix_name = str(matrix_name)[0:MATRIX_NAME_MAX_LENGTH]

            np.memmap(self.file_path, dtype="S" + str(MATRIX_NAME_MAX_LENGTH), offset=18, mode="r+", shape=1)[
                0
            ] = matrix_name

    def setDescription(self, matrix_description: str):
        """
        Sets description for the matrix

        Args:
            *matrix_description* (:obj:`str`): Text with matrix description . Maximum length is 144 characters

        ::

            mat = AequilibraeMatrix()
            mat.load('my/path/to/file')
            mat.setDescription('This is some text about this matrix of mine')
            mat.description
          'This is some text about this matrix of mine'
        """
        if self.__omx:
            raise NotImplementedError("This operation does not make sense for OMX matrices")

        if matrix_description is not None:
            if len(str(matrix_description)) > MATRIX_DESCRIPTION_MAX_LENGTH:
                matrix_description = str(matrix_description)[0:MATRIX_DESCRIPTION_MAX_LENGTH]

            np.memmap(
                self.file_path,
                dtype="S" + str(MATRIX_NAME_MAX_LENGTH),
                offset=18 + MATRIX_NAME_MAX_LENGTH,
                mode="r+",
                shape=1,
            )[0] = matrix_description

    @staticmethod
    def random_name() -> str:
        """
        Returns a random name for a matrix with root in the temp directory of the user

        ::

            name = AequilibraeMatrix().random_name()
          '/tmp/Aequilibrae_matrix_54625f36-bf41-4c85-80fb-7fc2e3f3d76e.aem'
        """
        return os.path.join(tempfile.gettempdir(), f"Aequilibrae_matrix_{uuid.uuid4()}.aem")
