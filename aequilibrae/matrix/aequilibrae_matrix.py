import importlib.util as iutil
import os
import tempfile
import uuid
from shutil import copyfile
from typing import List
import functools

import numpy as np

""""""
"""-----------------------------------------------------------------------------------------------------------
Package:    AequilibraE
Name:       AequilibraE Matrix
Purpose:    Implements a new class to represent multi-layer matrices
Original Author:  Pedro Camargo (c@margo.co)
Contributors: Pedro Camargo
Last edited by: Pedro Camargo
Website:    www.AequilibraE.com
Repository:  https://github.com/AequilibraE/AequilibraE
Created:    2017-10-02
Updated:    2018-07-08
Copyright:   (c) AequilibraE authors
Licence:     See LICENSE.TXT
-----------------------------------------------------------------------------------------------------------"""

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
        self.cores = None
        self.zones = None
        self.dtype = None
        self.names = None
        self.name = None
        self.description = None
        self.current_index = None
        self.__version__ = VERSION  # Writes file version

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

        Parameters
        ----------
        file_name: string
            Local path to the matrix file

        zones: integer
            Number of zones in the model (Integer). Maximum number of zones in a matrix is 4,294,967,296

        matrix_names: list
            A regular Python list of names of the matrix. Limit is 50 characters each. Maximum number of cores per
            matrix is 256

        data_type: np.dtype, optional
            Data type of the matrix as NUMPY data types (NP.int32, np.int64, np.float32, np.float64).
            Dafaultis np.float64

        index_names: list, optional
            A regular Python list of names for indices. Limit is 20 characters each).
            Maximum number of indices per matrix is 256

        compressed: bool, optional
            Whether it is a flat matrix or a compressed one(Boolean - Not yet implemented)

        ------------------------------------------------------------
        Example

        >>> zones_in_the_model = 3317
        >>> names_list = ['Car trips', 'pt trips', 'DRT trips', 'bike trips', 'walk trips']

        >>> mat = AequilibraeMatrix()
        >>> mat.create_empty(file_name='my/path/to/file', zones=zones_in_the_model, matrix_names= names_list)
        >>> mat.num_indices
        1
        >>> mat.zones
        3317
        >>> np.sum(mat[trips])
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
            if isinstance(index_names, list) or isinstance(index_names, tuple):
                self.index_names = index_names
                for ind_name in index_names:
                    if isinstance(ind_name, str):
                        if len(ind_name) > INDEX_NAME_MAX_LENGTH:
                            raise ValueError(
                                "Index names need to be be shorter "
                                "than {}: {}".format(INDEX_NAME_MAX_LENGTH, ind_name)
                            )
                    else:
                        raise ValueError("Index names need to be strings: " + str(ind_name))
            else:
                raise Exception("Index names need to be provided as a list")
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

    def create_from_omx(
        self,
        file_path: str,
        omx_path: str,
        cores: List[str] = None,
        mappings: List[str] = None,
        robust: bool = True,
        compressed: bool = False,
    ):
        """
        :param file_path: Path for the output AequilibraEMatrix
        :param omx_path: Path to the OMX file one wants to import
        :param cores: List of matrix cores to be imported
        :param mappings: List of the matrix mappings (i.e. indices, centroid numbers) to be imoprted
        :param: robust: Boolean for whether AequilibraE should try to adjust the names for cores and indices in case they are too long
        :param: compressed
        :return:
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

        spec = iutil.find_spec("openmatrix")
        if spec is None:
            print("Open Matrix is not installed. Cannot continue")
            return

        import openmatrix as omx

        src = omx.open_file(omx_path, "r")

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
            for nidx, idx in zip(idx_names, do_idx):
                ix = np.array(list(src.mapping(idx).keys()))
                self.indices[nidx][:] = ix[:]
        else:
            self.index[:] = np.arange(zones)

        self.indices.flush()

    def __load__(self):
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
        data_class = self.define_data_class()
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

    def set_index(self, index_to_set: str):
        """
                Sets the standard index to be the one the user wants to have be the one being used in all operations
                during run time. The first index is ALWAYS the default one every time the matrix is instantiated

                Parameters
                ----------
                index_to_set: string
                    Name of the index to be used. The default index name is 'main_index'

                ------------------------------------------------------------
                Example

                >>> zones_in_the_model = 3317
                >>> names_list = ['Car trips', 'pt trips', 'DRT trips', 'bike trips', 'walk trips']
                >>> index_list = ['tazs',  'census']

                >>> mat = AequilibraeMatrix()
                >>> mat.create_empty(file_name='my/path/to/file', zones=zones_in_the_model, matrix_names=names_list, index_names =index_list )
                >>> mat.num_indices
                2
                >>> mat.current_index
                'tazs'
                >>> mat.set_index('census')
                >>> mat.current_index
                'census'
                """

        if index_to_set in self.index_names:
            ind_index = self.index_names.index(index_to_set)
            self.index = self.indices[:, ind_index]
            self.current_index = index_to_set
        else:
            raise ValueError("Index {} needs to be a string or its integer index.".format(str(index_to_set)))

    def __getattr__(self, mat_name):
        if mat_name in object.__dict__:
            return self.__dict__[mat_name]

        if mat_name in self.names:
            return self.matrix[mat_name]

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
        Removes matrix from memory and flushes all data to disk
        """

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

        Parameters
        ----------
        output_name: Name of the output file

        cores: Names of the cores to be exported.

         ------------------------------------------------------------
        Example

        >>> zones_in_the_model = 3317
        >>> names_list = ['Car trips', 'pt trips', 'DRT trips', 'bike trips', 'walk trips']

        >>> mat = AequilibraeMatrix()
        >>> mat.create_empty(file_name='my/path/to/file', zones=zones_in_the_model, matrix_names= names_list)
        >>> mat.cores
        ['Car trips', 'pt trips', 'DRT trips', 'bike trips', 'walk trips']

        >>> mat.export('my_new_path', ['Car trips', 'bike trips'])

        >>> mat2 = AequilibraeMatrix()
        >>> mat2.load('my_new_path')
        >>> mat2.cores
        ['Car trips', 'bike trips']
        """
        fname, file_extension = os.path.splitext(output_name.upper())

        if file_extension == ".OMX":
            spec = iutil.find_spec("openmatrix")
            if spec is None:
                raise ValueError("Open Matrix is not installed. Cannot continue")
            import openmatrix as omx

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
            names = self.view_names
            self.computational_view(cores)
            output = open(output_name, "w")

            titles = ["row", "column"]
            for core in self.view_names:
                titles.append(core)
            output.write(",".join(titles))

            for i in range(self.zones):
                for j in range(self.zones):
                    record = [self.index[i], self.index[j]]
                    if len(self.view_names) > 1:
                        record.extend(self.matrix_view[i, j, :])
                    else:
                        record.append(self.matrix_view[i, j])
                    output.write(",".join(str(x) for x in record))
            output.flush()
            output.close()
            self.computational_view(names)

    def load(self, file_path: str):
        """
                Loads matrix from disk. All cores and indices are load. First index is default

                Parameters
                ----------
                file_path: Path to AEM file on disk
                ------------------------------------------------------------
                Example

                >>> zones_in_the_model = 3317
                >>> names_list = ['Car trips', 'pt trips', 'DRT trips', 'bike trips', 'walk trips']

                >>> mat = AequilibraeMatrix()
                >>> mat.create_empty(file_name='my/path/to/file', zones=zones_in_the_model, matrix_names= names_list)
                >>> mat.close()

                >>> mat2 = AequilibraeMatrix()
                >>> mat2.load('my/path/to/file')
                >>> mat2.zones
                3317
                """

        self.file_path = file_path
        self.__load__()

    def computational_view(self, core_list: List[str] = None):
        """
        Creates a memory view for a list of matrices that is compatible with Cython memory buffers

        It allows for AequilibraE matrices to be used in all parallelized algorithms within AequilibraE

        Parameters
        ----------
        core_list: List with the names of all matrices that need to be in the buffer
        ------------------------------------------------------------
        Example

        >>> zones_in_the_model = 3317
        >>> names_list = ['Car trips', 'pt trips', 'DRT trips', 'bike trips', 'walk trips']

        >>> mat = AequilibraeMatrix()
        >>> mat.create_empty(file_name='my/path/to/file', zones=zones_in_the_model, matrix_names= names_list)
        >>> mat.computational_view(['bike trips', 'walk trips'])
        >>> mat.view_names
        ['bike trips', 'walk trips']
        """

        self.matrix_view = None
        self.view_names = None
        if core_list is None:
            core_list = self.names
        else:
            if isinstance(core_list, list):
                for i in core_list:
                    if i not in self.names:
                        raise ValueError("Matrix core {} no available on this matrix".format(i))

                if len(core_list) > 1:
                    for i, x in enumerate(core_list[1:]):
                        k = self.names.index(x)  # index of the first element
                        k0 = self.names.index(core_list[i])  # index of the first element
                        if k - k0 != 1:
                            raise ValueError("Matrix cores {} and {} are not adjacent".format(core_list[i - 1], x))
            else:
                raise TypeError("Please provide a list of matrices")

        self.view_names = core_list
        if len(core_list) == 1:
            # self.matrix_view = self.matrix[:, :, self.names.index(core_list[0]):self.names.index(core_list[0])+1]
            self.matrix_view = self.matrices[:, :, self.names.index(core_list[0])]
        elif len(core_list) > 1:
            self.matrix_view = self.matrices[:, :, self.names.index(core_list[0]) : self.names.index(core_list[-1]) + 1]

    def copy(self, output_name: str = None, cores: List[str] = None, names: List[str] = None, compress: bool = None):
        """
        Copies a list of cores (or all cores) from one matrix file to another one

        Parameters
        ----------
        output_name: Name of the new matrix file

        cores: List (str)
            List of the matrix cores to be copied

        names: List(str), optional
            List with the new names for the cores (same list length as cores)

        compress: bool
            Whether you want to compress the matrix or not. NOT YET IMPLEMENTED
        ------------------------------------------------------------
        Example

        >>> zones_in_the_model = 3317
        >>> names_list = ['Car trips', 'pt trips', 'DRT trips', 'bike trips', 'walk trips']

        >>> mat = AequilibraeMatrix()
        >>> mat.create_empty(file_name='my/path/to/file', zones=zones_in_the_model, matrix_names= names_list)
        >>> mat.copy('my/new/path/to/file', cores=['bike trips', 'walk trips'], names=['bicycle', 'walking'])

        >>> mat2 = AequilibraeMatrix()
        >>> mat2.load('my/new/path/to/file')
        >>> mat.cores
        ['bicycle', 'walking']
        """
        if output_name is None:
            output_name = self.random_name()

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
            self.matrix_hash = output.__builds_hash__()
            output.matrices.flush()

        return output

    def rows(self):
        # type: () -> np.array()
        """
            Returns row vector for the matrix in the computational view

            Computational view needs to be set to a single matrix core

            ------------------------------------------------------------
            Example

            >>> mat = AequilibraeMatrix()
            >>> mat.load('my/path/to/file')
            >>> mat.computational_view(mat.cores[0])
            >>> mat.rows()
            array([0.,...,0.])
            """
        return self.__vector(axis=0)

    def columns(self):
        # type: () -> np.array()
        """
        Returns column vector for the matrix in the computational view

        Computational view needs to be set to a single matrix core

        ------------------------------------------------------------
        Example

        >>> mat = AequilibraeMatrix()
        >>> mat.load('my/path/to/file')
        >>> mat.computational_view(mat.cores[0])
        >>> mat.columns()
        array([0.,...,0.])
        """
        return self.__vector(axis=1)

    def nan_to_num(self):
        """
        Converts all NaN values in all cores in the computational view to zeros

        ------------------------------------------------------------
        Example

        >>> mat = AequilibraeMatrix()
        >>> mat.load('my/path/to/file')
        >>> mat.computational_view(mat.cores[0])
        >>> mat.nan_to_num()
        """
        if np.issubdtype(self.dtype, np.floating) and self.matrix_view is not None:
            for m in self.view_names:
                self.matrix[m][:, :] = np.nan_to_num(self.matrix[m])[:, :]

    def __vector(self, axis: int):
        if self.view_names is None:
            raise ReferenceError("Matrix is not set for computation")
        if len(self.view_names) > 1:
            raise ValueError("Vector for a multi-core matrix is ambiguous")
        return self.matrix_view.astype(np.float).sum(axis=axis)[:]

    def __builds_hash__(self):
        return {self.index[i]: i for i in range(self.zones)}

    def define_data_class(self):
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

        Parameters
        ----------
        matrix_name: str
            matrix name. Maximum length is 50 characters

        ------------------------------------------------------------
        Example


        >>> mat = AequilibraeMatrix()
        >>> mat.load('my/path/to/file')
        >>> mat.setName('This is my example')
        >>> mat.name
        'This is my example'
        """

        if matrix_name is not None:
            if len(str(matrix_name)) > MATRIX_NAME_MAX_LENGTH:
                matrix_name = str(matrix_name)[0:MATRIX_NAME_MAX_LENGTH]

            np.memmap(self.file_path, dtype="S" + str(MATRIX_NAME_MAX_LENGTH), offset=18, mode="r+", shape=1)[
                0
            ] = matrix_name

    def setDescription(self, matrix_description: str):
        """
        Sets description for the matrix

        Parameters
        ----------
        matrix_name: str
            Text with matrix description . Maximum length is 144 characters

        ------------------------------------------------------------
        Example


        >>> mat = AequilibraeMatrix()
        >>> mat.load('my/path/to/file')
        >>> mat.setDescription('This is some text about this matrix of mine')
        >>> mat.description
        'This is some text about this matrix of mine'
        """
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

        ------------------------------------------------------------
        Example


        >>> name = AequilibraeMatrix().random_name()
        '/tmp/Aequilibrae_matrix_54625f36-bf41-4c85-80fb-7fc2e3f3d76e.aem'
        """
        return os.path.join(tempfile.gettempdir(), "Aequilibrae_matrix_" + str(uuid.uuid4()) + ".aem")
