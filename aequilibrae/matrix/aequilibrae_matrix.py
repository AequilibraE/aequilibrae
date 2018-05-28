"""
 -----------------------------------------------------------------------------------------------------------
 Package:    AequilibraE

 Name:       AequilibraE Matrix
 Purpose:    Implements a new class to represent multi-layer matrices

 Original Author:  Pedro Camargo (c@margo.co)
 Contributors:
 Last edited by: Pedro Camargo

 Website:    www.AequilibraE.com
 Repository:  https://github.com/AequilibraE/AequilibraE

 Created:    2017-10-02
 Updated:
 Copyright:   (c) AequilibraE authors
 Licence:     See LICENSE.TXT
 -----------------------------------------------------------------------------------------------------------
 """

import numpy as np
import uuid
import tempfile
import os
from shutil import copyfile
import warnings

# CONSTANTS
VERSION = 1            # VERSION OF THE MATRIX FORMAT
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


"""
Matrix structure

What:   Version | Compress flag | # cells: compressed matrix | # of zones | # of cores | # of indices (Y) | Data type |
Size:    uint8  |    uint8      |        uint64              |  uint32    |   uint8    |     uint8        |  uint8    |
Shape:      1   |       1       |           1                |     1      |            |       1          |     1     |
Offset:     0   |       1       |           2                |     10     |     14     |      15          |    16     |


What:    Data size |  matrix name | matrix description | Core names |   index names    |
Size:     uint8    |     S20      |          S144      |    S50     |      S20         |
Shape:      1      |      1       |            1       |  [cores]   |    [indices]     |
Offset:     17     |     18       |          38        |     182    |  182 + 50*cores  |

What:         indices          |             Matrin-1                 |
Size:         uint64           |      f(Data type, Data size)         |
Shape:     [zones, indices]    |       [zones, zones, cores]          |
Offset:  18 + 50*cores + Y*20  |   18 + 50*cores + Y*20 + Y*zones*8   | 

"""
matrix_export_types = ["Aequilibrae matrix (*.aem)", "Comma-separated file (*.csv)"]


class AequilibraeMatrix(object):
    def __init__(self):

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
        self.num_indices = None
        self.matrix = None
        self.matrices = None
        self.cores = None
        self.zones = None
        self.dtype = None
        self.names = None
        self.name = None
        self.description = None
        self.__version__ = VERSION       # Writes file version


    def create_empty(self, file_name=None, zones=None, matrix_names=None, data_type=np.float64,
                     index_names=None, compressed=False):

        self.file_path = file_name
        self.zones = zones
        self.index_names = index_names
        self.dtype = data_type

        # Matrix compression still not supported
        if compressed:
            compressed = False
            print 'Matrix compression not yet supported'

        if compressed:
            self.compressed = COMPRESSED
        else:
            self.compressed = NOT_COMPRESSED

        if index_names is None:
            self.index_names = ['main_index']
        else:
            if isinstance(index_names, list) or isinstance(index_names, tuple):
                self.index_names = index_names
                for ind_name in index_names:
                    if isinstance(ind_name, str):
                        if len(ind_name) > INDEX_NAME_MAX_LENGTH:
                            raise ValueError('Index names need to be be shorter '
                                             'than {}: {}'.format(INDEX_NAME_MAX_LENGTH, ind_name))
                    else:
                        raise ValueError('Index names need to be strings: ' + str(ind_name))
            else:
                raise Exception('Index names need to be provided as a list')
        self.num_indices = len(self.index_names)

        if matrix_names is None:
            matrix_names = ['mat']
        else:
            if isinstance(matrix_names, list) or isinstance(matrix_names, tuple):
                for mat_name in matrix_names:
                    if isinstance(mat_name, str) or isinstance(mat_name, unicode):
                        if mat_name in object.__dict__:
                            raise ValueError(mat_name + ' is a reserved name')
                        if len(mat_name) > CORE_NAME_MAX_LENGTH:
                            raise ValueError('Matrix names need to be be shorter '
                                             'than {}: {}'.format(CORE_NAME_MAX_LENGTH, mat_name))
                    else:
                        raise ValueError('Matrix core names need to be strings: ' + str(mat_name))
            else:
                raise Exception('Matrix names need to be provided as a list')

        self.names = [x.encode('utf-8') for x in matrix_names]
        self.cores = len(self.names)
        if None not in [self.file_path, self.zones]:
            self.__write__()

    def __load__(self):
        # GET File version
        self.__version__ = np.memmap(self.file_path, dtype='uint8', offset=0, mode='r+', shape=1)[0]

        if self.__version__ != VERSION:
            raise ValueError('Matrix formats do not match')

        # If matrix is compressed or not
        self.compressed = np.memmap(self.file_path, dtype='uint8', offset=1, mode='r+', shape=1)[0]

        # number matrix cells if compressed
        matrix_cells = np.memmap(self.file_path, dtype='uint64', offset=2, mode='r+', shape=1)[0]

        # Zones
        self.zones = np.memmap(self.file_path, dtype='uint32', offset=10, mode='r+', shape=1)[0]

        # Matrix cores
        self.cores = np.memmap(self.file_path, dtype='uint8', offset=14, mode='r+', shape=1)[0]

        # Matrix indices
        self.num_indices = np.memmap(self.file_path, dtype='uint8', offset=15, mode='r+', shape=1)[0]

        # Data type
        data_class = np.memmap(self.file_path, dtype='uint8', offset=16, mode='r+', shape=1)[0]

        # Data size
        data_size = np.memmap(self.file_path, dtype='uint8', offset=17, mode='r+', shape=1)[0]

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
        self.name = np.memmap(self.file_path, dtype='S' + str(MATRIX_NAME_MAX_LENGTH), offset=18, mode='r+',
                                    shape=1)[0]

        # matrix description
        offset = 18 + MATRIX_NAME_MAX_LENGTH
        self.description = np.memmap(self.file_path, dtype='S' + str(MATRIX_DESCRIPTION_MAX_LENGTH), offset=offset,
                                          mode='r+', shape=1)[0]

        # core names
        offset += MATRIX_DESCRIPTION_MAX_LENGTH
        self.names = list(np.memmap(self.file_path, dtype='S' + str(CORE_NAME_MAX_LENGTH), offset=offset, mode='r+',
                                    shape=self.cores))

        # Index names
        offset += CORE_NAME_MAX_LENGTH * self.cores
        self.index_names = list(np.memmap(self.file_path, dtype='S' + str(INDEX_NAME_MAX_LENGTH), offset=offset,
                                          mode='r+', shape=self.num_indices))

        # Index
        offset += self.num_indices * INDEX_NAME_MAX_LENGTH
        self.indices = np.memmap(self.file_path, dtype='uint64', offset=offset, mode='r+',
                                 shape=(self.zones, self.num_indices))
        self.set_index(0)

        # DATA
        offset += self.zones * 8 * self.num_indices
        if self.compressed:
            self.matrices = np.memmap(self.file_path, dtype=self.dtype, offset=offset, mode='r+',
                                      shape=(matrix_cells, self.cores + 2))
        else:
            self.matrices = np.memmap(self.file_path, dtype=self.dtype, offset=offset, mode='r+',
                                      shape=(self.zones, self.zones, self.cores))

        self.matrix = {}
        for i, v in enumerate(self.names):
            self.matrix[v] = self.matrices[:, :, i]
        self.__builds_hash__()

    def __write__(self):
        np.memmap(self.file_path, dtype='uint8', offset=0, mode='w+', shape=1)[0] = self.__version__

        # If matrix is compressed or not
        np.memmap(self.file_path, dtype='uint8', offset=1, mode='r+', shape=1)[0] = self.compressed

        # number matrix cells if compressed
        matrix_cells = self.zones * self.zones
        np.memmap(self.file_path, dtype='uint64', offset=2, mode='r+', shape=1)[0] = matrix_cells

        # Zones
        np.memmap(self.file_path, dtype='uint32', offset=10, mode='r+', shape=1)[0] = self.zones

        # Matrix cores
        np.memmap(self.file_path, dtype='uint8', offset=14, mode='r+', shape=1)[0] = self.cores

        # Matrix indices
        np.memmap(self.file_path, dtype='uint8', offset=15, mode='r+', shape=1)[0] = self.num_indices

        # Data type
        data_class = self.define_data_class()
        np.memmap(self.file_path, dtype='uint8', offset=16, mode='r+', shape=1)[0] = data_class

        # Data size
        data_size = np.dtype(self.dtype).itemsize
        np.memmap(self.file_path, dtype='uint8', offset=17, mode='r+', shape=1)[0] = data_size

        # matrix name
        np.memmap(self.file_path, dtype='S' + str(MATRIX_NAME_MAX_LENGTH), offset=18, mode='r+',
                       shape=1)[0] = self.name

        # matrix description
        offset = 18 + MATRIX_NAME_MAX_LENGTH
        np.memmap(self.file_path, dtype='S' + str(MATRIX_DESCRIPTION_MAX_LENGTH), offset=offset, mode='r+',
                       shape=1)[0] = self.description

        # core names
        offset += MATRIX_DESCRIPTION_MAX_LENGTH
        fp = np.memmap(self.file_path, dtype='S' + str(CORE_NAME_MAX_LENGTH), offset=offset, mode='r+', shape=self.cores)
        for i, v in enumerate(self.names):
            fp[i] = v
        fp.flush()
        del fp

        # Index names
        offset += CORE_NAME_MAX_LENGTH * self.cores
        fp = np.memmap(self.file_path, dtype='S' + str(INDEX_NAME_MAX_LENGTH), offset=offset, mode='r+',
                       shape=self.num_indices)
        for i, v in enumerate(self.index_names):
            fp[i] = v
        fp.flush()
        del fp

        # Index
        offset += self.num_indices * INDEX_NAME_MAX_LENGTH
        self.indices = np.memmap(self.file_path, dtype='uint64', offset=offset, mode='r+',
                                 shape=(self.zones, self.num_indices))
        self.indices.fill(0)
        self.indices.flush()
        self.set_index(0)

        offset += self.zones * 8 * self.num_indices
        if self.compressed:
            self.matrices = np.memmap(self.file_path, dtype=self.dtype, offset=offset, mode='r+',
                                      shape=(matrix_cells, self.cores + 2))
        else:
            self.matrices = np.memmap(self.file_path, dtype=self.dtype, offset=offset, mode='r+',
                                      shape=(self.zones, self.zones, self.cores))

        if np.issubdtype(self.dtype, np.integer):
            self.matrices.fill(np.iinfo(self.dtype).min)
        else:
            self.matrices.fill(np.nan)
        self.matrices.flush()

        self.matrix = {}
        for i, v in enumerate(self.names):
            self.matrix[v] = self.matrices[:, :, i]

    def set_index(self, index_to_set):
        if isinstance(index_to_set, int):
            if index_to_set >= self.num_indices:
                raise ValueError('Index {} not available. Choose on interval [0, '
                                 '{}]'.format(index_to_set, self.num_indices-1))
        elif isinstance(index_to_set, str):
            if index_to_set in self.index:
                index_to_set = self.index_names.index(index_to_set)
            else:
                raise ValueError('Index {} needs to be a string or its integer index.'.format(str(index_to_set)))
        else:
            raise ValueError('Index {} not available. Choose one of {}'.format(index_to_set, str(self.index_names)))
        self.index = self.indices[:, index_to_set]

    def __getattr__(self, mat_name):
        if mat_name in object.__dict__:
            return self.__dict__[mat_name]

        if mat_name in self.names:
            return self.matrix[mat_name]

        raise AttributeError("No such method or matrix core! --> " + str(mat_name))

    # Transforms matrix from dense to CSR
    def compress(self):
        print (self.file_path + '. Method not implemented yet. All matrices are dense')

    # Transforms matrix from CSR to dense
    def decompress(self):
        print (self.file_path + '. Method not implemented yet. All matrices are dense')

    # Adds index to matrix
    def add_index(self):
        print (self.file_path + '. Method not implemented yet. All indices need to exist during the matrix creation')

    # Adds index to matrix
    def remove_index(self, index_number):
        print (self.file_path + '. Method not implemented. Indices are fixed on matrix creation: ' + str(index_number))

    def close(self, flush=True):
        if flush:
            self.matrices.flush()
            self.index.flush()
        del self.matrices
        del self.index
            
    def export(self, output_name, cores=None):
        fname, file_extension = os.path.splitext(output_name.upper())

        if file_extension not in ['.AEM', '.CSV']:
            raise ValueError('File extension %d not implemented yet', file_extension)

        if cores is None:
            cores = self.names

        if file_extension == '.AEM':
            self.copy(output_name=output_name, cores=cores)

        if file_extension == '.CSV':
            names = self.view_names
            self.computational_view(cores)
            output = open(output_name, 'w')

            titles = ['row', 'column']
            for core in self.view_names:
                titles.append(core)
            print >> output, ','.join(titles)

            for i in range(self.zones):
                for j in range(self.zones):
                    record = [self.index[i], self.index[j]]
                    if len(self.view_names) > 1:
                        record.extend(self.matrix_view[i, j, :])
                    else:
                        record.append(self.matrix_view[i, j])
                    print >> output, ','.join(str(x) for x in record)
            output.flush()
            output.close()
            self.computational_view(names)


    def load(self, file_path):
        self.file_path = file_path
        self.__load__()

    def computational_view(self, core_list=None):
        self.matrix_view = None
        self.view_names = None
        if core_list is None:
            core_list = self.names
        else:
            if isinstance(core_list, list):
                for i in core_list:
                    if i not in self.names:
                        raise ValueError('Matrix core {} no available on this matrix'.format(i))

                if len(core_list) > 1:
                    for i, x in enumerate(core_list[1:]):
                        k = self.names.index(x)   # index of the first element
                        k0 = self.names.index(core_list[i])   # index of the first element
                        if k-k0 != 1:
                            raise ValueError('Matrix cores {} and {} are not adjacent'.format(core_list[i - 1], x))
            else:
                raise TypeError('Please provide a list of matrices')

        self.view_names = core_list
        if len(core_list) == 1:
            # self.matrix_view = self.matrix[:, :, self.names.index(core_list[0]):self.names.index(core_list[0])+1]
            self.matrix_view = self.matrices[:, :, self.names.index(core_list[0])]
        elif len(core_list) > 1:
            self.matrix_view = self.matrices[:, :, self.names.index(core_list[0]):self.names.index(core_list[-1]) + 1]

    def copy(self, output_name=None, cores=None, names=None, compress=None):
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
                raise ValueError('Cores need to be presented as list')

            for i in cores:
                if i not in self.names:
                    raise ValueError('Matrix core {} not available on this matrix'.format(i))

            if names is None:
                names = cores
            else:
                if not isinstance(names, list):
                    raise ValueError('Names need to be presented as list')

                if len(names) != len(cores):
                    raise ValueError('Number of cores to cpy and list of names are not compatible')

            output = AequilibraeMatrix()
            output.create_empty(file_name=output_name,
                                zones=self.zones,
                                matrix_names=names,
                                index_names=self.index_names,
                                data_type=self.dtype,
                                compressed=bool(compress))

            output.indices[:] = self.indices[:]
            for i, c in enumerate(cores):
                output.matrices[:, :, i] = self.matrices[:, :, self.names.index(c)]
            output.__builds_hash__()
            output.matrices.flush()

        return output

    def rows(self):
        return self.vector(axis=0)

    def columns(self):
        return self.vector(axis=1)

    def nan_to_num(self):
        if np.issubdtype(self.dtype, np.float) and self.matrix_view is not None:
            for m in self.view_names:
                self.matrix[m][:,:] = np.nan_to_num(self.matrix[m])[:,:]

    def vector(self, axis):
        if self.view_names is None:
            raise ReferenceError('Matrix is not set for computation')
        if len(self.view_names) > 1:
            raise ValueError('Vector for a multi-core matrix is ambiguous')

        return self.matrix_view.astype(np.float).sum(axis=axis)[:]

    def __builds_hash__(self):
        return {self.index[i]: i for i in range(self.zones)}

    def define_data_class(self):
        if np.issubdtype(self.dtype, np.float):
            data_class = FLOAT
        elif np.issubdtype(self.dtype, np.integer):
            data_class = INT
        else:
            raise ValueError('Data type not supported. You can choose Integers of 8, 16, 32 and 64 bits, '
                             'or floats with 16, 32 or 64 bits')
        return data_class

    def setName(self, matrix_name):
        if matrix_name is not None:
            if len(str(matrix_name)) > MATRIX_NAME_MAX_LENGTH:
                matrix_name = str(matrix_name)[0:MATRIX_NAME_MAX_LENGTH]

            np.memmap(self.file_path, dtype='S' + str(MATRIX_NAME_MAX_LENGTH), offset=18, mode='r+',
                      shape=1)[0] = matrix_name

    def setDescription(self, matrix_description):
        if matrix_description is not None:
            if len(str(matrix_description)) > MATRIX_DESCRIPTION_MAX_LENGTH:
                matrix_description = str(matrix_description)[0:MATRIX_DESCRIPTION_MAX_LENGTH]

            np.memmap(self.file_path, dtype='S' + str(MATRIX_NAME_MAX_LENGTH), offset=18 + MATRIX_NAME_MAX_LENGTH,
                      mode='r+', shape=1)[0] = matrix_description

    @staticmethod
    def random_name():
        return os.path.join(tempfile.gettempdir(), 'Aequilibrae_matrix_' + str(uuid.uuid4()) + '.aem')
