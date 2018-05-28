from unittest import TestCase
import os
import datetime
import numpy as np
import random

from aequilibrae.matrix import AequilibraeMatrix

zones = 50
name_test = AequilibraeMatrix().random_name()
copy_matrix_name = AequilibraeMatrix().random_name()
csv_export_name = copy_matrix_name + '.csv'


class TestAequilibraeMatrix(TestCase):
    def test___init__(self):
        os.remove(name_test) if os.path.exists(name_test) else None
        args = {'file_name': name_test,
                'zones': zones,
                'matrix_names': ['mat', 'seed', 'dist'],
                'index_names': ['my indices']}

        matrix = AequilibraeMatrix()
        matrix.create_empty(**args)

        matrix.index[:] = np.arange(matrix.zones) + 100
        matrix.mat[:, :] = np.random.rand(matrix.zones, matrix.zones)[:, :]
        matrix.mat[:, :] = matrix.mat[:, :] * (1000 / np.sum(matrix.mat[:, :]))
        matrix.setName('Test matrix - ' + str(random.randint(1, 10)))
        matrix.setDescription('Generated at ' + datetime.datetime.now().strftime("%I:%M%p on %B %d, %Y"))
        matrix.close(True)
        del (matrix)

    def test_load(self):
        # self.test___init__()
        self.new_matrix = AequilibraeMatrix()
        self.new_matrix.load(name_test)

    def test_computational_view(self):
        self.test_load()
        self.new_matrix.computational_view(['mat', 'seed'])
        self.new_matrix.mat.fill(0)
        self.new_matrix.seed.fill(0)
        if self.new_matrix.matrix_view.shape[2] != 2:
            self.fail('Computational view returns the wrong number of matrices')

        self.new_matrix.computational_view(['mat'])
        self.new_matrix.matrix_view[:, :] = np.arange(zones ** 2).reshape(zones, zones)
        if np.sum(self.new_matrix.mat) != np.sum(self.new_matrix.matrix_view):
            self.fail('Assigning to matrix view did not work')
        self.new_matrix.setName('Test matrix - ' + str(random.randint(1, 10)))
        self.new_matrix.setDescription('Generated at ' + datetime.datetime.now().strftime("%I:%M%p on %B %d, %Y"))
        self.new_matrix.close(True)

    def test_copy(self):
        self.test_load()

        # test in-memory matrix_procedures copy

        matrix_copy = self.new_matrix.copy(copy_matrix_name, cores=['mat'])

        if not np.array_equal(matrix_copy.mat, self.new_matrix.mat):
            self.fail('Matrix copy was not perfect')
        matrix_copy.close(True)
        self.new_matrix.close(True)

    def test_export(self):
        self.test_load()
        self.new_matrix.export(csv_export_name)
        self.new_matrix.close(True)

    def test_nan_to_num(self):
        self.test_load()
        s = self.new_matrix.seed.sum() - self.new_matrix.seed[1, 1]
        m = self.new_matrix.mat.sum() - self.new_matrix.mat[1, 1]
        self.new_matrix.seed[1,1] = np.nan
        self.new_matrix.computational_view(['mat', 'seed'])
        self.new_matrix.nan_to_num()
        self.new_matrix.mat[1,1] = np.nan
        self.new_matrix.computational_view(['mat'])
        self.new_matrix.nan_to_num()

        if s != self.new_matrix.seed.sum():
            self.fail('Total for seed matrix not maintained')

        if m != self.new_matrix.mat.sum():
            self.fail('Total for mat matrix not maintained')
