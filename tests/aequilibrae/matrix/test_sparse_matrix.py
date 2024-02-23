from aequilibrae.matrix import SparseMatrix
from unittest import TestCase


class TestSparseMatrix(TestCase):
    def setUp(self) -> None:
        self.sparse = SparseMatrix()

    def tearDown(self) -> None:
        pass

    def test_debug(self) -> None:
        self.sparse.helloworld()
