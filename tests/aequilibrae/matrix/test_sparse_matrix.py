from tempfile import gettempdir
from aequilibrae.matrix import COO
from unittest import TestCase
from uuid import uuid4
import scipy.sparse
import numpy as np
import pathlib


class TestSparseMatrix(TestCase):
    def setUp(self) -> None:
        self.data = np.full((100, 100), 5.0)
        self.dir = pathlib.Path(gettempdir()) / uuid4().hex
        self.dir.mkdir()

    def tearDown(self) -> None:
        pass

    def test_round_trip(self):
        p = self.dir / "test.omx"

        coo = COO.from_matrix(
            self.data,
        )
        coo.to_disk(p, "m1")
        coo.to_disk(p, "m2")

        sp = coo.to_scipy()

        coo1 = COO.from_disk(p)
        coo2 = COO.from_disk(p, aeq=True)

        for m in ["m1", "m2"]:
            self.assertIsInstance(coo1[m], scipy.sparse.csr_matrix)
            self.assertIsInstance(coo2[m], COO)

            np.testing.assert_allclose(sp.A, coo1[m].A)
            np.testing.assert_allclose(sp.A, coo2[m].to_scipy().A)
