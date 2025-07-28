import numpy as np
import scipy.sparse

from aequilibrae.matrix import COO


def test_round_trip(test_folder):
    test_data = np.full((100, 100), 5.0)
    p = test_folder / "test.omx"

    coo = COO.from_matrix(test_data)
    coo.to_disk(p, "m1")
    coo.to_disk(p, "m2")

    sp = coo.to_scipy()

    coo1 = COO.from_disk(p)
    coo2 = COO.from_disk(p, aeq=True)

    for m in ["m1", "m2"]:
        assert isinstance(coo1[m], scipy.sparse.csr_matrix)
        assert isinstance(coo2[m], COO)

        np.testing.assert_allclose(sp.toarray(), coo1[m].toarray())
        np.testing.assert_allclose(sp.toarray(), coo2[m].to_scipy().toarray())
