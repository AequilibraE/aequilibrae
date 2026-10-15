import gc

import numpy as np
import pytest

from aequilibrae.matrix import AequilibraeMatrix, DenseMatrix, from_dict


@pytest.fixture(params=[np.float32, np.float64])
def dtype(request):
    return request.param


def test_construction(dtype):
    dense = DenseMatrix(3, ("time", "distance"), dtype=dtype)
    assert dense.dtype == np.dtype(dtype)
    assert (dense.origin_count, dense.destination_count, dense.field_count) == (3, 3, 2)
    assert dense.field_names == ("time", "distance")


@pytest.mark.parametrize("names", [[], ["time"]])
def test_export_rejects_wrong_index_length(names):
    dense = DenseMatrix(2, names)
    with pytest.raises(ValueError, match="index length"):
        dense.to_matrix([1])


@pytest.mark.parametrize("fill_value", [np.inf, 7])
def test_fill_value(dtype, fill_value):
    result = DenseMatrix(2, ["cost"], dtype=dtype, fill_value=fill_value).to_matrix([1, 2])
    np.testing.assert_array_equal(result["cost"], np.full((2, 2), fill_value, dtype=dtype))


def test_exports_share_buffer(dtype):
    dense = DenseMatrix(2, ["time", "distance"], dtype=dtype)
    first, second = dense.to_matrix([1, 2]), dense.to_matrix([1, 2])
    assert np.shares_memory(first["time"], second["time"])
    first["time"][0, 1] = 12
    assert second["time"][0, 1] == 12
    assert np.isinf(second["distance"]).all()


def test_export_keeps_buffer_alive(dtype):
    dense = DenseMatrix(2, ["time", "distance"], dtype=dtype, fill_value=4)
    result = dense.to_matrix([1, 2])
    # The exported arrays must keep the buffer alive without DenseMatrix.
    del dense
    gc.collect()
    np.testing.assert_array_equal(result["time"], np.full((2, 2), 4))
    result["time"][:] = 9
    np.testing.assert_array_equal(result["time"], np.full((2, 2), 9))


def test_from_matrix_copies_values_and_names(dtype):
    data = {
        "time": np.arange(4, dtype=dtype).reshape(2, 2),
        "distance": np.full((2, 2), 5, dtype=dtype),
    }
    source = from_dict(data, [1, 2])
    dense = DenseMatrix.from_matrix(source)
    result = dense.to_matrix([1, 2])
    assert dense.dtype == np.dtype(dtype)
    assert dense.field_names == ("time", "distance")
    for name in data:
        np.testing.assert_array_equal(result[name], data[name])
        assert not np.shares_memory(source[name], result[name])


def test_from_export_always_copies(dtype):
    dense = DenseMatrix(2, ["time", "distance"], dtype=dtype, fill_value=3)
    source = dense.to_matrix([1, 2])
    result = DenseMatrix.from_matrix(source).to_matrix([1, 2])
    for name in dense.field_names:
        np.testing.assert_array_equal(result[name], source[name])
        assert not np.shares_memory(result[name], source[name])


def test_mixed_dtypes_need_explicit_dtype(dtype):
    source = from_dict({"time": np.zeros((2, 2), np.float32), "distance": np.ones((2, 2), np.float64)}, [1, 2])
    with pytest.raises(TypeError, match="mixed dtypes"):
        DenseMatrix.from_matrix(source)
    dense = DenseMatrix.from_matrix(source, dtype=dtype)
    assert dense.dtype == np.dtype(dtype)
    result = dense.to_matrix([1, 2])
    for name in dense.field_names:
        np.testing.assert_array_equal(result[name], source[name])


def test_integer_input_needs_explicit_dtype(dtype):
    source = from_dict({"time": np.ones((2, 2), np.int32)}, [1, 2])
    with pytest.raises(TypeError, match="dtype"):
        DenseMatrix.from_matrix(source)
    dense = DenseMatrix.from_matrix(source, dtype=dtype)
    assert dense.dtype == np.dtype(dtype)
    np.testing.assert_array_equal(dense.to_matrix([1, 2])["time"], source["time"])


def test_explicit_dtype_rejects_complex_input(dtype):
    source = from_dict({"time": np.ones((1, 1), np.complex128)}, [1])
    with pytest.raises(TypeError):
        DenseMatrix.from_matrix(source, dtype=dtype)


@pytest.mark.parametrize("bad_dtype", [np.int32, np.complex128, np.float16, bool, ">f8"])
def test_bad_dtype(bad_dtype):
    with pytest.raises(TypeError, match="dtype"):
        DenseMatrix(1, ["time"], dtype=bad_dtype)


@pytest.mark.parametrize("names", ["time", [""], ["a/b"], [None], ["time", "time"]])
def test_bad_names(names):
    with pytest.raises((TypeError, ValueError)):
        DenseMatrix(1, names)


@pytest.mark.parametrize("zone_count", [-1, 1.5, [1], None])
def test_bad_zone_count(zone_count):
    with pytest.raises((TypeError, ValueError)):
        DenseMatrix(zone_count, ["time"])


def test_cannot_reinitialise():
    dense = DenseMatrix(1, ["time"])
    with pytest.raises(RuntimeError, match="reinitialised"):
        dense.__init__(2, ["distance"])
    assert dense.origin_count == 1
    assert dense.field_names == ("time",)


def test_from_matrix_requires_matrix():
    with pytest.raises(TypeError, match="AequilibraeMatrix"):
        DenseMatrix.from_matrix({"time": [[0]]})


@pytest.mark.parametrize("index,names", [([], ["time"]), ([1, 2], []), ([], [])])
def test_empty_buffers(dtype, index, names):
    dense = DenseMatrix(len(index), names, dtype=dtype)
    result = dense.to_matrix(index)
    assert dense.origin_count == dense.destination_count == len(index)
    assert dense.field_count == len(names)
    for name in names:
        assert result[name].shape == (0, 0)


def test_empty_source_defaults_to_float64():
    dense = DenseMatrix.from_matrix(AequilibraeMatrix([1, 2]))
    assert dense.dtype == np.dtype(np.float64)
    assert dense.origin_count == dense.destination_count == 2
    assert dense.field_count == 0
