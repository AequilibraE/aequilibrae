"""Packing, conversion and ownership contracts of the dense matrix buffer."""

import gc
from types import MappingProxyType

import numpy as np
import pytest
from aequilibrae.matrix.cython.dense_matrix import DenseMatrix

from aequilibrae.matrix import AequilibraeMatrix, AequilibraeSparseMatrix, from_dict


@pytest.fixture(params=[np.float32, np.float64])
def dtype(request):
    return request.param


def test_constructor_defaults_to_float64_and_infinite_values():
    packed = DenseMatrix(2, ["bus", "car"])
    matrix = packed.to_matrix([309, 0])

    assert (packed.origin_count, packed.destination_count, packed.field_count) == (2, 2, 2)
    assert packed.field_names == ("bus", "car")
    assert packed.dtype == np.dtype(np.float64)
    for field in matrix.values():
        assert field.dtype == np.dtype(np.float64)
        np.testing.assert_array_equal(field, np.full((2, 2), np.inf))


def test_constructor_initialises_every_field_with_the_requested_fill_value(dtype):
    packed = DenseMatrix(2, ["bus", "car"], dtype=dtype, fill_value=-7.5)
    matrix = packed.to_matrix([309, 0])

    assert packed.dtype == np.dtype(dtype)
    for field in matrix.values():
        assert field.dtype == np.dtype(dtype)
        np.testing.assert_array_equal(field, np.full((2, 2), -7.5))


def test_constructor_uses_explicit_origin_count(dtype):
    packed = DenseMatrix(3, ["bus", "car"], dtype=dtype, origin_count=2)

    assert (packed.origin_count, packed.destination_count, packed.field_count) == (2, 3, 2)
    assert packed.field_names == ("bus", "car")
    assert packed.dtype == np.dtype(dtype)


def test_from_dict_packs_fields_in_mapping_order(dtype):
    sources = {"bus": np.array([[0, 2], [3, 0]], dtype=dtype), "car": np.array([[4, 0], [0, 5]], dtype=dtype)}
    packed = DenseMatrix.from_dict(sources)
    matrix = packed.to_matrix([309, 0])

    assert (packed.origin_count, packed.destination_count, packed.field_count) == (2, 2, 2)
    assert packed.field_names == ("bus", "car")
    assert packed.dtype == np.dtype(dtype)
    assert tuple(matrix) == ("bus", "car")
    for name, expected in sources.items():
        assert matrix[name].dtype == np.dtype(dtype)
        np.testing.assert_array_equal(matrix[name], expected)


def test_from_dict_copies_strided_input_arrays(dtype):
    backing = np.arange(8, dtype=dtype).reshape(2, 4)
    packed = DenseMatrix.from_dict({"bus": backing[:, ::2], "car": backing[:, 1::2]})
    backing[:] = -1
    matrix = packed.to_matrix([309, 0])

    np.testing.assert_array_equal(matrix["bus"], [[0, 2], [4, 6]])
    np.testing.assert_array_equal(matrix["car"], [[1, 3], [5, 7]])


def test_from_dict_transposes_every_field(dtype):
    packed = DenseMatrix.from_dict(
        {"bus": np.array([[0, 2], [3, 0]], dtype=dtype), "car": np.array([[4, 5], [6, 7]], dtype=dtype)},
        transpose=True,
    )
    matrix = packed.to_matrix([309, 0])

    np.testing.assert_array_equal(matrix["bus"], [[0, 3], [2, 0]])
    np.testing.assert_array_equal(matrix["car"], [[4, 6], [5, 7]])


@pytest.mark.parametrize("transpose, expected_shape", [(False, (2, 3)), (True, (3, 2))], ids=["original", "transposed"])
def test_from_dict_uses_rectangular_source_dimensions(dtype, transpose, expected_shape):
    packed = DenseMatrix.from_dict(
        {"bus": np.arange(6, dtype=dtype).reshape(2, 3), "car": np.full((2, 3), 42, dtype=dtype)},
        transpose=transpose,
    )

    assert (packed.origin_count, packed.destination_count) == expected_shape
    assert packed.field_names == ("bus", "car")
    assert packed.field_count == 2
    assert packed.dtype == np.dtype(dtype)


def test_from_dict_accepts_read_only_mappings_of_array_like_values():
    sources = MappingProxyType({"bus": [[0.0, 2.0], [3.0, 0.0]]})
    matrix = DenseMatrix.from_dict(sources).to_matrix([309, 0])

    np.testing.assert_array_equal(matrix["bus"], [[0, 2], [3, 0]])


def test_from_dict_rejects_mixed_dtypes_without_a_target_dtype():
    with pytest.raises(TypeError, match="mixed dtypes"):
        DenseMatrix.from_dict({"bus": np.eye(2, dtype=np.int16), "car": np.eye(2, dtype=np.float64)})


def test_from_dict_converts_mixed_dtypes_to_the_requested_dtype(dtype):
    packed = DenseMatrix.from_dict(
        {"bus": np.array([[0, 2], [3, 0]], dtype=np.int16), "car": np.eye(2, dtype=np.float64)}, dtype=dtype
    )
    matrix = packed.to_matrix([309, 0])

    assert packed.dtype == np.dtype(dtype)
    for field in matrix.values():
        assert field.dtype == np.dtype(dtype)
    np.testing.assert_array_equal(matrix["bus"], [[0, 2], [3, 0]])
    np.testing.assert_array_equal(matrix["car"], np.eye(2))


@pytest.mark.parametrize(
    "sources, error, match",
    [
        pytest.param([], TypeError, "mapping", id="not_a_mapping"),
        pytest.param({}, ValueError, "at least one", id="empty_mapping"),
        pytest.param({"bus": np.array(1.0)}, ValueError, "two-dimensional", id="scalar_field"),
        pytest.param({"bus": np.ones(2)}, ValueError, "two-dimensional", id="1d_field"),
        pytest.param({"bus": np.ones((2, 2, 1))}, ValueError, "two-dimensional", id="3d_field"),
        pytest.param({"bus": np.ones((2, 3)), "car": np.ones((3, 2))}, ValueError, "same shape", id="different_shapes"),
        pytest.param(
            {"bus": np.ones((2, 2), dtype=bool)}, TypeError, "float32 or float64", id="boolean_without_conversion"
        ),
        pytest.param(
            {"bus": np.ones((2, 2), dtype=complex)}, TypeError, "float32 or float64", id="complex_without_conversion"
        ),
        pytest.param(
            {"bus": np.ones((2, 2), dtype=int)}, TypeError, "float32 or float64", id="integer_without_conversion"
        ),
        pytest.param({"bus": np.full((2, 2), "a")}, TypeError, "float32 or float64", id="string_field"),
    ],
)
def test_from_dict_rejects_invalid_inputs(sources, error, match):
    with pytest.raises(error, match=match):
        DenseMatrix.from_dict(sources)


def test_from_matrix_copies_fields_in_collection_order(dtype):
    sources = {"bus": np.array([[0, 2], [3, 0]], dtype=dtype), "car": np.array([[4, 0], [0, 5]], dtype=dtype)}
    source = from_dict(sources, index=[309, 0])
    packed = DenseMatrix.from_matrix(source)
    source["bus"][0, 1] = 99
    matrix = packed.to_matrix([309, 0])

    assert packed.field_names == ("bus", "car")
    for name, expected in sources.items():
        np.testing.assert_array_equal(matrix[name], expected)


def test_from_matrix_forwards_transpose(dtype):
    source = from_dict({"bus": np.array([[0, 2], [3, 0]], dtype=dtype)}, index=[309, 0])
    matrix = DenseMatrix.from_matrix(source, transpose=True).to_matrix([309, 0])

    np.testing.assert_array_equal(matrix["bus"], [[0, 3], [2, 0]])


def test_from_matrix_forwards_the_requested_conversion_dtype(dtype):
    source = from_dict(
        {"bus": np.array([[0, 2], [3, 0]], dtype=np.int16), "car": np.eye(2, dtype=np.float64)}, index=[309, 0]
    )
    packed = DenseMatrix.from_matrix(source, dtype=dtype)
    matrix = packed.to_matrix([309, 0])

    assert packed.dtype == np.dtype(dtype)
    for field in matrix.values():
        assert field.dtype == np.dtype(dtype)
    np.testing.assert_array_equal(matrix["bus"], [[0, 2], [3, 0]])
    np.testing.assert_array_equal(matrix["car"], np.eye(2))


def test_from_matrix_rejects_mixed_dtypes_without_a_target_dtype():
    source = from_dict({"bus": np.eye(2, dtype=np.int16), "car": np.eye(2, dtype=np.float64)}, index=[309, 0])
    with pytest.raises(TypeError, match="mixed dtypes"):
        DenseMatrix.from_matrix(source)


@pytest.mark.parametrize("source", [{"bus": np.eye(2)}, np.eye(2), AequilibraeSparseMatrix(index=[309, 0])])
def test_from_matrix_rejects_objects_that_are_not_dense_matrix_collections(source):
    with pytest.raises(TypeError, match="matrix must be an AequilibraeMatrix"):
        DenseMatrix.from_matrix(source)


@pytest.mark.parametrize("source_dtype", [np.bool_, np.complex128], ids=["boolean_values", "complex_values"])
def test_from_matrix_rejects_boolean_and_complex_fields_even_with_explicit_dtype(source_dtype):
    source = from_dict({"bus": np.ones((2, 2), dtype=source_dtype)}, index=[309, 0])
    with pytest.raises(TypeError, match="Matrix values must be real numbers"):
        DenseMatrix.from_matrix(source, dtype=np.float64)


@pytest.mark.parametrize(
    "dtype, expected_dtype",
    [(None, np.float64), (np.float32, np.float32), (np.float64, np.float64)],
    ids=["default_dtype", "float32", "float64"],
)
def test_from_matrix_without_fields_preserves_zone_count_and_requested_dtype(dtype, expected_dtype):
    packed = DenseMatrix.from_matrix(AequilibraeMatrix(index=[309, 0]), dtype=dtype)
    matrix = packed.to_matrix([309, 0])

    assert (packed.origin_count, packed.destination_count, packed.field_count) == (2, 2, 0)
    assert packed.field_names == ()
    assert packed.dtype == np.dtype(expected_dtype)
    assert len(matrix) == 0
    assert matrix.shape == (2, 2)


@pytest.mark.parametrize("shape", [(0, 0), (0, 3), (2, 0)], ids=["no_zones", "no_origins", "no_destinations"])
def test_from_dict_accepts_zero_sized_axes(dtype, shape):
    packed = DenseMatrix.from_dict({"bus": np.zeros(shape, dtype=dtype)})

    assert (packed.origin_count, packed.destination_count) == shape
    assert packed.field_names == ("bus",)
    assert packed.field_count == 1
    assert packed.dtype == np.dtype(dtype)


def test_to_matrix_accepts_zero_zones(dtype):
    matrix = DenseMatrix(0, ("bus", "car"), dtype=dtype).to_matrix([])

    assert tuple(matrix) == ("bus", "car")
    assert matrix.shape == (0, 0)
    for field in matrix.values():
        assert field.shape == (0, 0)
        assert field.dtype == np.dtype(dtype)


def test_to_matrix_preserves_the_supplied_zone_mapping(dtype):
    matrix = DenseMatrix(2, ["bus"], dtype=dtype).to_matrix([309, 0], index_name="zones")

    assert isinstance(matrix, AequilibraeMatrix)
    assert matrix.index_name == "zones"
    np.testing.assert_array_equal(matrix.index, [309, 0])


def test_to_matrix_exposes_origin_destination_field_order(dtype):
    matrix = DenseMatrix(2, ["bus", "car", "walk"], dtype=dtype).to_matrix([309, 0])
    itemsize = np.dtype(dtype).itemsize

    # All fields of a cell are adjacent; columns then rows advance across complete cells.
    for field in matrix.values():
        assert field.strides == (2 * 3 * itemsize, 3 * itemsize)
    assert matrix["car"].ctypes.data - matrix["bus"].ctypes.data == itemsize
    assert matrix["walk"].ctypes.data - matrix["car"].ctypes.data == itemsize


def test_to_matrix_returns_views_of_the_same_packed_values(dtype):
    packed = DenseMatrix.from_dict(
        {"bus": np.array([[0, 2], [3, 0]], dtype=dtype), "car": np.array([[4, 0], [0, 5]], dtype=dtype)}
    )
    first = packed.to_matrix([309, 0])
    second = packed.to_matrix([309, 0])
    assert np.shares_memory(first["bus"], second["bus"])
    first["bus"][0, 1] = 42

    assert second["bus"][0, 1] == 42
    np.testing.assert_array_equal(second["car"], [[4, 0], [0, 5]])


def test_to_matrix_arrays_remain_valid_after_owner_deletion(dtype):
    sources = {"bus": np.array([[0, 2], [3, 0]], dtype=dtype), "car": np.array([[4, 0], [0, 5]], dtype=dtype)}
    packed = DenseMatrix.from_dict(sources)
    matrix = packed.to_matrix([309, 0])
    retained = dict(matrix.items())
    del packed, matrix
    gc.collect()

    for name, values in retained.items():
        np.testing.assert_array_equal(values, sources[name])


@pytest.mark.parametrize("index", [[1], [1, 2, 3]], ids=["too_short", "too_long"])
def test_to_matrix_rejects_wrong_index_length(index):
    with pytest.raises(ValueError, match="index length"):
        DenseMatrix(2, ["bus"]).to_matrix(index)


def test_to_matrix_rejects_rectangular_buffers():
    with pytest.raises(ValueError, match="only square"):
        DenseMatrix(3, ["bus"], origin_count=2).to_matrix([1, 2])
