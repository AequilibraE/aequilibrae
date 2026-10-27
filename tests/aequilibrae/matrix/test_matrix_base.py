"""Constructor and fixed-attribute contracts shared by the native sparse owners."""

import numpy as np
import pytest
from aequilibrae.matrix.cython.sparse_matrix import SparseMatrix
from aequilibrae.matrix.cython.sparse_matrix_builder import SparseMatrixBuilder


@pytest.fixture(params=[SparseMatrix, SparseMatrixBuilder], ids=["packed_sparse", "builder"])
def owner(request):
    return request.param


@pytest.mark.parametrize(
    "options, error, match",
    [
        pytest.param({"zone_count": -1}, ValueError, "nonnegative", id="negative_destination_count"),
        pytest.param({"origin_count": -1}, ValueError, "nonnegative", id="negative_origin_count"),
        pytest.param({"zone_count": 1.5}, TypeError, None, id="non_integer_destination_count"),
        pytest.param({"origin_count": 1.5}, TypeError, None, id="non_integer_origin_count"),
        pytest.param({"field_names": "bus"}, TypeError, "sequence", id="names_as_string"),
        pytest.param({"field_names": ["bus", "bus"]}, ValueError, "unique", id="duplicate_names"),
        pytest.param({"field_names": [1]}, TypeError, "string", id="non_string_name"),
        pytest.param({"field_names": [""]}, ValueError, "not a path", id="empty_name"),
        pytest.param({"field_names": ["bus/car"]}, ValueError, "not a path", id="name_is_a_path"),
        pytest.param({"dtype": np.int64}, TypeError, "float32 or float64", id="integer_dtype"),
        pytest.param({"dtype": np.float16}, TypeError, "float32 or float64", id="float16_dtype"),
    ],
)
def test_constructor_rejects_invalid_dimensions_names_and_dtype(owner, options, error, match):
    arguments = {"zone_count": 3, "field_names": ["bus", "car"]}
    arguments.update(options)
    with pytest.raises(error, match=match):
        owner(**arguments)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_reinitialisation_is_rejected_without_changing_configuration(owner, dtype):
    instance = owner(3, ["bus", "car"], dtype=dtype, origin_count=2)
    with pytest.raises(RuntimeError, match="cannot be reinitialised"):
        instance.__init__(5, ["other"])

    assert (instance.origin_count, instance.destination_count, instance.field_count) == (2, 3, 2)
    assert instance.field_names == ("bus", "car")
    assert instance.dtype == np.dtype(dtype)


@pytest.mark.parametrize(
    "name, replacement",
    [
        ("origin_count", 5),
        ("destination_count", 5),
        ("field_count", 1),
        ("field_names", ("other",)),
        ("dtype", np.dtype(np.int64)),
    ],
    ids=["origins", "destinations", "field_count", "field_names", "dtype"],
)
def test_configuration_attributes_are_read_only(owner, name, replacement):
    instance = owner(3, ["bus", "car"], origin_count=2)
    with pytest.raises(AttributeError):
        setattr(instance, name, replacement)
