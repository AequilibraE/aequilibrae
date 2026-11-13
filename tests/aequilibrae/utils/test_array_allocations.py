"""Cython-backed allocations and read-only NumPy buffer exports."""

import numpy as np
import pytest

from aequilibrae.utils.cython.array_allocations import array, readonly_view


@pytest.mark.parametrize("element, value", [("double", np.inf), ("bool", True), ("uint16_t", 7), ("size_t", 19)])
@pytest.mark.parametrize("shape", [5, (2, 3), (2, 3, 4), 0, (0, 3), (2, 0, 4), (2, 3, 0)])
def test_array_shape_and_fill(element, value, shape):
    result = np.asarray(array[element](shape, True, value))
    assert result.shape == ((shape,) if isinstance(shape, int) else shape)
    assert result.flags.c_contiguous
    np.testing.assert_array_equal(result, value)


@pytest.mark.parametrize("dtype", [np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64])
@pytest.mark.parametrize("bound", ["min", "max"])
def test_fixed_width_integer_allocation(dtype, bound):
    dtype = np.dtype(dtype)
    value = int(getattr(np.iinfo(dtype), bound))
    result = np.asarray(array[f"{dtype.name}_t"]((2, 3), True, value))
    assert result.dtype == dtype
    assert result.flags.c_contiguous
    np.testing.assert_array_equal(result, np.full((2, 3), value, dtype=dtype))


@pytest.mark.parametrize("shape, error", [(-1, ValueError), ((2, -1), ValueError), ((), ValueError), (1.5, TypeError)])
def test_array_rejects_invalid_shapes(shape, error):
    with pytest.raises(error):
        array["double"](shape)


@pytest.mark.parametrize("buffer_kind", ["numpy", "python", "cython"])
def test_readonly_view_retains_shared_data(buffer_kind):
    source = array["double"]((2, 3), True, 7) if buffer_kind == "cython" else np.full((2, 3), 7.0)
    if buffer_kind == "python":
        source = memoryview(source)
    writable = np.asarray(source)
    view = readonly_view(source)
    assert np.shares_memory(view, writable)
    assert not view.flags.writeable
    with pytest.raises(ValueError):
        view.flags.writeable = True
    with pytest.raises(ValueError):
        view[0, 0] = 0
    writable[0, 0] = 42
    assert view[0, 0] == 42
    del source, writable
    assert view[0, 0] == 42
