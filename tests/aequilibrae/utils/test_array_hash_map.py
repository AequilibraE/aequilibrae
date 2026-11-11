"""Scalar and bulk lookups through the Cython-owned C++ hash map."""

import numpy as np
import pytest

from aequilibrae.utils.cython.array_hash_map import ArrayHashMap


@pytest.fixture
def mapping():
    return ArrayHashMap(np.array([7, 2, 9], dtype=np.uint64), np.array([70, 20, 90], dtype=np.uint64))


@pytest.mark.parametrize("key", [7, np.uint64(7), np.int64(7)])
def test_scalar_lookup(mapping, key):
    result = mapping[key]
    assert result == 70
    assert type(result) is int


def test_bulk_lookup(mapping):
    result = mapping[np.array([9, 7, 2, 9], dtype=np.uint64)]
    assert isinstance(result, np.ndarray)
    assert result.dtype == np.uint64
    assert result.flags.c_contiguous
    np.testing.assert_array_equal(result, [90, 70, 20, 90])
    result[0] = 0
    assert mapping[9] == 90
    del mapping
    np.testing.assert_array_equal(result, [0, 70, 20, 90])


def test_duplicate_keys_keep_last_pair():
    mapping = ArrayHashMap(np.array([3, 3, 5, 3], dtype=np.uint64), np.array([1, 2, 4, 8], dtype=np.uint64))
    assert mapping[3] == 8
    assert mapping[5] == 4


@pytest.mark.parametrize("key", [4, np.array([7, 4, 5], dtype=np.uint64)])
def test_missing_key_does_not_insert(mapping, key):
    for _ in range(2):
        with pytest.raises(KeyError) as error:
            mapping[key]
        assert error.value.args == (4,)
    assert mapping[7] == 70


def test_empty_arrays(mapping):
    empty = np.array([], dtype=np.uint64)
    empty_mapping = ArrayHashMap(empty, empty)
    for instance in (mapping, empty_mapping):
        result = instance[empty]
        assert result.shape == (0,)
        assert result.dtype == np.uint64
    with pytest.raises(KeyError) as error:
        empty_mapping[0]
    assert error.value.args == (0,)
    with pytest.raises(KeyError):
        empty_mapping[np.array([0], dtype=np.uint64)]


def test_full_uint64_range():
    maximum = np.iinfo(np.uint64).max
    keys = np.array([0, 2**63, maximum], dtype=np.uint64)
    values = np.array([maximum, 0, 2**63], dtype=np.uint64)
    mapping = ArrayHashMap(keys, values)
    np.testing.assert_array_equal(mapping[keys], values)
    for key, value in zip(keys, values, strict=True):
        assert mapping[int(key)] == int(value)


def test_readonly_buffers_and_input_lifetime():
    keys = np.array([1, 2], dtype=np.uint64)
    values = np.array([10, 20], dtype=np.uint64)
    keys.flags.writeable = False
    values.flags.writeable = False
    mapping = ArrayHashMap(memoryview(keys), memoryview(values))
    np.testing.assert_array_equal(mapping[memoryview(keys)], values)
    del keys, values
    assert mapping[2] == 20


def test_initialisation_copies_pairs():
    keys = np.array([1], dtype=np.uint64)
    values = np.array([10], dtype=np.uint64)
    mapping = ArrayHashMap(keys, values)
    keys[0], values[0] = 2, 20
    assert mapping[1] == 10
    with pytest.raises(KeyError):
        mapping[2]


def test_mismatched_lengths():
    with pytest.raises(ValueError, match="equal lengths"):
        ArrayHashMap(np.array([1, 2], dtype=np.uint64), np.array([10], dtype=np.uint64))


@pytest.mark.parametrize(
    "invalid",
    [
        np.array([1, 2], dtype=np.int64),
        np.array([1, 2], dtype=np.uint32),
        np.array([1, 2], dtype=np.float64),
        np.array([[1, 2]], dtype=np.uint64),
        np.arange(4, dtype=np.uint64)[::2],
        None,
    ],
)
def test_invalid_buffers(mapping, invalid):
    valid = np.array([1, 2], dtype=np.uint64)
    with pytest.raises((ValueError, TypeError)):
        ArrayHashMap(invalid, valid)
    with pytest.raises((ValueError, TypeError)):
        ArrayHashMap(valid, invalid)
    with pytest.raises((ValueError, TypeError)):
        mapping[invalid]


@pytest.mark.parametrize("key", [-1, 2**64])
def test_scalar_outside_uint64_range(mapping, key):
    with pytest.raises(OverflowError):
        mapping[key]
