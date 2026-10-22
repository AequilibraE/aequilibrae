import os
from abc import abstractmethod
from collections.abc import Iterator, MutableMapping
from copy import deepcopy
from typing import Any, Callable, Generic, Literal, Mapping, Self, Sequence, TypeAlias, TypeVar, overload
from uuid import uuid4

import h5py
import numpy as np
import openmatrix as omx
import pandas as pd
import scipy.sparse
from numpy.typing import ArrayLike, DTypeLike
from openmatrix import File as OMXFile
from scipy.sparse import sparray

OMX_SPECIAL_ATTRS = {"OMX_VERSION", "OMX_CREATED_WITH", "SHAPE", "TITLE"}
SPARSE_SPECIAL_ATTRS = {"FORMAT"}


def _copy_file_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    return deepcopy({key: value for key, value in metadata.items() if key not in OMX_SPECIAL_ATTRS})


def _cast_index(value: np.ndarray, dataset: h5py.Dataset) -> np.ndarray:
    if dataset.shape != value.shape:
        raise ValueError(f"index shape may not change, expected {dataset.shape}, got {value.shape}")
    return value.astype(dataset.dtype, casting="same_value")


def _write_index(file: OMXFile, name: str, value: np.ndarray) -> None:
    dataset = file.lookup.get(name)
    if dataset is None:
        file.lookup.create_dataset(name, data=value)
    else:
        # Keep the existing dataset and reject casts that would change zone IDs.
        value = _cast_index(value, dataset)
        dataset[...] = value


def _metadata_equal(left: Any, right: Any) -> bool:
    if isinstance(left, Mapping) or isinstance(right, Mapping):
        return (
            isinstance(left, Mapping)
            and isinstance(right, Mapping)
            and left.keys() == right.keys()
            and all(_metadata_equal(left[key], right[key]) for key in left)
        )

    left, right = np.asarray(left), np.asarray(right)
    equal_nan = left.dtype.kind in "fc" and right.dtype.kind in "fc"
    return np.array_equal(left, right, equal_nan=equal_nan)


def _matrix_values_equal(left: np.ndarray | sparray, right: np.ndarray | sparray) -> bool:
    if not isinstance(left, sparray) and not isinstance(right, sparray):
        return np.array_equal(left, right, equal_nan=True)
    if left.shape != right.shape:
        return False

    # Compare stored values without making dense copies. Ignore explicit zeros and duplicates.
    left, right = scipy.sparse.csr_array(left, copy=True), scipy.sparse.csr_array(right, copy=True)
    for value in (left, right):
        value.sum_duplicates()
        value.eliminate_zeros()
        value.sort_indices()
    return (
        np.array_equal(left.indptr, right.indptr)
        and np.array_equal(left.indices, right.indices)
        and np.array_equal(left.data, right.data, equal_nan=True)
    )


def _lookup_to_key_error(mapping, key):
    try:
        return mapping[key]
    except LookupError as e:
        # OMX raises the parent class LookupError instead of the more
        # appropriate KeyError.
        raise KeyError(key) from e


def _validate_name_available(file: OMXFile, name: str, group: str) -> None:
    other_group = "sparse" if group == "data" else "data"
    matrices = file["/"].get(other_group, {})
    if name in matrices:
        raise ValueError(f"matrix {name!r} already exists in the {other_group} group")


def _query_matrices(
    matrices: Mapping[str, Any],
    subset: Sequence[str | dict[str, Any]],
    matrix_metadata: Callable[[str], Mapping[str, Any]],
    *,
    other_matrices: Mapping[str, Any] | None = None,
    other_store: str | None = None,
) -> dict[str, Any]:
    mats = {}
    for key in subset:
        if isinstance(key, str):
            _validate_matrix_key(key)

            try:
                mats[key] = _lookup_to_key_error(matrices, key)
            except KeyError:
                if other_matrices is not None and key in other_matrices:
                    assert other_store is not None
                    raise KeyError(
                        f"{key!r} is missing from this collection, but exists in the other storage group. "
                        f"Use {other_store} to read it."
                    ) from None
                raise

        elif isinstance(key, dict):
            for name in matrices.keys():
                attrs = matrix_metadata(name)

                if all(attr in attrs and _metadata_equal(attrs[attr], value) for attr, value in key.items()):
                    _validate_matrix_key(name)
                    mats[name] = _lookup_to_key_error(matrices, name)
        else:
            raise TypeError("subset entries must be matrix names or attribute queries")

    return mats


_IndexKey: TypeAlias = str | tuple[str] | tuple[str, Any] | tuple[str, Any, Any]
_ArrayT = TypeVar("_ArrayT", bound=np.ndarray | sparray)


def _validate_name(name: object, kind: str) -> None:
    if not isinstance(name, str):
        raise TypeError(f"{kind} name must be a string")
    if not name or name in (".", "..") or any(char in name for char in ("/", "\\", "\0")):
        raise ValueError(f"{kind} name must be a non-empty name, not a path")


def _validate_matrix_key(key: object) -> None:
    _validate_name(key, "matrix")


def _zone_ids(value: ArrayLike) -> np.ndarray:
    # Keep integer IDs exact and booleans visible when reading Python sequences.
    value = np.asarray(value, dtype=object if isinstance(value, (list, tuple)) else None)
    if value.ndim != 1:
        raise ValueError(f"zone IDs must have one dimension, got {value.ndim}")

    if value.size:
        if value.dtype.kind == "O":
            if any(isinstance(zone, (bool, np.bool_)) or not isinstance(zone, (int, np.integer)) for zone in value):
                raise TypeError("zone IDs must be integers, not booleans")
        elif value.dtype.kind not in "iu":
            raise TypeError("zone IDs must be integers, not booleans")

        if int(value.min()) < 0 or int(value.max()) > np.iinfo(np.uint64).max:
            raise ValueError("zone IDs must be between 0 and 2**64 - 1")

    return np.array(value, dtype=np.uint64, copy=True, order="C")


def _validate_matrix_dtype(dtype: np.dtype) -> None:
    if dtype.kind not in "biufc":
        raise TypeError(f"matrix values must have an integer, float, bool or complex dtype, got {dtype}")


def _matrix_values(value: ArrayLike, shape: tuple[int, int]) -> np.ndarray:
    value = np.asarray(value, order="C")
    _validate_matrix_dtype(value.dtype)

    if value.shape != shape:
        raise ValueError(f"got bad shape, expected {shape}, got {value.shape}")
    return value


def _write_matrix(file: OMXFile, name: str, value: np.ndarray, attributes: Mapping[str, Any]) -> None:
    dataset = file.data.get(name)
    if dataset is not None and dataset.dtype == value.dtype:
        # Keep the dataset, including its dtype and compression, when replacing values.
        dataset.attrs.update(attributes)
        dataset[...] = value
        return

    # Keep destination attributes and apply source attributes before replacing the dataset.
    attrs = dict(dataset.attrs) if dataset is not None else {}
    attrs.update(attributes)
    temporary_name = f"__aequilibrae_{uuid4().hex}"

    try:
        file.create_matrix(temporary_name, obj=value, attrs=attrs)
        if dataset is not None:
            del file.data[name]

        file.data.move(temporary_name, name)
    finally:
        if temporary_name in file.data:
            del file.data[temporary_name]


def _sparse_values(value: sparray, shape: tuple[int, int]) -> sparray:
    if not isinstance(value, sparray):
        raise TypeError("matrix values must be SciPy sparse arrays")

    _validate_matrix_dtype(value.dtype)
    if value.shape != shape:
        raise ValueError(f"got bad shape, expected {shape}, got {value.shape}")

    return value


def _read_sparse_matrix(group: h5py.Group, expected_shape: tuple[int, int]) -> scipy.sparse.csr_array:
    if group.attrs.get("FORMAT") != "csr":
        raise ValueError(f"unsupported sparse format for matrix {group.name!r}, expected 'csr'")

    _validate_matrix_dtype(group["data"].dtype)

    value = scipy.sparse.csr_array(
        (group["data"][...], group["col_index"][...], group["row_index"][...]), shape=expected_shape
    )
    value.check_format(full_check=True)
    return value


def _write_sparse_matrix(file: OMXFile, name: str, value: sparray, attributes: Mapping[str, Any]) -> None:
    value = value.tocsr()
    value.check_format(full_check=True)
    # Work from the root group because OMX redirects file lookups to /data.
    sparse = file["/"].require_group("sparse")
    group = sparse.get(name)

    attrs = (
        {key: value for key, value in group.attrs.items() if key not in SPARSE_SPECIAL_ATTRS}
        if group is not None
        else {}
    )
    attrs.update({key: value for key, value in attributes.items() if key not in SPARSE_SPECIAL_ATTRS})
    attrs["FORMAT"] = "csr"
    temporary_name = f"__aequilibrae_{uuid4().hex}"

    try:
        replacement = sparse.create_group(temporary_name)
        replacement.attrs.update(attrs)
        replacement.create_dataset("row_index", data=value.indptr)
        replacement.create_dataset("col_index", data=value.indices)
        replacement.create_dataset("data", data=value.data)
        if group is not None:
            del sparse[name]
        sparse.move(temporary_name, name)
    finally:
        if temporary_name in sparse:
            del sparse[temporary_name]


class _MatrixIndexer(Generic[_ArrayT]):
    def __init__(self, matrix: "_Matrix[_ArrayT]", *, zones: bool) -> None:
        self._matrix = matrix
        self._zones = zones

    def _translate_zones(self, selection: Any) -> Any:
        if selection is None or selection is Ellipsis:
            return selection

        positions = self._matrix._zone_positions
        if isinstance(selection, slice):
            start = None if selection.start is None else positions[selection.start]
            stop = None if selection.stop is None else positions[selection.stop]

            if stop is not None:
                if selection.step is not None and selection.step < 0:
                    # A negative stop would count from the end, not include the first zone.
                    stop = stop - 1 if stop else None
                else:
                    stop += 1
            return slice(start, stop, selection.step)

        labels = np.asarray(selection)
        if labels.dtype.kind == "b":  # boolean selection
            return selection

        if isinstance(selection, (list, tuple)):
            # Keep integer labels exact when a list mixes small and large zone IDs.
            labels = np.asarray(selection, dtype=object)

        if labels.ndim == 0:
            return positions[labels.item()]

        return np.fromiter((positions[zone] for zone in labels.flat), dtype=np.intp, count=labels.size).reshape(
            labels.shape
        )

    def _parse_key(self, key: _IndexKey) -> tuple[str, tuple[Any, ...]]:
        args = key if isinstance(key, tuple) else (key,)
        if not 1 <= len(args) <= 3:
            raise TypeError("expected a matrix name, a name and row, or a name, row and column")

        name = args[0]
        _validate_matrix_key(name)

        selection = args[1:]
        if self._zones:
            selection = tuple(self._translate_zones(value) for value in selection)

        return name, selection

    # Normal lookup
    @overload
    def __getitem__(self, key: str | tuple[str]) -> _ArrayT: ...

    # Dataset slice (partial read)
    @overload
    def __getitem__(self, key: tuple[str, Any] | tuple[str, Any, Any]) -> _ArrayT | np.ndarray | np.generic: ...

    def __getitem__(self, key: _IndexKey) -> _ArrayT | np.ndarray | np.generic:
        name, selection = self._parse_key(key)
        if not selection:
            return self._matrix[name]

        return self._matrix._read_selection(name, selection)

    # Dataset slice (partial write), allows in-place modification without writing the whole matrix
    def __setitem__(self, key: _IndexKey, value: ArrayLike | sparray) -> None:
        name, selection = self._parse_key(key)
        if not selection:
            self._matrix[name] = value
        else:
            self._matrix._write_selection(name, selection, value)


class _Matrix(MutableMapping[str, _ArrayT]):
    def __init__(self, index: ArrayLike, index_name: str) -> None:
        self._backend: MutableMapping[str, Any]
        self._metadata: MutableMapping[str, Any]
        self._index: np.ndarray
        assert hasattr(self, "_backend") and hasattr(self, "_metadata")

        _validate_name(index_name, "index")
        self.__index_name = index_name
        self._index = self._validate_index(index)  # Validate without writing a store's index
        self._zone_positions = {int(zone): position for position, zone in enumerate(self._index)}

    def _read_selection(self, name: str, selection: tuple[Any, ...]) -> Any:
        dataset = _lookup_to_key_error(self._backend, name)
        return dataset[selection]

    def _write_selection(self, name: str, selection: tuple[Any, ...], value: Any) -> None:
        dataset = _lookup_to_key_error(self._backend, name)
        dataset[selection] = value

    def _query_subset(self, subset: Sequence[str | dict[str, Any]]) -> dict[str, Any]:
        return _query_matrices(self._backend, subset, self.matrix_metadata)

    def totals(self) -> dict[str, Any]:
        """Return each named matrix's total."""
        return {name: values.sum().item() for name, values in self.items()}

    def save_as_omx(
        self,
        path: str | os.PathLike,
        mode: Literal["w", "a", "r+", "w-", "x"] = "x",
        *,
        subset: Sequence[str | dict[str, Any]] | None = None,
        **kwargs,
    ) -> None:
        """
        Save named matrices, their attributes, the zone index and file
        metadata to OMX.

        By default, this saves all matrices to a new file and raises an error
        if the file already exists. Pass ``subset`` to select matrices by name
        or by their attributes. A query matches matrices with all the given
        attributes. Each matrix is saved once, even if it matches several
        entries.

        When updating a file, matrices outside the subset are left unchanged.
        Attributes provided here replace matching attributes in the file.
        Other attributes are kept. The zone mapping and file metadata are
        always written, apart from attributes managed by OMX.

        :Arguments:
            **path** (:obj:`str` or :obj:`os.PathLike`): Destination OMX file.

            **mode** (:obj:`str`, *Optional*): Use ``"x"`` or ``"w-"`` to
                create a new file, ``"w"`` to replace or create a file,
                ``"r+"`` to update an existing file, or ``"a"`` to update or
                create a file. Defaults to ``"x"``.

            **subset** (sequence of :obj:`str` or :obj:`dict`, *Optional*):
                Matrix names or attribute queries to save. Defaults to all
                matrices. If nothing matches, only the zone mapping and file
                metadata are written.

            **kwargs**: Additional keyword arguments passed to
              ``openmatrix.open_file``.

        >>> from pathlib import Path
        >>> from tempfile import TemporaryDirectory
        >>> demand = from_dict({"car": [[0, 12], [10, 0]]}, index=[101, 205])
        >>> demand.matrix_metadata("car")["mode"] = "car"
        >>> with TemporaryDirectory() as directory:
        ...     path = Path(directory) / "demand.omx"
        ...     demand.save_as_omx(path, subset=[{"mode": "car"}])
        ...     loaded = from_file(path)
        >>> loaded == demand
        True
        """
        if mode not in ("w", "a", "r+", "w-", "x"):
            raise ValueError("save_as_omx only supports modes 'w', 'a', 'r+', 'w-' and 'x'")

        if subset is not None:
            # Query backend arrays or dataset handles without loading store values.
            keys = set(self._query_subset(subset))
        else:
            keys = set(self)

        if (shape := kwargs.get("shape")) is not None and tuple(shape) != self.shape:
            raise ValueError(f"destination matrix shape does not match, expected {self.shape}, got {shape}")

        # OMX will try to save the OMX_SPECIAL_ATTRS when opened with write permissions, so we first open with read only
        # to validate an existing file before updating it with 'a' or 'r+'.
        if mode in ("a", "r+") and os.path.exists(path):
            with omx.open_file(path, mode="r") as file:
                if (shape := file.shape()) is not None and shape != self.shape:
                    raise ValueError(f"destination matrix shape does not match, expected {self.shape}, got {shape}")

                dataset = file.lookup.get(self.index_name)
                if dataset is not None:
                    _cast_index(self.index, dataset)

        # Sparse matrices do not pass through OMX's create_matrix, which normally sets the file shape.
        kwargs["shape"] = self.shape
        with omx.open_file(path, mode=mode, **kwargs) as file:
            # Save the index without replacing its existing dtype or attributes.
            _write_index(file, self.index_name, self.index)

            group = "sparse" if isinstance(self, AequilibraeSparseMatrix) else "data"
            for name in keys:
                _validate_name_available(file, name, group)
                # Keep destination attributes and apply source attributes when writing the matrix.
                self._write_omx_matrix(file, name)

            for key, value in self.metadata.items():
                # These attributes are managed by OMX so we shouldn't try to write them
                if key in OMX_SPECIAL_ATTRS:
                    continue

                file.attrs[key] = value

    @abstractmethod
    def _write_omx_matrix(self, file: OMXFile, name: str) -> None:
        pass

    @abstractmethod
    def __setitem__(self, key: str, value: Any) -> None:
        pass

    @abstractmethod
    def subset(
        self,
        subset: Sequence[str | dict[str, Any]],
    ) -> "_Matrix[_ArrayT]":
        """
        Copy selected matrices into a new in-memory matrix object.

        :Arguments:
            **subset** (sequence of :obj:`str` or :obj:`dict`): Matrix names or
                attribute queries.  A query matches matrices with all the given
                attributes. Each matching matrix is included once.
        """
        pass

    def _set_index(self, value: ArrayLike) -> None:
        self._index = self._validate_index(value)

    def _validate_index(self, value: ArrayLike) -> np.ndarray:
        value = _zone_ids(value)

        # hasattr because we might not have set it yet as this is used in the __init__ function
        if hasattr(self, "_index") and len(self._index) != len(value):
            raise ValueError(f"matrix and index shape may not change, expected {len(self._index)}, got {len(value)}")

        if len(np.unique(value)) != len(value):
            raise ValueError("found duplicate values in the index, the index must be unique")

        return value

    @property
    def index(self) -> np.ndarray:
        """
        Zone IDs in row and column order, returned as a read-only uint64 array.

        IDs must be unique integers between ``0`` and ``2**64 - 1``. To relabel
        zones, assign a new index of the same length. This does not move any
        matrix values.

        Changing a store's index requires write access. The mapping keeps its
        existing dtype and attributes.

        >>> demand = from_dict({"car": [[0, 12], [10, 0]]}, index=[101, 205])
        >>> demand.index.tolist()
        [101, 205]
        >>> demand.index.flags.writeable
        False
        >>> demand.index = [1001, 1002]
        >>> int(demand.mat["car", 1001, 1002])
        12
        """
        view = self._index.view()
        view.setflags(write=False)
        return view

    @index.setter
    def index(self, value: ArrayLike) -> None:
        """
        Relabel zones without changing the matrix shape or value order.

        :Arguments:
            **value** (array-like): One-dimensional sequence of unique integer
                zone IDs from ``0`` to ``2**64 - 1``, with the same length as
                the current index.
        """
        self._set_index(value)
        self._zone_positions = {int(zone): position for position, zone in enumerate(self._index)}

    @property
    def mat(self) -> _MatrixIndexer[_ArrayT]:
        """
        Read or change matrix values using zone IDs.

        Use ``mat[name]`` for a whole matrix, ``mat[name, row]`` for rows, or
        ``mat[name, row, column]`` to select both rows and columns. Slices
        follow the stored zone order and include the stop ID. An unknown zone
        ID raises ``KeyError``.

        Zone IDs are converted to positions before indexing. In-memory, dense
        matrices use NumPy selections while sparse matrices use SciPy
        selections. Dense stores support the selections allowed by h5py.
        Sparse stores only support whole-matrix access. Writing to a selection
        keeps the matrix's dtype. Values are broadcast and converted according
        to the array or dataset's rules.

        >>> demand = from_dict({"car": [[0, 12], [10, 0]]}, index=[101, 205])
        >>> demand.mat["car", 101:205].tolist()
        [[0, 12], [10, 0]]
        >>> demand.mat["car", 101, 205] = 42
        >>> int(demand.mat["car", 101, 205])
        42
        """
        return _MatrixIndexer(self, zones=True)

    @property
    def imat(self) -> _MatrixIndexer[_ArrayT]:
        """
        Read or change matrix values using zero-based positions.

        Use ``imat[name]`` for a whole matrix, ``imat[name, row]`` for rows, or
        ``imat[name, row, column]`` to select both rows and columns. Slices
        exclude the stop position.

        Zone IDs are converted to positions before indexing. In-memory, dense
        matrices use NumPy selections while sparse matrices use SciPy
        selections. Dense stores support the selections allowed by h5py.
        Sparse stores only support whole-matrix access. Writing to a selection
        keeps the matrix's dtype. Values are broadcast and converted according
        to the array or dataset's rules.

        >>> demand = from_dict({"car": [[0, 12], [10, 0]]}, index=[101, 205])
        >>> demand.imat["car", :, 1].tolist()
        [12, 0]
        >>> demand.imat["car", 0, :] = [1, 2]
        >>> demand.imat["car", :1].tolist()
        [[1, 2]]
        """
        return _MatrixIndexer(self, zones=False)

    @property
    def index_name(self) -> str:
        """
        Name of the zone mapping used when reading and writing OMX files.

        >>> AequilibraeMatrix(index=[101, 205]).index_name
        'main_index'
        >>> AequilibraeMatrix(index=[101, 205], index_name="zones").index_name
        'zones'
        """
        return self.__index_name

    @property
    def metadata(self) -> MutableMapping[str, Any]:
        """
        File-level attributes that can be read and updated by name.

        These are stored in a dictionary for in-memory matrices and in h5py
        attributes for stores. Assign a new value to an attribute to update it.

        Attributes managed by OMX are visible on a store but are ignored when
        loading, copying, saving or comparing file metadata.

        >>> demand = AequilibraeMatrix(index=[101, 205])
        >>> demand.metadata["description"] = "Morning demand"
        >>> demand.metadata["description"]
        'Morning demand'
        """
        return self._metadata

    @abstractmethod
    def matrix_metadata(self, name: str) -> MutableMapping[str, Any]:
        """
        Return mutable attributes for an existing named matrix.

        :Arguments:
            **name** (:obj:`str`): Name of an existing matrix.
        """
        pass

    @property
    def shape(self) -> tuple[int, int]:
        """
        Shape of each matrix, determined by the number of zone IDs.

        >>> demand = AequilibraeMatrix(index=[101, 205, 309])
        >>> demand.shape
        (3, 3)
        >>> len(demand)
        0
        """
        length = len(self._index)
        return (length, length)

    def __delitem__(self, key: str) -> None:
        _validate_matrix_key(key)
        del self._backend[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._backend.keys())

    def __len__(self) -> int:
        return len(self._backend)

    def __contains__(self, key: object) -> bool:
        _validate_matrix_key(key)
        return key in self._backend.keys()

    def __eq__(self, other: object) -> bool:
        """
        Compare matrix names and values, zone IDs, the index name and metadata.

        NaNs in the same positions are considered equal. Matrix insertion order
        and numeric dtypes do not affect the result. File attributes managed by
        OMX are ignored. Comparing a store reads its matrix values from disk.

        :Arguments:
            **other** (:obj:`object`): Object to compare with this matrix collection.

        >>> left = from_dict({"car": [[0, float("nan")], [10, 0]]}, index=[101, 205])
        >>> right = left.subset(["car"])
        >>> left == right
        True
        >>> right.matrix_metadata("car")["mode"] = "car"
        >>> left == right
        False
        """
        if not isinstance(other, _Matrix):
            return NotImplemented
        if self is other:
            return True
        if (
            self.index_name != other.index_name
            or not np.array_equal(self.index, other.index)
            or self.keys() != other.keys()
        ):
            return False

        # Ignore file attributes managed by OMX, as loading and copying do.
        left_metadata = {key: value for key, value in self.metadata.items() if key not in OMX_SPECIAL_ATTRS}
        right_metadata = {key: value for key, value in other.metadata.items() if key not in OMX_SPECIAL_ATTRS}
        if not _metadata_equal(left_metadata, right_metadata):
            return False

        return all(
            _metadata_equal(self.matrix_metadata(name), other.matrix_metadata(name))
            and _matrix_values_equal(self[name], other[name])
            for name in self
        )

    def clear(self) -> None:
        """
        Remove all named matrices and their attributes.

        The zone index and file-level metadata are kept.

        >>> demand = from_dict({"car": [[0]]}, index=[101], metadata={"year": 2030})
        >>> demand.clear()
        >>> list(demand), demand.index.tolist(), demand.metadata["year"]
        ([], [101], 2030)
        """
        for key in list(self):
            del self[key]

    def setdefault(self, key: str, default: ArrayLike | sparray | None = None) -> _ArrayT:
        """
        Return a named matrix, adding it from ``default`` if it is missing.

        An existing matrix is returned unchanged. If the name is missing,
        ``default`` must contain a full square matrix. ``None`` cannot be used
        to create a matrix.

        As with a normal lookup, the result is a view for a dense in-memory
        matrix, the stored array for a sparse matrix and a copy for a store.

        :Arguments:
            **key** (:obj:`str`): Matrix name to look up or insert.

            **default** (array-like, *Optional*): Full square matrix to insert
                if the name is missing.  Defaults to ``None``, which is only
                valid when the matrix already exists.

        >>> demand = AequilibraeMatrix(index=[101])
        >>> demand.setdefault("car", [[12]]).tolist()
        [[12]]
        >>> demand.setdefault("car", [[99]]).tolist()
        [[12]]
        """
        if key not in self:
            self[key] = default
        return self[key]


class AequilibraeMatrix(_Matrix[np.ndarray]):
    """
    A dictionary-like collection of square matrices held in memory.

    All matrices use the same zone index for rows and columns.

    Assigning a whole matrix copies the values and keeps their dtype. Reading a
    matrix or taking a basic slice returns a view of its data. Use ``.copy()``
    when you need an independent array. You can also use dictionary operations
    such as ``keys()``, ``items()`` and ``update()``.

    Matrix and index names must be non-empty strings, not paths. They cannot be
    ``"."`` or ``".."``, or contain a slash, backslash or null character.

    >>> demand = AequilibraeMatrix(index=[101, 205], index_name="zones")
    >>> demand["car"] = [[0, 12], [10, 0]]
    >>> demand["bus"] = [[0, 2], [2, 0]]
    >>> demand["total"] = demand["car"] + demand["bus"]
    >>> list(demand)
    ['car', 'bus', 'total']
    >>> demand["total"].tolist()
    [[0, 14], [12, 0]]
    >>> del demand["total"]
    >>> len(demand), demand.shape
    (2, (2, 2))
    """

    def __init__(
        self,
        index: ArrayLike,
        index_name: str = "main_index",
    ) -> None:
        """
        Create an empty in-memory collection with a shared zone index.

        :Arguments:
            **index** (array-like): One-dimensional sequence of unique integer
                zone IDs from ``0`` to ``2**64 - 1`` in row and column
                order.

            **index_name** (:obj:`str`, *Optional*): Name of the zone
                mapping. Defaults to ``"main_index"``.
        """
        self._backend: dict[str, np.ndarray] = {}
        self._metadata: dict[str, Any] = {}
        self._matrix_metadata: dict[str, dict[str, Any]] = {}
        super().__init__(index=index, index_name=index_name)

    @classmethod
    def _from_shared_arrays(
        cls,
        data: Mapping[str, np.ndarray],
        index: ArrayLike,
        index_name: str = "main_index",
    ) -> Self:
        """
        Use supplied arrays without copying.

        Used by DenseMatrix to share its buffer with an in-memory matrix.  The
        arrays keep the buffer alive. Normal matrix assignment still copies
        values.

        :Arguments:
            **data** (mapping of :obj:`str` to :obj:`numpy.ndarray`): Named
                square arrays to share.  Each array must match the index length
                and have a supported numeric or boolean dtype.

            **index** (array-like): Ordered zone IDs shared by rows and
                columns.

            **index_name** (:obj:`str`, *Optional*): Name of the zone
                mapping. Defaults to ``"main_index"``.
        """
        self = cls(index=index, index_name=index_name)
        for name, values in data.items():
            _validate_matrix_key(name)
            if not isinstance(values, np.ndarray):
                raise TypeError("shared matrix values must be NumPy arrays")

            _validate_matrix_dtype(values.dtype)
            if values.shape != self.shape:
                raise ValueError(f"got bad shape, expected {self.shape}, got {values.shape}")

            self._backend[name] = values
            self._matrix_metadata[name] = {}
        return self

    @classmethod
    def from_omx(
        cls,
        path: str | os.PathLike,
        index_name: str = "main_index",
        subset: Sequence[str | dict[str, Any]] | None = None,
        **kwargs,
    ) -> Self:
        """
        Load matrices from an OMX file into memory.

        The file must contain the mapping named by ``index_name``. By default,
        all matrices are loaded. Pass ``subset`` to choose matrices by name or
        by their attributes. A query matches matrices with all the given
        attributes, and each matching matrix is loaded once.

        Matrix values, the zone index and metadata are copied. Changing the
        loaded object does not change the file.

        :Arguments:
            **path** (:obj:`str` or :obj:`os.PathLike`): Existing OMX file to
                load.

            **index_name** (:obj:`str`, *Optional*): Existing zone mapping to
                read. Defaults to ``"main_index"``.

            **subset** (sequence of :obj:`str` or :obj:`dict`, *Optional*):
                Matrix names or attribute queries to load. Defaults to all
                matrices.

            **kwargs**: Additional keyword arguments passed to the matrix
                constructor.

        >>> from pathlib import Path
        >>> from tempfile import TemporaryDirectory
        >>> demand = from_dict({"car": [[12]], "bus": [[2]]}, index=[101])
        >>> demand.matrix_metadata("car")["mode"] = "car"
        >>> with TemporaryDirectory() as directory:
        ...     path = Path(directory) / "demand.omx"
        ...     demand.save_as_omx(path)
        ...     working = AequilibraeMatrix.from_omx(path, subset=[{"mode": "car"}])
        >>> list(working)
        ['car']
        >>> working["car"].tolist()
        [[12]]
        """
        _validate_name(index_name, "index")
        with omx.open_file(path, mode="r") as file:
            index = file.lookup.get(index_name)
            if index is None:
                raise ValueError(f"index not found, expected group at '/lookup/{index_name}'")

            if subset is None:
                subset = file.list_matrices()

            datasets = _query_matrices(
                file.data,
                subset,
                lambda name: file.data[name].attrs,
                other_matrices=file["/"].get("sparse", {}),
                other_store="SparseMatrixStore or AequilibraeSparseMatrix.from_omx()",
            )

            self = cls(index=index, index_name=index_name, **kwargs)

            # We should ignore the OMX special attributes
            attrs = _copy_file_metadata(file.attrs)

            self.update({key: np.asarray(dataset) for key, dataset in datasets.items()})
            for key, dataset in datasets.items():
                self.matrix_metadata(key).update(deepcopy(dict(dataset.attrs)))
            self.metadata.update(attrs)
            return self

    def matrix_metadata(self, name: str) -> MutableMapping[str, Any]:
        """
        Return the mutable attribute dictionary for an existing named
        matrix.

        Assign a new value to an attribute to update it. Attributes are kept
        when you replace a matrix and can be used to select matrices with
        ``subset()``.

        :Arguments:
            **name** (:obj:`str`): Name of an existing matrix.

        >>> demand = from_dict({"car": [[12]]}, index=[101])
        >>> demand.matrix_metadata("car")["mode"] = "car"
        >>> demand["car"] = [[15.0]]
        >>> demand.matrix_metadata("car")["mode"]
        'car'
        """
        _validate_matrix_key(name)
        return self._matrix_metadata[name]

    def __delitem__(self, key: str) -> None:
        super().__delitem__(key)
        del self._matrix_metadata[key]

    # Normal write
    def __setitem__(self, key: str, value: ArrayLike) -> None:
        """
        Copy a full square array into a named matrix, preserving its dtype.

        :Arguments:
            **key** (:obj:`str`): Matrix name to create or replace.

            **value** (array-like): Values matching the collection's square
                shape, with an integer, float, boolean or complex dtype.

        >>> values = np.array([[12]])
        >>> demand = AequilibraeMatrix(index=[101])
        >>> demand["car"] = values
        >>> values[0, 0] = 99
        >>> demand["car"].tolist()
        [[12]]
        """
        _validate_matrix_key(key)
        value = _matrix_values(value, self.shape)
        self._backend[key] = np.asarray(value, copy=True, order="C")
        self._matrix_metadata.setdefault(key, {})

    def _write_omx_matrix(self, file: OMXFile, name: str) -> None:
        _write_matrix(file, name, self[name], self.matrix_metadata(name))

    # Normal lookup
    def __getitem__(self, key: str) -> np.ndarray:
        """
        Return a view of a named matrix.

        Changing values in the returned array also changes the matrix.

        :Arguments:
            **key** (:obj:`str`): Name of an existing matrix.

        >>> demand = from_dict({"car": [[12]]}, index=[101])
        >>> values = demand["car"]
        >>> values[0, 0] = 42
        >>> demand["car"].tolist()
        [[42]]
        """
        _validate_matrix_key(key)
        return self._backend[key].view()

    def subset(
        self,
        subset: Sequence[str | dict[str, Any]],
    ) -> Self:
        """
        Copy selected matrices into a new in-memory matrix object.

        Select matrices by name or by their attributes. A query matches
        matrices with all the given attributes. Each matrix is included once,
        even if it matches more than one entry.

        The arrays, zone index and metadata are copied. Changes to the result
        do not affect the original object.

        :Arguments:
            **subset** (sequence of :obj:`str` or :obj:`dict`): Matrix names or
              attribute queries to copy.

        >>> demand = from_dict({"car": [[12]], "bus": [[2]]}, index=[101])
        >>> demand.matrix_metadata("car")["mode"] = "car"
        >>> working = demand.subset([{"mode": "car"}, "car"])
        >>> list(working)
        ['car']
        >>> working["car"] *= 2
        >>> demand["car"].tolist(), working["car"].tolist()
        ([[12]], [[24]])
        """
        mats = _query_matrices(self, subset, self.matrix_metadata)

        other = self.__class__(index=self.index, index_name=self.index_name)
        other.update(mats)
        for key in mats:
            other.matrix_metadata(key).update(deepcopy(dict(self.matrix_metadata(key))))
        other.metadata.update(_copy_file_metadata(self.metadata))

        return other


class AequilibraeSparseMatrix(_Matrix[sparray]):
    """
    A dictionary-like collection of square SciPy sparse arrays held in memory.

    All matrices use the same zone index for rows and columns. Assigning a
    matrix copies the array and keeps its sparse format and dtype. Reading a
    whole matrix returns the stored array, so changes to it change the matrix.
    Use ``.copy()`` when you need an independent array. Selections through
    ``mat`` and ``imat`` follow the sparse format's own SciPy indexing rules.

    Saving writes each array in the CSR format.  This extends the OMX format
    with a ``/sparse/<name>`` group for each matrix. The group has a
    ``row_index`` (CSR row pointers), ``col_index`` (column positions) and
    ``data`` datasets. The group attribute ``FORMAT`` is set to ``"csr"``. Zone
    mappings, file metadata, and shape use the standard OMX attributes.

    When reading, only ``/sparse`` is read from, dense matrices in ``/data``
    are ignored. Use :class:`MatrixStore` or
    :method:`AequilibraeMatrix.from_omx()` instead.

    Matrix and index names follow the same rules as :class:`AequilibraeMatrix`.

    >>> from scipy.sparse import coo_array
    >>> demand = AequilibraeSparseMatrix(index=[101, 205])
    >>> demand["car"] = coo_array([[0, 12], [10, 0]])
    >>> demand["car"].format
    'coo'
    >>> demand["car"].toarray().tolist()
    [[0, 12], [10, 0]]
    """

    def __init__(self, index: ArrayLike, index_name: str = "main_index") -> None:
        """
        Create an empty sparse collection with a shared zone index.

        :Arguments:
            **index** (array-like): Unique integer zone IDs from ``0`` to
                ``2**64 - 1`` in row and column order.

            **index_name** (:obj:`str`, *Optional*): Name of the zone mapping.
                Defaults to ``"main_index"``.
        """
        self._backend: dict[str, sparray] = {}
        self._metadata: dict[str, Any] = {}
        self._matrix_metadata: dict[str, dict[str, Any]] = {}

        super().__init__(index=index, index_name=index_name)

    @classmethod
    def _from_shared_arrays(
        cls,
        data: Mapping[str, sparray],
        index: ArrayLike,
        index_name: str = "main_index",
    ) -> Self:
        """
        Use supplied sparse arrays without copying.

        :Arguments:
            **data** (mapping of :obj:`str` to :obj:`scipy.sparse.sparray`):
                Named square sparse arrays to share. Each array must match
                the index length and have a supported numeric or boolean dtype.

            **index** (array-like): Ordered zone IDs shared by rows and columns.

            **index_name** (:obj:`str`, *Optional*): Name of the zone mapping.
                Defaults to ``"main_index"``.
        """
        self = cls(index=index, index_name=index_name)

        for name, value in data.items():
            _validate_matrix_key(name)
            if not isinstance(value, sparray):
                raise TypeError("shared matrix values must be SciPy sparse arrays")

            _validate_matrix_dtype(value.dtype)
            if value.shape != self.shape:
                raise ValueError(f"got bad shape, expected {self.shape}, got {value.shape}")

            self._backend[name] = value
            self._matrix_metadata[name] = {}
        return self

    def __setitem__(self, key: str, value: sparray) -> None:
        """Copy a square sparse array, keeping its format, dtype and metadata."""
        _validate_matrix_key(key)
        self._backend[key] = _sparse_values(value, self.shape).copy()
        self._matrix_metadata.setdefault(key, {})

    def __getitem__(self, key: str) -> sparray:
        """Return the stored sparse array. Changing it also changes the matrix."""
        _validate_matrix_key(key)
        return self._backend[key]

    def __delitem__(self, key: str) -> None:
        super().__delitem__(key)
        del self._matrix_metadata[key]

    def matrix_metadata(self, name: str) -> MutableMapping[str, Any]:
        """Return mutable attributes for an existing named sparse matrix."""
        _validate_matrix_key(name)
        return self._matrix_metadata[name]

    def subset(self, subset: Sequence[str | dict[str, Any]]) -> Self:
        """
        Copy selected sparse matrices, their zone index and metadata.

        Select by matrix name or attribute query. Each matching matrix is
        included once. Changes to the result do not change the original.
        Sparse formats and dtypes are kept.
        """
        mats = _query_matrices(self, subset, self.matrix_metadata)
        other = self.__class__(index=self.index, index_name=self.index_name)

        other.update(mats)
        for key in mats:
            other.matrix_metadata(key).update(deepcopy(dict(self.matrix_metadata(key))))
        other.metadata.update(_copy_file_metadata(self.metadata))

        return other

    def _write_omx_matrix(self, file: OMXFile, name: str) -> None:
        _write_sparse_matrix(file, name, self[name], self.matrix_metadata(name))

    @classmethod
    def from_omx(
        cls,
        path: str | os.PathLike,
        index_name: str = "main_index",
        subset: Sequence[str | dict[str, Any]] | None = None,
        **kwargs,
    ) -> Self:
        """
        Load CSR arrays from the OMX extension group ``/sparse`` into memory.

        Dense matrices in ``/data`` are ignored. Values, the zone index and
        metadata are copied. Changes to the result do not change the file.

        :Arguments:
            **path** (:obj:`str` or :obj:`os.PathLike`): Existing OMX file.

            **index_name** (:obj:`str`, *Optional*): Existing zone mapping to
                read. Defaults to ``"main_index"``.

            **subset** (sequence of :obj:`str` or :obj:`dict`, *Optional*):
                Matrix names or attribute queries. Defaults to all sparse
                matrices. Each matching matrix is loaded once.

            **kwargs**: Additional keyword arguments passed to the constructor.
        """
        _validate_name(index_name, "index")
        with omx.open_file(path, mode="r") as file:
            index = file.lookup.get(index_name)
            if index is None:
                raise ValueError(f"index not found, expected group at '/lookup/{index_name}'")

            sparse = file["/"].get("sparse", {})

            if subset is None:
                subset = list(sparse)

            groups = _query_matrices(
                sparse,
                subset,
                lambda name: sparse[name].attrs,
                other_matrices=file.data,
                other_store="MatrixStore or AequilibraeMatrix.from_omx()",
            )
            self = cls(index=index, index_name=index_name, **kwargs)

            for name, group in groups.items():
                self[name] = _read_sparse_matrix(group, self.shape)
                self.matrix_metadata(name).update(
                    deepcopy({key: value for key, value in group.attrs.items() if key not in SPARSE_SPECIAL_ATTRS})
                )

            self.metadata.update(_copy_file_metadata(file.attrs))
            return self


class _MatrixStore(_Matrix[_ArrayT]):
    """Share file access and zone mappings between dense and sparse stores."""

    _storage_group: str
    _other_group: str
    _other_store: str

    def __init__(
        self,
        path: str | os.PathLike,
        index: ArrayLike | None = None,
        index_name: str = "main_index",
        mode: Literal["r", "w", "a", "r+", "w-", "x"] = "r",
        **kwargs,
    ) -> None:
        """
        Open or create an OMX-backed matrix collection.

        :Arguments:

            **path** (:obj:`str` or :obj:`os.PathLike`): OMX file to open or
                create.

            **index** (array-like, *Optional*): One-dimensional sequence of
                unique integer zone IDs from ``0`` to ``2**64 - 1`` in row and
                column order. Its length must match the size of any existing
                matrices. You must use a writable mode when providing an
                index. If omitted, the mapping named by ``index_name`` is read
                without changing it.

            **index_name** (:obj:`str`, *Optional*): Zone mapping to read or
                write. Defaults to ``"main_index"``.  Must be a non-empty name,
                not a path.

            **mode** (:obj:`str`, *Optional*): ``"r"`` for read-only access,
                ``"r+"`` to update an existing file, ``"x"`` or ``"w-"`` to
                create a new file, ``"w"`` to replace or create a file, or
                ``"a"`` to update or create a file. Defaults to ``"r"``.

            **kwargs**: Additional keyword arguments passed to
                ``openmatrix.open_file``.
        """
        _validate_name(index_name, "index")
        if mode == "r" and index is not None:
            raise ValueError("cannot supply an index when opening a read-only matrix store")

        file = omx.open_file(
            path,
            mode=mode,
            **kwargs,
        )
        try:
            supplied_index = index is not None
            if not supplied_index:
                index = file.lookup.get(index_name)
                if index is None:
                    raise ValueError(
                        f"index not found, expected group at '/lookup/{index_name}'. "
                        "Either provide an index directly when opening a writable store, "
                        "or an index name of one that exists in the OMX file"
                    )

            self._file = file

            # the group may not exist if a non-omx file is opened in read only.
            self._backend = file["/"].get(self._storage_group, {})

            self._metadata: h5py.AttributeManager = file.attrs

            super().__init__(index=index, index_name=index_name)
            if (shape := file.shape()) is not None and shape != self.shape:
                raise ValueError(f"matrix shape does not match index, expected {self.shape}, got {shape}")

            if overlap := file["/"].get("data", {}).keys() & file["/"].get("sparse", {}).keys():
                raise ValueError(f"matrix names {sorted(overlap)} exist in both the data and sparse groups")

            if supplied_index:
                self.index = index
            # Sparse writes do not use OMX's create_matrix, which normally sets the shape.
            if file.shape() is None and file.mode != "r":
                file.attrs["SHAPE"] = self.shape
        except:
            file.close()
            raise

    def _query_subset(self, subset: Sequence[str | dict[str, Any]]) -> dict[str, Any]:
        return _query_matrices(
            self._backend,
            subset,
            self.matrix_metadata,
            other_matrices=self._file["/"].get(self._other_group, {}),
            other_store=self._other_store,
        )

    @property
    def omx(self) -> OMXFile:
        """
        Underlying open OMX file for advanced operations.

        >>> from pathlib import Path
        >>> from tempfile import TemporaryDirectory
        >>> with TemporaryDirectory() as directory:
        ...     with MatrixStore(Path(directory) / "demand.omx", index=[101], mode="x") as store:
        ...         store["car"] = [[12]]
        ...         print(store.omx.list_matrices())
        ['car']
        """
        return self._file

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_args: Any, **_kwargs: Any) -> None:
        self.close()

    def close(self) -> None:
        """
        Close the OMX file.

        This is called automatically when leaving a ``with`` block.

        >>> from pathlib import Path
        >>> from tempfile import TemporaryDirectory
        >>> with TemporaryDirectory() as directory:
        ...     store = MatrixStore(Path(directory) / "demand.omx", index=[101], mode="x")
        ...     store["car"] = [[12]]
        ...     working = store.subset(["car"])
        ...     store.close()
        >>> working["car"].tolist()
        [[12]]
        """
        if hasattr(self, "_file"):
            self.omx.close()

    def __del__(self) -> None:
        self.close()

    def _set_index(self, value: ArrayLike) -> None:
        if self._file.mode == "r":
            raise PermissionError("cannot change the index of a read-only matrix store")

        value = self._validate_index(value)
        _write_index(self._file, self.index_name, value)
        self._index = value


class MatrixStore(_MatrixStore[np.ndarray]):
    """
    A dictionary-like collection of square matrices backed by an OMX file.

    Open an existing file by choosing a zone mapping with ``index_name``.
    To create a file or relabel its zones, supply ``index`` and a writable
    mode. Stores are read-only by default.

    Each read loads values from disk into a new array. Changing that array does
    not update the file. To write changes, assign a whole matrix by name or use
    ``mat`` or ``imat`` to update part of a matrix.

    Use ``subset()`` to copy selected matrices into memory for calculations.
    Keep the store open while using it. Using a ``with`` block is recommended.

    >>> from pathlib import Path
    >>> from tempfile import TemporaryDirectory
    >>> with TemporaryDirectory() as directory:
    ...     path = Path(directory) / "demand.omx"
    ...     with MatrixStore(path, index=[101, 205], mode="x") as store:
    ...         store["car"] = [[0, 12], [10, 0]]
    ...         values = store["car"]
    ...         values[0, 1] = 99
    ...         print(int(store.imat["car", 0, 1]))
    ...         store.mat["car", 101, 205] = 42
    ...         working = store.subset(["car"])
    12
    >>> working["car"].tolist()
    [[0, 42], [10, 0]]
    """

    _storage_group = "data"
    _other_group = "sparse"
    _other_store = "SparseMatrixStore"

    def matrix_metadata(self, name: str) -> h5py.AttributeManager:
        """
        Return h5py attributes for an existing matrix dataset.

        Use a writable store to change attributes, and assign each new value as
        a whole. Replacing a matrix with a different dtype creates a new
        dataset. Call this method again afterwards to get its attributes.

        :Arguments:
            **name** (:obj:`str`): Name of an existing matrix dataset.

        >>> from pathlib import Path
        >>> from tempfile import TemporaryDirectory
        >>> with TemporaryDirectory() as directory:
        ...     with MatrixStore(Path(directory) / "demand.omx", index=[101], mode="x") as store:
        ...         store["car"] = [[12]]
        ...         store.matrix_metadata("car")["mode"] = "car"
        ...         print(store.matrix_metadata("car")["mode"])
        car
        """
        _validate_matrix_key(name)
        return self._backend[name].attrs

    def __setitem__(self, key: str, value: ArrayLike) -> None:
        """
        Write a full square matrix using its supplied dtype, keeping attributes.
        :Arguments:

            **key** (:obj:`str`): Matrix name to create or replace.

            **value** (array-like): Values matching the collection's square
                shape, with an integer, float, boolean or complex dtype.

        >>> from pathlib import Path
        >>> from tempfile import TemporaryDirectory
        >>> with TemporaryDirectory() as directory:
        ...     with MatrixStore(Path(directory) / "demand.omx", index=[101], mode="x") as store:
        ...         store["car"] = [[12]]
        ...         store["car"] = [[12.5]]
        ...         print(store["car"].tolist())
        [[12.5]]
        """
        _validate_matrix_key(key)
        _validate_name_available(self._file, key, self._storage_group)
        value = _matrix_values(value, self.shape)

        # Keep the dataset's existing attributes, even when its dtype changes.
        _write_matrix(self._file, key, value, {})

    def _write_omx_matrix(self, file: OMXFile, name: str) -> None:
        _write_matrix(file, name, self[name], self.matrix_metadata(name))

    # Normal lookup
    def __getitem__(self, key: str) -> np.ndarray:
        """
        Read a whole matrix into an independent array, without changing the file.

        :Arguments:
            **key** (:obj:`str`): Name of an existing matrix dataset.

        >>> from pathlib import Path
        >>> from tempfile import TemporaryDirectory
        >>> with TemporaryDirectory() as directory:
        ...     with MatrixStore(Path(directory) / "demand.omx", index=[101], mode="x") as store:
        ...         store["car"] = [[12]]
        ...         values = store["car"]
        ...         values[0, 0] = 99
        ...         print(store["car"].tolist())
        [[12]]
        """
        _validate_matrix_key(key)
        value = _lookup_to_key_error(self._backend, key)
        return np.asarray(value)

    def subset(
        self,
        subset: Sequence[str | dict[str, Any]],
    ) -> AequilibraeMatrix:
        """
        Copy selected matrices from the file into memory.

        Select matrices by name or by their attributes. A query matches
        matrices with all the given attributes. Each matrix is included once,
        even if it matches more than one entry.

        The zone index and metadata are also copied. You can keep using the
        result after the store closes, and changes to it do not affect the
        file.

        :Arguments:
            **subset** (sequence of :obj:`str` or :obj:`dict`): Matrix names or
              attribute queries to load.

        >>> from pathlib import Path
        >>> from tempfile import TemporaryDirectory
        >>> with TemporaryDirectory() as directory:
        ...     with MatrixStore(Path(directory) / "demand.omx", index=[101], mode="x") as store:
        ...         store["car"] = [[12]]
        ...         store["bus"] = [[2]]
        ...         store.matrix_metadata("car")["mode"] = "car"
        ...         working = store.subset([{"mode": "car"}])
        >>> list(working), working["car"].tolist()
        (['car'], [[12]])
        """
        datasets = self._query_subset(subset)

        other = AequilibraeMatrix(index=self.index, index_name=self.index_name)
        other.update({key: np.asarray(dataset) for key, dataset in datasets.items()})

        for key, dataset in datasets.items():
            other.matrix_metadata(key).update(deepcopy(dict(dataset.attrs)))
        other.metadata.update(_copy_file_metadata(self.metadata))

        return other


class SparseMatrixStore(_MatrixStore[sparray]):
    """
    A dictionary-like collection of sparse matrices backed by an OMX file.

    Matrices use the CSR extension in ``/sparse``. All matrices share the
    selected zone mapping.

    Each read returns an independent CSR array. To save changes, assign the
    whole matrix back to the store. Partial reads and writes through ``mat`` or
    ``imat`` are not supported, even for selections covering all values.
    Whole-matrix access through ``mat[name]`` and ``imat[name]`` is supported.

    Stores are read-only by default. Supply an index and a writable mode to
    create a file. Keep the store open while using it, preferably with a
    ``with`` block. ``subset()`` copies selected matrices into memory.

    >>> from pathlib import Path
    >>> from tempfile import TemporaryDirectory
    >>> from scipy.sparse import csr_array
    >>> with TemporaryDirectory() as directory:
    ...     with SparseMatrixStore(Path(directory) / "demand.omx", index=[101, 205], mode="x") as store:
    ...         store["car"] = csr_array([[0, 12], [10, 0]])
    ...         values = store["car"]
    ...         values *= 2
    ...         store["car"] = values
    ...         working = store.subset(["car"])
    >>> working["car"].toarray().tolist()
    [[0, 24], [20, 0]]
    """

    _storage_group = "sparse"
    _other_group = "data"
    _other_store = "MatrixStore"

    def __getitem__(self, key: str) -> scipy.sparse.csr_array:
        """Read a whole matrix into an independent CSR array."""
        _validate_matrix_key(key)
        return _read_sparse_matrix(_lookup_to_key_error(self._backend, key), self.shape)

    def __setitem__(self, key: str, value: sparray) -> None:
        """Write a whole sparse array as CSR, keeping its dtype and stored attributes."""
        _validate_matrix_key(key)
        _validate_name_available(self._file, key, self._storage_group)
        value = _sparse_values(value, self.shape)

        _write_sparse_matrix(self._file, key, value, {})
        self._backend = self._file["/"]["sparse"]

    def _read_selection(self, name: str, selection: tuple[Any, ...]) -> Any:
        raise NotImplementedError("SparseMatrixStore only supports whole-matrix reads. Use 'store[name]' instead")

    def _write_selection(self, name: str, selection: tuple[Any, ...], value: Any) -> None:
        raise NotImplementedError(
            "SparseMatrixStore only supports whole-matrix writes. Use 'store[name] = value' instead"
        )

    def matrix_metadata(self, name: str) -> MutableMapping[str, Any]:
        """
        Return the mutable HDF5 attributes for this matrix. The sparse storage
        attribute ``FORMAT`` is ignored when writing.

        Assign each attribute as a whole to update it. Call this method again
        after replacing a matrix, as replacement creates a new storage group.
        """
        _validate_matrix_key(name)
        return _lookup_to_key_error(self._backend, name).attrs

    def _write_omx_matrix(self, file: OMXFile, name: str) -> None:
        _write_sparse_matrix(file, name, self[name], self.matrix_metadata(name))

    def subset(self, subset: Sequence[str | dict[str, Any]]) -> AequilibraeSparseMatrix:
        """
        Copy selected matrices, their zone index and metadata into memory.

        Select by matrix names or attribute queries. Each matrix is copied
        once. The result can be used after the store closes. Missing names
        raise KeyError, with a hint if the name exists in the dense group.
        """
        groups = self._query_subset(subset)
        other = AequilibraeSparseMatrix(index=self.index, index_name=self.index_name)
        for name, group in groups.items():
            other[name] = _read_sparse_matrix(group, self.shape)
            other.matrix_metadata(name).update(deepcopy(dict(self.matrix_metadata(name))))
        other.metadata.update(_copy_file_metadata(self.metadata))
        return other


@overload
def from_file(
    path: str | os.PathLike,
    mode: None = None,
    *,
    index_name: str | None = None,
    subset: Sequence[str | dict[str, Any]] | None = None,
    store: Literal[False] = False,
    sparse: Literal[False] = False,
) -> AequilibraeMatrix: ...


@overload
def from_file(
    path: str | os.PathLike,
    mode: Literal["r", "r+"] | None = None,
    *,
    index_name: str | None = None,
    subset: None = None,
    store: Literal[True],
    sparse: Literal[False] = False,
) -> MatrixStore: ...


@overload
def from_file(
    path: str | os.PathLike,
    mode: None = None,
    *,
    index_name: str | None = None,
    subset: Sequence[str | dict[str, Any]] | None = None,
    store: Literal[False] = False,
    sparse: Literal[True],
) -> AequilibraeSparseMatrix: ...


@overload
def from_file(
    path: str | os.PathLike,
    mode: Literal["r", "r+"] | None = None,
    *,
    index_name: str | None = None,
    subset: None = None,
    store: Literal[True],
    sparse: Literal[True],
) -> SparseMatrixStore: ...


def from_file(
    path: str | os.PathLike,
    mode: Literal["r", "r+"] | None = None,
    *,
    index_name: str | None = None,
    subset: Sequence[str | dict[str, Any]] | None = None,
    store: bool = False,
    sparse: bool = False,
) -> AequilibraeMatrix | MatrixStore | AequilibraeSparseMatrix | SparseMatrixStore:
    """
    Load an OMX file as an in-memory matrix or an on-disk store.

    By default, this copies all matrices and metadata into an in-memory
    :class:`AequilibraeMatrix`. Pass ``subset`` to load only the matrices you
    need.

    Use ``store=True`` to work with the file through a :class:`MatrixStore`
    instead. The store opens read-only unless you pass ``mode="r+"``.  Stores
    do not accept ``subset``. Use their ``subset()`` method to copy selected
    matrices into memory.

    Pass ``sparse=True`` to read only the CSR extension in ``/sparse`` instead
    of dense matrices in ``/data``. This returns an AequilibraeSparseMatrix,
    or a SparseMatrixStore when ``store=True``.

    If the file has only one zone mapping, it is selected automatically. If
    there are several, ``main_index`` is used when available. Otherwise, choose
    a mapping with ``index_name``.

    :Arguments:
        **path** (:obj:`str` or :obj:`os.PathLike`): Existing OMX file to open.

        **mode** (:obj:`str`, *Optional*): Store access mode, ``"r"`` or
            ``"r+"``. Defaults to ``None``, which opens stores read-only. Must
            be omitted for in-memory loading.

        **index_name** (:obj:`str`, *Optional*): Zone mapping to read. Defaults
            to automatic selection of the only mapping, or ``"main_index"``
            when several exist. Required if neither choice is available.

        **subset** (sequence of :obj:`str` or :obj:`dict`, *Optional*): Matrix
            names or attribute queries to load into memory. Defaults to all
            matrices. Cannot be combined with ``store=True``.

        **store** (:obj:`bool`, *Optional*): Return an open
            :class:`MatrixStore` instead of an in-memory
            :class:`AequilibraeMatrix`. Defaults to ``False``.

        **sparse** (:obj:`bool`, *Optional*): Read sparse matrices instead of
            dense matrices. Defaults to ``False``.

    >>> from pathlib import Path
    >>> from tempfile import TemporaryDirectory
    >>> demand = from_dict({"car": [[0, 12], [10, 0]]}, index=[101, 205], index_name="zones")
    >>> with TemporaryDirectory() as directory:
    ...     path = Path(directory) / "demand.omx"
    ...     demand.save_as_omx(path)
    ...     loaded = from_file(path, subset=["car"])
    ...     with from_file(path, store=True, mode="r+") as store:
    ...         store.mat["car", 101, 205] = 42
    ...         working = store.subset(["car"])
    >>> loaded["car"].tolist()
    [[0, 12], [10, 0]]
    >>> working["car"].tolist()
    [[0, 42], [10, 0]]
    """
    if store:
        if subset is not None:
            raise ValueError("store=True and subset may not be provided at once")
        if mode is not None and mode not in ("r", "r+"):
            raise ValueError("from_file stores only support modes 'r' and 'r+' (file must exist)")
    elif mode is not None:
        raise ValueError("an in-memory matrix cannot have a mode")

    if index_name is None:
        with omx.open_file(path, mode="r") as file:
            index_names = list(file.lookup.keys())

        if len(index_names) == 1:
            index_name = index_names[0]
        elif "main_index" in index_names:
            index_name = "main_index"
        elif not index_names:
            raise ValueError(f"No zone index found in OMX matrix file {str(path)!r}")
        else:
            raise ValueError(f"Multiple zone indices found in OMX matrix file {str(path)!r}; please specify index_name")

    if store:
        store_type = SparseMatrixStore if sparse else MatrixStore
        return store_type(path=path, index_name=index_name, mode=mode if mode is not None else "r")
    else:
        matrix_type = AequilibraeSparseMatrix if sparse else AequilibraeMatrix
        return matrix_type.from_omx(path=path, index_name=index_name, subset=subset)


@overload
def from_dict(
    data: Mapping[str, ArrayLike],
    index: ArrayLike,
    *,
    path: None = None,
    mode: None = None,
    index_name: str = "main_index",
    metadata: Mapping[str, Any] | None = None,
    store: Literal[False] = False,
) -> AequilibraeMatrix: ...


@overload
def from_dict(
    data: Mapping[str, ArrayLike],
    index: ArrayLike,
    *,
    path: str | os.PathLike,
    mode: Literal["w", "a", "r+", "w-", "x"] | None = None,
    index_name: str = "main_index",
    metadata: Mapping[str, Any] | None = None,
    store: Literal[True],
) -> MatrixStore: ...


def from_dict(
    data: Mapping[str, ArrayLike],
    index: ArrayLike,
    *,
    path: str | os.PathLike | None = None,
    mode: Literal["w", "a", "r+", "w-", "x"] | None = None,
    index_name: str = "main_index",
    metadata: Mapping[str, Any] | None = None,
    store: bool = False,
) -> AequilibraeMatrix | MatrixStore:
    """
    Create named square matrices from a mapping of names to array-like values.

    The result is an in-memory :class:`AequilibraeMatrix` unless you pass
    ``store=True``. Arrays are copied with their original dtypes. File metadata
    is also copied, apart from attributes managed by OMX.

    To create a :class:`MatrixStore`, pass ``store=True`` and a file path.  The
    default mode is ``"x"``, which raises an error if the file already
    exists. Close the store when finished, or use a ``with`` block. In-memory
    matrices do not accept ``path`` or ``mode``.

    :Arguments:

        **data** (mapping of :obj:`str` to array-like): Named square matrices
            matching the index length.  Values must have integer, float,
            boolean or complex dtypes.

        **index** (array-like): One-dimensional sequence of unique integer zone
            IDs from ``0`` to ``2**64 - 1`` in row and column order.

        **path** (:obj:`str` or :obj:`os.PathLike`, *Optional*): Destination
            OMX file. Required when ``store=True`` and not accepted
            otherwise. Defaults to ``None``.

        **mode** (:obj:`str`, *Optional*): Writable store mode: ``"x"``,
            ``"w-"``, ``"w"``, ``"r+"`` or ``"a"``. Defaults to ``None``, which
            uses ``"x"`` for stores. Must be omitted for in-memory matrices.

        **index_name** (:obj:`str`, *Optional*): Zone mapping name. Defaults to
          ``"main_index"``.

        **metadata** (:obj:`Mapping`, *Optional*): File-level attributes to
            copy, excluding OMX-managed attributes. Defaults to ``None``.

        **store** (:obj:`bool`, *Optional*): Write to an OMX file and return an
            open :class:`MatrixStore` instead of an in-memory
            :class:`AequilibraeMatrix`. Defaults to ``False``.

    >>> demand = from_dict(
    ...     {"car": [[0, 12], [10, 0]], "bus": [[0, 2], [2, 0]]},
    ...     index=[101, 205], index_name="zones", metadata={"year": 2030},
    ... )
    >>> demand["car"].tolist(), demand.metadata["year"]
    ([[0, 12], [10, 0]], 2030)
    >>> from pathlib import Path
    >>> from tempfile import TemporaryDirectory
    >>> with TemporaryDirectory() as directory:
    ...     with from_dict(
    ...         {"car": demand["car"]}, index=demand.index,
    ...         path=Path(directory) / "demand.omx", store=True,
    ...     ) as store:
    ...         print(store["car"].tolist())
    [[0, 12], [10, 0]]
    """
    mat: AequilibraeMatrix | MatrixStore
    if store:
        if path is None:
            raise ValueError("a matrix store must have a path")

        if mode is not None and mode not in ("w", "a", "r+", "w-", "x"):
            raise ValueError("from_dict stores only support modes 'w', 'a', 'r+', 'w-' and 'x'")

        # Check names, shapes and dtypes before opening a file or changing its index.
        validated_index = AequilibraeMatrix(index=index, index_name=index_name).index
        prepared = {}
        for name, value in data.items():
            _validate_matrix_key(name)
            prepared[name] = _matrix_values(value, (len(validated_index), len(validated_index)))

        data = prepared

        mat = MatrixStore(
            path=path,
            index=validated_index,
            index_name=index_name,
            mode=mode if mode is not None else "x",
        )
    else:
        if path is not None:
            raise ValueError("an in-memory matrix cannot have a path")

        if mode is not None:
            raise ValueError("an in-memory matrix cannot have a mode")
        mat = AequilibraeMatrix(index=index, index_name=index_name)

    try:
        for name, value in data.items():
            mat[name] = value
        if metadata is not None:
            mat.metadata.update(_copy_file_metadata(metadata))

    except Exception:
        if isinstance(mat, MatrixStore):
            mat.close()
        raise

    return mat


def from_df(
    df: pd.DataFrame,
    row: str,
    col: str,
    subset: Sequence[str] | None = None,
    *,
    index: ArrayLike | None = None,
    index_name: str = "main_index",
    fill_value: float = 0,
    dtype: DTypeLike = float,
) -> AequilibraeMatrix:
    """
    Create dense in-memory matrices from an origin-destination DataFrame.

    Choose the origin and destination columns with ``row`` and ``col``.  Each
    remaining column becomes a named matrix, or you can choose columns with
    ``subset``. Values for repeated origin-destination pairs are added
    together. Missing pairs are filled with ``fill_value``.

    Pass ``index`` to set the zone order and include zones with no trips.  If
    omitted, the index contains the sorted unique origin and destination
    IDs. All IDs must be valid integer zone IDs and appear in the index.

    :Arguments:
        **df** (:obj:`pandas.DataFrame`): Origin-destination records with zone
          IDs and matrix value columns.

        **row** (:obj:`str`): Column containing origin zone IDs.

        **col** (:obj:`str`): Column containing destination zone IDs.

        **subset** (sequence of :obj:`str`, *Optional*): Value columns to
            convert to named matrices.  Defaults to all columns except ``row``
            and ``col``.

        **index** (array-like, *Optional*): Unique integer zone IDs shared by
            rows and columns.  Must include all origin and destination
            IDs. Defaults to their sorted union.

        **index_name** (:obj:`str`, *Optional*): Zone mapping name. Defaults to
          ``"main_index"``.

        **fill_value** (:obj:`float`, *Optional*): Value for pairs without
          records. Defaults to ``0``.

        **dtype** (NumPy dtype-like, *Optional*): Dtype used for all output
          matrices. Defaults to :obj:`float`.

    >>> trips = pd.DataFrame({"origin": [101, 101, 309], "destination": [205, 205, 101], "car": [2, 3, 7]})
    >>> demand = from_df(
    ...     trips, row="origin", col="destination", subset=["car"],
    ...     index=[101, 205, 309], index_name="zones", fill_value=-1,
    ... )
    >>> demand["car"].tolist()
    [[-1.0, 5.0, -1.0], [-1.0, -1.0, -1.0], [7.0, -1.0, -1.0]]
    >>> from_df(trips, row="origin", col="destination").index.tolist()
    [101, 205, 309]
    """
    dtype = np.dtype(dtype)
    if subset is None:
        subset = [name for name in df.columns if name not in (row, col)]

    row_ids = _zone_ids(df[row].to_numpy())
    col_ids = _zone_ids(df[col].to_numpy())
    if index is None:
        index = np.union1d(row_ids, col_ids)

    mat = AequilibraeMatrix(index=index, index_name=index_name)
    matrix_index = pd.Index(mat.index)
    rows = matrix_index.get_indexer(row_ids)
    cols = matrix_index.get_indexer(col_ids)
    if np.any(rows == -1) or np.any(cols == -1):
        raise ValueError("found row or column values that are not in the index")

    for name in subset:
        values = df[name].to_numpy(dtype=dtype, copy=False)
        values_matrix: np.ndarray = np.full(mat.shape, fill_value, dtype=dtype)
        values_matrix[rows, cols] = 0
        np.add.at(values_matrix, (rows, cols), values)
        mat[name] = values_matrix

    return mat


@overload
def from_scipy(
    data: Mapping[str, sparray],
    index: ArrayLike,
    *,
    path: None = None,
    mode: None = None,
    index_name: str = "main_index",
    metadata: Mapping[str, Any] | None = None,
    store: Literal[False] = False,
) -> AequilibraeSparseMatrix: ...


@overload
def from_scipy(
    data: Mapping[str, sparray],
    index: ArrayLike,
    *,
    path: str | os.PathLike,
    mode: Literal["w", "a", "r+", "w-", "x"] | None = None,
    index_name: str = "main_index",
    metadata: Mapping[str, Any] | None = None,
    store: Literal[True],
) -> SparseMatrixStore: ...


def from_scipy(
    data: Mapping[str, sparray],
    index: ArrayLike,
    *,
    path: str | os.PathLike | None = None,
    mode: Literal["w", "a", "r+", "w-", "x"] | None = None,
    index_name: str = "main_index",
    metadata: Mapping[str, Any] | None = None,
    store: bool = False,
) -> AequilibraeSparseMatrix | SparseMatrixStore:
    """
    Copy named SciPy sparse arrays to sparse in-memory matrices or a store.

    Each in-memory sparse array keeps its format and dtype. Its shape must
    match the number of zones in ``index``, which sets the row and column order.

    Pass ``store=True`` and a file path to write CSR arrays to a SparseMatrixStore,
    keeping their dtypes. The default mode is ``"x"``, which raises an error if
    the file exists. Close the store when finished, or use a ``with`` block.
    In-memory matrices do not accept ``path`` or ``mode``.

    :Arguments:

        **data** (mapping of :obj:`str` to :obj:`scipy.sparse.sparray`): Named
            square sparse arrays.  Each array's shape must match the index
            length.

        **index** (array-like): One-dimensional sequence of unique integer zone
            IDs from ``0`` to ``2**64 - 1`` in row and column order.

        **index_name** (:obj:`str`, *Optional*): Zone mapping name. Defaults to
          ``"main_index"``.

        **path** (:obj:`str` or :obj:`os.PathLike`, *Optional*): Destination OMX
            file. Required when ``store=True`` and not accepted otherwise.

        **mode** (:obj:`str`, *Optional*): Writable store mode: ``"x"``, ``"w-"``,
            ``"w"``, ``"r+"`` or ``"a"``. Defaults to ``"x"`` for stores.
            Must be omitted for in-memory matrices.

        **metadata** (:obj:`Mapping`, *Optional*): File-level attributes to
            copy, excluding OMX-managed attributes. Defaults to ``None``.

        **store** (:obj:`bool`, *Optional*): Return an open SparseMatrixStore
            instead of in-memory matrices. Defaults to ``False``.

    >>> from scipy.sparse import csr_array
    >>> demand = from_scipy(
    ...     {"car": csr_array([[0, 12], [10, 0]])},
    ...     index=[101, 205], index_name="zones",
    ... )
    >>> demand["car"].toarray().tolist()
    [[0, 12], [10, 0]]
    >>> demand.index.tolist()
    [101, 205]
    """
    mat: AequilibraeSparseMatrix | SparseMatrixStore
    if store:
        if path is None:
            raise ValueError("a matrix store must have a path")
        if mode is not None and mode not in ("w", "a", "r+", "w-", "x"):
            raise ValueError("from_scipy stores only support modes 'w', 'a', 'r+', 'w-' and 'x'")

        # Check names, shapes and dtypes before opening a file or changing its index.
        validated_index = AequilibraeSparseMatrix(index=index, index_name=index_name).index
        prepared = {}
        for name, value in data.items():
            _validate_matrix_key(name)
            value = _sparse_values(value, (len(validated_index), len(validated_index))).tocsr()
            value.check_format(full_check=True)
            prepared[name] = value

        data = prepared
        mat = SparseMatrixStore(
            path=path,
            index=validated_index,
            index_name=index_name,
            mode=mode if mode is not None else "x",
        )
    else:
        if path is not None:
            raise ValueError("an in-memory matrix cannot have a path")
        if mode is not None:
            raise ValueError("an in-memory matrix cannot have a mode")
        mat = AequilibraeSparseMatrix(index=index, index_name=index_name)

    try:
        mat.update(data)
        if metadata is not None:
            mat.metadata.update(_copy_file_metadata(metadata))
    except Exception:
        if isinstance(mat, SparseMatrixStore):
            mat.close()
        raise
    return mat
